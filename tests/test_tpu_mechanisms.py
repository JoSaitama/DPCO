"""Offline acceptance for the opt-in real TPU observer, not TPU certification."""

import argparse
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from src.config import MechanismConfig, MethodConfig, ReferenceConfig, ResultsConfig, load_config
from src.data.batch import classification_batch
from src.data.diagnostic_batches import TrainingAuditBatch
from src.methods import create_method
from src.probes.fixed_training_bank import FixedTrainingBank, tree_fingerprint
from src.probes.mechanism_config import overrides
from src.probes.tpu_mechanisms import TPUMechanisms, SCHEMAS
from src.training import TrainState, Trainer
from src.training.experiment import ExperimentRunner
from src.training.launch import add_overrides, resolve_config
from src.training.recording import SCHEMAS as OLD_SCHEMAS
from src.training.updates import SGDUpdate
from tests.test_observation import ROOT, fixture_config, fixture_components, read_rows


def components(root, *, name="sage", reference="error", enabled=True):
    config = replace(fixture_config(root), method=MethodConfig(name), reference=ReferenceConfig(reference),
        mechanism=MechanismConfig(enabled=enabled, steps=(0, 1, 3), moment_steps=(0,), oracle_steps=(0,),
            absolute_lrs=(0., .01, .02), lr_scales=(1.,), windows=((0, 4),), bank_batches=1, norm_matched=True))
    trainer, source, evaluation, state, _ = fixture_components(config)
    trainer = trainer.with_method(create_method(name, reference_name=reference, gradient_backend="jvp"))
    observer = TPUMechanisms(trainer.task, trainer.method, config.mechanism, reference_name=reference,
        trajectory_method=name, bank_batches=(TrainingAuditBatch("fixed", source.batches[0], (0,1,2)),),
        fixed_buffers=state.model_state)
    if not enabled:
        config = replace(config, mechanism=MechanismConfig())
    return config, trainer, source, evaluation, state, observer


def rows_for(observer, trainer, before, batch, eta=.01):
    after, _ = trainer.compile_step()(before, batch, jax.random.key(int(before.step)))
    rows = observer.run(before, after, batch, learning_rate=eta, rng_key=jax.random.key(int(before.step)),
        context={"batch_id": "real_source", "sample_indices": [0,1,2]}, cumulative_eta_before=0.)
    for name, row in rows:
        assert set(row) <= set(SCHEMAS[name]), (name, set(row)-set(SCHEMAS[name]))
    return after, rows


class TPUMechanismTests(unittest.TestCase):
    def test_same_state_references_identity_absolute_lr_and_actual_endpoints(self):
        for method, reference in (("sgd", "loss"), ("sage", "error"), ("sage", "loss")):
            with self.subTest(method=method, reference=reference), TemporaryDirectory() as tmp:
                config, trainer, source, _, before, observer = components(tmp, name=method, reference=reference)
                with patch("jax.jacrev", side_effect=AssertionError("Dense Jacobian forbidden")):
                    _, rows = rows_for(observer, trainer, before, next(source))
                t1 = [r for s,r in rows if s == "mechanism_t1"]
                self.assertEqual({r["reference"] for r in t1}, {"error", "loss", "exact_bank_reference_oracle"})
                for row in t1:
                    self.assertEqual(row["trajectory_method"], "sgd" if method=="sgd" else "sage_e" if reference=="error" else "sage_l")
                    self.assertAlmostEqual(row["reference_delta"], row["delta_formula"], delta=2e-6)
                    self.assertEqual(row["negative_count"], row["flip_count"])
                pairs = [r for s,r in rows if s == "mechanism_pairs"]
                for row in pairs:
                    if row["lr_kind"] == "absolute":
                        self.assertIn(row["probe_lr"], config.mechanism.absolute_lrs)
                    if row["candidate_kind"] == "actual_pair":
                        self.assertEqual(row["probe_lr"], .01)
                        self.assertEqual(row["normalization"], "original")
                        if method == "sage":
                            self.assertEqual(row["reference"], reference)
                        else:
                            self.assertEqual(row["actual_endpoint"], "sgd")
                    if row["probe_lr"] == 0 and row.get("defined"):
                        self.assertIsNone(row["first_order_residual"])
                        self.assertIsNone(row["difference_over_lr"])
                for row in [r for s,r in rows if s == "mechanism_moments"]:
                    self.assertEqual(row["estimator"], "exact_finite_bank")
                    self.assertIsNone(row["standard_error"])
                    self.assertLess(row["mean_gd_h_l2_error"], 2e-6)
                oracle = next(r for s,r in rows if s == "mechanism_transfer" and r["reference"] == "exact_bank_reference_oracle")
                self.assertAlmostEqual(oracle["target_lambda"], oracle["reference_delta"], places=6)

    def test_independent_directions_match_dense_original_oracle_and_valid_B(self):
        _, trainer, source, _, state, observer = components("unused")
        for batch in source.batches:
            both = observer._sources(state.params, batch, state.model_state)
            for name, stats in both.items():
                dense = create_method("sage", reference_name=name, gradient_backend="audit")
                direction = dense.compute_direction(trainer.task, state.params, batch).direction
                for a,b in zip(jax.tree_util.tree_leaves(direction), jax.tree_util.tree_leaves(stats.sage)):
                    np.testing.assert_allclose(a, b, atol=2e-6)

    def test_frozen_buffers_target_id_and_read_only_inputs(self):
        from src.tasks.classification import ClassificationTask
        def model(p, x, s):
            z = (x-jnp.mean(x, axis=0)) @ p["w"] + s["offset"]
            return jnp.concatenate((jnp.zeros_like(z), z), axis=1), {"offset": s["offset"]+.7}
        task = ClassificationTask(model, 2)
        method = create_method("sage", reference_name="error", gradient_backend="jvp")
        trainer = Trainer(task, method, SGDUpdate(.01))
        batch = classification_batch([[1.],[-2.],[3.],[-1.]], [1,0,0,1])
        before = TrainState({"w": jnp.array([[.3]])}, {"offset": jnp.array([.2])})
        cfg = MechanismConfig(enabled=True, bank_batches=1)
        observer = TPUMechanisms(task, method, cfg, reference_name="error", trajectory_method="sage",
            bank_batches=(TrainingAuditBatch("global", batch),), fixed_buffers=before.model_state)
        target = observer.target
        original_hash = tree_fingerprint(target.buffers)
        f1, h = target.evaluate(before.params, observer.ad, observer._call)
        after, _ = trainer.step(before, batch)
        f2, _ = target.evaluate(before.params, observer.ad, observer._call)
        self.assertEqual(float(f1), float(f2))
        self.assertEqual(tree_fingerprint(target.buffers), original_hash)
        self.assertNotEqual(tree_fingerprint(after.model_state), original_hash)
        other = FixedTrainingBank(target.batches, after.model_state, signature="explicit_task_v1")
        self.assertNotEqual(target.identity, other.identity)
        result = task.evaluate(before.params, batch, before.model_state)
        self.assertAlmostEqual(float(f1), float(jnp.mean(result.losses)), places=6)

    def test_on_off_trajectory_private_cursors_windows_and_exact_resume(self):
        final_states = []
        for enabled in (False, True):
            with TemporaryDirectory() as tmp:
                c, trainer, source, evaluation, initial, observer = components(tmp, enabled=enabled)
                runner = ExperimentRunner(c, trainer, source, evaluation, base_dir=ROOT,
                    train_sample_count=5, metadata={}, mechanism=observer, console=False, run_name="run")
                final = runner.run(initial)
                final_states.append((final, source.index,
                    [(r["train_ce"],r["train_top1"]) for r in read_rows(runner.paths.files/"train_steps.csv")]))
                if not enabled:
                    self.assertEqual(set(runner.recorder.row_counts), set(OLD_SCHEMAS))
                    self.assertFalse(list(runner.paths.files.glob("mechanism_*.csv")))
                    self.assertEqual(observer._cache, {})
                    self.assertNotIn("mechanism_state", runner.store.load())
                    # Historical config without the new default is still valid.
                    path=runner.paths.logs/"metadata.json"
                    metadata=json.loads(path.read_text()); del metadata["config"]["mechanism"]
                    path.write_text(json.dumps(metadata))
                else:
                    windows=read_rows(runner.paths.files/"mechanism_windows.csv")
                    self.assertEqual([int(r["step"]) for r in windows], [0,1,2,3])
                    self.assertEqual(windows[-1]["window_complete"], "True")
                    self.assertLess(abs(float(windows[-1]["identity_residual"])), 1e-10)
                    self.assertIn("mechanism_state", runner.store.load())
                c2,t2,s2,e2,_,o2=components(tmp, enabled=enabled)
                resumed=ExperimentRunner(c2,t2,s2,e2,base_dir=ROOT,train_sample_count=5,
                    metadata={},mechanism=o2,console=False,run_name="run",resume=True)
                restored=resumed.run(initial,resume_slot="p50")
                for a,b in zip(jax.tree_util.tree_leaves(final),jax.tree_util.tree_leaves(restored)):
                    np.testing.assert_array_equal(a,b)
                if enabled:
                    self.assertEqual(len(read_rows(runner.paths.files/"mechanism_windows.csv")),4)
                    changed=replace(c2,mechanism=replace(c2.mechanism,absolute_lrs=(.2,)))
                    with self.assertRaisesRegex(ValueError,"configuration differs"):
                        ExperimentRunner(changed,t2,s2,e2,base_dir=ROOT,train_sample_count=5,
                            metadata={},mechanism=o2,console=False,run_name="run",resume=True)
                    different=TPUMechanisms(t2.task,t2.method,c2.mechanism,reference_name="error",trajectory_method="sage",
                        bank_batches=o2.target.batches,fixed_buffers=initial.model_state,signature="changed_target")
                    with self.assertRaisesRegex(ValueError,"fixed mechanism target differs"):
                        ExperimentRunner(c2,t2,s2,e2,base_dir=ROOT,train_sample_count=5,
                            metadata={},mechanism=different,console=False,run_name="run",resume=True)
        self.assertEqual(final_states[0][1:],final_states[1][1:])
        for a,b in zip(jax.tree_util.tree_leaves(final_states[0][0]),jax.tree_util.tree_leaves(final_states[1][0])):
            np.testing.assert_array_equal(a,b)

    def test_numerical_fault_is_finite_but_unreliable(self):
        _,trainer,source,_,before,observer=components("unused")
        batch=next(source)
        after,_=trainer.step(before,batch)
        after=after._replace(params={**after.params, "w":after.params["w"].at[0,1].add(.02)})
        rows=observer.run(before,after,batch,learning_rate=.01,rng_key=jax.random.key(0),
            context={"batch_id":"fault"},cumulative_eta_before=0.)
        check=next(r for s,r in rows if s=="mechanism_update")
        self.assertTrue(check["finite"]); self.assertFalse(check["reliable"])
        actual=[r for s,r in rows if s=="mechanism_pairs" and r["candidate_kind"]=="actual_pair"]
        self.assertTrue(actual)
        self.assertTrue(any(r["endpoint_loss_error"]>0 for r in actual))
        self.assertTrue(all(not r["signal_reliable"] for r in actual))

    def test_empty_error_zero_score_zero_norm_and_zero_executed_lr(self):
        from src.tasks.classification import ClassificationTask
        for balanced in (False,True):
            def model(p,x,s):
                z=x@p["w"]
                return jnp.concatenate((jnp.zeros_like(z),z),axis=1),s
            task=ClassificationTask(model,2)
            batch=classification_batch([[1.],[1.]] if balanced else [[1.],[-1.]],
                [0,1] if balanced else [1,0])
            initial=TrainState({"w":jnp.array([[0. if balanced else 1.]])})
            method=create_method("sgd")
            trainer=Trainer(task,method,SGDUpdate(lambda step:jnp.asarray(0.,jnp.float32)))
            cfg=MechanismConfig(enabled=True,steps=(0,),bank_moments=False,bank_batches=1,
                absolute_lrs=(0.,.01),lr_scales=(),norm_matched=True)
            observer=TPUMechanisms(task,method,cfg,reference_name="loss",trajectory_method="sgd",
                bank_batches=(TrainingAuditBatch("bank",batch),),fixed_buffers=None)
            _,rows=rows_for(observer,trainer,initial,batch,eta=0.)
            update=next(r for s,r in rows if s=="mechanism_update")
            self.assertEqual(update["actual_displacement_norm"],0.)
            self.assertIsNone(update["parameter_relative_error"])
            if balanced:
                loss=next(r for s,r in rows if s=="mechanism_t1" and r["reference"]=="loss")
                self.assertEqual(loss["flip_count"],0)
                self.assertEqual(loss["negative_count"],0)
                undefined=[r for s,r in rows if s=="mechanism_pairs" and r["reference"]=="loss"
                    and r["normalization"]=="norm_matched"]
                self.assertTrue(all(not r["defined"] and r["undefined_reason"]=="zero_sage_norm" for r in undefined))
            else:
                error=next(r for s,r in rows if s=="mechanism_t1" and r["reference"]=="error")
                self.assertFalse(error["reference_valid"])
                self.assertEqual(error["flip_count"],0)
                self.assertEqual(error["norm_sgd_sq"],error["norm_sage_sq"])
                self.assertFalse(any(s=="mechanism_pairs" and r["reference"]=="error"
                    and r["objective"]=="source_frozen_error_ce" for s,r in rows))

    def test_error_mask_is_frozen_when_candidate_predictions_change(self):
        _,trainer,source,_,before,observer=components("unused")
        observer.config=replace(observer.config,absolute_lrs=(2.,),lr_scales=(),norm_matched=False)
        batch=next(source)
        _,rows=rows_for(observer,trainer,before,batch)
        stats=observer._sources(before.params,batch,before.model_state)["error"]
        candidate=jax.tree_util.tree_map(lambda p,g:p-2*g,before.params,stats.sage)
        result=trainer.task.evaluate(candidate,batch,before.model_state)
        self.assertFalse(np.array_equal(np.asarray(stats.errors),np.asarray(result.feedback["error_mask"])))
        expected=float(jnp.sum(jnp.where(stats.errors,result.losses,0.))/jnp.sum(stats.errors))
        row=next(r for s,r in rows if s=="mechanism_pairs" and r["reference"]=="error"
            and r["objective"]=="source_frozen_error_ce" and r["candidate_kind"]=="independent_pair")
        self.assertAlmostEqual(row["j_sage"],expected,places=6)

    def test_duplicate_lr_labels_reuse_candidate_forwards_and_original_gain_semantics(self):
        counts=[]
        lengths=[]
        for scales in ((),(1.,)):
            _,trainer,source,_,before,observer=components("unused")
            observer.config=replace(observer.config,absolute_lrs=(.01,),lr_scales=scales,
                bank_moments=False,windows=(),oracle_steps=(),norm_matched=False)
            original=observer._call
            calls=[]
            def tracked(name,fn,*args):
                if name in ("objective","bank_value"):calls.append(name)
                return original(name,fn,*args)
            observer._call=tracked
            _,rows=rows_for(observer,trainer,before,next(source))
            pairs=[r for s,r in rows if s=="mechanism_pairs"]
            for row in pairs:
                self.assertEqual(row["effective_objective_gain"],row["objective_gain"])
            counts.append(len(calls)); lengths.append(len(pairs))
        self.assertEqual(counts[0],counts[1])
        self.assertEqual(lengths[1],2*lengths[0])

    def test_all_error_references_coincide_and_bfloat16_is_not_falsely_qualified(self):
        _,trainer,source,_,before,observer=components("unused")
        batch=source.batches[0]
        predicted=trainer.task.evaluate(before.params,batch).feedback["error_mask"]
        # Original labels are zero; prediction error is therefore class1.
        wrong=classification_batch(batch.images,1-predicted.astype(jnp.int32),batch.valid_mask)
        both=observer._sources(before.params,wrong,before.model_state)
        np.testing.assert_allclose(both["error"].sage["w"],both["loss"].sage["w"],atol=2e-6)
        observer.compute_dtype="bfloat16"
        _,rows=rows_for(observer,trainer,before,batch)
        update=next(r for s,r in rows if s=="mechanism_update")
        self.assertFalse(update["tolerance_supported"])
        self.assertFalse(update["reliable"])

    def test_window_gap_is_not_silently_reported_as_a_complete_sum(self):
        _,trainer,source,_,initial,observer=components("unused")
        observer.config=replace(observer.config,steps=(),moment_steps=(),oracle_steps=(),
            update_check=False,paired_loss=False,bank_moments=False)
        before=initial._replace(step=1)
        with self.assertRaisesRegex(ValueError,"continuous-window prefix"):
            observer.restore(dict(target_id=observer.target.identity,windows={}),step=1)
        after,_=trainer.step(before,next(source))
        with self.assertRaisesRegex(ValueError,"incomplete cumulative window"):
            observer.run(before,after,source.batches[0],learning_rate=.01,rng_key=jax.random.key(1),
                context={},cumulative_eta_before=.01)

    def test_real_classification_entry_with_tiny_resnet_and_existing_data_adapter(self):
        from scripts.train_classification import run
        from src.data.imagenet_hf import DataPlan,prepare
        from src.models.resnet import ResNet
        from tests.test_tpu_pipeline import fake_stream
        with TemporaryDirectory() as tmp:
            images=prepare(DataPlan("fixture","a"*40,{0:0,1:1},"fixture",8,3),
                Path(tmp)/"data",stream_factory=fake_stream)
            config=resolve_config(ROOT/"configs/tpu/imagenet100_resnet50_sgd.toml",mode="smoke",
                steps=2,batch_size=2,mechanisms=True,mechanism_steps=[0,1],mechanism_moment_steps=[0],
                mechanism_windows=[(0,2)],mechanism_absolute_lrs=[0.,.01],mechanism_lr_scales=[],mechanism_bank_batches=1)
            config=replace(config,data=replace(config.data,image_size=8),results=ResultsConfig(str(Path(tmp)/"results")))
            args=SimpleNamespace(mode="smoke",run_name="real_mechanism",resume=False,reserved=False,resume_slot="latest")
            with patch("src.data.classification.open_training_images",return_value=(images,{})), \
                 patch("src.models.resnet.create_resnet",return_value=ResNet((1,),100,num_filters=2)), redirect_stdout(StringIO()):
                run(args,config,jax.local_devices(),status=None)
            files=Path(tmp)/"results/files/real_mechanism"
            self.assertEqual(len(read_rows(files/"mechanism_update.csv")),2)
            self.assertEqual(len(read_rows(files/"mechanism_windows.csv")),2)
            self.assertEqual(len(read_rows(files/"mechanism_moments.csv")),2)
            info=json.loads((Path(tmp)/"results/logs/real_mechanism/metadata.json").read_text())
            self.assertEqual(info["tpu_mechanisms"]["target"]["buffer_origin"],"initial_model_state")
            self.assertEqual(info["tpu_mechanisms"]["target"]["bn_mode"],"train_mode_global_logical_batch")

    def test_cli_toml_forwarding_and_production_defaults(self):
        path=ROOT/"configs/tpu/imagenet100_resnet50_sgd.toml"
        self.assertFalse(load_config(path).mechanism.enabled)
        env={**os.environ,"SAGE_PYTHON":sys.executable}
        for suite in ("run_sage_suite.sh","run_inaturalist_suite.sh","run_sage_mechanism_suite.sh"):
            command=["bash",str(ROOT/suite),"--config",str(path),"--mode","train","--methods","sgd","sage_e","sage_l",
                "--steps","4","--mechanisms","--mechanism-steps","0","2","--mechanism-moment-steps","1",
                "--mechanism-oracle-steps","2","--mechanism-windows","0:4","--mechanism-absolute-lrs","0",".003",".02",
                "--mechanism-norm-matched","--no-theorem-probes","--dry-run"]
            plans=json.loads(subprocess.check_output(command,cwd=ROOT,env=env,text=True))
            parser=argparse.ArgumentParser(allow_abbrev=False);add_overrides(parser)
            for plan in plans:
                args,_=parser.parse_known_args(plan["command"][4:])
                child=resolve_config(path,mode="train",steps=4,method=args.method,**overrides(args))
                self.assertEqual(json.loads(json.dumps(child.as_dict()["mechanism"])),plan["mechanism"])
                self.assertEqual(plan["active_mechanism_steps"],[0,1,2,3])
        with TemporaryDirectory() as tmp:
            custom=Path(tmp)/"custom.toml"
            custom.write_text(path.read_text()+"\n[mechanism]\nenabled=true\nsteps=[0,2]\nwindows=[[0,4]]\nbank_batches=1\nabsolute_lrs=[0.0,0.02]\n")
            self.assertTrue(load_config(custom).mechanism.enabled)
        with self.assertRaisesRegex(ValueError,"sorted, disjoint"):
            replace(fixture_config("unused"),mechanism=MechanismConfig(windows=((1,3),(2,4)))).validate()
        with self.assertRaisesRegex(ValueError,"original SGD/SAGE"):
            resolve_config(path,mode="train",method="sage_l_half",mechanisms=True)

    def test_two_device_global_bn_and_dense_oracle(self):
        env={**os.environ,"JAX_PLATFORMS":"cpu","JAX_PLATFORM_NAME":"cpu",
             "XLA_FLAGS":"--xla_force_host_platform_device_count=2"}
        output=subprocess.check_output([sys.executable,"-m","tests.mechanism_mesh_check"],
            cwd=ROOT,env=env,text=True,timeout=120)
        self.assertIn("PASS: mechanism global BN",output)


if __name__ == "__main__":
    unittest.main()
