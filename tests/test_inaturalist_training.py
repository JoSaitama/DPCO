"""Offline local-subset, shared-entry and shell-suite integration checks."""

from contextlib import closing, redirect_stdout
import csv
from dataclasses import asdict, replace
import hashlib
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory, gettempdir
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import jax
from PIL import Image

from scripts.train_classification import run
from src.config import EvaluationConfig, ResultsConfig, load_config
from src.data.classification import open_training_images
from src.data.images import ImageBatchSource, evaluation_batches
from src.data.inaturalist import INaturalistImages
from src.data.inaturalist_subset import ALGORITHM, SOURCES, canonical
from src.models.resnet import ResNet, create_resnet
from src.training.launch import resolve_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/tpu/inaturalist2021_resnet50.toml"


def prepared_fixture(root, classes=4):
    """Write a tiny completed local selection with genuine encoded image bytes."""
    root.mkdir()
    stream = BytesIO()
    Image.new("RGB", (12, 14), (127, 61, 33)).save(stream, format="PNG")
    blob = stream.getvalue()
    samples = [{"split": split, "id": i, "image_id": i,
                "category_id": 100 + i % classes, "label": i % classes,
                "path": f"{split}/{i}.png"}
               for split, n in (("train", 2 * classes), ("validation", classes)) for i in range(n)]
    manifest = {"identity": {"algorithm": ALGORITHM, "selection_seed": 42,
        "subset_counts": {"train": 2 * classes, "validation": classes},
        "sources": {split: asdict(source) for split, source in SOURCES.items()},
        "num_classes": classes, "image_encoding": "official_original_bytes"},
        "categories": [{"category_id": i + 100, "label": i, "name": f"species_{i}"}
                       for i in range(classes)], "samples": samples}
    content = canonical(manifest)
    digest = hashlib.sha256(content).hexdigest()
    (root / "manifest.json").write_bytes(content)
    (root / "manifest.sha256").write_text(digest + "\n")
    with closing(sqlite3.connect(root / "images.sqlite")) as connection, connection:
        connection.executescript("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT);"
            "CREATE TABLE samples(split TEXT,id INTEGER,image_id INTEGER,label INTEGER,path TEXT,"
            "image BLOB,sha256 TEXT,PRIMARY KEY(split,id),UNIQUE(path));")
        meta = {"manifest_sha256": digest, "complete": True,
                **{split + "_verified": source.image_md5 for split, source in SOURCES.items()}}
        connection.executemany("INSERT INTO meta VALUES (?,?)", [(k, json.dumps(v)) for k, v in meta.items()])
        connection.executemany("INSERT INTO samples VALUES (?,?,?,?,?,?,?)",
            [(s["split"], s["id"], s["image_id"], s["label"], s["path"], blob,
              hashlib.sha256(blob).hexdigest()) for s in samples])
    (root / "complete.json").write_bytes(canonical({"manifest_sha256": digest,
        "sample_count": len(samples), "payload_bytes": len(blob) * len(samples)}))
    return digest


class INaturalistTrainingTests(unittest.TestCase):
    def test_real_resnet_heads_support_all_ten_thousand_classes(self):
        import jax.numpy as jnp
        for name, expected in (("resnet50", 43998032), ("resnet101", 62990160)):
            model = create_resnet(name, 10000)
            shapes = jax.eval_shape(lambda key: model.init(key, jnp.zeros((1, 8, 8, 3)), train=True),
                                    jax.random.key(0))
            self.assertEqual(sum(leaf.size for leaf in jax.tree_util.tree_leaves(shapes["params"])), expected)

    def test_local_read_only_view_no_download_and_exact_iterator_resume(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "subset"
            digest = prepared_fixture(root)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            config = replace(load_config(CONFIG).data, num_classes=4, prepared_dir=str(root))
            with patch("src.data.imagenet_hf.prepare", side_effect=AssertionError("No ImageNet/HF preparation")):
                images, identity = open_training_images(config, ROOT)
            try:
                self.assertEqual(images.counts, {"train": 8, "validation": 4})
                self.assertEqual(images.fingerprint, digest)
                self.assertEqual(identity["selection"]["selection_seed"], 42)
                with self.assertRaises(sqlite3.OperationalError):
                    images.connection.execute("DELETE FROM samples")
                source = ImageBatchSource(images, 3, 8, 5)
                next(source)
                snapshot = source.snapshot()
                expected = next(source)
                restored = ImageBatchSource(images, 3, 8, 5)
                restored.restore(snapshot)
                import numpy as np
                for a, b in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(next(restored))):
                    np.testing.assert_array_equal(a, b)
                batches = list(evaluation_batches(images, 3, 8))
                self.assertEqual(sum(int(batch.valid_mask.sum()) for batch in batches), 4)
            finally:
                images.close()
            smoke = INaturalistImages(root, num_classes=4, limit=3)
            try:
                self.assertEqual(smoke.counts, {"train": 3, "validation": 3})
                self.assertNotEqual(smoke.fingerprint, digest)
                with self.assertRaises(IndexError):
                    smoke.get("train", 3)
                source = ImageBatchSource(smoke, 3, 8, 5)
                with self.assertRaisesRegex(ValueError, "snapshot"):
                    source.restore(snapshot)
            finally:
                smoke.close()
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})

    def test_incomplete_mismatched_or_tampered_selections_fail_closed(self):
        mutations = {
            "missing_completion": lambda r: (r / "complete.json").unlink(),
            "wrong_manifest": lambda r: (r / "manifest.sha256").write_text("0" * 64),
            "wrong_completion": lambda r: (r / "complete.json").write_text("{}"),
            "missing_member": "DELETE FROM samples WHERE split='train' AND id=0",
            "wrong_label": "UPDATE samples SET label=3 WHERE split='train' AND id=0",
            "empty_image": "UPDATE samples SET image=X'' WHERE split='train' AND id=0",
            "unverified": "DELETE FROM meta WHERE key='train_verified'",
            "unfinished": "UPDATE meta SET value='false' WHERE key='complete'",
            "wrong_fingerprint": "UPDATE meta SET value='\"wrong\"' WHERE key='manifest_sha256'",
        }
        with TemporaryDirectory() as tmp:
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    root = Path(tmp) / name
                    prepared_fixture(root)
                    if isinstance(mutate, str):
                        with closing(sqlite3.connect(root / "images.sqlite")) as connection, connection:
                            connection.execute(mutate)
                    else:
                        mutate(root)
                    with self.assertRaises((ValueError, FileNotFoundError)):
                        INaturalistImages(root, num_classes=4)
            root = Path(tmp) / "wrong_head"
            prepared_fixture(root)
            with self.assertRaisesRegex(ValueError, "classifier head"):
                INaturalistImages(root)

    def test_new_shell_defaults_and_all_common_overrides_reach_children(self):
        def dry_run(*flags):
            return json.loads(subprocess.check_output([str(ROOT / "run_inaturalist_suite.sh"),
                "--dry-run", *flags], cwd=gettempdir(),
                env={**os.environ, "DPCO_PYTHON": sys.executable}, text=True))
        defaults = dry_run()
        self.assertEqual([(p["method"], p["reference"]) for p in defaults],
                         [("dpco", "error"), ("sgd", "loss"), ("dpco", "loss")])
        for plan in defaults:
            self.assertEqual((plan["seed"], plan["num_classes"], plan["total_steps"], plan["batch_size"]),
                             (5, 10000, 2000, 256))
            self.assertEqual(plan["dataset"], "inaturalist2021")
            self.assertEqual(plan["worker"], "worker0_only")
            self.assertTrue(plan["wait"])
            self.assertFalse(plan["probes_enabled"])
        plans = dry_run("--mode", "smoke", "--methods", "sgd", "dpco_l", "--seeds", "2", "7",
            "--data-dir", "/tmp/local subset", "--steps", "5", "--batch-size", "16",
            "--model", "resnet101", "--datamode", "clean", "--lr", ".005", "--momentum", ".9",
            "--nesterov", "--l2-coefficient", ".0001", "--lr-schedule", "cosine", "--min-lr", ".000005",
            "--eval-every-steps", "3", "--checkpoint-every-steps", "2")
        self.assertEqual(len(plans), 4)
        for plan in plans:
            command = plan["command"]
            value = lambda flag: command[command.index(flag) + 1]
            resolved = resolve_config(value("--config"), mode=value("--mode"),
                seed=int(value("--seed")), method=value("--method"), data_dir=value("--data-dir"),
                steps=int(value("--steps")), batch_size=int(value("--batch-size")), model=value("--model"),
                data_mode=value("--data-mode"), learning_rate=float(value("--lr")),
                momentum=float(value("--momentum")), nesterov="--nesterov" in command,
                l2_coefficient=float(value("--l2-coefficient")), lr_schedule=value("--lr-schedule"),
                min_learning_rate=float(value("--min-lr")),
                checkpoint_every_steps=int(value("--checkpoint-every-steps")),
                eval_every_steps=int(value("--eval-every-steps")))
            self.assertEqual(resolved.data.prepared_dir, "/tmp/local subset")
            self.assertEqual(resolved.model.name, "resnet101")
            self.assertEqual(resolved.training.total_steps, 5)
            self.assertEqual(resolved.training.min_learning_rate, .000005)
            self.assertEqual(resolved.checkpoints.latest_every_steps, 2)
            self.assertEqual(resolved.evaluation, EvaluationConfig("steps", 3))
        single = dry_run("--mode", "validate", "--methods", "dpco_e", "--eval-every-epochs", ".5")
        self.assertTrue(single["probes_enabled"])
        self.assertEqual(single["evaluation_interval"], .5)

    def test_dataset_switches_keep_classifier_mapping_and_directory_consistent(self):
        original = load_config(ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml")
        legacy_switch = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml", mode="train",
                                       dataset="imagenet1k")
        self.assertEqual(legacy_switch.data, replace(original.data, name="imagenet1k", num_classes=1000))
        config = resolve_config(ROOT / "configs/tpu/imagenet100_resnet50_sgd.toml", mode="train",
                                dataset="inaturalist2021")
        self.assertEqual(config.data.num_classes, 10000)
        self.assertTrue(config.data.prepared_dir.endswith("_s42"))
        self.assertIsNone(config.data.class_manifest)
        config = resolve_config(CONFIG, mode="train", dataset="imagenet100")
        self.assertEqual(config.data.num_classes, 100)
        self.assertEqual(config.data.prepared_dir, "../data")
        self.assertTrue(config.data.class_manifest)

    def test_shared_training_entry_runs_three_methods_and_resumes_checkpoint(self):
        # CPU-only service check: real entry/mesh/BN/records, tiny ResNet backbone.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = prepared_fixture(root / "subset", classes=10000)
            for label in ("dpco_e", "sgd", "dpco_l"):
                config = resolve_config(CONFIG, mode="smoke", method=label,
                    data_dir=str(root / "subset"), steps=2, batch_size=2)
                config = replace(config, data=replace(config.data, image_size=8),
                    results=ResultsConfig(str(root / "results")))
                args = SimpleNamespace(mode="smoke", run_name=label, resume=False,
                                       reserved=False, resume_slot="latest")
                with patch("src.models.resnet.create_resnet", return_value=ResNet((1,), 10000, num_filters=2)), redirect_stdout(StringIO()):
                    run(args, config, jax.local_devices(), status=None)
                    args.resume, args.resume_slot = True, "p50"
                    run(args, config, jax.local_devices(), status=None)
                metadata = json.loads((root / "results/logs" / label / "metadata.json").read_text())
                self.assertEqual(metadata["experiment"]["data_plan"]["manifest_sha256"], digest)
                self.assertEqual(metadata["experiment"]["split_counts"], {"train": 32, "validation": 32})
                with (root / "results/files" / label / "summary.csv").open() as stream:
                    summary = list(csv.DictReader(stream))[-1]
                self.assertEqual(int(summary["step"]), 2)
                self.assertEqual(summary["status"], "completed")


if __name__ == "__main__":
    unittest.main()
