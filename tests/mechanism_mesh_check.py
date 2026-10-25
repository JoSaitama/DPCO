"""Fresh two-CPU GSPMD check; does not assert anything about actual TPU numerics."""

import jax
import jax.numpy as jnp
import numpy as np

from src.config import MechanismConfig
from src.data.batch import classification_batch
from src.data.diagnostic_batches import TrainingAuditBatch
from src.methods import create_method
from src.probes.tpu_mechanisms import TPUMechanisms
from src.tasks.classification import ClassificationTask
from src.training import TrainState,Trainer
from src.training.local_mesh import LocalMeshTrainer,local_mesh
from src.training.updates import SGDUpdate


def main():
    assert len(jax.local_devices())==2 and jax.default_backend()=="cpu"
    def model(params,x,state):
        mean=x.mean(0); variance=((x-mean)**2).mean(0)
        z=((x-mean)/jnp.sqrt(variance+1e-5))@params["w"]+state["offset"]
        return jnp.concatenate((jnp.zeros_like(z),z),axis=1),{"offset":state["offset"]+.03*mean}
    task=ClassificationTask(model,2)
    initial=TrainState({"w":jnp.array([[.5]])},{"offset":jnp.array([.2])})
    b1=classification_batch(np.array([[5.],[1.],[-2.],[-4.]],np.float32),np.array([1,1,0,1],np.int32))
    b2=classification_batch(np.array([[2.],[-6.],[1.],[3.]],np.float32),np.array([1,0,1,0],np.int32))
    method=create_method("sage",reference_name="loss",gradient_backend="jvp")
    trainer=LocalMeshTrainer(task,method,SGDUpdate(.01),local_mesh(jax.local_devices()))
    before=jax.device_put(initial,trainer.replicated)
    batch=jax.device_put(b1,trainer.batch_sharding)
    bank=tuple(TrainingAuditBatch(str(i),jax.device_put(b,trainer.batch_sharding)) for i,b in enumerate((b1,b2)))
    cfg=MechanismConfig(enabled=True,steps=(0,),moment_steps=(0,),oracle_steps=(0,),
        absolute_lrs=(0.,.01),lr_scales=(),bank_batches=2,windows=((0,1),))
    observer=TPUMechanisms(task,method,cfg,reference_name="loss",trajectory_method="sage",
        bank_batches=bank,fixed_buffers=before.model_state,signature="global_bn_fixture")
    after,_=trainer.compile_step()(before,batch,jax.random.key(0))
    rows=observer.run(before,after,batch,learning_rate=.01,rng_key=jax.random.key(0),context={},cumulative_eta_before=0.)
    for reference in ("loss","error"):
        independent=observer._sources(before.params,batch,before.model_state)[reference]
        dense=create_method("sage",reference_name=reference,gradient_backend="audit")
        expected=dense.compute_direction(task,initial.params,b1,initial.model_state)
        for a,b in zip(jax.tree_util.tree_leaves(independent.sage),jax.tree_util.tree_leaves(expected.direction)):
            np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
    def objective(p):
        return .5*sum(jnp.mean(task.evaluate(p,b,initial.model_state).losses) for b in (b1,b2))
    expected_value,expected_h=jax.value_and_grad(objective)(initial.params)
    actual_value,actual_h=observer.target.evaluate(before.params,observer.ad,observer._call)
    np.testing.assert_allclose(actual_value,expected_value,atol=2e-6)
    for a,b in zip(jax.tree_util.tree_leaves(actual_h),jax.tree_util.tree_leaves(expected_h)):
        np.testing.assert_allclose(a,b,rtol=2e-5,atol=2e-6)
    for stream,row in rows:
        if stream=="mechanism_moments": assert row["mean_gd_h_l2_error"]<2e-6,row
        if stream=="mechanism_windows": assert row["identity_pass"],row
    print("PASS: mechanism global BN, fixed buffers and independent dense target oracle")


if __name__=="__main__":main()
