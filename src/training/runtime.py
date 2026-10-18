"""Configure isolated worker0 TPU execution before importing/initializing JAX."""

import os
import re
import socket
import threading


def configure_worker0(config, *, environ=None, hostname=None):
    env = os.environ if environ is None else environ
    hostname = socket.gethostname() if hostname is None else hostname
    if re.search(r"(?:-w-|worker)([1-9][0-9]*)$", hostname):
        raise RuntimeError("This launch protocol is restricted to worker0; do not start on worker1")
    if env.get("SAGE_BACKGROUND_LAUNCH") != "1":
        raise RuntimeError("Use scripts/launch_tpu.py: TPU smoke/training must start through nohup")
    # Restrict libtpu to one process on this physical host. Never call
    # jax.distributed.initialize or discover peers from the original pod slice.
    settings = {
        "JAX_PLATFORMS": "tpu", "JAX_PLATFORM_NAME": "tpu",
        "TPU_PROCESS_BOUNDS": "1,1,1",
        "TPU_CHIPS_PER_PROCESS_BOUNDS": config.chips_per_process_bounds,
        "TPU_VISIBLE_CHIPS": config.visible_chips,
        "TPU_VISIBLE_DEVICES": config.visible_chips,
        "TPU_PROCESS_ADDRESSES": f"localhost:{config.port}",
        "TPU_PROCESS_PORT": str(config.port), "CLOUD_TPU_TASK_ID": "0",
        "TPU_WORKER_ID": "0", "TPU_WORKER_HOSTNAMES": "localhost",
    }
    env.update(settings)
    for name in ("JAX_COORDINATOR_ADDRESS", "JAX_PROCESS_COUNT", "JAX_PROCESS_INDEX"):
        env.pop(name, None)
    return settings


def require_local_tpu(config):
    """Fail closed, including a bounded timeout instead of waiting for worker1."""
    def expired():
        print("TPU initialization timed out. Check isolated-host libtpu topology settings; no CPU fallback.", flush=True)
        os._exit(124)
    timer = threading.Timer(config.startup_timeout_s, expired)
    timer.daemon = True
    timer.start()
    try:
        import jax
        jax.config.update("jax_platforms", "tpu")
        devices = jax.local_devices(backend="tpu")
        if (jax.default_backend() != "tpu" or jax.process_count() != 1 or
                len(jax.devices()) != len(devices) or
                any(d.platform != "tpu" or d.process_index != 0 for d in devices)):
            raise RuntimeError("Expected local TPU devices in one isolated process; CPU/remote devices are forbidden")
        return devices
    finally:
        timer.cancel()
