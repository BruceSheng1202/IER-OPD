# Adapted from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ray/utils.py#L1
import os

import ray
import torch
from slime.ray.ray_actor import RayActor


# Refer to
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/nvidia_gpu.py#L95-L96
NOSET_VISIBLE_DEVICES_ENV_VARS_LIST = [
    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
]


def ray_noset_visible_devices(env_vars=os.environ):
    return any(env_vars.get(env_var) for env_var in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST)


def get_physical_gpu_id():
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return str(props.uuid)


@ray.remote
class Lock(RayActor):
    def __init__(self):
        self._locked = False  # False: unlocked, True: locked

    def acquire(self):
        """
        Try to acquire the lock. Returns True if acquired, False otherwise.
        Caller should retry until it returns True.
        """
        if not self._locked:
            self._locked = True
            return True
        return False

    def release(self):
        """Release the lock, allowing others to acquire."""
        assert self._locked, "Lock is not acquired, cannot release."
        self._locked = False
