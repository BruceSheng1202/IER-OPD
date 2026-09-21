"""Local Qwen architecture adapters used by the paper models."""
from .qwen2_local import Qwen2LocalBridge
from .qwen3_local import Qwen3LocalBridge

__all__ = ["Qwen2LocalBridge", "Qwen3LocalBridge"]
