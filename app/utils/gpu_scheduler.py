"""GPU Scheduler – simple memory‑based resource guard.

The scheduler maintains a global counter of used VRAM (in MB) and provides an
``async acquire``/``release`` API.  It is deliberately lightweight – it does not
track individual processes, only the total memory requested by all active
runtimes.  For production you may replace it with a more sophisticated
CUDA‑MPS or external resource manager.
"""

import asyncio
from typing import ClassVar

import torch

class GPUScheduler:
    _total_vram_mb: ClassVar[int] = 0
    _used_vram_mb: ClassVar[int] = 0
    _lock: ClassVar[asyncio.Lock] = asyncio.Lock()
    _condition: ClassVar[asyncio.Condition] = asyncio.Condition(_lock)

    @classmethod
    async def init(cls) -> None:
        """Detect total VRAM on the primary GPU.  Called once at app start."""
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            cls._total_vram_mb = torch.cuda.get_device_properties(device).total_memory // (1024 * 1024)
        else:
            cls._total_vram_mb = 0

    @classmethod
    async def acquire(cls, mem_mb: int) -> None:
        """Block until at least ``mem_mb`` free VRAM is available.

        The function reserves the requested memory once the condition is met.
        If no CUDA/GPU total VRAM is detected, it returns immediately without blocking.
        """
        if cls._total_vram_mb == 0 and torch.cuda.is_available():
            await cls.init()

        if cls._total_vram_mb == 0:
            return

        async with cls._condition:
            while cls._used_vram_mb + mem_mb > cls._total_vram_mb:
                await cls._condition.wait()
            cls._used_vram_mb += mem_mb

    @classmethod
    async def release(cls, mem_mb: int) -> None:
        """Release previously reserved VRAM and wake waiting coroutines."""
        async with cls._condition:
            cls._used_vram_mb = max(cls._used_vram_mb - mem_mb, 0)
            cls._condition.notify_all()

    @classmethod
    def status(cls) -> dict:
        """Return current memory usage statistics for monitoring."""
        return {
            "total_vram_mb": cls._total_vram_mb,
            "used_vram_mb": cls._used_vram_mb,
            "free_vram_mb": cls._total_vram_mb - cls._used_vram_mb,
        }
