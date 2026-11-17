"""Runtime Manager Service – V3.0 architecture

This module implements the runtime abstraction layer that isolates model
execution environments (CPU, CUDA, ROCm, or external server). It works
together with the GPU scheduler to ensure that GPU memory constraints are
respected before a runtime is started.
"""

import asyncio
import subprocess
from pathlib import Path
from typing import Optional, Dict

import torch

class Runtime:
    """Represent a single running instance of a model.

    Attributes
    ----------
    model_id: str
        Identifier of the model (matches the ``id`` field in the Manifest).
    backend: str
        Execution backend, e.g. ``pytorch``, ``onnxruntime``.
    device: str
        Target device – ``cpu``, ``cuda``, ``rocm`` etc.
    precision: str
        Precision hint – ``auto``, ``float16``, ``int8``.
    env_path: Path
        Path to the isolated Python/Conda environment that contains the model
        dependencies.  For pure in‑process execution this can be ``None``.
    process: Optional[subprocess.Popen]
        If the model is launched as an external service (e.g. a gRPC server),
        the subprocess handle is stored here so that it can be terminated.
    """

    def __init__(self, model_id: str, backend: str, device: str, precision: str, env_path: Optional[Path] = None):
        self.model_id = model_id
        self.backend = backend
        self.device = device
        self.precision = precision
        self.env_path = env_path
        self.process: Optional[subprocess.Popen] = None
        self._started = False

    async def start(self) -> "Runtime":
        """Start the runtime.

        For most local PyTorch models we simply load the model in‑process, but
        the method is kept ``async`` so that future implementations can spawn a
        separate process (e.g. a Docker container) without changing the caller.
        """
        # Placeholder: actual model loading is delegated to the provider.
        # Here we only mark the runtime as started.
        self._started = True
        return self

    async def stop(self) -> None:
        """Terminate the runtime.

        If a subprocess was created, terminate it gracefully.  If the runtime
        runs in‑process, cleanup hooks can be added later.
        """
        if self.process:
            self.process.terminate()
            await asyncio.get_event_loop().run_in_executor(None, self.process.wait)
        self._started = False

    def is_running(self) -> bool:
        return self._started

class RuntimeManagerService:
    """Central manager for all Runtime objects.

    The manager keeps a dictionary ``self.runtimes`` mapping ``model_id`` to
    ``Runtime`` instances.  It uses the :class:`~app.utils.gpu_scheduler.GPUScheduler`
    to ensure that a new runtime does not exceed the available GPU memory.
    """

    def __init__(self, plugins_root: Path):
        self.plugins_root = plugins_root
        self.runtimes: Dict[str, Runtime] = {}

    async def get_or_create_runtime(self, manifest) -> Runtime:
        """Return an existing runtime for ``manifest.id`` or create a new one.

        The function checks GPU availability via ``GPUScheduler`` before creating
        the runtime.  ``manifest`` is a :class:`app.core.model_manifest.ModelManifest`
        instance.
        """
        model_id = manifest.id
        if model_id in self.runtimes and self.runtimes[model_id].is_running():
            return self.runtimes[model_id]

        # Estimate memory requirement – a simple heuristic from manifest.hardware.
        memory_needed_mb = manifest.hardware.get("minimum_vram_mb") or 0
        from app.utils.gpu_scheduler import GPUScheduler
        await GPUScheduler.acquire(memory_needed_mb)

        runtime = Runtime(
            model_id=model_id,
            backend=manifest.runtime.get("backend", "pytorch"),
            device=manifest.runtime.get("device", "auto"),
            precision=manifest.runtime.get("precision", "auto"),
            env_path=Path(manifest.runtime.get("environment", "")) if manifest.runtime.get("environment") else None,
        )
        await runtime.start()
        self.runtimes[model_id] = runtime
        return runtime

    async def release_runtime(self, model_id: str) -> None:
        """Stop and remove a runtime; releases GPU resources back to the scheduler."""
        runtime = self.runtimes.get(model_id)
        if runtime and runtime.is_running():
            await runtime.stop()
            # In a real implementation we would also inform the scheduler about
            # the released memory.  For now we simply delete the entry.
            del self.runtimes[model_id]
