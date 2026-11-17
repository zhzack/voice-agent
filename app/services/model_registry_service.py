# app/services/model_registry_service.py
"""Model Registry Service – loads model manifests from the plugin directory.

The service scans ``model_plugins`` for ``manifest.yaml`` files, parses them
using :func:`app.core.model_manifest.load_manifest`, and keeps a dictionary of
``ModelManifest`` objects keyed by ``model_id``.  It also exposes simple lookup
APIs used by the Runtime manager and by the API layer.
"""

from pathlib import Path
from typing import Dict, List

from app.core.model_manifest import ModelManifest, load_manifest

class ModelRegistryService:
    """Central registry for all model plugins.

    Attributes
    ----------
    plugins_root: Path
        Root directory containing ``model_plugins`` sub‑folders.
    models: Dict[str, ModelManifest]
        Mapping from ``model_id`` to the parsed manifest.
    """

    def __init__(self, plugins_root: Path):
        self.plugins_root = plugins_root
        self.models: Dict[str, ModelManifest] = {}
        self._load_all_plugins()

    def _load_all_plugins(self) -> None:
        """Iterate over ``model_plugins/**/manifest.yaml`` and register models."""
        if not self.plugins_root.exists():
            return
        for manifest_path in self.plugins_root.glob("**/manifest.yaml"):
            try:
                manifest = load_manifest(manifest_path)
                if manifest.id in self.models:
                    # Duplicate IDs – keep the first and warn.
                    print(f"[ModelRegistry] Duplicate model id '{manifest.id}' ignored (" +
                          f"{manifest_path})")
                else:
                    self.models[manifest.id] = manifest
            except Exception as e:
                print(f"[ModelRegistry] Failed to load manifest {manifest_path}: {e}")

    def get_model(self, model_id: str) -> ModelManifest:
        """Return the manifest for ``model_id`` or raise ``KeyError``."""
        return self.models[model_id]

    def list_models(self) -> List[ModelManifest]:
        """Return a list of all loaded model manifests."""
        return list(self.models.values())
