# app/core/model_manifest.py
"""Model Manifest parsing utilities for V3.0 plugin system.

The Manifest is a YAML file placed under each plugin directory (e.g.
`model_plugins/tts/chattts/manifest.yaml`).  It describes the model ID,
name, version, runtime requirements and capabilities.  The parser validates
required fields and converts the YAML into a typed ``Model`` dataclass.
"""

import yaml
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, validator

try:
    from pydantic import field_validator
    class ModelManifest(BaseModel):
        schema_version: str
        id: str
        name: str
        provider: str
        task: str
        version: Dict[str, Any]
        source: Dict[str, Any]
        runtime: Dict[str, Any]
        capabilities: Dict[str, Any]
        parameters: Dict[str, Any]
        license: Dict[str, Any]
        hardware: Dict[str, Any]

        @field_validator("schema_version")
        @classmethod
        def _check_schema(cls, v: str) -> str:
            if not v.startswith("1."):
                raise ValueError("Unsupported manifest schema version")
            return v
except ImportError:
    from pydantic import validator
    class ModelManifest(BaseModel):
        schema_version: str
        id: str
        name: str
        provider: str
        task: str
        version: Dict[str, Any]
        source: Dict[str, Any]
        runtime: Dict[str, Any]
        capabilities: Dict[str, Any]
        parameters: Dict[str, Any]
        license: Dict[str, Any]
        hardware: Dict[str, Any]

        @validator("schema_version")
        def _check_schema(cls, v: str) -> str:
            if not v.startswith("1."):
                raise ValueError("Unsupported manifest schema version")
            return v

def load_manifest(manifest_path: Path) -> ModelManifest:
    """Load and validate a manifest YAML file.

    Args:
        manifest_path: Path to ``manifest.yaml``.
    Returns:
        ModelManifest instance.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return ModelManifest(**data)
