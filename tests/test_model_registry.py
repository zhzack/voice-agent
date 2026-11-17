from pathlib import Path
from app.services.model_registry_service import ModelRegistryService

def test_model_registry_discovery():
    repo_root = Path(__file__).resolve().parent.parent
    plugins_dir = repo_root / "model_plugins"
    registry = ModelRegistryService(plugins_root=plugins_dir)
    models = registry.list_models()
    assert len(models) >= 1
    chattts = registry.get_model("chattts")
    assert chattts.id == "chattts"
    assert chattts.task == "tts"
