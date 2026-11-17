import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "VoiceStudio"

def test_get_telemetry(client):
    response = client.get("/api/v1/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "cpu" in data
    assert "ram" in data
    assert "gpu" in data
    assert "usage_percent" in data["cpu"]
    assert "name" in data["cpu"]

def test_get_voices(client):
    response = client.get("/api/v1/voices")
    assert response.status_code == 200
    voices = response.json()
    assert isinstance(voices, list)
    assert len(voices) > 0
    # Every voice metadata should conform to VoiceMetadata schema
    first_voice = voices[0]
    assert "id" in first_voice
    assert "name" in first_voice
    assert "engine" in first_voice

def test_benchmark_endpoints(client):
    # Categories
    resp_cat = client.get("/api/v1/benchmark/categories")
    assert resp_cat.status_code == 200
    cats = resp_cat.json()
    assert "daily" in cats

    # Status
    resp_stat = client.get("/api/v1/benchmark/status")
    assert resp_stat.status_code == 200
    assert "is_running" in resp_stat.json()

def test_clean_audio_cache_endpoint(client):
    response = client.delete("/api/v1/tts/cache?max_age_hours=0")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "cleaned" in data

def test_call_history_endpoint(client):
    response = client.get("/api/v1/call/history")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_static_vendor_assets(client):
    resp_vue = client.get("/static/vendor/vue.global.prod.js")
    assert resp_vue.status_code == 200
    assert len(resp_vue.content) > 50000

    resp_tw = client.get("/static/vendor/tailwindcss.js")
    assert resp_tw.status_code == 200
    assert len(resp_tw.content) > 50000

def test_stt_devices_endpoint(client):
    response = client.get("/api/v1/stt/devices")
    assert response.status_code == 200
    data = response.json()
    assert "devices" in data
    assert "preferred_device" in data
    assert isinstance(data["devices"], list)

def test_stt_transcribe_endpoint(client):
    # Test file transcribe with existing sample
    import os
    sample_file = os.path.join(os.path.dirname(__file__), "..", "speakers", "spk_7869.wav")
    if os.path.exists(sample_file):
        with open(sample_file, "rb") as f:
            response = client.post("/api/v1/stt/transcribe", files={"file": ("spk_7869.wav", f, "audio/wav")})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "text" in data
        assert len(data["text"]) > 0
