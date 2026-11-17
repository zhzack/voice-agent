import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from typing import List

from app.core.base import VoiceMetadata
from app.providers.tts.edge_provider import EdgeTTSProvider
from app.providers.tts.chattts_provider import ChatTTSProvider
from app.providers.tts.melotts_provider import MeloTTSProvider
from app.providers.tts.f5tts_provider import F5TTSProvider
from app.providers.tts.kokoro_provider import KokoroTTSProvider
from app.providers.tts.piper_provider import PiperTTSProvider

router = APIRouter(prefix="/api/v1/voices", tags=["Voices"])

edge_provider = EdgeTTSProvider()
chattts_provider = ChatTTSProvider()
melotts_provider = MeloTTSProvider()
f5tts_provider = F5TTSProvider()
kokoro_provider = KokoroTTSProvider()
piper_provider = PiperTTSProvider()

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPEAKERS_DIR = os.path.join(BASE_DIR, "speakers")

@router.get("", response_model=List[VoiceMetadata])
def get_all_voices(engine: str = None):
    all_voices = []
    engine_filter = engine.lower() if engine else None

    if engine_filter is None or engine_filter == "edge":
        all_voices.extend(edge_provider.list_voices())
    if engine_filter is None or engine_filter == "chattts":
        all_voices.extend(chattts_provider.list_voices())
    if engine_filter is None or engine_filter == "melotts":
        all_voices.extend(melotts_provider.list_voices())
    if engine_filter is None or engine_filter == "f5tts":
        all_voices.extend(f5tts_provider.list_voices())
    if engine_filter is None or engine_filter == "kokoro":
        all_voices.extend(kokoro_provider.list_voices())
    if engine_filter is None or engine_filter == "piper":
        all_voices.extend(piper_provider.list_voices())

    return all_voices

@router.get("/sample/{engine}/{voice_id}")
def get_voice_sample(engine: str, voice_id: str):
    engine = engine.lower()
    if engine == "edge":
        path = os.path.join(SPEAKERS_DIR, "edge", f"{voice_id}.mp3")
        media_type = "audio/mpeg"
    elif engine == "chattts":
        seed_map = {
            "spk_warm_female": "spk_2222.wav",
            "spk_young_male": "spk_7869.wav",
            "spk_sweet_female": "spk_4099.wav",
            "spk_deep_male": "spk_1111.wav",
            "spk_gentle_female": "spk_6653.wav",
            "spk_vivid_boy": "spk_5555.wav"
        }
        filename = seed_map.get(voice_id, f"{voice_id}.wav")
        path = os.path.join(SPEAKERS_DIR, filename)
        media_type = "audio/wav"
    elif engine == "melotts":
        audio_assets_dir = os.path.join(BASE_DIR, "assets", "audio")
        path = os.path.join(audio_assets_dir, "test_melotts.wav")
        if not os.path.exists(path):
            path = os.path.join(BASE_DIR, "test_melotts.wav")
        media_type = "audio/wav"
    elif engine == "f5tts":
        f5_map = {
            "f5_clone_warm_female": "spk_2222.wav",
            "f5_clone_young_male": "spk_7869.wav",
            "f5_clone_deep_male": "spk_1111.wav",
            "f5_custom_clone": "spk_2222.wav"
        }
        filename = f5_map.get(voice_id, "spk_2222.wav")
        path = os.path.join(SPEAKERS_DIR, filename)
        media_type = "audio/wav"
    elif engine == "kokoro":
        audio_assets_dir = os.path.join(BASE_DIR, "assets", "audio")
        sample_path = os.path.join(audio_assets_dir, f"test_kokoro_{voice_id}.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(audio_assets_dir, "test_kokoro.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, f"test_kokoro_{voice_id}.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "test_kokoro.wav")
        path = sample_path
        media_type = "audio/wav"
    elif engine == "piper":
        audio_assets_dir = os.path.join(BASE_DIR, "assets", "audio")
        sample_path = os.path.join(audio_assets_dir, f"test_{voice_id}.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(audio_assets_dir, "test_piper_ok.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(audio_assets_dir, "test_piper.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, f"test_{voice_id}.wav")
        if not os.path.exists(sample_path):
            sample_path = os.path.join(BASE_DIR, "test_piper_ok.wav")
        path = sample_path
        media_type = "audio/wav"
    else:
        raise HTTPException(status_code=400, detail=f"未知引擎: {engine}")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"音频样本未找到: {path}")

    return FileResponse(path, media_type=media_type)
