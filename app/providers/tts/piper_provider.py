import os
import time
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

class PiperTTSProvider(BaseTTSProvider):
    VOICES_DEF = [
        {
            "id": "piper_huayan",
            "name": "华严 (Piper 轻量快速中文女声)",
            "model_path": os.path.join(MODELS_DIR, "vits-piper-zh_CN-huayan-medium", "zh_CN-huayan-medium.onnx"),
            "gender": "female",
            "locale": "zh-CN",
            "desc": "轻量级 VITS ONNX 离线推理，极低 CPU 消耗，发音清晰自然",
            "tags": ["轻量", "极速", "离线", "中文"]
        },
        {
            "id": "piper_lessac",
            "name": "Lessac (Piper 纯正美音英文女声)",
            "model_path": os.path.join(MODELS_DIR, "piper-en-lessac", "en", "en_US", "lessac", "medium", "en_US-lessac-medium.onnx"),
            "gender": "female",
            "locale": "en-US",
            "desc": "Piper 经典英文 Lessac 音色，发音地道流畅，低延迟",
            "tags": ["轻量", "英语", "美音", "离线"]
        }
    ]

    def __init__(self):
        self._loaded_voices = {}

    def _get_voice(self, voice_id: str):
        if voice_id not in self._loaded_voices:
            import piper
            preset = next((v for v in self.VOICES_DEF if v["id"] == voice_id), self.VOICES_DEF[0])
            m_path = preset["model_path"]
            c_path = m_path + ".json"
            if not os.path.exists(m_path):
                raise FileNotFoundError(f"Piper 模型文件未找到: {m_path}")
            self._loaded_voices[voice_id] = piper.PiperVoice.load(m_path, config_path=c_path)
        return self._loaded_voices[voice_id]

    def list_voices(self) -> List[VoiceMetadata]:
        return [
            VoiceMetadata(
                id=v["id"],
                name=v["name"],
                engine="piper",
                gender=v["gender"],
                locale=v["locale"],
                desc=v["desc"],
                sample_url=f"/api/v1/voices/sample/piper/{v['id']}",
                tags=v["tags"],
                is_default=False
            )
            for v in self.VOICES_DEF
        ]

    def synthesize(
        self,
        text: str,
        voice_id: str = "piper_huayan",
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        apply_polishing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        t0 = time.time()
        voice = self._get_voice(voice_id)

        chunks = list(voice.synthesize(text))
        if not chunks:
            raise RuntimeError("Piper 合成未输出音频切片")

        # Combine audio data
        audio_arrays = [c.audio_float_array for c in chunks if hasattr(c, 'audio_float_array')]
        if not audio_arrays:
            raise RuntimeError("Piper 未能提取音频浮点数组")
        
        audio = np.concatenate(audio_arrays)
        sr = chunks[0].sample_rate
        cost_sec = time.time() - t0

        if apply_polishing:
            audio = polish_audio_cadence(audio, sr)

        duration = len(audio) / sr
        rtf = cost_sec / duration if duration > 0 else 0.0

        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        sf.write(output_path, audio, sr)

        return {
            "audio": audio,
            "sample_rate": sr,
            "duration": duration,
            "cost_sec": cost_sec,
            "rtf": rtf,
            "output_path": output_path
        }
