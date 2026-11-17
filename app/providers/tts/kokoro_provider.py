import os
import time
import tempfile
import soundfile as sf
from typing import List, Dict, Any, Optional

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro-multi-lang-v1_0")

class KokoroTTSProvider(BaseTTSProvider):
    """
    Kokoro-82M Ultra-Lightweight High-Fidelity Multi-Lingual Neural TTS.
    StyleTTS2-distilled architecture. 100% CPU Offline inference, 0 MB VRAM, RTF ~0.2.
    """
    PRESETS = [
        {
            "id": "kokoro_zh_xiaoxiao",
            "name": "Kokoro 晓晓 (中英双语 知性女声)",
            "sid": 42,
            "gender": "female",
            "locale": "zh-CN",
            "desc": "基于 Kokoro-82M 蒸馏架构，中英双语极速生成，音色温和清晰",
            "tags": ["Kokoro-82M", "纯离线", "纯CPU", "0显存", "中英双语"]
        },
        {
            "id": "kokoro_zh_yunxi",
            "name": "Kokoro 云希 (中英双语 阳光男声)",
            "sid": 45,
            "gender": "male",
            "locale": "zh-CN",
            "desc": "青年阳光男声，中英混合朗读极其自然流畅，低延迟响应",
            "tags": ["Kokoro-82M", "纯离线", "纯CPU", "0显存", "中英双语"]
        },
        {
            "id": "kokoro_en_heart",
            "name": "Kokoro Heart (美式旗舰 优雅女声)",
            "sid": 0,
            "gender": "female",
            "locale": "en-US",
            "desc": "Kokoro 官方排名第一的标志性美式女声，抑扬顿挫如真人播报",
            "tags": ["Kokoro-82M", "美语标杆", "高拟真", "轻量之王"]
        },
        {
            "id": "kokoro_en_michael",
            "name": "Kokoro Michael (美式磁性 叙事男声)",
            "sid": 9,
            "gender": "male",
            "locale": "en-US",
            "desc": "磁性深沉美式男声，适合有声书朗诵、纪录片与科技旁白",
            "tags": ["Kokoro-82M", "美音", "磁性男声", "低延迟"]
        }
    ]

    def __init__(self):
        self._tts = None

    def _ensure_loaded(self):
        if self._tts is None:
            if not os.path.exists(KOKORO_DIR):
                raise RuntimeError(f"Kokoro 模型目录未找到: {KOKORO_DIR}")

            import sherpa_onnx
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=os.path.join(KOKORO_DIR, "model.onnx"),
                        voices=os.path.join(KOKORO_DIR, "voices.bin"),
                        tokens=os.path.join(KOKORO_DIR, "tokens.txt"),
                        data_dir=os.path.join(KOKORO_DIR, "espeak-ng-data"),
                        dict_dir=os.path.join(KOKORO_DIR, "dict"),
                        lexicon=os.path.join(KOKORO_DIR, "lexicon-us-en.txt," + os.path.join(KOKORO_DIR, "lexicon-zh.txt"))
                    ),
                    provider="cpu",
                    num_threads=4
                )
            )
            self._tts = sherpa_onnx.OfflineTts(tts_config)

    def list_voices(self) -> List[VoiceMetadata]:
        return [
            VoiceMetadata(
                id=p["id"],
                name=p["name"],
                engine="kokoro",
                gender=p["gender"],
                locale=p["locale"],
                desc=p["desc"],
                sample_url=f"/api/v1/voices/sample/kokoro/{p['id']}",
                tags=p["tags"]
            )
            for p in self.PRESETS
        ]

    def synthesize(
        self,
        text: str,
        voice_id: str = "kokoro_zh_xiaoxiao",
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        apply_polishing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        t0 = time.time()

        preset = next((p for p in self.PRESETS if p["id"] == voice_id), self.PRESETS[0])
        sid = preset["sid"]

        # Run Kokoro Sherpa-ONNX generation
        audio = self._tts.generate(text, sid=sid, speed=rate)
        sample_rate = audio.sample_rate
        raw_samples = np.array(audio.samples, dtype=np.float32)

        if apply_polishing:
            raw_samples = polish_audio_cadence(raw_samples, sample_rate)

        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        sf.write(output_path, raw_samples, sample_rate)

        cost_time = time.time() - t0
        duration = len(raw_samples) / sample_rate if sample_rate > 0 else 0.0
        rtf = cost_time / duration if duration > 0 else 0.0

        return {
            "audio": raw_samples,
            "output_path": output_path,
            "cost_sec": cost_time,
            "duration": duration,
            "rtf": rtf,
            "sample_rate": sample_rate,
            "engine": "kokoro",
            "voice_id": voice_id
        }
