import os
import time
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MELO_DIR = os.path.join(BASE_DIR, "models", "vits-melo-tts-zh_en")

class MeloTTSProvider(BaseTTSProvider):
    """
    Local Lightweight Offline VITS Model (MeloTTS via Sherpa-ONNX).
    44.1kHz high-fidelity, Chinese-English bilingual, pure CPU inference (~1s), 0 VRAM.
    """
    def __init__(self, model_dir: str = MELO_DIR):
        self.model_dir = model_dir
        self._tts = None

    def _ensure_loaded(self):
        if self._tts is None:
            import sherpa_onnx
            config = sherpa_onnx.OfflineTtsConfig()
            config.model.vits.model = os.path.join(self.model_dir, "model.int8.onnx")
            config.model.vits.tokens = os.path.join(self.model_dir, "tokens.txt")
            config.model.vits.lexicon = os.path.join(self.model_dir, "lexicon.txt")
            config.model.vits.dict_dir = os.path.join(self.model_dir, "dict")
            config.model.num_threads = 4
            self._tts = sherpa_onnx.OfflineTts(config)

    def list_voices(self) -> List[VoiceMetadata]:
        return [
            VoiceMetadata(
                id="melotts_zh_en",
                name="MeloTTS (离线极速中英双语女声)",
                engine="melotts",
                gender="female",
                locale="zh-CN",
                desc="纯本地 CPU 极速离线推理，44.1kHz 高保真，中英文混合朗读，零显存占用",
                sample_url="/api/v1/voices/sample/melotts/melotts_zh_en",
                tags=["纯本地", "纯离线", "中英双语", "零显存", "44.1kHz"]
            )
        ]

    def synthesize(
        self,
        text: str,
        voice_id: str = "melotts_zh_en",
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        apply_polishing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        t0 = time.time()

        audio = self._tts.generate(text, sid=0, speed=rate)
        cost_sec = time.time() - t0

        data = np.array(audio.samples, dtype=np.float32)
        sr = audio.sample_rate

        if apply_polishing:
            data = polish_audio_cadence(data, sr)

        duration = len(data) / sr
        rtf = cost_sec / duration if duration > 0 else 0.0

        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        sf.write(output_path, data, sr)

        return {
            "audio": data,
            "sample_rate": sr,
            "duration": duration,
            "cost_sec": cost_sec,
            "rtf": rtf,
            "output_path": output_path
        }
