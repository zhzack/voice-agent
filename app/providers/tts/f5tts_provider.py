import os
import time
import tempfile
import threading
import numpy as np
import soundfile as sf
import torch
import torchaudio
from typing import List, Dict, Any, Optional

# Compatibility monkeypatch for torchaudio 2.9 on Windows
def _safe_torchaudio_load(uri, **kwargs):
    d, sr = sf.read(uri)
    t = torch.from_numpy(d.T if d.ndim > 1 else d.reshape(1, -1)).float()
    return t, sr

torchaudio.load = _safe_torchaudio_load

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPEAKERS_DIR = os.path.join(BASE_DIR, "speakers")

class F5TTSProvider(BaseTTSProvider):
    """
    Local Non-Autoregressive Flow Matching Diffusion Zero-Shot Voice Cloning TTS (F5-TTS).
    100% Local GPU accelerated, zero hallucinations, clones any voice in 5 seconds.
    """
    PRESETS = [
        {
            "id": "f5_clone_warm_female",
            "name": "F5-TTS 声音克隆 (知性女声)",
            "ref_audio": "spk_2222.wav",
            "ref_text": "春天的微风轻拂过大地，带来了温暖与生机。",
            "gender": "female",
            "desc": "基于 3 秒知性女声录音进行流匹配零样本克隆，发音自然生动",
            "tags": ["流匹配", "零样本克隆", "纯本地", "GPU加速", "零幻觉"]
        },
        {
            "id": "f5_clone_young_male",
            "name": "F5-TTS 声音克隆 (磁性男声)",
            "ref_audio": "spk_7869.wav",
            "ref_text": "海浪拍打着金色的沙滩，夕阳染红了天边。",
            "gender": "male",
            "desc": "基于 3 秒阳光男声录音进行流匹配零样本克隆，充满朝气",
            "tags": ["流匹配", "零样本克隆", "纯本地", "GPU加速", "零幻觉"]
        },
        {
            "id": "f5_clone_deep_male",
            "name": "F5-TTS 声音克隆 (成熟叔音)",
            "ref_audio": "spk_1111.wav",
            "ref_text": "在深夜的电台里，听一段古老的故事。",
            "gender": "male",
            "desc": "基于成熟低沉男声参考音频进行零样本克隆，胸腔共鸣浓厚",
            "tags": ["流匹配", "零样本克隆", "纯本地", "GPU加速", "零幻觉"]
        },
        {
            "id": "f5_custom_clone",
            "name": "✨ 自定义克隆 (上传或录制 3~5 秒你自己的声音)",
            "ref_audio": "",
            "ref_text": "",
            "gender": "custom",
            "desc": "上传或录制您自己的 3~5 秒语音，SenseVoice 自动转写并以您的音色合成任意文本",
            "tags": ["我的声音", "声音克隆", "零样本", "纯本地"]
        }
    ]

    def __init__(self):
        self._f5 = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._f5 is None:
            print(f"正在加载 F5-TTS 非自回归流匹配扩散模型至 {self.device.upper()}...")
            from f5_tts.api import F5TTS
            self._f5 = F5TTS(device=self.device)

    def list_voices(self) -> List[VoiceMetadata]:
        return [
            VoiceMetadata(
                id=p["id"],
                name=p["name"],
                engine="f5tts",
                gender=p["gender"],
                locale="zh-CN",
                desc=p["desc"],
                sample_url=f"/api/v1/voices/sample/f5tts/{p['id']}",
                tags=p["tags"]
            )
            for p in self.PRESETS
        ]

    def synthesize(
        self,
        text: str,
        voice_id: str = "f5_clone_warm_female",
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        apply_polishing: bool = True,
        custom_ref_audio: Optional[str] = None,
        custom_ref_text: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        t0 = time.time()

        preset = next((p for p in self.PRESETS if p["id"] == voice_id), self.PRESETS[0])

        if custom_ref_audio and os.path.exists(custom_ref_audio):
            ref_file = custom_ref_audio
            ref_text = custom_ref_text or ""
        else:
            ref_file = os.path.join(SPEAKERS_DIR, preset["ref_audio"])
            ref_text = preset["ref_text"]

        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

        # Execute F5-TTS Flow Matching Inference
        with self._lock:
            wav, sr, _ = self._f5.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=text,
                file_wave=output_path,
                speed=rate,
                nfe_step=kwargs.get("nfe_step", 32),
                cfg_strength=kwargs.get("cfg_strength", 2.0),
                sway_sampling_coef=kwargs.get("sway_sampling_coef", -1.0)
            )
        cost_sec = time.time() - t0

        data, sr = sf.read(output_path)
        data = np.array(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)

        if apply_polishing:
            data = polish_audio_cadence(data, sr)
            sf.write(output_path, data, sr)

        duration = len(data) / sr
        rtf = cost_sec / duration if duration > 0 else 0.0

        return {
            "audio": data,
            "sample_rate": sr,
            "duration": duration,
            "cost_sec": cost_sec,
            "rtf": rtf,
            "output_path": output_path
        }
