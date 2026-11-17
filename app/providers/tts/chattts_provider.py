import os
import time
import tempfile
import threading
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

class ChatTTSProvider(BaseTTSProvider):
    PRESETS = [
        {"id": "spk_warm_female", "name": "知性温润女声 (Seed 2222)", "seed": 2222, "gender": "female", "desc": "温润亲切、音质清晰，日常陪伴与知识问答", "tags": ["温润", "知性", "潜变量"]},
        {"id": "spk_young_male", "name": "磁性阳光青年男声 (Seed 7869)", "seed": 7869, "gender": "male", "desc": "声音清爽明亮、充满朝气，男声助理", "tags": ["磁性", "阳光", "潜变量"]},
        {"id": "spk_sweet_female", "name": "甜美灵动少女音 (Seed 4099)", "seed": 4099, "gender": "female", "desc": "语调轻快活泼，充满少女感", "tags": ["少女", "甜美", "潜变量"]},
        {"id": "spk_deep_male", "name": "成熟深沉低音叔音 (Seed 1111)", "seed": 1111, "gender": "male", "desc": "低沉磁性、胸腔共鸣丰富，叙事电台", "tags": ["大叔", "深沉", "低音"]},
        {"id": "spk_gentle_female", "name": "温柔从容暖心女声 (Seed 6653)", "seed": 6653, "gender": "female", "desc": "轻缓柔和，治愈感十足，睡前夜聊", "tags": ["温柔", "治愈", "慢速"]},
        {"id": "spk_vivid_boy", "name": "活力朝气少年音 (Seed 5555)", "seed": 5555, "gender": "male", "desc": "清脆利落、元气满满，年轻化场景", "tags": ["少年", "元气", "清脆"]}
    ]

    def __init__(self):
        self._chat = None
        self._torch = None
        self._cached_spk_embs = {}
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._chat is None:
            import torch
            import ChatTTS
            self._torch = torch
            self._chat = ChatTTS.Chat()
            self._chat.load(source="huggingface", compile=False)

    def list_voices(self) -> List[VoiceMetadata]:
        return [
            VoiceMetadata(
                id=p["id"],
                name=p["name"],
                engine="chattts",
                gender=p["gender"],
                locale="zh-CN",
                desc=p["desc"],
                sample_url=f"/api/v1/voices/sample/chattts/{p['id']}",
                tags=p["tags"]
            )
            for p in self.PRESETS
        ]

    def synthesize(
        self,
        text: str,
        voice_id: str,
        rate: float = 1.0,
        pitch: float = 0.0,
        output_path: Optional[str] = None,
        apply_polishing: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        self._ensure_loaded()
        t0 = time.time()

        # Find seed
        preset = next((p for p in self.PRESETS if p["id"] == voice_id), None)
        seed = preset["seed"] if preset else 2222

        if seed not in self._cached_spk_embs:
            self._torch.manual_seed(seed)
            self._cached_spk_embs[seed] = self._chat.sample_random_speaker()

        spk_emb = self._cached_spk_embs[seed]
        temperature = kwargs.get("temperature", 0.3)
        top_p = kwargs.get("top_p", 0.75)

        oral = kwargs.get("oral", 2)
        laugh = kwargs.get("laugh", 0)
        bk = kwargs.get("bk", 4)

        params_infer = self._chat.InferCodeParams(
            spk_emb=spk_emb,
            temperature=float(temperature),
            top_P=float(top_p),
            top_K=20,
            prompt=f"[oral_{oral}][laugh_{laugh}][break_{bk}]"
        )

        with self._lock:
            wavs = self._chat.infer(
                [text],
                skip_refine_text=True,
                params_infer_code=params_infer
            )
        cost_sec = time.time() - t0

        audio = np.array(wavs[0], dtype=np.float32).squeeze()
        sr = 24000

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
