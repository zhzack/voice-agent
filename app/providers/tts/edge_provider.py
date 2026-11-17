import os
import time
import asyncio
import concurrent.futures
import tempfile
import numpy as np
import soundfile as sf
from typing import List, Dict, Any, Optional

from app.core.base import BaseTTSProvider, VoiceMetadata
from app.core.sound_processor import polish_audio_cadence

class EdgeTTSProvider(BaseTTSProvider):
    VOICES_DEF = [
        {"id": "yunxi", "name": "云希 (清澈阳光青年男声)", "short": "zh-CN-YunxiNeural", "gender": "male", "locale": "zh-CN", "desc": "阳光自然、干净利落，实时通话最佳默认首选", "tags": ["青年", "阳光", "日常"], "is_default": True},
        {"id": "xiaoxiao", "name": "晓晓 (知性温柔暖心女声)", "short": "zh-CN-XiaoxiaoNeural", "gender": "female", "locale": "zh-CN", "desc": "微软旗舰女声，温润亲和，生活助理与有声书首选", "tags": ["知性", "温柔", "旗舰"]},
        {"id": "yunjian", "name": "云健 (沉稳磁性叙事叔音)", "short": "zh-CN-YunjianNeural", "gender": "male", "locale": "zh-CN", "desc": "短视频电影解说标配，胸腔共鸣浓厚，极具叙事力", "tags": ["解说", "大叔", "磁性"]},
        {"id": "xiaoyi", "name": "小艺 (活泼灵动少女音)", "short": "zh-CN-XiaoyiNeural", "gender": "female", "locale": "zh-CN", "desc": "轻快活泼、生动元气，动漫陪伴与趣味闲聊", "tags": ["少女", "活泼", "元气"]},
        {"id": "yunxia", "name": "云夏 (朝气少年正太音)", "short": "zh-CN-YunxiaNeural", "gender": "male", "locale": "zh-CN", "desc": "清亮稚嫩，少儿伴读与青春活力问答", "tags": ["少年", "正太", "清脆"]},
        {"id": "yunyang", "name": "云扬 (专业新闻播音男声)", "short": "zh-CN-YunyangNeural", "gender": "male", "locale": "zh-CN", "desc": "严肃沉稳、抑扬规范，政企播报与科技发布", "tags": ["新闻", "沉稳", "正式"]},
        {"id": "xiaobei", "name": "小北 (东北方言特色女声)", "short": "zh-CN-liaoning-XiaobeiNeural", "gender": "female", "locale": "zh-CN-liaoning", "desc": "纯正东北味儿，幽默接地气，唠嗑脱口秀", "tags": ["东北话", "幽默", "方言"]},
        {"id": "xiaoni", "name": "小妮 (陕西方言特色女声)", "short": "zh-CN-shaanxi-XiaoniNeural", "gender": "female", "locale": "zh-CN-shaanxi", "desc": "质朴爽朗西北腔调，民俗风情与贴心唠嗑", "tags": ["陕西话", "质朴", "方言"]},
        {"id": "hsiaochen", "name": "小陈 (台湾软萌甜美女声)", "short": "zh-TW-HsiaoChenNeural", "gender": "female", "locale": "zh-TW", "desc": "甜美软萌台湾腔，温柔细腻，慢聊陪伴", "tags": ["台湾腔", "甜美", "治愈"]},
        {"id": "yunjhe", "name": "云哲 (台湾自然青年男声)", "short": "zh-TW-YunJheNeural", "gender": "male", "locale": "zh-TW", "desc": "台湾腔青年小哥，语气从容平缓、轻松日常", "tags": ["台湾腔", "青年", "自然"]},
        {"id": "hiugaai", "name": "晓佳 (粤语香港知性女声)", "short": "zh-HK-HiuGaaiNeural", "gender": "female", "locale": "zh-HK", "desc": "地道粤语发音，知性优雅，粤语区专属", "tags": ["粤语", "知性", "地道"]},
        {"id": "wanlung", "name": "云龙 (粤语香港成熟男声)", "short": "zh-HK-WanLungNeural", "gender": "male", "locale": "zh-HK", "desc": "磁性成熟港风男声，沉稳大气", "tags": ["粤语", "成熟", "港风"]},
        {"id": "andrew", "name": "Andrew (美式纯正青年男声)", "short": "en-US-AndrewNeural", "gender": "male", "locale": "en-US", "desc": "标准美语发音，地道纯正，英语教学与跨语种交流", "tags": ["英语", "美音", "国际"]},
        {"id": "ava", "name": "Ava (美式优雅端庄女声)", "short": "en-US-AvaNeural", "gender": "female", "locale": "en-US", "desc": "优雅知性、流畅清晰的美式女声", "tags": ["英语", "美音", "优雅"]}
    ]

    def __init__(self):
        try:
            import edge_tts
            self.edge_tts = edge_tts
        except ImportError:
            raise ImportError("请运行 pip install edge-tts 安装支持")

        self.id_to_short = {item["id"]: item["short"] for item in self.VOICES_DEF}

    def list_voices(self) -> List[VoiceMetadata]:
        result = []
        for v in self.VOICES_DEF:
            result.append(VoiceMetadata(
                id=v["id"],
                name=v["name"],
                engine="edge",
                gender=v["gender"],
                locale=v["locale"],
                desc=v["desc"],
                sample_url=f"/api/v1/voices/sample/edge/{v['id']}",
                tags=v["tags"],
                is_default=v.get("is_default", False)
            ))
        return result

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
        t0 = time.time()
        short_name = self.id_to_short.get(voice_id, voice_id)

        # Format rate and pitch
        rate_percent = int(round((rate - 1.0) * 100))
        rate_str = f"{rate_percent:+d}%" if rate_percent != 0 else "+0%"

        pitch_hz = int(round(pitch))
        pitch_str = f"{pitch_hz:+d}Hz" if pitch_hz != 0 else "+0Hz"

        need_cleanup = False
        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            need_cleanup = True

        async def _run():
            communicate = self.edge_tts.Communicate(
                text=text,
                voice=short_name,
                rate=rate_str,
                pitch=pitch_str
            )
            await communicate.save(output_path)

        # Handle async inside sync or async loops
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, _run()).result()
        else:
            asyncio.run(_run())

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
