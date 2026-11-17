import os
import sys
import time
import asyncio
import numpy as np
import soundfile as sf
import tempfile

class BaseTTSEngine:
    def synthesize(self, text: str, output_path: str = None) -> dict:
        """
        Synthesizes speech from text.
        Returns: {
            "audio": np.ndarray (float32),
            "sample_rate": int,
            "duration": float,
            "cost_sec": float,
            "output_path": str
        }
        """
        raise NotImplementedError

class EdgeTTSEngine(BaseTTSEngine):
    """
    Microsoft Edge Neural TTS Engine.
    High accuracy (99%+), ultra-realistic, zero VRAM usage, very low latency.
    """
    VOICE_MAP = {
        # === 普通话核心音色 (Mandarin Core) ===
        "yunxi": "zh-CN-YunxiNeural",       # 阳光青年男声 (清爽自然、日常对话首选)
        "xiaoxiao": "zh-CN-XiaoxiaoNeural", # 知性温柔女声 (标准播音/贴心陪伴/有声书)
        "yunjian": "zh-CN-YunjianNeural",   # 沉稳磁性叔音 (影视解说/纪录片/激情叙事)
        "xiaoyi": "zh-CN-XiaoyiNeural",     # 活泼灵动少女 (热情轻快/动漫元气)
        "yunxia": "zh-CN-YunxiaNeural",     # 少年清脆男声 (正太音/活力朝气)
        "yunyang": "zh-CN-YunyangNeural",   # 专业新闻播音男声 (沉稳大气/正式严肃)
        # === 特色方言音色 (Dialects) ===
        "xiaobei": "zh-CN-liaoning-XiaobeiNeural", # 东北方言女声 (幽默风趣/接地气)
        "xiaoni": "zh-CN-shaanxi-XiaoniNeural",   # 陕西方言女声 (质朴亲切/西北腔调)
        "hsiaochen": "zh-TW-HsiaoChenNeural",     # 台湾甜美女声 (机车软萌/温柔亲切)
        "yunjhe": "zh-TW-YunJheNeural",           # 台湾自然青年男声 (日常从容)
        "hiugaai": "zh-HK-HiuGaaiNeural",         # 粤语香港女声 (知性地道)
        "wanlung": "zh-HK-WanLungNeural",         # 粤语香港男声 (成熟磁性)
        # === 英语与多语言推荐 (English) ===
        "andrew": "en-US-AndrewNeural",           # 美语青年男声 (自然流畅)
        "ava": "en-US-AvaNeural",                 # 美语温柔女声 (优雅知性)
    }

    def __init__(self, voice="yunxi", rate="+0%", pitch="+0Hz", volume="+0%"):
        self.voice_name = self.VOICE_MAP.get(voice, voice)
        self.rate = rate
        self.pitch = pitch
        self.volume = volume
        try:
            import edge_tts
            self.edge_tts = edge_tts
        except ImportError:
            raise ImportError("请运行 `pip install edge-tts` 安装 Edge TTS 支持！")

    def synthesize(self, text: str, output_path: str = None) -> dict:
        t0 = time.time()
        need_cleanup = False
        if not output_path:
            fd, output_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            need_cleanup = True

        async def _run():
            communicate = self.edge_tts.Communicate(
                text=text,
                voice=self.voice_name,
                rate=self.rate,
                pitch=self.pitch,
                volume=self.volume
            )
            await communicate.save(output_path)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                executor.submit(asyncio.run, _run()).result()
        else:
            asyncio.run(_run())

        cost_sec = time.time() - t0

        data, sr = sf.read(output_path)
        data = np.array(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration = len(data) / sr

        if need_cleanup and os.path.exists(output_path):
            try:
                os.remove(output_path)
                output_path = None
            except Exception:
                pass

        return {
            "audio": data,
            "sample_rate": sr,
            "duration": duration,
            "cost_sec": cost_sec,
            "output_path": output_path
        }

class ChatTTSEngine(BaseTTSEngine):
    """
    ChatTTS Local Autoregressive Engine.
    """
    def __init__(self, speaker_seed=7869, temperature=0.3, top_p=0.7):
        import torch
        import ChatTTS
        self.torch = torch
        self.chat = ChatTTS.Chat()
        print("正在加载 ChatTTS 权重...")
        self.chat.load(source="huggingface", compile=False)
        self.speaker_seed = speaker_seed
        self.temperature = temperature
        self.top_p = top_p
        torch.manual_seed(speaker_seed)
        self.spk_emb = self.chat.sample_random_speaker()

    def synthesize(self, text: str, output_path: str = None) -> dict:
        t0 = time.time()
        params_infer_code = {
            "spk_emb": self.spk_emb,
            "temperature": self.temperature,
            "top_P": self.top_p,
            "top_K": 20,
            "prompt": "[oral_2][laugh_0][break_4]"
        }
        params_refine_text = {
            "prompt": ""
        }

        wavs = self.chat.infer(
            [text],
            skip_refine_text=True,
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code
        )
        cost_sec = time.time() - t0

        audio = np.array(wavs[0], dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        sr = 24000
        duration = len(audio) / sr

        if output_path:
            sf.write(output_path, audio, sr)

        return {
            "audio": audio,
            "sample_rate": sr,
            "duration": duration,
            "cost_sec": cost_sec,
            "output_path": output_path
        }

def get_tts_engine(engine_type: str = "edge", **kwargs) -> BaseTTSEngine:
    engine_type = engine_type.lower()
    if engine_type in ["edge", "edge-tts", "edgetts"]:
        return EdgeTTSEngine(**kwargs)
    elif engine_type in ["chattts", "chat"]:
        return ChatTTSEngine(**kwargs)
    else:
        raise ValueError(f"未知的 TTS 引擎类型: {engine_type}，支持: 'edge', 'chattts'")
