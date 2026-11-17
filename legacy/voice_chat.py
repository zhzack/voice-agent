import os
import sys
import time
import re
import queue
import threading
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import soundfile as sf
import torch

# Ensure local voice_agent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from tts_engine import get_tts_engine

# Windows ROCm / PyTorch Distributed Compatibility Patch
import torch.distributed as dist
class _DummyGroup:
    WORLD = None
    SUM = 0
if not hasattr(dist, 'group'):
    dist.group = _DummyGroup
if not hasattr(dist, 'ReduceOp'):
    dist.ReduceOp = _DummyGroup

DEFAULT_LLM_CACHE = os.path.expanduser(r"~/.cache/modelscope/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master")
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", DEFAULT_LLM_CACHE)
DEFAULT_SPEAKER_SEED = 2222
SAMPLE_RATE = 24000

# Edge-TTS Voice Presets
EDGE_VOICES = {
    # 普通话核心声线
    "yunxi": {"name": "云希 (清澈阳光青年男声 - 默认推荐)", "voice": "zh-CN-YunxiNeural"},
    "xiaoxiao": {"name": "晓晓 (知性温柔暖心女声 - 旗舰)", "voice": "zh-CN-XiaoxiaoNeural"},
    "yunjian": {"name": "云健 (沉稳磁性叙事叔音 - 解说标配)", "voice": "zh-CN-YunjianNeural"},
    "xiaoyi": {"name": "小艺 (活泼灵动少女音 - 元气)", "voice": "zh-CN-XiaoyiNeural"},
    "yunxia": {"name": "云夏 (朝气少年男声 - 正太)", "voice": "zh-CN-YunxiaNeural"},
    "yunyang": {"name": "云扬 (专业新闻播音男声 - 严肃稳重)", "voice": "zh-CN-YunyangNeural"},
    # 特色方言
    "xiaobei": {"name": "小北 (东北方言女声 - 幽默唠嗑)", "voice": "zh-CN-liaoning-XiaobeiNeural"},
    "xiaoni": {"name": "小妮 (陕西方言女声 - 质朴爽朗)", "voice": "zh-CN-shaanxi-XiaoniNeural"},
    "hsiaochen": {"name": "小陈 (台湾软萌甜美女声)", "voice": "zh-TW-HsiaoChenNeural"},
    "yunjhe": {"name": "云哲 (台湾自然青年男声)", "voice": "zh-TW-YunJheNeural"},
    "hiugaai": {"name": "晓佳 (粤语香港知性女声)", "voice": "zh-HK-HiuGaaiNeural"},
    "wanlung": {"name": "云龙 (粤语香港成熟男声)", "voice": "zh-HK-WanLungNeural"},
    # 英语
    "andrew": {"name": "Andrew (美式地道青年男声)", "voice": "en-US-AndrewNeural"},
    "ava": {"name": "Ava (美式优雅从容女声)", "voice": "en-US-AvaNeural"}
}

# ChatTTS Speaker Timbre Presets
SPEAKER_PRESETS = {
    "warm_female": {"seed": 2222, "name": "知性温润女声 (ChatTTS推荐)", "desc": "温润亲切、音质清晰"},
    "young_male": {"seed": 7869, "name": "磁性阳光青年男声", "desc": "声音清爽明亮、充满朝气"},
    "sweet_female": {"seed": 4099, "name": "甜美灵动少女音", "desc": "语调轻快活泼"},
    "deep_male": {"seed": 1111, "name": "成熟深沉低音叔音", "desc": "低沉磁性、胸腔共鸣丰富"},
    "gentle_female": {"seed": 6653, "name": "温柔从容暖心女声", "desc": "轻缓柔和，治愈感十足"},
    "vivid_boy": {"seed": 5555, "name": "活力朝气少年音", "desc": "清脆利落、元气满满"}
}

# Emotion & Prosody Presets
EMOTION_PRESETS = {
    "warm": {
        "name": "温暖亲切 (日常陪伴)",
        "prompt": "[oral_2][laugh_0][break_4]",
        "temperature": 0.28,
        "top_P": 0.70,
        "top_K": 20,
        "desc": "发音饱满清晰，语气亲切温和，节奏自然舒适"
    },
    "happy": {
        "name": "欢快活泼 (阳光互动)",
        "prompt": "[oral_5][laugh_2][break_3]",
        "temperature": 0.40,
        "top_P": 0.75,
        "top_K": 20,
        "desc": "轻快上扬，带有轻微自然笑意，生动活泼"
    },
    "gentle": {
        "name": "沉静舒缓 (晚安哄睡)",
        "prompt": "[oral_1][laugh_0][break_6]",
        "temperature": 0.22,
        "top_P": 0.65,
        "top_K": 20,
        "desc": "语速稍缓，轻柔平稳，带有安抚陪伴质感"
    },
    "story": {
        "name": "深情朗诵 (有声书/散文)",
        "prompt": "[oral_2][laugh_0][break_5]",
        "temperature": 0.30,
        "top_P": 0.72,
        "top_K": 20,
        "desc": "抑扬顿挫分明，重音有力，气口从容，极富叙事感染力"
    }
}

SYSTEM_PROMPT = (
    "你是一个本地全双工实时语音助手，声音富有感情、自然温和。"
    "在回答时，请像正常人说话一样交谈，语言简明扼要，直接切入重点，"
    "单次回答尽量控制在1到3句话以内，避免列举长列表，使用口语化表达。"
    "严禁输出任何括号形式的旁白或动作说明（例如不要输出'（微笑）'或'（深情地说）'）。"
)

def polish_audio_cadence(audio, sr=SAMPLE_RATE, fade_ms=50, pad_ms=280):
    """
    Polishes audio for human-like cadence and warmth.
    """
    if audio is None or len(audio) == 0:
        return audio

    try:
        w0_low = 250.0 / (sr / 2.0)
        b_low, a_low = signal.iirpeak(w0_low, Q=1.0)
        warm_filtered = signal.lfilter(b_low, a_low, audio)
        audio = audio + 0.12 * warm_filtered

        w0_high = 3500.0 / (sr / 2.0)
        b_high, a_high = signal.iirpeak(w0_high, Q=1.2)
        clarity_filtered = signal.lfilter(b_high, a_high, audio)
        audio = audio + 0.10 * clarity_filtered

        audio = np.tanh(audio * 1.1) / 1.1

        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.92 / max(peak, 1e-4))
    except Exception:
        pass

    fade_len = int(sr * fade_ms / 1000)
    pad_len = int(sr * pad_ms / 1000)

    if len(audio) > fade_len:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        audio[-fade_len:] *= fade_curve

    head_silence = np.zeros(int(sr * 0.02), dtype=np.float32)
    tail_silence = np.zeros(pad_len, dtype=np.float32)

    return np.concatenate([head_silence, audio, tail_silence])

class RealtimeVoiceAgent:
    def __init__(self, engine_type="edge", voice="yunxi", speaker="warm_female", speaker_seed=None, mood="warm"):
        print("\n" + "="*50)
        print("  Realtime Voice Agent 初始化中...")
        print("="*50)
        
        self.engine_type = engine_type.lower()
        self.voice = voice
        self.mood = mood if mood in EMOTION_PRESETS else "warm"
        self.preset = EMOTION_PRESETS[self.mood]

        # 1. Initialize TTS Engine
        print(f"[1/2] 正在加载 TTS 语音合成引擎: [{self.engine_type.upper()}]...")
        t0 = time.time()
        if self.engine_type == "edge":
            v_info = EDGE_VOICES.get(voice, {"name": voice, "voice": voice})
            self.speaker_info = v_info
            self.speaker_seed = None
            self.engine = get_tts_engine("edge", voice=voice)
            print(f"       人声模式: 【{v_info['name']}】 (微软生产级神经语音)")
            print(f"       特性优势: 准确率 99%+, 零吞字, 零幻觉, 极速流式生成")
        else:
            if speaker in SPEAKER_PRESETS:
                self.speaker_info = SPEAKER_PRESETS[speaker]
                actual_seed = self.speaker_info["seed"] if speaker_seed is None else speaker_seed
            else:
                actual_seed = speaker_seed if speaker_seed is not None else DEFAULT_SPEAKER_SEED
                self.speaker_info = {"name": f"自定义音色 (Seed: {actual_seed})", "desc": "用户自定义种子"}
            self.speaker_seed = actual_seed
            self.engine = get_tts_engine("chattts", speaker_seed=self.speaker_seed, temperature=self.preset["temperature"], top_p=self.preset["top_P"])
            print(f"       人声模式: 【{self.speaker_info['name']}】 (Seed: {self.speaker_seed})")

        print(f"       情感模式: 【{self.preset['name']}】")
        print(f"TTS 引擎就绪，耗时: {time.time() - t0:.2f}s")
        
        # 2. Load LLM
        print("[2/2] 正在加载 Qwen2.5-1.5B-Instruct 大语言模型到 GPU...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        print(f"大语言模型加载完成，耗时: {time.time() - t0:.2f}s")
        
        # Audio Playback Queue
        self.audio_queue = queue.Queue()
        self.stop_signal = threading.Event()
        
        # Start background playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        print("\n 全双工语音对话引擎就绪！(已开启声学温暖增强与自然情感律动)")

    def _playback_worker(self):
        while not self.stop_signal.is_set():
            try:
                item = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            if item is None:
                self.audio_queue.task_done()
                continue
            
            audio_data, sample_rate = item
            try:
                sd.play(audio_data, sample_rate)
                sd.wait()
            except Exception:
                pass
            finally:
                self.audio_queue.task_done()

    def _clean_text_for_tts(self, text):
        # Remove parenthetical stage directions completely
        text = re.sub(r'（.*?）|\(.*?\)|\[.*?\]|【.*?】|《.*?》', '', text)
        text = re.sub(r'\b(uv_break|laugh|uv|break|oral|smile)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[“”"\'‘’《》〈〉[\]#*`~^\\/|]', ' ', text)
        text = text.replace("！", "。").replace("!", "。")
        text = text.replace("？", "。").replace("?", "。")
        text = text.replace("：", "，").replace(":", "，")
        text = re.sub(r'\s+', ' ', text).strip()
        if text and text[-1] not in ["。", "！", "？"]:
            text += "。"
        return text

    def _synthesize_chunk(self, chunk):
        clean_chunk = self._clean_text_for_tts(chunk)
        if not clean_chunk or len(clean_chunk) <= 1:
            return None
        try:
            res = self.engine.synthesize(clean_chunk)
            if res and "audio" in res and len(res["audio"]) > 0:
                audio = polish_audio_cadence(res["audio"], res["sample_rate"])
                return audio, res["sample_rate"]
        except Exception:
            pass
        return None

    def chat_turn(self, user_query, history=None):
        if history is None:
            history = []
            
        print(f"\n用户: {user_query}")
        print("AI: ", end="", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in history:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_query})

        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=140,
            temperature=0.7,
            top_p=0.9,
        )

        llm_thread = threading.Thread(target=self.llm.generate, kwargs=generation_kwargs)
        t_start = time.time()
        llm_thread.start()

        full_reply = ""
        sentence_buffer = ""
        first_token_time = None
        first_audio_time = None
        first_chunk_emitted = False
        delimiters = set("。！？；…\n")
        soft_delimiters = set("，,")

        for token in streamer:
            if first_token_time is None:
                first_token_time = time.time() - t_start
                
            print(token, end="", flush=True)
            full_reply += token
            sentence_buffer += token

            should_flush = False
            last_char = token[-1] if len(token) > 0 else ""
            
            # Fast-first-chunk
            if not first_chunk_emitted:
                if last_char in delimiters or (last_char in soft_delimiters and len(sentence_buffer.strip()) >= 5):
                    should_flush = True
                    first_chunk_emitted = True
            else:
                if last_char in delimiters:
                    should_flush = True
                elif last_char in soft_delimiters and len(sentence_buffer.strip()) >= 8:
                    should_flush = True
                elif len(sentence_buffer.strip()) >= 22:
                    should_flush = True

            if should_flush:
                chunk_to_speak = sentence_buffer.strip()
                sentence_buffer = ""
                synth_res = self._synthesize_chunk(chunk_to_speak)
                if synth_res is not None:
                    audio_data, sr = synth_res
                    if first_audio_time is None:
                        first_audio_time = time.time() - t_start
                    self.audio_queue.put((audio_data, sr))

        # Flush any trailing text
        if sentence_buffer.strip():
            synth_res = self._synthesize_chunk(sentence_buffer.strip())
            if synth_res is not None:
                audio_data, sr = synth_res
                if first_audio_time is None:
                    first_audio_time = time.time() - t_start
                self.audio_queue.put((audio_data, sr))

        llm_thread.join()
        t_total = time.time() - t_start

        print("\n" + "-"*40)
        ttft_str = f"{first_token_time*1000:.0f}ms" if first_token_time else "N/A"
        ttfa_str = f"{first_audio_time*1000:.0f}ms" if first_audio_time else "N/A"
        print(f"⏱️ 性能指标: LLM 首字: {ttft_str} | 首包出声: {ttfa_str} | 全句完成: {t_total:.2f}s")
        print("-"*40)

        self.audio_queue.join()
        return full_reply

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Realtime Voice Agent")
    parser.add_argument("--engine", type=str, default="edge", choices=["edge", "chattts"], help="TTS Engine: edge (default, 99%+ accurate) or chattts")
    parser.add_argument("--voice", type=str, default="yunxi", choices=list(EDGE_VOICES.keys()),
                        help="Voice for Edge-TTS: yunxi (default, 青年男声), xiaoxiao (温润女声), yunjian (叔音), xiaoyi (少女)")
    parser.add_argument("--speaker", type=str, default="warm_female", choices=list(SPEAKER_PRESETS.keys()),
                        help="Speaker for ChatTTS: warm_female (default), young_male, sweet_female, deep_male, gentle_female, vivid_boy")
    parser.add_argument("--seed", type=int, default=None, help="Explicit speaker voice seed number for ChatTTS")
    parser.add_argument("--mood", type=str, default="warm", choices=list(EMOTION_PRESETS.keys()),
                        help="Voice emotion style: warm (default), happy, gentle, story")
    args = parser.parse_args()

    agent = RealtimeVoiceAgent(engine_type=args.engine, voice=args.voice, speaker=args.speaker, speaker_seed=args.seed, mood=args.mood)
    
    print("\n" + "="*50)
    print(" 开始体验大模型 + 实时语音对话！")
    print(f" 当前引擎: 【{args.engine.upper()}】")
    print(f" 人声音色: 【{agent.speaker_info['name']}】")
    print(f" 情感风格: 【{agent.preset['name']}】")
    print(" 输入 'exit' 或 'quit' 退出。")
    print("="*50)

    history = []
    while True:
        try:
            user_input = input("\n请输入你的问题 (按 Enter 发送): ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("会话结束，再见！")
                agent.stop_signal.set()
                break
            reply = agent.chat_turn(user_input, history)
            history.append((user_input, reply))
        except KeyboardInterrupt:
            print("\n会话结束，再见！")
            agent.stop_signal.set()
            break
