import os
import sys
import time
import re
import queue
import threading
import argparse
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

# Windows ROCm / PyTorch Distributed Compatibility Patch
import torch.distributed as dist
class _DummyGroup:
    WORLD = None
    SUM = 0
if not hasattr(dist, 'group'):
    dist.group = _DummyGroup
if not hasattr(dist, 'ReduceOp'):
    dist.ReduceOp = _DummyGroup

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import sherpa_onnx
from tts_engine import get_tts_engine

MODELS_DIR = os.path.join(BASE_DIR, "models")
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")
SENSE_VOICE_MODEL_PATH = os.path.join(MODELS_DIR, "model.int8.onnx")
SENSE_VOICE_TOKENS_PATH = os.path.join(MODELS_DIR, "tokens.txt")
DEFAULT_LLM_CACHE = os.path.expanduser(r"~/.cache/modelscope/models/qwen--Qwen2.5-1.5B-Instruct/snapshots/master")
LLM_MODEL_PATH = os.environ.get("LLM_MODEL_PATH", DEFAULT_LLM_CACHE)

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
DEFAULT_SPEAKER_SEED = 2222

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

SPEAKER_PRESETS = {
    "warm_female": {"seed": 2222, "name": "知性温润女声 (ChatTTS推荐)", "desc": "温润亲切、音质清晰"},
    "young_male": {"seed": 7869, "name": "磁性阳光青年男声", "desc": "声音清爽明亮、充满朝气"},
    "sweet_female": {"seed": 4099, "name": "甜美灵动少女音", "desc": "语调轻快活泼"},
    "deep_male": {"seed": 1111, "name": "成熟深沉低音叔音", "desc": "低沉磁性、胸腔共鸣丰富"},
    "gentle_female": {"seed": 6653, "name": "温柔从容暖心女声", "desc": "轻缓柔和，治愈感十足"},
    "vivid_boy": {"seed": 5555, "name": "活力朝气少年音", "desc": "清脆利落、元气满满"}
}

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

def find_preferred_microphone():
    devices = sd.query_devices()
    preferred_idx = None
    for idx, dev in enumerate(devices):
        if dev['max_input_channels'] > 0 and "智能音箱 Pro-8313" in dev['name']:
            preferred_idx = idx
            break
    if preferred_idx is None:
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0 and ("智能音箱" in dev['name'] or "Pro-8313" in dev['name']):
                preferred_idx = idx
                break
    if preferred_idx is None:
        preferred_idx = sd.default.device[0]
    return preferred_idx, devices[preferred_idx]['name']

def polish_audio_cadence(audio, sr=OUTPUT_SAMPLE_RATE, fade_ms=50, pad_ms=280):
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
            audio = audio * (0.91 / max(peak, 1e-4))
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

class RealtimeVoiceCallAgent:
    def __init__(self, engine_type="edge", voice="yunxi", input_device_id=None, speaker="warm_female", speaker_seed=None, mood="warm", enable_barge_in=False):
        print("\n" + "="*60)
        print("   Realtime Voice Call Agent (全双工实时语音通话系统)")
        print("="*60)
        
        self.engine_type = engine_type.lower()
        self.voice = voice
        self.mood = mood if mood in EMOTION_PRESETS else "warm"
        self.preset = EMOTION_PRESETS[self.mood]

        if input_device_id is None:
            self.input_device_id, dev_name = find_preferred_microphone()
        else:
            self.input_device_id = input_device_id
            dev_name = sd.query_devices(self.input_device_id)['name']
            
        print(f" 自动绑定麦克风设备: [{self.input_device_id}] {dev_name}")
        
        # 1. Init Silero VAD on CPU
        print("[1/3] 加载 Silero-VAD 语音活动检测模块 (CPU)...")
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = VAD_MODEL_PATH
        vad_config.silero_vad.min_silence_duration = 0.45
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.silero_vad.threshold = 0.35
        vad_config.sample_rate = INPUT_SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
        
        # 2. Init SenseVoice ASR on CPU
        print("[2/3] 加载 SenseVoice-Small 语音识别大模型 (强制中文精准模式)...")
        self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=SENSE_VOICE_MODEL_PATH,
            tokens=SENSE_VOICE_TOKENS_PATH,
            num_threads=4,
            use_itn=True,
            language="zh"
        )
        
        # 3. Init TTS Engine
        print(f"[3/3] 加载 TTS 语音合成引擎: [{self.engine_type.upper()}]...")
        t0 = time.time()
        if self.engine_type == "edge":
            v_info = EDGE_VOICES.get(voice, {"name": voice, "voice": voice})
            self.speaker_info = v_info
            self.speaker_seed = None
            self.engine = get_tts_engine("edge", voice=voice)
            print(f"       人声模式: 【{v_info['name']}】 (微软生产级神经语音)")
            print(f"       优势特性: 准确率 99%+, 极速生成(~1s), 零吞字, 零幻觉")
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

        print(f"       情感风格: 【{self.preset['name']}】")
        print(f"TTS 引擎初始化完毕，耗时: {time.time() - t0:.2f}s")
        
        # 4. Init LLM on GPU
        print("加载 Qwen2.5-1.5B 对话大语言模型 (RX 9070 XT GPU)...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)
        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="cuda"
        )
        print(f"大语言模型加载完毕，耗时: {time.time() - t0:.2f}s")
        
        # Queues & State
        self.mic_queue = queue.Queue()
        self.playback_queue = queue.Queue()
        
        self.is_ai_speaking = False
        self.enable_barge_in = enable_barge_in
        self.interrupt_event = threading.Event()
        self.stop_system = threading.Event()
        self.history = []
        
        # Start background playback thread
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        self.playback_thread.start()
        
        print("\n 全双工通话系统全部就绪！\n")

    def _playback_worker(self):
        while not self.stop_system.is_set():
            try:
                item = self.playback_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if item is None:
                self.playback_queue.task_done()
                continue

            audio_data, sr = item
            
            if self.interrupt_event.is_set():
                self.playback_queue.task_done()
                continue

            self.is_ai_speaking = True
            try:
                sd.play(audio_data, sr, blocking=True)
            except Exception:
                pass
            finally:
                self.playback_queue.task_done()
                if self.playback_queue.empty():
                    time.sleep(0.20)
                    self.is_ai_speaking = False

    def _clean_text_for_tts(self, text):
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

    def _clean_asr_text(self, text):
        text = re.sub(r'<\|.*?\|>', '', text)
        return text.strip()

    def _synthesize_sentence(self, sentence_text):
        clean_text = self._clean_text_for_tts(sentence_text)
        if not clean_text or len(clean_text) <= 1 or self.interrupt_event.is_set():
            return None
        try:
            res = self.engine.synthesize(clean_text)
            if res and "audio" in res and len(res["audio"]) > 0 and not self.interrupt_event.is_set():
                audio = polish_audio_cadence(res["audio"], res["sample_rate"])
                return audio, res["sample_rate"]
        except Exception:
            pass
        return None

    def _ai_respond_and_stream_tts(self, user_text):
        self.interrupt_event.clear()
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for u, a in self.history[-4:]:
            messages.append({"role": "user", "content": u})
            messages.append({"role": "assistant", "content": a})
        messages.append({"role": "user", "content": user_text})

        text_input = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to("cuda")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

        llm_thread = threading.Thread(target=self.llm.generate, kwargs=gen_kwargs)
        llm_thread.start()

        full_reply = ""
        sentence_buffer = ""
        sentence_delimiters = set("。！？；…\n")
        
        print("🔵 AI: ", end="", flush=True)
        for token in streamer:
            if self.interrupt_event.is_set():
                print(" [已打断]", flush=True)
                break
                
            print(token, end="", flush=True)
            full_reply += token
            sentence_buffer += token

            last_char = token[-1] if len(token) > 0 else ""
            
            if last_char in sentence_delimiters and len(sentence_buffer.strip()) >= 3:
                synth_res = self._synthesize_sentence(sentence_buffer)
                if synth_res is not None and not self.interrupt_event.is_set():
                    audio, sr = synth_res
                    self.playback_queue.put((audio, sr))
                sentence_buffer = ""

        if sentence_buffer.strip() and not self.interrupt_event.is_set():
            synth_res = self._synthesize_sentence(sentence_buffer)
            if synth_res is not None and not self.interrupt_event.is_set():
                audio, sr = synth_res
                self.playback_queue.put((audio, sr))

        llm_thread.join()
        print()
        
        if full_reply.strip():
            self.history.append((user_text, full_reply.strip()))

    def start_call(self):
        print("\n" + "="*60)
        print(" 通话已接通！你可以开始说话了 (智能音箱实时拾音中)")
        print(f" • 当前引擎: 【{self.engine_type.upper()}】")
        print(f" • 当前人声: 【{self.speaker_info['name']}】")
        print(f" • 当前情感: 【{self.preset['name']}】")
        print(" • 已启用【温暖人声音学增强】：胸腔共鸣提升 + 辅音清晰度 + 动态柔和")
        print(" • 已启用【余音平滑淡出 + 自然呼吸延展】：语速从容不仓促")
        print(" • 已启用【顺序无缝播放】：每句必然完整播放完毕，下一句绝不打断上一句")
        print(" • 按 Ctrl + C 随时挂断退出")
        print("="*60 + "\n")

        def mic_callback(indata, frames, time_info, status):
            self.mic_queue.put(indata[:, 0].copy())

        try:
            stream = sd.InputStream(
                device=self.input_device_id,
                channels=1,
                samplerate=INPUT_SAMPLE_RATE,
                blocksize=1024,
                dtype="float32",
                callback=mic_callback
            )
            with stream:
                last_status_time = time.time()
                was_speaking_prev = False

                while not self.stop_system.is_set():
                    try:
                        raw_samples = self.mic_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    # If AI is speaking, prevent echo loops
                    if self.is_ai_speaking:
                        if not was_speaking_prev:
                            sys.stdout.write("\r🔵 AI 正在说话... (麦克风处于静音回声消除中)       ")
                            sys.stdout.flush()
                            was_speaking_prev = True
                        while not self.vad.empty():
                            self.vad.pop()
                        continue

                    if was_speaking_prev:
                        was_speaking_prev = False
                        print("🟢 正在倾听... (请说话)")

                    # Feed clean mic samples to VAD
                    self.vad.accept_waveform(raw_samples)

                    now = time.time()
                    if now - last_status_time > 0.4:
                        max_amp = np.max(np.abs(raw_samples))
                        bars = int(min(max_amp * 40, 20))
                        meter = "█" * bars + "-" * (20 - bars)
                        status_text = " 🎙️ 捕捉到声音" if max_amp > 0.015 else ""
                        sys.stdout.write(f"\r🟢 正在倾听... 音量: [{meter}] {max_amp:.3f}{status_text}   ")
                        sys.stdout.flush()
                        last_status_time = now

                    # Process speech segments
                    while not self.vad.empty():
                        segment = self.vad.front
                        speech_samples = np.array(segment.samples, dtype=np.float32)
                        self.vad.pop()

                        if len(speech_samples) >= int(INPUT_SAMPLE_RATE * 0.3):
                            peak = np.max(np.abs(speech_samples))
                            if peak > 0.005:
                                speech_samples = speech_samples / peak * 0.85

                            stream_asr = self.asr.create_stream()
                            stream_asr.accept_waveform(INPUT_SAMPLE_RATE, speech_samples)
                            self.asr.decode_stream(stream_asr)
                            raw_result = stream_asr.result.text
                            user_text = self._clean_asr_text(raw_result)
                            
                            if user_text and len(user_text) >= 1:
                                print(f"\n\n🟢 用户: \"{user_text}\"")
                                self._ai_respond_and_stream_tts(user_text)
                                print("\n")
                                last_status_time = time.time()

        except KeyboardInterrupt:
            print("\n\n正在挂断通话...")
            self.stop_system.set()
            while not self.playback_queue.empty():
                try:
                    self.playback_queue.get_nowait()
                    self.playback_queue.task_done()
                except Exception:
                    break
            print("通话已结束，系统安全退出。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realtime Voice Call Agent")
    parser.add_argument("--engine", type=str, default="edge", choices=["edge", "chattts"], help="TTS Engine: edge (default) or chattts")
    parser.add_argument("--voice", type=str, default="yunxi", choices=list(EDGE_VOICES.keys()), help="Voice for Edge-TTS (yunxi, xiaoxiao, yunjian, xiaoyi, etc.)")
    parser.add_argument("--device", type=int, default=None, help="Explicit microphone device ID")
    parser.add_argument("--speaker", type=str, default="warm_female", choices=list(SPEAKER_PRESETS.keys()),
                        help="Speaker for ChatTTS: warm_female (default), young_male, sweet_female, deep_male, gentle_female, vivid_boy")
    parser.add_argument("--seed", type=int, default=None, help="Explicit speaker voice seed number for ChatTTS")
    parser.add_argument("--mood", type=str, default="warm", choices=list(EMOTION_PRESETS.keys()),
                        help="Voice emotion style: warm (default), happy, gentle, story")
    parser.add_argument("--barge-in", action="store_true", help="Enable barge-in interruption")
    args = parser.parse_args()

    agent = RealtimeVoiceCallAgent(
        engine_type=args.engine,
        voice=args.voice,
        input_device_id=args.device,
        speaker=args.speaker,
        speaker_seed=args.seed,
        mood=args.mood,
        enable_barge_in=args.barge_in
    )
    agent.start_call()
