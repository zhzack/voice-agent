import os
import sys
import time
import re
import argparse
import numpy as np

# Windows ROCm / PyTorch Distributed Compatibility Patch
import torch.distributed as dist
class _DummyGroup:
    WORLD = None
    SUM = 0
if not hasattr(dist, 'group'):
    dist.group = _DummyGroup
if not hasattr(dist, 'ReduceOp'):
    dist.ReduceOp = _DummyGroup

import torch
import sounddevice as sd
import soundfile as sf
import scipy.signal as signal
import ChatTTS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "prosody_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPEAKER_PRESETS = {
    "warm_female": {"seed": 2222, "name": "知性温润女声 (温和清晰)"},
    "young_male":  {"seed": 7869, "name": "磁性阳光青年男声 (开朗利落，起伏鲜明)"},
    "sweet_female":{"seed": 4099, "name": "甜美灵动少女音 (语调活泼灵巧)"},
    "deep_male":   {"seed": 1111, "name": "成熟深沉低音叔音 (胸腔共鸣浓厚)"},
    "gentle_female":{"seed": 6653, "name": "温柔从容暖心女声 (慢节奏治愈)"},
    "vivid_boy":   {"seed": 5555, "name": "活力朝气少年音 (清脆明快)"},
}

PRESET_TEXTS = {
    "1": ("高尔基《海燕》节选 (激昂抑扬顿挫)", "乌云在天空中翻滚，暴风雨即将来临！我要勇敢地飞翔，即使翅膀被撕裂，也绝不放弃！"),
    "2": ("朱自清《春》节选 (轻柔舒缓节奏)", "盼望着，盼望着，东风来了，春天的脚步近了。小草偷偷地从土里钻出来，嫩嫩的，绿绿的。"),
    "3": ("贴心闺蜜日常问候 (口语化生动起伏)", "好呀！今天外面的阳光可真好，我们待会儿一起去喝杯咖啡吧，你最近过得怎么样呢？"),
    "4": ("深夜电台深情叙事 (低沉深情顿挫)", "夜深了，外面的街道渐渐安静下来。无论今天经历了什么，都请放慢呼吸，给自己一个温柔的拥抱。"),
}

def clean_and_inject_prosody(text, inject_breaks=True):
    """
    Cleans invalid characters and injects natural prosodic breath markers.
    """
    text = re.sub(r'（.*?）|\(.*?\)|\[.*?\]|【.*?】|〔.*?〕', '', text)
    text = re.sub(r'\b(uv_break|laugh|uv|break|oral|smile)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[“”"\'‘’《》〈〉_#*`~^\\/|]', ' ', text)
    text = text.replace("！", "。").replace("!", "。")
    text = text.replace("？", "。").replace("?", "。")
    text = text.replace("：", "，").replace(":", "，")
    
    if inject_breaks:
        # [lbreak] gives ~190ms natural clause breath pause
        # [llbreak] gives ~350ms sentence breath pause
        text = re.sub(r'；|;', '。[llbreak]', text)
        text = re.sub(r'——|—|-', '[lbreak]', text)
        text = re.sub(r'，|,', '，[lbreak]', text)
        
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in ["。", "，", "]"]:
        text += "。"
    return text

def polish_audio(audio, sr=24000):
    """Acoustic resonance dressing & cadence breath expansion."""
    audio = np.array(audio, dtype=np.float32).squeeze()
    if len(audio) == 0:
        return audio

    try:
        # Warmth resonance filter (250Hz, +1.5dB)
        w0_low = 250.0 / (sr / 2.0)
        b_low, a_low = signal.iirpeak(w0_low, Q=1.0)
        audio = audio + 0.15 * signal.lfilter(b_low, a_low, audio)

        # Articulation filter (3500Hz, +1.0dB)
        w0_high = 3500.0 / (sr / 2.0)
        b_high, a_high = signal.iirpeak(w0_high, Q=1.2)
        audio = audio + 0.12 * signal.lfilter(b_high, a_high, audio)

        # Soft compression
        audio = np.tanh(audio * 1.15) / 1.15

        # Normalize
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.91 / max(peak, 1e-4))
    except Exception:
        pass

    # Half-cosine tail fade (45ms) + breath silence (280ms)
    fade_len = int(sr * 0.045)
    pad_len = int(sr * 0.280)
    if len(audio) > fade_len:
        fade_curve = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, fade_len, dtype=np.float32)))
        audio[-fade_len:] *= fade_curve

    head_silence = np.zeros(int(sr * 0.02), dtype=np.float32)
    tail_silence = np.zeros(pad_len, dtype=np.float32)
    return np.concatenate([head_silence, audio, tail_silence])

class ProsodyStudio:
    def __init__(self):
        print("\n" + "="*60)
        print("   ChatTTS 抑扬顿挫与拟人韵律实验室 (Prosody Studio)")
        print("="*60)
        print("正在加载 ChatTTS 声学引擎 (AMD RX 9070 XT GPU)...")
        self.chat = ChatTTS.Chat()
        self.chat.load(source="huggingface", compile=False)
        print("引擎就绪！\n")

    def synthesize(self, text, seed=7869, oral_level=7, laugh_level=0, temperature=0.32, top_p=0.75, inject_breaks=True, play=True, verify=False):
        """
        Synthesizes text with explicit prosody and intonation controls.
        """
        cleaned_text = clean_and_inject_prosody(text, inject_breaks=inject_breaks)
        
        torch.manual_seed(seed)
        spk = self.chat.sample_random_speaker()
        
        # Build prosody prompt
        prompt = f"[oral_{oral_level}]"
        if laugh_level > 0:
            prompt += f"[laugh_{laugh_level}]"
        prompt += "[break_4]"
        
        params = ChatTTS.Chat.InferCodeParams(
            spk_emb=spk,
            prompt=prompt,
            temperature=temperature,
            top_P=top_p,
            top_K=20,
            manual_seed=seed
        )
        
        t0 = time.time()
        wavs = self.chat.infer(
            [cleaned_text],
            params_infer_code=params,
            skip_refine_text=True
        )
        cost_time = time.time() - t0
        
        if not wavs or len(wavs) == 0:
            print("❌ 合成失败！")
            return None
            
        raw_audio = np.array(wavs[0], dtype=np.float32).squeeze()
        audio = polish_audio(raw_audio, 24000)
        duration = len(audio) / 24000
        rtf = cost_time / duration if duration > 0 else 0
        
        # Save output
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prosody_s{seed}_t{int(temperature*100)}_{timestamp}.wav"
        out_path = os.path.join(OUTPUT_DIR, filename)
        sf.write(out_path, audio, 24000)
        
        print("\n" + "-"*50)
        print(f"🎵 合成成功: {filename}")
        print(f" • 音频时长: {duration:.2f} 秒 | 耗时: {cost_time:.2f} 秒 | RTF: {rtf:.3f}")
        print(f" • 参数配置: Seed={seed} | Oral={oral_level} | Temp={temperature:.2f} | TopP={top_p:.2f}")
        print(f" • 韵律标记: {'已启用 [lbreak] 逗号生理换气' if inject_breaks else '纯文字连读模式'}")
        print(f" • 保存路径: {out_path}")
        print("-"*50)
        
        if verify:
            try:
                from test_stt import AudioTranscriber
                transcriber = AudioTranscriber()
                transcriber.verify(out_path, text)
            except Exception as e:
                print(f"语音校验提示: {e}")

        if play:
            print("🔊 正在播放合成语音...")
            try:
                sd.play(audio, 24000, blocking=True)
            except Exception as e:
                print(f"播放受阻: {e}")
        return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatTTS Prosody & Intonation Studio")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="young_male", choices=list(SPEAKER_PRESETS.keys()), help="Speaker timbre preset")
    parser.add_argument("--seed", type=int, default=None, help="Explicit speaker seed")
    parser.add_argument("--oral", type=int, default=7, choices=range(0, 10), help="Oral liveliness level (0-9)")
    parser.add_argument("--laugh", type=int, default=0, choices=range(0, 3), help="Smiling/laugh level (0-2)")
    parser.add_argument("--temp", type=float, default=0.32, help="Temperature for pitch dynamics (0.2-0.5)")
    parser.add_argument("--no-break", action="store_true", help="Disable [lbreak] pause injection")
    parser.add_argument("--no-play", action="store_true", help="Do not play audio after synthesis")
    parser.add_argument("--verify", action="store_true", help="Auto-verify generated audio with Speech-To-Text (ASR)")
    args = parser.parse_args()

    studio = ProsodyStudio()
    
    # Resolve speaker seed
    if args.seed is not None:
        actual_seed = args.seed
        spk_name = f"自定义 Seed {actual_seed}"
    else:
        actual_seed = SPEAKER_PRESETS[args.speaker]["seed"]
        spk_name = SPEAKER_PRESETS[args.speaker]["name"]

    if args.text:
        studio.synthesize(
            text=args.text,
            seed=actual_seed,
            oral_level=args.oral,
            laugh_level=args.laugh,
            temperature=args.temp,
            inject_breaks=not args.no_break,
            play=not args.no_play,
            verify=args.verify
        )
    else:
        print("请选择要体验的经典抑扬顿挫段落 (输入 1~4，或直接输入自定义文本):")
        for k, (title, content) in PRESET_TEXTS.items():
            print(f"  [{k}] {title}:")
            print(f"      \"{content}\"")
        
        while True:
            try:
                user_choice = input("\n请输入测试编号 [1~4] 或自定义文字 (输入 'exit' 退出): ").strip()
                if not user_choice:
                    user_choice = "1"
                if user_choice.lower() in ["exit", "quit", "q"]:
                    print("退出实验室。")
                    break
                
                target_text = PRESET_TEXTS[user_choice][1] if user_choice in PRESET_TEXTS else user_choice
                
                print(f"\n正在使用【{spk_name}】合成: \"{target_text}\"")
                studio.synthesize(
                    text=target_text,
                    seed=actual_seed,
                    oral_level=args.oral,
                    laugh_level=args.laugh,
                    temperature=args.temp,
                    inject_breaks=not args.no_break,
                    play=not args.no_play,
                    verify=args.verify
                )
            except (KeyboardInterrupt, EOFError):
                print("\n已退出。")
                break
