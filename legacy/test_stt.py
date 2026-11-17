import os
import sys
import time
import re
import argparse
import numpy as np
import soundfile as sf
import scipy.signal as signal
import sherpa_onnx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if LEGACY_DIR not in sys.path:
    sys.path.insert(0, LEGACY_DIR)
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.int8.onnx")
TOKENS_PATH = os.path.join(MODELS_DIR, "tokens.txt")
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

def normalize_text(text):
    """Normalizes text for character comparison (removes punctuation, spaces, lowercases)."""
    text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    return text.lower().strip()

def calculate_cer(reference, hypothesis):
    """
    Computes Character Error Rate (CER) and returns similarity percentage,
    along with detailed difference summary.
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    
    if not ref:
        return 0.0, 100.0, "参考文本为空"

    # Levenshtein distance
    n = len(ref)
    m = len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1   # Substitution
                )

    dist = dp[n][m]
    cer = dist / max(n, 1)
    accuracy = max(0.0, (1.0 - cer) * 100.0)
    return cer, accuracy, (ref, hyp)

class AudioTranscriber:
    def __init__(self, model_path=MODEL_PATH, tokens_path=TOKENS_PATH, threads=4):
        if not os.path.exists(model_path) or not os.path.exists(tokens_path):
            raise FileNotFoundError(f"ASR 模型未找到: {model_path} 或 {tokens_path}")
            
        self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=threads,
            use_itn=True,
            language="zh"
        )

    def transcribe(self, audio_path):
        """
        Transcribes an audio file (.wav, .flac, etc.) to text.
        Returns: (text, duration_sec, cost_sec, rtf)
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        data, sr = sf.read(audio_path)
        data = np.array(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)  # convert to mono

        duration_sec = len(data) / sr

        # Resample to 16kHz for SenseVoice
        if sr != 16000:
            target_len = int(len(data) * 16000 / sr)
            samples_16k = signal.resample(data, target_len).astype(np.float32)
        else:
            samples_16k = data

        t0 = time.time()
        stream = self.asr.create_stream()
        stream.accept_waveform(16000, samples_16k)
        self.asr.decode_stream(stream)
        cost_sec = time.time() - t0

        raw_text = stream.result.text
        # Clean SenseVoice event/emotion tags
        cleaned_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        rtf = cost_sec / duration_sec if duration_sec > 0 else 0.0

        return cleaned_text, duration_sec, cost_sec, rtf

    def verify(self, audio_path, expected_text):
        """
        Verifies if the audio accurately pronounces expected_text.
        """
        hyp_text, duration, cost, rtf = self.transcribe(audio_path)
        cer, accuracy, (ref_norm, hyp_norm) = calculate_cer(expected_text, hyp_text)
        
        print("\n" + "="*60)
        print(" 语音生成正确性校验报告 (Audio Accuracy Report)")
        print("="*60)
        print(f" • 音频文件: {audio_path}")
        print(f" • 音频时长: {duration:.2f}s | 识别耗时: {cost*1000:.1f}ms | 实时率 (RTF): {rtf:.3f}")
        print("-"*60)
        print(f" [期望文本]: {expected_text}")
        print(f" [识别结果]: {hyp_text}")
        print("-"*60)
        
        if accuracy >= 95.0:
            status = " 完美匹配 (发音完全正确)"
        elif accuracy >= 80.0:
            status = " 良好匹配 (个别轻微吞音/同音差异)"
        else:
            status = "❌ 存在明显发音错误或幻觉"

        print(f" • 准确率 (Accuracy)   : {accuracy:.1f}%")
        print(f" • 字符错误率 (CER)     : {cer*100:.1f}%")
        print(f" • 综合评定            : {status}")
        
        if ref_norm != hyp_norm:
            print("\n [逐字核对]:")
            print(f"   规范参考: {ref_norm}")
            print(f"   规范识别: {hyp_norm}")
        print("="*60 + "\n")
        
        return {
            "transcription": hyp_text,
            "expected": expected_text,
            "accuracy": accuracy,
            "cer": cer,
            "duration": duration,
            "cost": cost
        }

def live_mic_transcription():
    """Realtime live microphone listening & transcription test."""
    import sounddevice as sd
    
    print("\n" + "="*60)
    print(" 实时麦克风语音转文字测试 (Live Microphone ASR)")
    print("="*60)
    
    # Auto-detect smart speaker or default mic
    devices = sd.query_devices()
    dev_id = sd.default.device[0]
    dev_name = devices[dev_id]['name']
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0 and '智能音箱' in d['name']:
            dev_id = idx
            dev_name = d['name']
            break
            
    print(f" 麦克风设备: [{dev_id}] {dev_name}")
    print(" 正在加载 VAD 与 SenseVoice-Small 识别引擎...")
    
    vad_config = sherpa_onnx.VadModelConfig()
    vad_config.silero_vad.model = VAD_MODEL_PATH
    vad_config.silero_vad.min_silence_duration = 0.45
    vad_config.silero_vad.min_speech_duration = 0.25
    vad_config.silero_vad.threshold = 0.35
    vad_config.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)
    
    asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL_PATH,
        tokens=TOKENS_PATH,
        num_threads=4,
        use_itn=True,
        language="zh"
    )
    
    import queue
    q = queue.Queue()
    def callback(indata, frames, time_info, status):
        q.put(indata[:, 0].copy())

    print(" 倾听就绪！请对着麦克风说话 (按 Ctrl + C 结束)...")
    print("-" * 60)
    
    try:
        with sd.InputStream(device=dev_id, channels=1, samplerate=16000, blocksize=1024, dtype='float32', callback=callback):
            while True:
                try:
                    samples = q.get(timeout=0.05)
                except queue.Empty:
                    continue
                vad.accept_waveform(samples)
                while not vad.empty():
                    segment = vad.front
                    speech = np.array(segment.samples, dtype=np.float32)
                    vad.pop()
                    if len(speech) >= int(16000 * 0.25):
                        peak = np.max(np.abs(speech))
                        if peak > 0.005:
                            speech = speech / peak * 0.85
                        s = asr.create_stream()
                        s.accept_waveform(16000, speech)
                        asr.decode_stream(s)
                        text = re.sub(r'<\|.*?\|>', '', s.result.text).strip()
                        if text:
                            t_now = time.strftime("%H:%M:%S")
                            print(f"[{t_now}] 🎙️ 识别: {text}")
    except KeyboardInterrupt:
        print("\n\n麦克风识别已结束。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speech-To-Text (ASR) Audio Accuracy Verification Tool")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--text", type=str, default=None, help="Expected ground-truth text for accuracy verification")
    parser.add_argument("--dir", type=str, default=None, help="Batch transcribe all audio files in a directory")
    parser.add_argument("--mic", action="store_true", help="Start live microphone speech-to-text test")
    args = parser.parse_args()

    if args.mic:
        live_mic_transcription()
    elif args.dir:
        transcriber = AudioTranscriber()
        if not os.path.exists(args.dir):
            print(f"❌ 目录不存在: {args.dir}")
            sys.exit(1)
        files = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))]
        if not files:
            print(f"目录中未找到音频文件: {args.dir}")
            sys.exit(0)
            
        print(f"\n批量转文字中: {args.dir} (共 {len(files)} 个文件)")
        print("="*75)
        print(f"{'文件名':<35} | {'时长':<6} | {'识别内容'}")
        print("-"*75)
        for f in sorted(files):
            try:
                text, dur, cost, rtf = transcriber.transcribe(f)
                bname = os.path.basename(f)
                if len(bname) > 33:
                    bname = bname[:30] + "..."
                print(f"{bname:<35} | {dur:4.1f}s | {text}")
            except Exception as e:
                print(f"{os.path.basename(f):<35} | 失败: {e}")
        print("="*75 + "\n")
    elif args.audio:
        transcriber = AudioTranscriber()
        if args.text:
            transcriber.verify(args.audio, args.text)
        else:
            text, dur, cost, rtf = transcriber.transcribe(args.audio)
            print("\n" + "="*50)
            print(" 语音转文字结果 (Transcription Result)")
            print("="*50)
            print(f" • 文件路径: {args.audio}")
            print(f" • 时长: {dur:.2f}s | 耗时: {cost*1000:.1f}ms | RTF: {rtf:.3f}")
            print(f" • 识别文字:\n   \"{text}\"")
            print("="*50 + "\n")
    else:
        # Default: verify one of the generated speaker files as a demo
        demo_audio = os.path.join(BASE_DIR, "speakers", "spk_2222.wav")
        demo_text = "你好呀！我是你的智能语音助手，今天由我来为你服务，你觉得我的声音怎么样呢？"
        if os.path.exists(demo_audio):
            print("未传入参数，正在对预设示例进行校验测试:")
            transcriber = AudioTranscriber()
            transcriber.verify(demo_audio, demo_text)
            print("💡 使用提示:")
            print(" 1. 验证单个音频: python test_stt.py --audio <音频文件.wav>")
            print(" 2. 带参考文本比对准确率: python test_stt.py --audio <音频.wav> --text \"期望文本\"")
            print(" 3. 批量转录目录: python test_stt.py --dir speakers/")
            print(" 4. 实时麦克风转文字: python test_stt.py --mic")
        else:
            parser.print_help()
