import os
import sys
import time
import re
import queue
import threading
import numpy as np
import soundfile as sf
import scipy.signal as signal
from typing import Dict, Any, Tuple, List, Optional, Iterator

from app.core.base import BaseSTTProvider

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.int8.onnx")
TOKENS_PATH = os.path.join(MODELS_DIR, "tokens.txt")
VAD_MODEL_PATH = os.path.join(MODELS_DIR, "silero_vad.onnx")

def normalize_text(text: str) -> str:
    # 过滤控制标记如 [laugh], [sigh], [break_500ms], [uv_break] 等
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fa5]', '', text)
    return text.lower().strip()

def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, float, Tuple[str, str]]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0, 100.0, ("", "")

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
                    dp[i - 1][j] + 1,
                    dp[i][j - 1] + 1,
                    dp[i - 1][j - 1] + 1
                )

    dist = dp[n][m]
    cer = dist / max(n, 1)
    accuracy = max(0.0, (1.0 - cer) * 100.0)
    return cer, accuracy, (ref, hyp)

def query_microphones() -> List[Dict[str, Any]]:
    """获取本机所有可用音频输入麦克风设备列表"""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_in = sd.default.device[0]
        results = []
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                results.append({
                    "id": idx,
                    "name": d.get("name", f"Microphone-{idx}"),
                    "channels": d.get("max_input_channels", 1),
                    "default_samplerate": d.get("default_samplerate", 16000),
                    "is_default": (idx == default_in)
                })
        return results
    except Exception as e:
        print(f"[Microphone Query Warning] {e}")
        return []

def find_preferred_microphone() -> Tuple[int, str]:
    """智能优选麦克风（音箱/外接降噪麦/系统默认设备）"""
    try:
        import sounddevice as sd
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
    except Exception as e:
        return 0, "Default Microphone"

class SenseVoiceSTTProvider(BaseSTTProvider):
    def __init__(self, model_path: str = MODEL_PATH, tokens_path: str = TOKENS_PATH):
        import sherpa_onnx
        if not os.path.exists(model_path) or not os.path.exists(tokens_path):
            raise FileNotFoundError(f"ASR 模型未找到: {model_path}")

        self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=4,
            use_itn=True,
            language="zh"
        )

    def transcribe(self, audio_path: str) -> Dict[str, Any]:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        data, sr = sf.read(audio_path)
        data = np.array(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration_sec = len(data) / sr

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
        cleaned_text = re.sub(r'<\|.*?\|>', '', raw_text).strip()
        rtf = cost_sec / duration_sec if duration_sec > 0 else 0.0

        return {
            "text": cleaned_text,
            "duration": duration_sec,
            "cost_sec": cost_sec,
            "rtf": rtf
        }

    def verify(self, audio_path: str, expected_text: str) -> Dict[str, Any]:
        res = self.transcribe(audio_path)
        hyp_text = res["text"]
        cer, accuracy, (ref_norm, hyp_norm) = calculate_cer(expected_text, hyp_text)

        status = "完美" if accuracy >= 95.0 else ("良好" if accuracy >= 80.0 else "瑕疵")
        return {
            "expected_text": expected_text,
            "recognized_text": hyp_text,
            "accuracy": round(accuracy, 2),
            "cer": round(cer, 4),
            "status": status,
            "duration": round(res["duration"], 2),
            "cost_sec": round(res["cost_sec"], 3),
            "rtf": round(res["rtf"], 3)
        }

class LiveMicRecognizer:
    """
    麦克风实时边录音边识别处理器
    结合 Silero-VAD 实时语音端点检测与 SenseVoice-Small 极速精准解码
    """
    def __init__(
        self,
        model_path: str = MODEL_PATH,
        tokens_path: str = TOKENS_PATH,
        vad_model_path: str = VAD_MODEL_PATH,
        threads: int = 4
    ):
        import sherpa_onnx
        if not os.path.exists(model_path) or not os.path.exists(tokens_path):
            raise FileNotFoundError(f"SenseVoice 模型未找到: {model_path}")
        if not os.path.exists(vad_model_path):
            raise FileNotFoundError(f"Silero VAD 模型未找到: {vad_model_path}")

        self.model_path = model_path
        self.tokens_path = tokens_path
        self.vad_model_path = vad_model_path

        self.asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=threads,
            use_itn=True,
            language="zh"
        )

    def _create_vad(self, min_silence_duration: float = 0.45, min_speech_duration: float = 0.25, threshold: float = 0.35):
        import sherpa_onnx
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = self.vad_model_path
        vad_config.silero_vad.min_silence_duration = min_silence_duration
        vad_config.silero_vad.min_speech_duration = min_speech_duration
        vad_config.silero_vad.threshold = threshold
        vad_config.sample_rate = 16000
        return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)

    def listen(
        self,
        device_id: Optional[int] = None,
        min_silence_duration: float = 0.45,
        min_speech_duration: float = 0.25,
        threshold: float = 0.35,
        max_duration: Optional[float] = None,
        stop_event: Optional[threading.Event] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        调起本地麦克风，开启流式监听，每检测并转写出一句话即 yield 出结果
        """
        import sounddevice as sd

        if device_id is None:
            dev_id, dev_name = find_preferred_microphone()
        else:
            dev_id = device_id
            devices = sd.query_devices()
            dev_name = devices[dev_id]['name'] if 0 <= dev_id < len(devices) else f"Device-{dev_id}"

        vad = self._create_vad(
            min_silence_duration=min_silence_duration,
            min_speech_duration=min_speech_duration,
            threshold=threshold
        )

        audio_queue: queue.Queue = queue.Queue()

        def audio_callback(indata, frames, time_info, status):
            if status:
                pass
            audio_queue.put(indata[:, 0].copy())

        t_start = time.time()
        stream = sd.InputStream(
            device=dev_id,
            channels=1,
            samplerate=16000,
            blocksize=1024,
            dtype='float32',
            callback=audio_callback
        )

        with stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_duration is not None and (time.time() - t_start) >= max_duration:
                    break

                try:
                    samples = audio_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                vad.accept_waveform(samples)
                while not vad.empty():
                    segment = vad.front
                    speech = np.array(segment.samples, dtype=np.float32)
                    vad.pop()

                    if len(speech) >= int(16000 * min_speech_duration):
                        peak = np.max(np.abs(speech))
                        if peak > 0.005:
                            speech = speech / peak * 0.85
                        t0 = time.time()
                        s = self.asr.create_stream()
                        s.accept_waveform(16000, speech)
                        self.asr.decode_stream(s)
                        cost_sec = time.time() - t0
                        text = re.sub(r'<\|.*?\|>', '', s.result.text).strip()
                        if text:
                            dur_sec = len(speech) / 16000.0
                            rtf = cost_sec / dur_sec if dur_sec > 0 else 0.0
                            yield {
                                "type": "speech",
                                "text": text,
                                "duration": round(dur_sec, 2),
                                "cost_sec": round(cost_sec, 3),
                                "rtf": round(rtf, 3),
                                "timestamp": time.time(),
                                "device_id": dev_id,
                                "device_name": dev_name
                            }

    def record_and_transcribe(
        self,
        duration: float = 5.0,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """录制指定秒数并执行 ASR 识别"""
        import sounddevice as sd

        if device_id is None:
            dev_id, dev_name = find_preferred_microphone()
        else:
            dev_id = device_id
            devices = sd.query_devices()
            dev_name = devices[dev_id]['name'] if 0 <= dev_id < len(devices) else f"Device-{dev_id}"

        frames = int(duration * 16000)
        t_record_start = time.time()
        recording = sd.rec(frames, samplerate=16000, channels=1, dtype='float32', device=dev_id)
        sd.wait()
        record_cost = time.time() - t_record_start

        speech = recording[:, 0]
        peak = np.max(np.abs(speech))
        if peak > 0.005:
            speech = speech / peak * 0.85

        t0 = time.time()
        s = self.asr.create_stream()
        s.accept_waveform(16000, speech)
        self.asr.decode_stream(s)
        cost_sec = time.time() - t0
        text = re.sub(r'<\|.*?\|>', '', s.result.text).strip()
        dur_sec = len(speech) / 16000.0

        return {
            "text": text,
            "duration": round(dur_sec, 2),
            "cost_sec": round(cost_sec, 3),
            "record_time_sec": round(record_cost, 2),
            "rtf": round(cost_sec / dur_sec, 3) if dur_sec > 0 else 0.0,
            "device_id": dev_id,
            "device_name": dev_name
        }
