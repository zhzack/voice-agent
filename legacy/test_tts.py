import time
import os
import numpy as np

# --- Windows ROCm / PyTorch Distributed Compatibility Patch ---
import torch.distributed as dist
class _DummyGroup:
    WORLD = None
    SUM = 0
if not hasattr(dist, 'group'):
    dist.group = _DummyGroup
if not hasattr(dist, 'ReduceOp'):
    dist.ReduceOp = _DummyGroup
# -------------------------------------------------------------

import torch
import soundfile as sf
import sounddevice as sd
import ChatTTS

print("=== ChatTTS Environment Check ===")
print("PyTorch Version:", torch.__version__)
print("CUDA / ROCm Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

print("\n[1/3] Initializing ChatTTS instance...")
chat = ChatTTS.Chat()

print("[2/3] Loading models (source='huggingface', compile=False)...")
t0 = time.time()
# The weights are already cached locally, so loading will be super fast!
chat.load(source="huggingface", compile=False)
t1 = time.time()
print(f"Model loaded successfully in {t1 - t0:.2f}s!")

# Test conversational text with laughter and pauses
text = "你好啊！很高兴认识你。[laugh] 当前的文字转语音系统已经准备就绪了，在你的全新显卡上运行得非常流畅！[uv_break] 接下来我们还可以加入大模型，实现超低延迟的实时对话哦。"

print(f"\n[3/3] Inferencing text to speech on GPU...")
print(f"Input text: \"{text}\"")

t0 = time.time()
wavs = chat.infer([text])
t1 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "assets", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)
output_file = os.path.join(AUDIO_DIR, "output_test.wav")
if wavs and len(wavs) > 0:
    # Ensure audio_data is a 1D float numpy array
    raw = wavs[0]
    audio_data = np.array(raw, dtype=np.float32).squeeze()
    
    sample_rate = 24000
    sf.write(output_file, audio_data, sample_rate)
    audio_duration = len(audio_data) / sample_rate
    infer_time = t1 - t0
    rtf = infer_time / audio_duration if audio_duration > 0 else 0
    print(f"\n Synthesis Complete!")
    print(f"  - Generated Audio File: {os.path.abspath(output_file)}")
    print(f"  - Audio Duration: {audio_duration:.2f}s")
    print(f"  - Inference Time: {infer_time:.2f}s")
    print(f"  - Real-Time Factor (RTF): {rtf:.3f} (Values < 1.0 are faster than real-time)")
    
    # Try playing the audio on default speaker
    try:
        print("\n Playing audio on default speaker...")
        sd.play(audio_data, sample_rate)
        sd.wait()
        print(" Playback finished!")
    except Exception as e:
        print(f"Audio playback note (can be played manually): {e}")
else:
    print("Inference completed but no audio returned.")
