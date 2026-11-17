"""
实时麦克风边录音边识别 (Live Microphone Speech-To-Text)
基于 SenseVoice-Small INT8 + Silero VAD 本地极速解码
"""
import sys
import os
import time

# Ensure voice_agent directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.providers.stt.sensevoice_provider import LiveMicRecognizer, query_microphones, find_preferred_microphone

def main():
    print("\n" + "=" * 65)
    print(" 🎙️ VoiceStudio - 本地麦克风实时边录音边识别系统")
    print("=" * 65)

    # 1. 查找麦克风设备
    pref_id, pref_name = find_preferred_microphone()
    print(f" • 默认麦克风设备: [{pref_id}] {pref_name}")

    # 2. 初始化引擎
    print(" • 正在初始化 Silero-VAD 与 SenseVoice-Small 识别引擎...")
    t0 = time.time()
    recognizer = LiveMicRecognizer()
    print(f" • 引擎加载就绪，耗时: {time.time() - t0:.2f}s")
    print("-" * 65)
    print(" 🔊 倾听中... 请直接对着麦克风说话（支持连续说话，按 Ctrl + C 退出）")
    print("-" * 65)

    try:
        for res in recognizer.listen(device_id=pref_id):
            time_str = time.strftime("%H:%M:%S")
            print(f"[{time_str}] 🗣️ 识别: {res['text']}  (时长: {res['duration']}s | 推理耗时: {int(res['cost_sec']*1000)}ms | RTF: {res['rtf']})")
    except KeyboardInterrupt:
        print("\n\n👋 麦克风实时监听已结束。")

if __name__ == "__main__":
    main()
