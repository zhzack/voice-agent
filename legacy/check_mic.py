import time
import sounddevice as sd
import numpy as np

def list_and_test_microphones():
    devices = sd.query_devices()
    input_devices = []
    default_input = sd.default.device[0]
    
    print("=" * 60)
    print("  系统麦克风设备扫描与音量检测")
    print("=" * 60)
    print(f"当前系统默认输入设备编号: [{default_input}]\n")
    
    for idx, d in enumerate(devices):
        if d['max_input_channels'] > 0:
            is_default = " (当前系统默认)" if idx == default_input else ""
            input_devices.append((idx, d['name'], d['hostapi']))
            print(f"[{idx}] {d['name']}{is_default}")
            
    print("\n" + "-" * 60)
    print("现在请对着麦克风发出声音（例如说话、咳嗽或拍手）...")
    print("正在逐一检测各个设备是否有实际声音输入 (各测试 1.5 秒)...\n")
    
    active_devices = []
    
    for idx, name, hostapi in input_devices:
        try:
            # Record 1.5s at 16kHz
            rec = sd.rec(int(16000 * 1.5), samplerate=16000, channels=1, dtype='float32', device=idx)
            sd.wait()
            
            rms = np.sqrt(np.mean(rec**2))
            max_amp = np.max(np.abs(rec))
            
            # Simple ascii volume bar
            bars = int(min(max_amp * 50, 30))
            meter = "█" * bars + "-" * (30 - bars)
            
            status = " 有明显声音" if max_amp > 0.05 else " 声音微弱或静音"
            print(f"设备 [{idx:2d}] {name[:24]:<24} | 峰值: {max_amp:.4f} | [{meter}] {status}")
            
            if max_amp > 0.03:
                active_devices.append((idx, name, max_amp))
        except Exception as e:
            # Some virtual or disconnected devices may fail to open
            pass
            
    print("\n" + "=" * 60)
    if active_devices:
        # Sort by signal strength
        active_devices.sort(key=lambda x: x[2], reverse=True)
        best_id, best_name, best_amp = active_devices[0]
        print(f" 推荐使用麦克风: 设备编号 [{best_id}] - {best_name} (信号强度峰值: {best_amp:.4f})")
    else:
        print("⚠️ 未在任何设备上检测到明显声波输入。请检查麦克风是否已开启静音开关，或在 Windows 声音设置中提高麦克风音量。")
    print("=" * 60)

if __name__ == "__main__":
    list_and_test_microphones()
