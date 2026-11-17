import numpy as np
import scipy.signal as signal

def polish_audio_cadence(audio: np.ndarray, sr: int = 24000, fade_ms: int = 45, pad_ms: int = 240) -> np.ndarray:
    """
    Polishes synthesized audio for organic human warmth:
    1. Low-mid warmth boost around 250Hz (adds chest resonance)
    2. Upper-mid articulation peaking around 3500Hz (adds consonant presence)
    3. Soft-knee tanh saturation (tames peaks, enriches whispers)
    4. Half-cosine fade out (eliminates sudden click or cutoff)
    5. Natural breath silence padding (ensures relaxed cadence)
    """
    if audio is None or len(audio) == 0:
        return audio

    audio = np.array(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    try:
        # 1. Chest resonance warmth EQ
        w0_low = 250.0 / (sr / 2.0)
        b_low, a_low = signal.iirpeak(w0_low, Q=1.0)
        warm_filtered = signal.lfilter(b_low, a_low, audio)
        audio = audio + 0.12 * warm_filtered

        # 2. Articulation peaking
        w0_high = 3500.0 / (sr / 2.0)
        b_high, a_high = signal.iirpeak(w0_high, Q=1.2)
        clarity_filtered = signal.lfilter(b_high, a_high, audio)
        audio = audio + 0.10 * clarity_filtered

        # 3. Dynamics limiter
        audio = np.tanh(audio * 1.1) / 1.1

        # 4. Peak normalize to -0.8 dBFS
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
