import numpy as np
import pytest
from app.core.sound_processor import polish_audio_cadence

def test_polish_empty():
    res = polish_audio_cadence(np.array([], dtype=np.float32))
    assert len(res) == 0

def test_polish_mono():
    sr = 24000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    polished = polish_audio_cadence(sine, sr=sr, fade_ms=45, pad_ms=240)
    assert polished is not None
    assert polished.ndim == 1
    # Check that head silence and tail padding were added
    head_silence_len = int(sr * 0.02)
    tail_silence_len = int(sr * 240 / 1000)
    assert len(polished) == len(sine) + head_silence_len + tail_silence_len
    # Peak amplitude bounded under 1.0
    assert np.max(np.abs(polished)) <= 1.0

def test_polish_stereo():
    sr = 24000
    stereo = np.zeros((1000, 2), dtype=np.float32)
    polished = polish_audio_cadence(stereo, sr=sr)
    assert polished.ndim == 1
