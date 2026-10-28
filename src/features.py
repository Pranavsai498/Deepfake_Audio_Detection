import numpy as np
import scipy.fftpack
import scipy.signal
from typing import Tuple, Optional
_MEL_FB_CACHE = {}

def compute_delta(feat: np.ndarray, N: int=2) -> np.ndarray:
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    denominator = 2 * sum((i ** 2 for i in range(1, N + 1)))
    padded = np.pad(feat, ((0, 0), (N, N)), mode='edge')
    delta = np.zeros_like(feat)
    n_frames = feat.shape[1]
    for n in range(1, N + 1):
        delta += n * (padded[:, N + n:N + n + n_frames] - padded[:, N - n:N - n + n_frames])
    return delta / denominator

def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def get_mel_filterbank(sr: int=16000, n_fft: int=1024, n_mels: int=80, f_min: float=20.0, f_max: float=8000.0) -> np.ndarray:
    cache_key = (sr, n_fft, n_mels, f_min, f_max)
    if cache_key in _MEL_FB_CACHE:
        return _MEL_FB_CACHE[cache_key]
    f_max = min(f_max, sr / 2.0)
    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_m_minus = bins[m - 1]
        f_m = bins[m]
        f_m_plus = bins[m + 1]
        for k in range(f_m_minus, f_m):
            if f_m > f_m_minus:
                fb[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
        for k in range(f_m, f_m_plus):
            if f_m_plus > f_m:
                fb[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
    _MEL_FB_CACHE[cache_key] = fb
    return fb

def extract_120dim_mfcc(audio: np.ndarray, sr: int=16000, n_mfcc: int=40, n_mels: int=80, n_fft: int=1024, hop_length: int=160, win_length: int=400, target_frames: int=300) -> np.ndarray:
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    window = np.hanning(win_length)
    n_samples = len(audio)
    pad_amount = (n_fft - hop_length) // 2
    padded_audio = np.pad(audio, (pad_amount, pad_amount), mode='reflect')
    n_frames = 1 + int((len(padded_audio) - win_length) // hop_length)
    if n_frames < 1:
        padded_audio = np.pad(padded_audio, (0, win_length), mode='constant')
        n_frames = 1 + int((len(padded_audio) - win_length) // hop_length)
    frames = np.lib.stride_tricks.as_strided(padded_audio, shape=(n_frames, win_length), strides=(padded_audio.strides[0] * hop_length, padded_audio.strides[0]))
    windowed_frames = frames * window
    stft = np.fft.rfft(windowed_frames, n=n_fft, axis=1)
    power_spectrum = np.abs(stft) ** 2 / n_fft
    mel_fb = get_mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, f_min=20.0, f_max=8000.0)
    mel_energy = np.dot(power_spectrum, mel_fb.T)
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    log_mel_T = log_mel.T
    mfcc = scipy.fftpack.dct(log_mel_T, type=2, axis=0, norm='ortho')[:n_mfcc, :]
    mean = np.mean(mfcc, axis=1, keepdims=True)
    std = np.std(mfcc, axis=1, keepdims=True) + 1e-08
    mfcc = (mfcc - mean) / std
    delta1 = compute_delta(mfcc, N=2)
    delta2 = compute_delta(delta1, N=2)
    feat_120 = np.concatenate([mfcc, delta1, delta2], axis=0).astype(np.float32)
    cur_frames = feat_120.shape[1]
    if cur_frames < target_frames:
        pad_len = target_frames - cur_frames
        feat_120 = np.pad(feat_120, ((0, 0), (0, pad_len)), mode='wrap')
    elif cur_frames > target_frames:
        feat_120 = feat_120[:, :target_frames]
    return feat_120
