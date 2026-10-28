import numpy as np
import scipy.signal
import os
from typing import Tuple

def read_audio_file(file_path: str, target_sr: int=16000) -> Tuple[np.ndarray, int]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Audio file not found: {file_path}')
    try:
        import imageio_ffmpeg
        import subprocess
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [exe, '-y', '-v', 'error', '-i', file_path, '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(target_sr), '-ac', '1', '-']
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = proc.communicate(timeout=30)
        if proc.returncode == 0 and len(out) >= 4:
            audio = np.frombuffer(out, dtype=np.float32).copy()
            if len(audio) > 1600 and float(np.max(np.abs(audio))) < 1e-07:
                print(f'[WARN] imageio_ffmpeg returned all-zero audio for {os.path.basename(file_path)} ({len(audio)} samples). File may be a broken Gradio temp WAV. Trying fallbacks.')
            else:
                return (audio, target_sr)
    except Exception as e:
        print(f'[WARN] imageio_ffmpeg failed for {os.path.basename(file_path)}: {e}')
    try:
        import soundfile as sf
        audio, sr = sf.read(file_path, dtype='float32')
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            n = int(round(len(audio) * target_sr / sr))
            audio = scipy.signal.resample(audio, n).astype(np.float32)
        return (audio.astype(np.float32), target_sr)
    except Exception:
        pass
    try:
        import librosa
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)
        return (audio.astype(np.float32), int(target_sr))
    except Exception:
        pass
    try:
        import torchaudio
        waveform, sr = torchaudio.load(file_path)
        audio = waveform.cpu().numpy()
        if audio.ndim == 2:
            audio = audio.mean(axis=0)
        if sr != target_sr:
            n = int(round(len(audio) * target_sr / sr))
            audio = scipy.signal.resample(audio, n).astype(np.float32)
        return (audio.astype(np.float32), target_sr)
    except Exception:
        pass
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(file_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != target_sr:
            n = int(round(len(audio) * target_sr / sr))
            audio = scipy.signal.resample(audio, n).astype(np.float32)
        return (audio.astype(np.float32), target_sr)
    except Exception as e:
        raise ValueError(f"Unable to read '{os.path.basename(file_path)}'. All decoders failed. Last error: {e}")

def load_and_preprocess_audio(file_path: str, target_sr: int=16000, target_duration_sec: float=3.0, normalize_rms: bool=True, target_rms: float=0.1, trim_silence: bool=False) -> np.ndarray:
    audio, sr = read_audio_file(file_path, target_sr=target_sr)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if trim_silence and len(audio) > target_sr * 0.5:
        frame_len = int(target_sr * 0.02)
        energy = np.array([np.sum(audio[i:i + frame_len] ** 2) for i in range(0, len(audio) - frame_len, frame_len)])
        if len(energy) > 0:
            thresh = np.max(energy) * 0.001
            active_frames = np.where(energy > thresh)[0]
            if len(active_frames) > 0:
                start_sample = active_frames[0] * frame_len
                end_sample = min(len(audio), (active_frames[-1] + 1) * frame_len)
                audio = audio[start_sample:end_sample]
    target_samples = int(target_sr * target_duration_sec)
    current_samples = len(audio)
    if current_samples == 0:
        audio = np.zeros(target_samples, dtype=np.float32)
    elif current_samples < target_samples:
        pad_len = target_samples - current_samples
        if pad_len <= current_samples:
            audio = np.pad(audio, (0, pad_len), mode='reflect')
        else:
            repeats = int(np.ceil(target_samples / current_samples))
            audio = np.tile(audio, repeats)[:target_samples]
    elif current_samples > target_samples:
        hop = target_sr // 4
        best_start = 0
        max_energy = -1.0
        for s in range(0, current_samples - target_samples + 1, hop):
            segment = audio[s:s + target_samples]
            energy = float(np.sum(segment ** 2))
            if energy > max_energy:
                max_energy = energy
                best_start = s
        audio = audio[best_start:best_start + target_samples]
    if normalize_rms:
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-08))
        if rms > 0.003:
            gain = min(target_rms / rms, 5.0)
            audio = audio * gain
            audio = np.clip(audio, -0.98, 0.98)
    return audio.astype(np.float32)
