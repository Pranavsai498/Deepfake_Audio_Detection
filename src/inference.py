import os
import sys
import argparse
import time
import torch
import numpy as np
import scipy.signal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import HybridCNNLSTMDetector
from src.features import extract_120dim_mfcc
from src.preprocessing.audio_loader import read_audio_file
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

class DeepfakeAudioPredictor:

    def __init__(self, checkpoint_path: str='checkpoints/best_detector.pt', device: str=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.tau = 0.6901
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found at: {self.checkpoint_path}')
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model = HybridCNNLSTMDetector().to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        self.tau = float(ckpt.get('optimal_tau', 0.6901))
        if self.tau < 0.01 or self.tau > 0.99:
            self.tau = 0.6901

    def predict(self, audio_path: str, threshold: float=None, language: str=None) -> dict:
        if threshold is not None:
            active_tau = float(threshold)
        else:
            active_tau = self.tau
        audio, sr = read_audio_file(audio_path, target_sr=16000)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        dur = len(audio) / float(sr)
        raw_rms = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
        raw_peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
        print(f'[PREDICT] file={os.path.basename(audio_path)}  sr={sr}  samples={len(audio)}  dur={dur:.2f}s  peak={raw_peak:.5f}  rms={raw_rms:.5f}')
        if sr != 16000 and len(audio) > 0:
            num_samples = int(round(len(audio) * 16000.0 / sr))
            audio = scipy.signal.resample(audio, num_samples).astype(np.float32)
            sr = 16000
        is_silence = raw_peak < 0.002 and raw_rms < 0.0003 or raw_peak < 0.0005 or len(audio) < 1600
        if is_silence:
            return {'verdict': 'SILENCE_DETECTED', 'is_fake': False, 'is_silent': True, 'confidence_percent': 0.0, 'operating_threshold': round(active_tau, 4), 'calibrated_probabilities': {'Genuine / Real': 0.0, 'AI Deepfake': 0.0}, 'raw_scores': {'raw_real': 0.0, 'raw_fake': 0.0}, 'acoustic_stats': {'sample_rate': sr, 'duration_sec': round(dur, 2), 'rms_energy': round(raw_rms, 5), 'peak_amplitude': round(raw_peak, 5)}, 'processed_audio': np.zeros(48000, dtype=np.float32), 'message': 'No audible speech detected in recording. The microphone input appears silent or muted.'}
        target_samples = int(sr * 3.0)
        if len(audio) > target_samples:
            hop = sr // 4
            best_start = 0
            max_energy = -1.0
            for s in range(0, len(audio) - target_samples + 1, hop):
                seg = audio[s:s + target_samples]
                seg_energy = float(np.sum(seg ** 2))
                if seg_energy > max_energy:
                    max_energy = seg_energy
                    best_start = s
            active_audio = audio[best_start:best_start + target_samples]
        elif len(audio) < target_samples:
            pad_len = target_samples - len(audio)
            active_audio = np.pad(audio, (0, pad_len), mode='reflect')
        else:
            active_audio = audio.copy()
        active_rms = float(np.sqrt(np.mean(active_audio ** 2) + 1e-10))
        active_peak = float(np.max(np.abs(active_audio)))
        if active_rms > 0.0005:
            gain = min(0.1 / active_rms, 25.0)
            active_audio = np.clip(active_audio * gain, -0.98, 0.98)
        feats = extract_120dim_mfcc(active_audio, sr=sr, target_frames=300)
        feat_tensor = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(feat_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prob_real = float(probs[0])
        prob_fake = float(probs[1])
        is_fake = prob_fake >= active_tau
        verdict = 'FAKE' if is_fake else 'REAL'
        confidence = (prob_fake if is_fake else prob_real) * 100.0
        res = {'verdict': verdict, 'is_fake': bool(is_fake), 'is_silent': False, 'confidence_percent': round(confidence, 2), 'operating_threshold': round(active_tau, 4), 'calibrated_probabilities': {'Genuine / Real': round(prob_real, 4), 'AI Deepfake': round(prob_fake, 4)}, 'raw_scores': {'raw_real': round(prob_real, 4), 'raw_fake': round(prob_fake, 4)}, 'acoustic_stats': {'sample_rate': sr, 'duration_sec': round(dur, 2), 'rms_energy': round(active_rms, 4), 'peak_amplitude': round(active_peak, 4)}, 'processed_audio': active_audio}
        if language:
            res['language'] = language.capitalize()
        return res

def main():
    parser = argparse.ArgumentParser(description='Multilingual Audio Deepfake Inference CLI')
    parser.add_argument('--audio', type=str, required=True, help='Path to input audio file (.wav, .mp3, .flac)')
    parser.add_argument('--language', type=str, default=None, help='Target language (optional, e.g. hindi, telugu, malayalam, bengali)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_detector.pt', help='Path to model checkpoint')
    args = parser.parse_args()
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found at '{args.audio}'")
        sys.exit(1)
    predictor = DeepfakeAudioPredictor(checkpoint_path=args.checkpoint)
    res = predictor.predict(args.audio, language=args.language)
    if res.get('is_silent', False):
        print(f"Warning: {res['message']}")
        print(f"Peak: {res['acoustic_stats']['peak_amplitude']}, RMS: {res['acoustic_stats']['rms_energy']}")
        sys.exit(0)
    print(f"Prediction: {res['verdict']}")
    print(f"Confidence: {res['confidence_percent']:.2f}%")
    if 'language' in res:
        print(f"Language:   {res['language']}")
if __name__ == '__main__':
    main()
