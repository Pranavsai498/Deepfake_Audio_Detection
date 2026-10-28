import os
import torch
import numpy as np
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset
from typing import Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from src.features import extract_120dim_mfcc
LABEL_MAP = {'real': 0, 'bonafide': 0, 0: 0, 'fake': 1, 'spoof': 1, 'synthetic': 1, 1: 1}

def apply_spec_augment(features: np.ndarray, freq_mask_max: int=15, time_mask_max: int=30, p: float=0.5) -> np.ndarray:
    feat = features.copy()
    n_freq, n_time = feat.shape
    if np.random.rand() < p:
        f = np.random.randint(1, min(freq_mask_max, n_freq // 4))
        f0 = np.random.randint(0, n_freq - f)
        feat[f0:f0 + f, :] = 0.0
    if np.random.rand() < p:
        t = np.random.randint(1, min(time_mask_max, n_time // 4))
        t0 = np.random.randint(0, n_time - t)
        feat[:, t0:t0 + t] = 0.0
    if np.random.rand() < 0.3:
        std = np.std(feat) + 1e-06
        noise = np.random.randn(*feat.shape).astype(np.float32) * (0.01 * std)
        feat += noise
    return feat

class AudioDeepfakeDataset(Dataset):

    def __init__(self, metadata_df: pd.DataFrame, is_train: bool=False, target_frames: int=300, target_sr: int=16000, preload: bool=True, max_workers: int=8):
        self.df = metadata_df.reset_index(drop=True)
        self.is_train = is_train
        self.target_frames = target_frames
        self.target_sr = target_sr
        self.cache: Dict[int, np.ndarray] = {}
        if preload and len(self.df) > 0:
            self._preload_all_features(max_workers=max_workers)

    def _extract_single(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        file_path = row['file_path']
        try:
            audio, sr = sf.read(file_path, dtype='float32')
        except Exception:
            audio = np.zeros(self.target_sr * 3, dtype=np.float32)
            sr = self.target_sr
        feat = extract_120dim_mfcc(audio, sr=sr, n_mfcc=40, target_frames=self.target_frames)
        return (idx, feat)

    def _preload_all_features(self, max_workers: int=8):
        from concurrent.futures import as_completed
        split_name = 'Train' if self.is_train else 'Val/Test'
        print(f'Pre-loading {len(self.df)} {split_name} acoustic features into memory ({max_workers} threads)...')
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._extract_single, i): i for i in range(len(self.df))}
            for future in tqdm(as_completed(futures), total=len(futures), desc=f'Loading {split_name} Features'):
                idx, feat = future.result()
                self.cache[idx] = feat
        print(f'Preloaded {len(self.cache)} feature arrays into RAM successfully.')

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if idx in self.cache:
            features = self.cache[idx]
        else:
            _, features = self._extract_single(idx)
            self.cache[idx] = features
        if self.is_train:
            features = apply_spec_augment(features)
        lbl_raw = row['label']
        label = LABEL_MAP.get(lbl_raw, int(lbl_raw))
        lang = str(row.get('language', 'unknown'))
        model = str(row.get('generative_model', 'real'))
        spk = str(row.get('speaker_id', f'spk_{idx}'))
        feat_tensor = torch.from_numpy(features).float()
        label_tensor = torch.tensor(label, dtype=torch.long)
        return (feat_tensor, label_tensor, lang, model, spk)
