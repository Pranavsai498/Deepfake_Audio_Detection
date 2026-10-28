import os
import sys
import glob
import json
import math
import hashlib
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import scipy.signal
import soundfile as sf
from tqdm import tqdm
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def compute_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while (chunk := f.read(65536)):
            h.update(chunk)
    return h.hexdigest()

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int=16000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return scipy.signal.resample_poly(audio, up, down).astype(np.float32)

def process_audio_file(input_path: str, output_path: str, target_sr: int=16000, min_dur: float=1.0, max_dur: float=10.0) -> dict:
    qc = {'status': 'FAILED', 'reason': '', 'orig_sr': 0, 'orig_duration': 0.0, 'final_duration': 0.0, 'channels': 0, 'has_nan_inf': False, 'is_silent': False, 'is_clipped': False, 'peak': 0.0, 'rms': 0.0, 'sha256': ''}
    try:
        audio, sr = sf.read(input_path, dtype='float32')
    except Exception as e:
        qc['reason'] = f'Decode error: {str(e)[:80]}'
        return qc
    qc['orig_sr'] = sr
    qc['channels'] = 1 if audio.ndim == 1 else audio.shape[1]
    dur = len(audio) / float(sr)
    qc['orig_duration'] = round(dur, 3)
    if dur < min_dur:
        qc['reason'] = f'Too short ({dur:.2f}s < {min_dur}s)'
        return qc
    if dur > max_dur:
        audio = audio[:int(max_dur * sr)]
        dur = max_dur
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr
    if not np.isfinite(audio).all():
        qc['has_nan_inf'] = True
        qc['reason'] = 'Contains NaN or Inf'
        return qc
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-10))
    qc['rms'] = round(rms, 5)
    if rms < 0.001:
        qc['is_silent'] = True
        qc['reason'] = f'Near-silent audio (RMS={rms:.5f})'
        return qc
    peak = float(np.max(np.abs(audio)))
    qc['peak'] = round(peak, 4)
    if peak > 0.999:
        qc['is_clipped'] = True
    if peak > 0.01:
        target_rms = 0.1
        gain = min(target_rms / (rms + 1e-08), 4.0)
        audio = audio * gain
        audio = np.clip(audio, -0.98, 0.98)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, target_sr, subtype='PCM_16')
    qc['status'] = 'PASSED'
    qc['final_duration'] = round(len(audio) / float(target_sr), 3)
    qc['sha256'] = compute_file_sha256(output_path)
    return qc

def _worker_task(row_dict, output_dir, target_sr, min_dur, max_dur):
    src_path = str(row_dict['file_path']).replace('\\', '/')
    if not os.path.exists(src_path):
        return (None, 'File missing on disk')
    lang = str(row_dict.get('language', 'unknown')).lower()
    lbl = int(row_dict['label'])
    lbl_dir = 'fake' if lbl == 1 else 'real'
    fname = os.path.basename(src_path)
    dst_path = os.path.join(output_dir, lbl_dir, lang, fname).replace('\\', '/')
    qc = process_audio_file(src_path, dst_path, target_sr=target_sr, min_dur=min_dur, max_dur=max_dur)
    if qc['status'] != 'PASSED':
        return (None, qc['reason'])
    new_rec = dict(row_dict)
    new_rec['file_path'] = dst_path
    new_rec['duration'] = qc['final_duration']
    new_rec['sample_rate'] = target_sr
    new_rec['rms_energy'] = qc['rms']
    new_rec['peak_amplitude'] = qc['peak']
    new_rec['audio_sha256'] = qc['sha256']
    return (new_rec, None)

def run_pipeline(raw_dirs: list, output_dir: str='data/processed', target_sr: int=16000, max_workers: int=8):
    print('=' * 60)
    print('  AUDIO PREPROCESSING & QUALITY CONTROL PIPELINE')
    print(f'  Target Sample Rate: {target_sr} Hz | Workers: {max_workers}')
    print('=' * 60)
    all_records = []
    seen_hashes = {}
    stats = {'total_scanned': 0, 'passed': 0, 'failed': 0, 'duplicates_removed': 0, 'reasons': Counter()}
    all_rows = []
    for raw_meta_file in raw_dirs:
        if not os.path.exists(raw_meta_file):
            continue
        df = pd.read_csv(raw_meta_file)
        print(f'Loaded {len(df)} entries from {raw_meta_file}')
        all_rows.extend(df.to_dict('records'))
    stats['total_scanned'] = len(all_rows)
    print(f'Total entries to process: {len(all_rows)}')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_task, r, output_dir, target_sr, 1.0, 10.0): r for r in all_rows}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Preprocessing'):
            rec, err = future.result()
            if err is not None:
                stats['failed'] += 1
                stats['reasons'][err] += 1
            else:
                sha = rec['audio_sha256']
                dst_path = rec['file_path']
                if sha in seen_hashes:
                    stats['duplicates_removed'] += 1
                    stats['reasons']['Duplicate audio hash'] += 1
                    try:
                        os.remove(dst_path)
                    except OSError:
                        pass
                else:
                    seen_hashes[sha] = dst_path
                    stats['passed'] += 1
                    all_records.append(rec)
    proc_df = pd.DataFrame(all_records)
    all_meta_path = 'data/metadata/all_metadata.csv'
    os.makedirs('data/metadata', exist_ok=True)
    proc_df.to_csv(all_meta_path, index=False)
    os.makedirs('results', exist_ok=True)
    report_path = 'results/quality_report.json'
    report_data = {'total_scanned': stats['total_scanned'], 'passed_quality_control': stats['passed'], 'rejected': stats['failed'], 'duplicates_removed': stats['duplicates_removed'], 'rejection_reasons': {str(k): int(v) for k, v in stats['reasons'].items()}, 'language_breakdown': {str(k): int(v) for k, v in proc_df['language'].value_counts().items()} if not proc_df.empty else {}, 'label_breakdown': {str(k): int(v) for k, v in proc_df['label'].value_counts().items()} if not proc_df.empty else {}, 'generative_model_breakdown': {str(k): int(v) for k, v in proc_df['generative_model'].value_counts().items()} if not proc_df.empty else {}}
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2)
    print('\n================ QUALITY CONTROL SUMMARY ================')
    print(f"Total Audio Files Scanned: {stats['total_scanned']}")
    print(f"Passed Quality Checks:     {stats['passed']}")
    print(f"Rejected / Corrupted:      {stats['failed']}")
    print(f"Duplicate Files Filtered:  {stats['duplicates_removed']}")
    print('Rejection Breakdown:')
    for r, c in stats['reasons'].most_common():
        print(f'  - {r}: {c}')
    print(f'Saved verified dataset to {all_meta_path} & {report_path}')
    return proc_df
if __name__ == '__main__':
    meta_files = ['data/raw/real/all_real_metadata.csv', 'data/raw/fake/all_fake_metadata.csv']
    run_pipeline(meta_files)
