import os
import sys
import io
import argparse
import time
from collections import defaultdict, Counter
import pandas as pd
import soundfile as sf
from tqdm import tqdm
import requests
from huggingface_hub import HfFileSystem
import pyarrow.parquet as pq
SVQ_FILES = {'hindi': ['utts_hi_in_clean.parquet', 'utts_hi_in_background_speech.parquet', 'utts_hi_in_traffic_noise.parquet', 'utts_hi_in_media_noise.parquet'], 'telugu': ['utts_te_in_clean.parquet', 'utts_te_in_background_speech.parquet', 'utts_te_in_traffic_noise.parquet', 'utts_te_in_media_noise.parquet'], 'malayalam': ['utts_ml_in_clean.parquet', 'utts_ml_in_background_speech.parquet', 'utts_ml_in_traffic_noise.parquet', 'utts_ml_in_media_noise.parquet'], 'bengali': ['utts_bn_in_clean.parquet', 'utts_bn_in_background_speech.parquet', 'utts_bn_in_traffic_noise.parquet', 'utts_bn_in_media_noise.parquet', 'utts_bn_bd_clean.parquet', 'utts_bn_bd_background_speech.parquet']}

def download_real_language(language: str, target_total: int=3000, output_root: str='data/raw/real') -> pd.DataFrame:
    lang_lower = language.lower()
    save_dir = os.path.join(output_root, lang_lower)
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, 'metadata.csv')
    existing_df = pd.DataFrame()
    collected_paths = set()
    speaker_counts = Counter()
    if os.path.exists(meta_path):
        existing_df = pd.read_csv(meta_path)
        collected_paths = set(existing_df['file_path'].tolist())
        for spk in existing_df['speaker_id']:
            speaker_counts[str(spk)] += 1
        print(f'[{language}] Found existing metadata with {len(existing_df)} clips across {len(speaker_counts)} speakers.')
    if len(existing_df) >= target_total:
        print(f'[{language}] Target of {target_total} already satisfied! Skipping.')
        return existing_df
    files = SVQ_FILES.get(lang_lower, [])
    if not files:
        print(f'[{language}] No parquet files configured! Skipping.')
        return existing_df
    print(f'\n========================================================')
    print(f'  Streaming Real Human Speech for: {language.upper()}')
    print(f'  Target: {target_total} clips (Google SVQ Corpus)')
    print(f'========================================================')
    records = existing_df.to_dict('records') if not existing_df.empty else []
    pbar = tqdm(total=target_total, initial=len(records), desc=f'{language} Real')

    def download_shard_with_resume(url: str, local_path: str, max_retries: int=10) -> bool:
        headers = {'User-Agent': 'Mozilla/5.0'}
        for attempt in range(max_retries):
            try:
                downloaded = os.path.getsize(local_path) if os.path.exists(local_path) else 0
                head = requests.head(url, headers=headers, allow_redirects=True, timeout=15)
                total = int(head.headers.get('content-length', 0))
                if total > 0 and downloaded >= total:
                    return True
                req_headers = dict(headers)
                if downloaded > 0:
                    req_headers['Range'] = f'bytes={downloaded}-'
                with requests.get(url, headers=req_headers, stream=True, timeout=25) as r:
                    if r.status_code not in (200, 206):
                        time.sleep(2)
                        continue
                    mode = 'ab' if downloaded > 0 and r.status_code == 206 else 'wb'
                    if mode == 'wb':
                        downloaded = 0
                    with open(local_path, mode) as f:
                        for chunk in r.iter_content(chunk_size=2 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                if total > 0 and os.path.getsize(local_path) >= total:
                    return True
                elif total == 0 and os.path.getsize(local_path) > 0:
                    return True
            except Exception as e:
                print(f'\n  [Retry {attempt + 1}/{max_retries}] Download error for {os.path.basename(local_path)}: {e}')
                time.sleep(2)
        return os.path.exists(local_path) and os.path.getsize(local_path) > 0
    for fname in files:
        if len(records) >= target_total:
            break
        url = f'https://huggingface.co/datasets/google/svq/resolve/main/1.0.0/audio/{fname}'
        temp_shard = f'temp_svq_{lang_lower}_{fname}'
        print(f'\n  -> Fetching shard: {fname}...')
        success = download_shard_with_resume(url, temp_shard)
        if not success:
            print(f'  Failed to download {fname}, skipping to next...')
            continue
        try:
            pf = pq.ParquetFile(temp_shard)
            for rg in range(pf.metadata.num_row_groups):
                if len(records) >= target_total:
                    break
                table = pf.read_row_group(rg, columns=['waveform', 'speaker_id', 'speaker_gender'])
                for row in table.to_pylist():
                    if len(records) >= target_total:
                        break
                    wf = row.get('waveform')
                    if not wf or not isinstance(wf, dict) or 'bytes' not in wf:
                        continue
                    raw_bytes = wf['bytes']
                    if not raw_bytes or len(raw_bytes) < 4000:
                        continue
                    try:
                        audio_arr, sr = sf.read(io.BytesIO(raw_bytes))
                    except Exception:
                        continue
                    dur = len(audio_arr) / float(sr)
                    if dur < 1.5 or dur > 15.0:
                        continue
                    idx = len(records)
                    spk = str(row.get('speaker_id', f'spk_{idx}'))
                    out_fname = f'svq_{lang_lower}_real_{idx:05d}.wav'
                    filepath = os.path.join(save_dir, out_fname).replace('\\', '/')
                    if filepath in collected_paths:
                        continue
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    sf.write(filepath, audio_arr, sr, subtype='PCM_16')
                    collected_paths.add(filepath)
                    speaker_counts[spk] += 1
                    rec = {'file_path': filepath, 'label': 0, 'language': lang_lower, 'speaker_id': spk, 'gender': str(row.get('speaker_gender', 'unknown')), 'source_dataset': 'google/svq', 'generative_model': 'real', 'duration': round(dur, 3), 'sample_rate': sr}
                    records.append(rec)
                    pbar.update(1)
                    if len(records) % 200 == 0:
                        pd.DataFrame(records).to_csv(meta_path, index=False)
            try:
                pf.reader.close()
            except Exception:
                pass
            del pf
            import gc
            gc.collect()
            if os.path.exists(temp_shard):
                try:
                    os.remove(temp_shard)
                except Exception:
                    pass
        except Exception as e:
            print(f'  Warning: Shard processing error ({e}). Moving to next shard...')
            if os.path.exists(temp_shard):
                try:
                    os.remove(temp_shard)
                except Exception:
                    pass
            continue
            continue
    pbar.close()
    final_df = pd.DataFrame(records)
    final_df.to_csv(meta_path, index=False)
    print(f'[{language}] Finished! Saved {len(final_df)} real clips to {save_dir}. Speakers: {len(speaker_counts)}')
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Selective Streaming Sampler for Real Human Speech')
    parser.add_argument('--languages', nargs='+', default=['Hindi', 'Telugu', 'Malayalam', 'Bengali'], help='Languages to stream')
    parser.add_argument('--target_per_lang', type=int, default=3000, help='Target number of authentic human clips per language')
    parser.add_argument('--output_root', type=str, default='data/raw/real', help='Directory to save raw real audio')
    args = parser.parse_args()
    all_dfs = []
    for lang in args.languages:
        df = download_real_language(language=lang, target_total=args.target_per_lang, output_root=args.output_root)
        all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv('data/raw/real/all_real_metadata.csv', index=False)
        print(f'\n[ALL REALS COMPLETE] Total clips: {len(combined)}')
        print(combined.groupby(['language']).size())
if __name__ == '__main__':
    main()
