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
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
LANGUAGE_SHARD_TARGETS = {'hindi': [('freevc24', [0, 1, 2], 1500), ('xtts_v2', [60, 61, 62, 63], 1500)], 'telugu': [('freevc24', [0, 1, 2], 1500), ('xtts_v2', [70, 71, 72, 73], 1500)], 'malayalam': [('freevc24', [0, 1, 2], 1500), ('xtts_v2', [18, 19, 20, 21], 1500)], 'bengali': [('freevc24', [0, 1], 1000), ('vits', [28, 29], 1000), ('xtts_v2', [45, 46], 1000)]}

def download_synthetic_language(language: str, target_total: int=3000, output_root: str='data/raw/fake') -> pd.DataFrame:
    lang_lower = language.lower()
    lang_title = language.capitalize()
    save_dir = os.path.join(output_root, lang_lower)
    os.makedirs(save_dir, exist_ok=True)
    meta_path = os.path.join(save_dir, 'metadata.csv')
    existing_df = pd.DataFrame()
    collected_paths = set()
    model_counts = Counter()
    if os.path.exists(meta_path):
        existing_df = pd.read_csv(meta_path)
        collected_paths = set(existing_df['file_path'].tolist())
        for m in existing_df['generative_model']:
            model_counts[str(m).lower()] += 1
        print(f'[{language}] Found existing metadata with {len(existing_df)} clips. Models: {dict(model_counts)}')
    if len(existing_df) >= target_total:
        print(f'[{language}] Target of {target_total} already satisfied! Skipping.')
        return existing_df
    print(f'\n========================================================')
    print(f'  Streaming IndicSynth for: {language.upper()}')
    print(f'  Target: {target_total} clips (FreeVC24, XTTS-v2, VITS)')
    print(f'========================================================')

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
                with requests.get(url, headers=req_headers, stream=True, timeout=30) as r:
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
    fs = HfFileSystem()
    shards = sorted(fs.glob(f'datasets/vdivyasharma/IndicSynth/{lang_title}/*.parquet'))
    if not shards:
        print(f'[{language}] Error: No shards found in datasets/vdivyasharma/IndicSynth/{lang_title}/*.parquet')
        return existing_df
    records = existing_df.to_dict('records') if not existing_df.empty else []
    pbar = tqdm(total=target_total, initial=len(records), desc=f'{language} Fakes')

    def process_shard(shard_path: str, model_target_key: str, sub_target: int) -> bool:
        nonlocal records
        shard_name = shard_path.split('/')[-1]
        url = f'https://huggingface.co/datasets/vdivyasharma/IndicSynth/resolve/main/{lang_title}/{shard_name}'
        temp_shard = f'temp_indicsynth_{lang_lower}_{shard_name}'
        print(f'\n  -> Fetching shard: {shard_name} (Model: {model_target_key}, target: {sub_target})...')
        success = download_shard_with_resume(url, temp_shard)
        if not success:
            print(f'  Failed to download {shard_name}, skipping...')
            return False
        try:
            pf = pq.ParquetFile(temp_shard)
            for rg in range(pf.metadata.num_row_groups):
                if len(records) >= target_total:
                    break
                table = pf.read_row_group(rg, columns=['audio', 'Generative Model', 'Source Speaker_ID', 'Target Speaker ID', 'Gender', 'TTS Transcript'])
                for item in table.to_pylist():
                    if len(records) >= target_total:
                        break
                    model_raw = str(item.get('Generative Model', 'unknown')).lower()
                    if 'xtts' in model_raw:
                        model_key = 'xtts_v2'
                    elif 'vits' in model_raw:
                        model_key = 'vits'
                    elif 'freevc' in model_raw:
                        model_key = 'freevc24'
                    else:
                        model_key = model_raw
                    if model_counts[model_key] >= sub_target and len(records) < target_total - 100:
                        continue
                    audio_dict = item.get('audio')
                    if not audio_dict or not isinstance(audio_dict, dict) or 'bytes' not in audio_dict:
                        continue
                    raw_bytes = audio_dict['bytes']
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
                    filename = f'indicsynth_{lang_lower}_{model_key}_{idx:05d}.wav'
                    filepath = os.path.join(save_dir, filename).replace('\\', '/')
                    if filepath in collected_paths:
                        continue
                    if audio_arr.ndim > 1:
                        audio_arr = audio_arr.mean(axis=1)
                    sf.write(filepath, audio_arr, sr, subtype='PCM_16')
                    collected_paths.add(filepath)
                    model_counts[model_key] += 1
                    rec = {'file_path': filepath, 'label': 1, 'language': lang_lower, 'speaker_id': str(item.get('Source Speaker_ID', f'src_{idx}')), 'target_speaker_id': str(item.get('Target Speaker ID', f'tgt_{idx}')), 'gender': str(item.get('Gender', 'unknown')), 'source_dataset': 'IndicSynth', 'generative_model': model_key, 'transcript': str(item.get('TTS Transcript', '')), 'duration': round(dur, 3), 'sample_rate': sr}
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
            return True
        except Exception as e:
            print(f'  Warning: Shard processing error ({e}). Skipping...')
            if os.path.exists(temp_shard):
                try:
                    os.remove(temp_shard)
                except Exception:
                    pass
            return False
    targets = LANGUAGE_SHARD_TARGETS.get(lang_lower, [])
    processed_shards = set()
    for model_name, shard_indices, sub_target in targets:
        if len(records) >= target_total:
            break
        if model_counts[model_name] >= sub_target:
            print(f'[{language}] Quota for {model_name} satisfied ({model_counts[model_name]}/{sub_target}).')
            continue
        for s_idx in shard_indices:
            if s_idx >= len(shards):
                continue
            shard = shards[s_idx]
            if shard in processed_shards:
                continue
            processed_shards.add(shard)
            process_shard(shard, model_target_key=model_name, sub_target=sub_target)
            if model_counts[model_name] >= sub_target or len(records) >= target_total:
                break
    if len(records) < target_total:
        print(f'[{language}] Current total {len(records)}/{target_total}. Topping off from remaining shards...')
        for shard in shards:
            if len(records) >= target_total:
                break
            if shard in processed_shards:
                continue
            processed_shards.add(shard)
            process_shard(shard, model_target_key='top-off', sub_target=target_total)
    pbar.close()
    final_df = pd.DataFrame(records)
    final_df.to_csv(meta_path, index=False)
    print(f'[{language}] Finished! Saved {len(final_df)} clips to {save_dir}. Distribution: {dict(model_counts)}')
    return final_df

def main():
    parser = argparse.ArgumentParser(description='Selective Streaming Sampler for IndicSynth')
    parser.add_argument('--languages', nargs='+', default=['Hindi', 'Telugu', 'Malayalam', 'Bengali'], help='Languages to stream')
    parser.add_argument('--target_per_lang', type=int, default=3000, help='Target number of synthetic clips per language')
    parser.add_argument('--target_per_model', type=int, default=1000, help='Target number of clips per generative model')
    parser.add_argument('--output_root', type=str, default='data/raw/fake', help='Directory to save raw fake audio')
    args = parser.parse_args()
    all_dfs = []
    for lang in args.languages:
        df = download_synthetic_language(language=lang, target_total=args.target_per_lang, output_root=args.output_root)
        all_dfs.append(df)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv('data/raw/fake/all_fake_metadata.csv', index=False)
        print(f'\n[ALL FAKES COMPLETE] Total clips: {len(combined)}')
        print(combined.groupby(['language', 'generative_model']).size())
if __name__ == '__main__':
    main()
