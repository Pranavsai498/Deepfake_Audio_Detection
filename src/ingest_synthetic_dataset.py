import os
import shutil
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.model_selection import StratifiedShuffleSplit
data_dir = 'data/raw/synthetic_voices_dataset'
meta_csv = os.path.join(data_dir, 'metadata.csv')
audio_dir = os.path.join(data_dir, 'audio')
metadata_dir = 'data/metadata'
train_path = os.path.join(metadata_dir, 'train.csv')
val_path = os.path.join(metadata_dir, 'val.csv')
test_path = os.path.join(metadata_dir, 'test.csv')
all_meta_path = os.path.join(metadata_dir, 'all_metadata.csv')
for p in [train_path, val_path, test_path, all_meta_path]:
    bak = p + '.backup_orig'
    if not os.path.exists(bak):
        shutil.copy2(p, bak)
        print(f'Backed up: {p} -> {bak}')
new_df = pd.read_csv(meta_csv)
print(f'Loaded new dataset: {len(new_df)} rows')
lang_map = {'Indian_English': 'english', 'Telugu': 'telugu', 'Telugu_English_CodeSwitch': 'telugu', 'Hindi': 'hindi', 'Bengali': 'bengali', 'Malayalam': 'malayalam'}
records = []
for idx, row in new_df.iterrows():
    fname = row['file_name']
    fpath = os.path.join(audio_dir, fname).replace('\\', '/')
    if not os.path.exists(fpath):
        print(f'Warning: file not found {fpath}')
        continue
    audio, sr = sf.read(fpath, dtype='float32')
    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    dur = float(len(audio) / sr)
    raw_lang = row.get('language', 'english')
    std_lang = lang_map.get(raw_lang, raw_lang.lower())
    records.append({'file_path': fpath, 'label': 1, 'language': std_lang, 'speaker_id': f'espeak_{std_lang}_{idx:04d}', 'gender': 'synthetic', 'source_dataset': 'synthetic_voices_dataset', 'generative_model': 'espeak_tts', 'duration': round(dur, 2), 'sample_rate': sr, 'rms_energy': round(rms, 5), 'peak_amplitude': round(peak, 5), 'audio_sha256': f'espeak_{fname}', 'target_speaker_id': 'synthetic', 'transcript': str(row.get('text', '')), 'speaker_cluster': f'cluster_espeak_{std_lang}', 'condition': str(row.get('condition', 'wideband')), 'scenario': str(row.get('scenario', 'general'))})
df_new_full = pd.DataFrame(records)
print(f'Processed {len(df_new_full)} valid audio clips.')
df_new_full['strat_key'] = df_new_full['language'] + '_' + df_new_full['condition']
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
for train_idx, temp_idx in sss1.split(df_new_full, df_new_full['strat_key']):
    train_subset = df_new_full.iloc[train_idx].copy()
    temp_subset = df_new_full.iloc[temp_idx].copy()
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
for val_idx, test_idx in sss2.split(temp_subset, temp_subset['strat_key']):
    val_subset = temp_subset.iloc[val_idx].copy()
    test_subset = temp_subset.iloc[test_idx].copy()
train_subset['split'] = 'train'
val_subset['split'] = 'val'
test_subset['split'] = 'test'
print(f'Split counts: Train={len(train_subset)}, Val={len(val_subset)}, Test={len(test_subset)}')
train_subset = train_subset.drop(columns=['strat_key', 'condition', 'scenario'])
val_subset = val_subset.drop(columns=['strat_key', 'condition', 'scenario'])
test_subset = test_subset.drop(columns=['strat_key', 'condition', 'scenario'])
orig_train = pd.read_csv(train_path)
orig_val = pd.read_csv(val_path)
orig_test = pd.read_csv(test_path)
common_cols = [c for c in orig_train.columns if c in train_subset.columns]
merged_train = pd.concat([orig_train[common_cols], train_subset[common_cols]], ignore_index=True)
merged_val = pd.concat([orig_val[common_cols], val_subset[common_cols]], ignore_index=True)
merged_test = pd.concat([orig_test[common_cols], test_subset[common_cols]], ignore_index=True)
merged_train.to_csv(train_path, index=False)
merged_val.to_csv(val_path, index=False)
merged_test.to_csv(test_path, index=False)
merged_all = pd.concat([merged_train, merged_val, merged_test], ignore_index=True)
merged_all.to_csv(all_meta_path, index=False)
print('\n========================================================')
print(f'  DATASET INGESTION COMPLETED SUCCESSFULLY')
print(f'========================================================')
print(f"Train Set: {len(merged_train)} rows (Real: {sum(merged_train['label'] == 0)}, Fake: {sum(merged_train['label'] == 1)})")
print(f"Val Set:   {len(merged_val)} rows (Real: {sum(merged_val['label'] == 0)}, Fake: {sum(merged_val['label'] == 1)})")
print(f"Test Set:  {len(merged_test)} rows (Real: {sum(merged_test['label'] == 0)}, Fake: {sum(merged_test['label'] == 1)})")
print(f"Total:     {len(merged_all)} rows (Real: {sum(merged_all['label'] == 0)}, Fake: {sum(merged_all['label'] == 1)})")
