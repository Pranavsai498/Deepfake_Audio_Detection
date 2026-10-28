import os
import glob
import pandas as pd
LANGUAGES = ['hindi', 'telugu', 'malayalam', 'bengali']

def main():
    print('=' * 60)
    print('VERIFYING MULTILINGUAL DATASET INTEGRITY')
    print('=' * 60)
    temp_files = glob.glob('temp_indicsynth_*') + glob.glob('temp_shard_*')
    for tf in temp_files:
        try:
            os.remove(tf)
            print(f'Removed temporary file: {tf}')
        except Exception as e:
            print(f'Could not remove {tf}: {e}')
    summary = []
    for lang in LANGUAGES:
        real_dir = os.path.join('data', 'raw', 'real', lang)
        files = glob.glob(os.path.join(real_dir, '*.wav'))
        summary.append({'Category': 'Real (Google SVQ)', 'Language': lang, 'Target': 3000, 'Count': len(files), 'Status': 'PASS' if len(files) == 3000 else 'WARN' if len(files) >= 2950 else 'FAIL'})
    for lang in LANGUAGES:
        fake_dir = os.path.join('data', 'raw', 'fake', lang)
        files = glob.glob(os.path.join(fake_dir, '*.wav'))
        summary.append({'Category': 'Fake (IndicSynth)', 'Language': lang, 'Target': 3000, 'Count': len(files), 'Status': 'PASS' if len(files) == 3000 else 'WARN' if len(files) >= 2950 else 'FAIL'})
    df_summary = pd.DataFrame(summary)
    print(df_summary.to_string(index=False))
    total_real = sum((s['Count'] for s in summary if s['Category'].startswith('Real')))
    total_fake = sum((s['Count'] for s in summary if s['Category'].startswith('Fake')))
    total_all = total_real + total_fake
    print('\n' + '=' * 60)
    print(f'TOTAL REAL CLIPS: {total_real} / 12,000')
    print(f'TOTAL FAKE CLIPS: {total_fake} / 12,000')
    print(f'GRAND TOTAL:      {total_all} / 24,000')
    print('=' * 60)
    fake_meta_dfs = []
    for lang in LANGUAGES:
        meta_file = os.path.join('data', 'raw', 'fake', f'{lang}_fake_metadata.csv')
        if os.path.exists(meta_file):
            df_m = pd.read_csv(meta_file)
            fake_meta_dfs.append(df_m)
            print(f'Found fake metadata for {lang}: {len(df_m)} rows')
        else:
            print(f'Warning: Fake metadata not found for {lang} at {meta_file}')
    if fake_meta_dfs:
        all_fake_meta = pd.concat(fake_meta_dfs, ignore_index=True)
        out_meta_path = os.path.join('data', 'raw', 'fake', 'all_fake_metadata.csv')
        all_fake_meta.to_csv(out_meta_path, index=False)
        print(f'\nSaved master fake metadata to {out_meta_path} ({len(all_fake_meta)} rows)')
        print('\nGenerative Model breakdown across fake dataset:')
        print(all_fake_meta.groupby(['language', 'generative_model']).size())
if __name__ == '__main__':
    main()
