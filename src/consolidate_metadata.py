import os
import pandas as pd
LANGS = ['hindi', 'telugu', 'malayalam', 'bengali']

def main():
    print('=' * 60)
    print('CONSOLIDATING METADATA FOR 24,000 AUDIO CORPUS')
    print('=' * 60)
    fake_dfs = []
    for lang in LANGS:
        p = os.path.join('data', 'raw', 'fake', lang, 'metadata.csv')
        if os.path.exists(p):
            df = pd.read_csv(p)
            print(f"[FAKE] {lang.upper():<10}: {len(df)} rows. Models: {dict(df['generative_model'].value_counts())}")
            fake_dfs.append(df)
        else:
            print(f'[FAKE] Missing {p}!')
    if fake_dfs:
        all_fake = pd.concat(fake_dfs, ignore_index=True)
        all_fake['file_path'] = all_fake['file_path'].str.replace('\\', '/')
        out_fake = 'data/raw/fake/all_fake_metadata.csv'
        all_fake.to_csv(out_fake, index=False)
        print(f'\n=> Successfully written {out_fake} ({len(all_fake)} rows)')
    real_dfs = []
    real_comb = 'data/raw/real/all_real_metadata.csv'
    if os.path.exists(real_comb):
        all_real = pd.read_csv(real_comb)
        all_real['file_path'] = all_real['file_path'].str.replace('\\', '/')
        print(f'\n=> Found master real metadata at {real_comb} ({len(all_real)} rows)')
    else:
        for lang in LANGS:
            p = os.path.join('data', 'raw', 'real', lang, 'metadata.csv')
            if os.path.exists(p):
                df = pd.read_csv(p)
                print(f'[REAL] {lang.upper():<10}: {len(df)} rows')
                real_dfs.append(df)
        all_real = pd.concat(real_dfs, ignore_index=True)
        all_real['file_path'] = all_real['file_path'].str.replace('\\', '/')
        all_real.to_csv(real_comb, index=False)
        print(f'\n=> Saved master real metadata at {real_comb} ({len(all_real)} rows)')
    print('\n' + '=' * 60)
    print('DATASET BREAKDOWN SUMMARY:')
    print('=' * 60)
    print('REAL SPEECH (by language):')
    print(all_real['language'].value_counts())
    print('\nSYNTHETIC SPEECH (by language & generator):')
    print(all_fake.groupby(['language', 'generative_model']).size())
    print('\nGRAND TOTALS:')
    print(f'Total Real Rows: {len(all_real)}')
    print(f'Total Fake Rows: {len(all_fake)}')
    print(f'Combined Total:  {len(all_real) + len(all_fake)}')
    print('=' * 60)
if __name__ == '__main__':
    main()
