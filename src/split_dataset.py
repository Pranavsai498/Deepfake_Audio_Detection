import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def partition_speaker_disjoint(df: pd.DataFrame, train_ratio: float=0.7, val_ratio: float=0.15, test_ratio: float=0.15, seed: int=42) -> tuple:
    df = df.copy()

    def get_cluster(row):
        spk = str(row.get('speaker_id', 'spk_unknown'))
        tgt = str(row.get('target_speaker_id', ''))
        if tgt and tgt not in ['nan', 'None', '', spk]:
            return f'{min(spk, tgt)}_{max(spk, tgt)}'
        return spk
    df['speaker_cluster'] = df.apply(get_cluster, axis=1)
    train_rows, val_rows, test_rows = ([], [], [])
    for (lang, lbl), group in df.groupby(['language', 'label']):
        n_clusters = group['speaker_cluster'].nunique()
        if n_clusters >= 3:
            gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
            train_val_idx, test_idx = next(gss_test.split(group, groups=group['speaker_cluster']))
            test_subset = group.iloc[test_idx]
            train_val_subset = group.iloc[train_val_idx]
            val_rel_ratio = val_ratio / (train_ratio + val_ratio)
            gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=seed + 1)
            train_idx, val_idx = next(gss_val.split(train_val_subset, groups=train_val_subset['speaker_cluster']))
            train_subset = train_val_subset.iloc[train_idx]
            val_subset = train_val_subset.iloc[val_idx]
        else:
            shuffled = group.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n_tot = len(shuffled)
            n_train = int(round(train_ratio * n_tot))
            n_val = int(round(val_ratio * n_tot))
            train_subset = shuffled.iloc[:n_train]
            val_subset = shuffled.iloc[n_train:n_train + n_val]
            test_subset = shuffled.iloc[n_train + n_val:]
        train_rows.append(train_subset)
        val_rows.append(val_subset)
        test_rows.append(test_subset)
    train_df = pd.concat(train_rows, ignore_index=True)
    val_df = pd.concat(val_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    return (train_df, val_df, test_df)

def audit_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    report = {'leakage_detected': False, 'issues': []}
    train_files = set(train_df['file_path'])
    val_files = set(val_df['file_path'])
    test_files = set(test_df['file_path'])
    tv_files = train_files.intersection(val_files)
    tt_files = train_files.intersection(test_files)
    vt_files = val_files.intersection(test_files)
    if tv_files or tt_files or vt_files:
        report['leakage_detected'] = True
        report['issues'].append(f'File path overlap: Train-Val={len(tv_files)}, Train-Test={len(tt_files)}, Val-Test={len(vt_files)}')
    if 'audio_sha256' in train_df.columns:
        train_hashes = set(train_df['audio_sha256'])
        val_hashes = set(val_df['audio_sha256'])
        test_hashes = set(test_df['audio_sha256'])
        tv_hash = train_hashes.intersection(val_hashes)
        tt_hash = train_hashes.intersection(test_hashes)
        vt_hash = val_hashes.intersection(test_hashes)
        if tv_hash or tt_hash or vt_hash:
            report['leakage_detected'] = True
            report['issues'].append(f'Audio hash collision: Train-Val={len(tv_hash)}, Train-Test={len(tt_hash)}, Val-Test={len(vt_hash)}')
    train_spks = set(train_df['speaker_cluster'])
    val_spks = set(val_df['speaker_cluster'])
    test_spks = set(test_df['speaker_cluster'])
    tv_spks = train_spks.intersection(val_spks)
    tt_spks = train_spks.intersection(test_spks)
    vt_spks = val_spks.intersection(test_spks)
    if tv_spks or tt_spks or vt_spks:
        report['leakage_detected'] = True
        report['issues'].append(f'Speaker cluster overlap: Train-Val={len(tv_spks)}, Train-Test={len(tt_spks)}, Val-Test={len(vt_spks)}')
    return report

def plot_dataset_statistics(all_df: pd.DataFrame, output_dir: str='results/dataset_statistics'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='darkgrid')
    plt.figure(figsize=(9, 4.5))
    ax = sns.countplot(data=all_df, x='language', hue='label', palette=['#10B981', '#EF4444'])
    plt.title('Multilingual Dataset Balance (Real = 0 vs. Fake = 1)', fontsize=12, fontweight='bold')
    plt.xlabel('Language', fontweight='bold')
    plt.ylabel('Number of Clips', fontweight='bold')
    plt.legend(title='Label', labels=['Genuine (Real)', 'Synthetic (Fake)'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_balance_per_language.png'), dpi=200)
    plt.close()
    fakes_only = all_df[all_df['label'] == 1]
    if not fakes_only.empty:
        plt.figure(figsize=(8, 4.2))
        sns.countplot(data=fakes_only, x='language', hue='generative_model', palette='viridis')
        plt.title('Synthetic Data Generation Architecture Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Language', fontweight='bold')
        plt.ylabel('Number of Clips', fontweight='bold')
        plt.legend(title='Generative Model')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'generative_model_distribution.png'), dpi=200)
        plt.close()
    plt.figure(figsize=(8, 4.2))
    sns.histplot(data=all_df, x='duration', hue='split', multiple='stack', bins=30, palette='magma')
    plt.title('Audio Duration Distribution Across Dataset Splits', fontsize=12, fontweight='bold')
    plt.xlabel('Duration (seconds)', fontweight='bold')
    plt.ylabel('Clip Count', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_histogram.png'), dpi=200)
    plt.close()
    if 'gender' in all_df.columns:
        valid_gender = all_df[all_df['gender'].isin(['Male', 'Female', 'male', 'female', 'M', 'F'])]
        if not valid_gender.empty:
            plt.figure(figsize=(7, 4.2))
            sns.countplot(data=valid_gender, x='language', hue='gender', palette='Blues')
            plt.title('Speaker Gender Distribution', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gender_distribution.png'), dpi=200)
            plt.close()

def run_splitting(metadata_path: str='data/metadata/all_metadata.csv'):
    if not os.path.exists(metadata_path):
        print(f'Error: {metadata_path} not found. Please run prepare_dataset.py first.')
        return
    df = pd.read_csv(metadata_path)
    print(f"Loaded total metadata: {len(df)} samples across languages: {df['language'].unique().tolist()}")
    train_df, val_df, test_df = partition_speaker_disjoint(df)
    audit = audit_leakage(train_df, val_df, test_df)
    print('\n================ ANTI-LEAKAGE AUDIT REPORT ================')
    if audit['leakage_detected']:
        print('⚠️ LEAKAGE DETECTED:')
        for issue in audit['issues']:
            print(f'  - {issue}')
    else:
        print('✅ PASS: Zero speaker overlap, zero duplicate file paths, and zero hash collisions across splits!')
    train_path = 'data/metadata/train.csv'
    val_path = 'data/metadata/val.csv'
    test_path = 'data/metadata/test.csv'
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    full_split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_split_df.to_csv(metadata_path, index=False)
    print(f'\n================ SPLIT COUNTS ================')
    print(f'TRAIN Split (70%):      {len(train_df)} clips')
    print(f'VALIDATION Split (15%): {len(val_df)} clips')
    print(f'TEST Split (15%):       {len(test_df)} clips')
    print(f'Total Corpus:           {len(full_split_df)} clips')
    plot_dataset_statistics(full_split_df)
    print('Saved statistical distribution plots to results/dataset_statistics/')
if __name__ == '__main__':
    run_splitting()
