import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import HybridCNNLSTMDetector
from src.dataset import AudioDeepfakeDataset
from src.evaluation.eer import compute_eer
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def evaluate_dataset(model: nn.Module, test_df: pd.DataFrame, device: torch.device, batch_size: int=32) -> dict:
    model.eval()
    dataset = AudioDeepfakeDataset(test_df, is_train=False, target_frames=300)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_preds, all_labels, all_fake_probs = ([], [], [])
    all_langs, all_models, all_spks = ([], [], [])
    with torch.no_grad():
        for feats, labels, langs, models, spks in loader:
            feats = feats.to(device)
            logits = model(feats)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_fake_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_langs.extend(langs)
            all_models.extend(models)
            all_spks.extend(spks)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_fake_probs = np.array(all_fake_probs)
    acc = accuracy_score(all_labels, all_preds) * 100.0
    prec = precision_score(all_labels, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_labels, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100.0
    try:
        roc_auc = roc_auc_score(all_labels, all_fake_probs)
    except Exception:
        roc_auc = 0.5
    try:
        eer_res = compute_eer(all_labels, all_fake_probs)
        eer = float(eer_res['eer']) * 100.0
        tau = float(eer_res['threshold'])
    except Exception:
        eer = 50.0
        tau = 0.5
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    calib_preds = (all_fake_probs >= tau).astype(int)
    calib_acc = accuracy_score(all_labels, calib_preds) * 100.0
    calib_prec = precision_score(all_labels, calib_preds, zero_division=0) * 100.0
    calib_rec = recall_score(all_labels, calib_preds, zero_division=0) * 100.0
    calib_f1 = f1_score(all_labels, calib_preds, zero_division=0) * 100.0
    c_cm = confusion_matrix(all_labels, calib_preds)
    c_tn, c_fp, c_fn, c_tp = c_cm.ravel() if c_cm.size == 4 else (0, 0, 0, 0)
    results = {'overall': {'accuracy': round(calib_acc, 2), 'calibrated_accuracy': round(calib_acc, 2), 'default_accuracy': round(acc, 2), 'precision': round(calib_prec, 2), 'recall': round(calib_rec, 2), 'f1_score': round(calib_f1, 2), 'roc_auc': round(float(roc_auc), 4), 'eer': round(eer, 2), 'threshold_tau': round(float(tau), 4), 'confusion_matrix': {'tn': int(c_tn), 'fp': int(c_fp), 'fn': int(c_fn), 'tp': int(c_tp)}, 'default_confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}, 'total_clips': len(all_labels)}, 'per_language': {}, 'per_generative_model': {}, 'raw_predictions': {'labels': all_labels.tolist(), 'fake_probs': all_fake_probs.tolist(), 'calib_preds': calib_preds.tolist(), 'languages': all_langs, 'models': all_models}}
    df_eval = pd.DataFrame({'label': all_labels, 'pred': calib_preds, 'default_pred': all_preds, 'prob': all_fake_probs, 'language': all_langs, 'model': all_models})
    for lang, grp in df_eval.groupby('language'):
        l_acc = accuracy_score(grp['label'], grp['pred']) * 100.0
        l_f1 = f1_score(grp['label'], grp['pred'], zero_division=0) * 100.0
        try:
            l_eer = float(compute_eer(grp['label'].values, grp['prob'].values)['eer']) * 100.0
        except Exception:
            l_eer = 50.0
        results['per_language'][lang] = {'accuracy': round(l_acc, 2), 'f1_score': round(l_f1, 2), 'eer': round(l_eer, 2), 'total': len(grp), 'real_count': int((grp['label'] == 0).sum()), 'fake_count': int((grp['label'] == 1).sum())}
    fakes_only = df_eval[df_eval['label'] == 1]
    for gm, grp in fakes_only.groupby('model'):
        gm_rec = recall_score(grp['label'], grp['pred'], zero_division=0) * 100.0
        results['per_generative_model'][gm] = {'detection_rate': round(gm_rec, 2), 'count': len(grp)}
    return results

def plot_evaluation_charts(results: dict, output_dir: str='results'):
    os.makedirs(output_dir, exist_ok=True)
    raw = results['raw_predictions']
    labels = np.array(raw['labels'])
    probs = np.array(raw['fake_probs'])
    cm_dict = results['overall']['confusion_matrix']
    cm = np.array([[cm_dict['tn'], cm_dict['fp']], [cm_dict['fn'], cm_dict['tp']]])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred Real (0)', 'Pred Fake (1)'], yticklabels=['Actual Real (0)', 'Actual Fake (1)'], annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    plt.title(f"Confusion Matrix (Acc: {results['overall']['accuracy']}%, EER: {results['overall']['eer']}%)", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='#2563EB', lw=2.5, label=f"ROC Curve (AUC = {results['overall']['roc_auc']:.4f})")
        plt.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance (AUC = 0.50)')
        eer_val = results['overall']['eer'] / 100.0
        plt.scatter([eer_val], [1.0 - eer_val], color='#DC2626', s=80, zorder=5, label=f"EER Operating Point ({results['overall']['eer']}%)")
        plt.xlabel('False Positive Rate (FAR)', fontweight='bold')
        plt.ylabel('True Positive Rate (1 - FRR)', fontweight='bold')
        plt.title('ROC Curve & EER Operating Point', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=200)
        plt.close()
    except Exception as e:
        print('ROC curve plot note:', e)
    per_lang = results['per_language']
    if per_lang:
        lang_names = list(per_lang.keys())
        accs = [per_lang[l]['accuracy'] for l in lang_names]
        eers = [per_lang[l]['eer'] for l in lang_names]
        x = np.arange(len(lang_names))
        width = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - width / 2, accs, width, label='Accuracy (%)', color='#3B82F6')
        plt.bar(x + width / 2, eers, width, label='EER (%)', color='#EF4444')
        plt.xticks(x, [l.capitalize() for l in lang_names], fontweight='bold')
        plt.ylabel('Percentage (%)', fontweight='bold')
        plt.title('Deepfake Audio Detection Accuracy & EER Across Indic Languages', fontweight='bold')
        plt.legend()
        plt.ylim(0, 105)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'language_performance.png'), dpi=200)
        plt.close()

def run_evaluation(checkpoint_path: str='checkpoints/best_detector.pt', test_csv: str='data/metadata/test.csv', results_dir: str='results', batch_size: int=64):
    if not os.path.exists(checkpoint_path):
        print(f'Error: Checkpoint {checkpoint_path} not found.')
        return
    if not os.path.exists(test_csv):
        print(f'Error: Test file {test_csv} not found.')
        return
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Loading best model from {checkpoint_path}...')
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = HybridCNNLSTMDetector().to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    test_df = pd.read_csv(test_csv)
    print(f'Evaluating held-out unseen test set ({len(test_df)} clips)...')
    results = evaluate_dataset(model, test_df, device, batch_size=batch_size)
    metrics_summary = {'overall': results['overall'], 'per_language': results['per_language'], 'per_generative_model': results['per_generative_model']}
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    plot_evaluation_charts(results, output_dir=results_dir)
    print('\n================ FINAL BENCHMARK EVALUATION ================')
    ov = results['overall']
    print(f"Calibrated Accuracy: {ov['calibrated_accuracy']}% (Default 0.50 Acc: {ov['default_accuracy']}%)")
    print(f"Overall EER:         {ov['eer']}% (tau = {ov['threshold_tau']})")
    print(f"Precision:         {ov['precision']}%")
    print(f"Recall:            {ov['recall']}%")
    print(f"F1-Score:          {ov['f1_score']}%")
    print(f"ROC-AUC:           {ov['roc_auc']}")
    print(f"Confusion Matrix:  TN={ov['confusion_matrix']['tn']}, FP={ov['confusion_matrix']['fp']}, FN={ov['confusion_matrix']['fn']}, TP={ov['confusion_matrix']['tp']}")
    print('\n--- Per-Language Diagnostics ---')
    for l, d in results['per_language'].items():
        print(f"  {l.capitalize():<12} | Acc: {d['accuracy']:>6.2f}% | EER: {d['eer']:>6.2f}% | Total: {d['total']} clips")
    print('\n--- Synthetic Generator Detection Rate (Recall) ---')
    for gm, d in results['per_generative_model'].items():
        print(f"  {gm:<12} | Detection Rate: {d['detection_rate']:>6.2f}% ({d['count']} clips)")
    print(f'\nSaved figures and metrics to {results_dir}/')
    return results
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate deepfake detector on held-out test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_detector.pt', help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, default='data/metadata/test.csv', help='Path to test metadata CSV')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save evaluation artifacts')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    args = parser.parse_args()
    run_evaluation(checkpoint_path=args.checkpoint, test_csv=args.test_csv, results_dir=args.results_dir, batch_size=args.batch_size)
