# Multilingual Indic Audio Deepfake Detection

A production-grade, from-scratch deep learning system for detecting synthetic, cloned, and telephony-transmitted speech across major Indian languages (**Telugu, Hindi, Malayalam, Bengali, and Indian English**).

---

## 🌟 Highlights

- **Corpus**: Trained on **23,090 balanced audio clips** across 5 languages:
  - **11,548 Authentic Human Clips** from Google Crowdsourced High-Quality Speech (`google/svq_speech` v1.0.0).
  - **11,542 Synthetic Speech Clones & Formant Synthesizers** spanning:
    - **IndicSynth** (`vdivyasharma/IndicSynth`): FreeVC24 (Voice Conversion), XTTS-v2 (Zero-Shot Neural Cloning), and VITS (End-to-End Neural TTS).
    - **Synthetic Voices Dataset**: eSpeak formant TTS, Telugu code-switching, and dual acoustic conditioning:
      - *Clean Wideband 16 kHz*
      - *Telephone Bandpass Filtered (300–3,400 Hz)* simulating GSM/PSTN mobile voice channels.
- **Strict Anti-Leakage Protocol**: Disjoint partitioning by speaker ID (`GroupShuffleSplit`) with zero speaker overlap, zero duplicate hashes, and zero identity leakage across Train (14,300), Validation (3,579), and Test (5,211) splits.
- **Lightweight Architecture**: **342,211 parameters** ($<0.35\text{M}$ params, well below the $<5\text{M}$ limit; disk size: 1.38 MB).
- **Acoustic Features**: 120-dimensional representations (40 Static MFCCs + 40 Velocity Deltas + 40 Acceleration Delta-Deltas) with Mel filterbank caching and online SpecAugment.
- **Held-Out Test Benchmark (5,211 Unseen Clips)**:
  - **Calibrated Accuracy**: **`99.71%`** (Default 0.50 Threshold Accuracy: `98.79%`)
  - **Equal Error Rate (EER)**: **`0.29%`** ($\tau = 0.0215$)
  - **ROC-AUC**: **`0.9999`**
  - **Precision / Recall / F1**: **`99.75% / 99.71% / 99.73%`**
  - **Confusion Matrix**: TN = 2,423 | FP = 7 | FN = 8 | TP = 2,773

---

## 📊 Held-Out Test Set Performance (5,211 Unseen Clips)

Evaluated on unseen held-out speakers across all supported languages and generative architectures:

| Language | Test Clips | Real (Human) | Fake (Synthetic) | Accuracy | Equal Error Rate (EER) | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Malayalam (മലയാളം)** | 1,719 | 479 | 1,240 | **`100.00%`** | **`0.00%`** | **`1.0000`** |
| **Bengali (বাংলা)** | 900 | 711 | 189 | **`99.78%`** | **`0.00%`** | **`0.9947`** |
| **Telugu (తెలుగు)** | 1,926 | 760 | 1,166 | **`99.58%`** | **`0.26%`** | **`0.9966`** |
| **Hindi (हिन्दी)** | 652 | 480 | 172 | **`99.23%`** | **`0.00%`** | **`0.9857`** |
| **English / Code-Switching** | 14 | 0 | 14 | **`100.00%`** | **`—`** (Fake Only) | **`1.0000`** |
| **Combined Benchmark** | **5,211** | **2,430** | **2,781** | **`99.71%`** | **`0.29%`** | **`0.9973`** |

### Synthetic Generator Detection Rate (Recall)
- **FreeVC24 (Voice Conversion)**: **`100.00%`** (881 / 881 detected)
- **eSpeak TTS (Telephone Bandpass & Clean Wideband)**: **`100.00%`** (75 / 75 detected)
- **XTTS-v2 (Zero-Shot Neural Cloning)**: **`99.56%`** (1,817 / 1,825 detected)

---

## 🏗️ Neural Architecture

The detector is implemented from scratch in pure PyTorch (`src/model.py`) without pre-trained black-box checkpoints:

1. **2D Residual Convolutional Frontend**: 3-layer gated 2D convolutions with Batch Normalization, residual skip connections, and Mish activation functions extracting fine-grained time-frequency patterns.
2. **Bidirectional LSTM**: 2-layer BiLSTM (128 hidden units per direction = 256 temporal context representation) capturing forward and reverse temporal phonetic transitions.
3. **Multi-Head Temporal Self-Attention**: Scaled dot-product self-attention across time frames, directing gradient sensitivity toward transient vocoder artifacts and phase discontinuities.
4. **Classification MLP**: Linear projection layer with Dropout ($p=0.30$) and label smoothing ($0.03$) to prevent overconfident calibration errors.

```
Input Audio (WAV, 16 kHz)
   │
   ▼
[120-dim MFCC Engine] ─── (40 Static + 40 Delta + 40 Delta-Delta)
   │
   ▼
[2D ResConv Blocks] ──── (3 Gated Conv Layers + BatchNorm + Mish + Residual Skips)
   │
   ▼
[Bidirectional LSTM] ─── (2 Layers, 128 Hidden Dim per direction)
   │
   ▼
[Temporal Self-Attention] (Multi-Head Attention across frame sequences)
   │
   ▼
[Dense Classification] ─ (Dropout 0.30 ➔ Linear ➔ Calibrated Softmax)
   │
   ▼
Verdict: REAL / FAKE (with confidence % and forensic acoustic metrics)
```

---

## 🛡️ Robust Audio Preprocessing & Channel Conditioning

Real-world voice calls and messaging apps frequently alter acoustic characteristics. The ingestion and inference pipelines (`src/preprocessing/audio_loader.py`) apply standard conditioning:
- **Resampling**: Polyphase filter resampling to 16,000 Hz mono PCM.
- **RMS Energy Normalization**: Scaled to target RMS of 0.100 with dynamic range limiting ($-1.0$ to $+1.0$) to prevent hard digital clipping.
- **Telephone Bandpass Filtering (300–3,400 Hz)**: The model is specifically fine-tuned with telephony-conditioned audio to eliminate blind spots caused by telephone network bandwidth compression.

---

## 🚀 Quickstart

### 1. Installation
```powershell
pip install -r requirements.txt
```

### 2. Launch Interactive Forensic Studio (Gradio Web UI)
```powershell
python demo/app.py
```
Open **`http://127.0.0.1:7860`** in your browser to inspect audio recordings, file uploads, spectrograms, RMS energy profiles, and 1-click regional presets across Telugu, Hindi, Malayalam, and Bengali.

### 3. Command-Line Inference
Run fast inference on any WAV audio file:

```powershell
# Analyze Telugu speech
python src/inference.py --audio "path/to/telugu_voice.wav" --language telugu

# Analyze Hindi, Malayalam, Bengali, or Indian English
python src/inference.py --audio "path/to/voice_call.wav" --language hindi
```

### 4. Evaluate Held-Out Benchmark
Re-run the full evaluation suite and regenerate all figures and metrics:

```powershell
python src/evaluate.py --checkpoint checkpoints/best_detector.pt --test_csv data/metadata/test.csv --batch_size 128
```
Artifacts saved to `results/`:
- `results/confusion_matrix.png`
- `results/roc_curve.png`
- `results/language_performance.png`
- `results/metrics.json`

### 5. Ingesting New Datasets & Incremental Retraining
To add new synthetic or real datasets without catastrophic forgetting:

```powershell
# Ingest new synthetic samples into metadata splits
python src/ingest_synthetic_dataset.py

# Warm-start fine-tuning from existing checkpoint
python src/train.py --resume checkpoints/best_detector_v1_backup.pt --epochs 10 --lr 0.0003
```

---

## 📁 Repository Structure

```
deepfake-audio-indic/
├── checkpoints/
│   ├── best_detector.pt             # Active fine-tuned model (342K params, 1.38 MB)
│   └── best_detector_v1_backup.pt   # Original base model backup
├── configs/
│   └── default_config.yaml          # Hyperparameters and pipeline configuration
├── data/
│   ├── metadata/                    # Train (14,300), Val (3,579), Test (5,211) CSVs
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   └── all_metadata.csv         # Consolidated dataset (23,090 clips)
│   ├── processed/                   # QC-verified standardized 16kHz WAV files
│   └── raw/
│       ├── google_svq/              # Google Crowdsourced High-Quality Speech (Real)
│       ├── indicsynth/              # FreeVC24, XTTS-v2, VITS clones (Fake)
│       └── synthetic_voices_dataset/# eSpeak telephone & wideband dataset (Fake)
├── demo/
│   └── app.py                       # Cyber-Forensic Gradio Studio UI
├── results/                         # Official benchmark evaluation artifacts
│   ├── confusion_matrix.png         # Calibrated confusion matrix
│   ├── roc_curve.png                # ROC curve with EER operating point
│   ├── language_performance.png     # Accuracy & EER bar charts per language
│   └── metrics.json                 # Machine-readable evaluation metrics
├── src/
│   ├── dataset.py                   # PyTorch Dataset with in-memory caching & SpecAugment
│   ├── features.py                  # 120-dim MFCC feature extraction engine
│   ├── model.py                     # Hybrid CNN-BiLSTM-Attention architecture
│   ├── train.py                     # Training & fine-tuning pipeline with cosine annealing
│   ├── evaluate.py                  # Full test benchmark suite & plotting
│   ├── inference.py                 # CLI and programmatic inference engine
│   ├── ingest_synthetic_dataset.py  # Ingestion script for new synthetic datasets
│   ├── download_real.py             # Google SVQ downloader
│   ├── download_indicsynth.py       # IndicSynth downloader
│   ├── prepare_dataset.py           # 16kHz resampling and QC filter
│   ├── split_dataset.py             # Speaker-disjoint partitioning
│   ├── evaluation/
│   │   └── eer.py                   # Equal Error Rate (EER) calculator
│   └── preprocessing/
│       └── audio_loader.py          # Resampling, RMS calibration, & peak limiting
├── README.md                        # Project documentation
└── requirements.txt                 # Python dependencies
```

---

## 📜 Citation & Credits

- **Google Crowdsourced High-Quality Speech Data**: `google/svq_speech` (Telugu, Hindi, Malayalam, Bengali).
- **IndicSynth Dataset**: Divya Sharma et al., `vdivyasharma/IndicSynth` (FreeVC24, XTTS-v2, VITS).
- **Evaluation Standards**: ASVspoof Equal Error Rate (EER) protocol and Scikit-Learn evaluation suites.