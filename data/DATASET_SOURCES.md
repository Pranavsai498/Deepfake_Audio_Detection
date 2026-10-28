# Multilingual Indian Speech Dataset Sources & Licensing

This document records the provenance, licensing, and citations for all datasets utilized in the development of the Indian Language Audio Deepfake Detection System (Hindi, Telugu, Malayalam, Bengali).

---

## 1. Synthetic / Deepfake Speech Source (Class: FAKE = 1)

### IndicSynth
* **Dataset Name**: IndicSynth: A Large-Scale Multilingual Synthetic Speech Dataset for Low-Resource Indian Languages
* **Repository**: [`vdivyasharma/IndicSynth`](https://huggingface.co/datasets/vdivyasharma/IndicSynth)
* **Authors**: V. Divya Sharma, Vijval Ekbote, Anubha Gupta (IIIT-Delhi)
* **Academic Recognition**: Outstanding Paper Award, ACL 2025
* **License**: Creative Commons Attribution NonCommercial 4.0 International (**CC-BY-NC-4.0**)
* **Languages Extracted**:
  * Hindi (`hi`)
  * Telugu (`te`)
  * Malayalam (`ml`)
  * Bengali (`bn`)
* **Generative Architectures Sampled**:
  * **XTTS-v2** (Auto-regressive neural voice cloning with latent diffusion vocoding)
  * **VITS** (Conditional variational autoencoder with adversarial learning & flow-based duration predictor)
  * **FreeVC24** (Zero-shot one-to-one voice conversion)
* **Sampling Method**: Selective streaming of Parquet shards without full corpus download.
* **Target Count**: ~3,000 synthetic audio clips per language (~12,000 clips total).
* **Citation**:
  ```bibtex
  @inproceedings{sharma2025indicsynth,
    title={IndicSynth: A Large-Scale Multilingual Synthetic Speech Dataset for Low-Resource Indian Languages},
    author={Sharma, V. Divya and Ekbote, Vijval and Gupta, Anubha},
    booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)},
    year={2025}
  }
  ```

---

## 2. Authentic Human Speech Source (Class: REAL = 0)

### eQOURSE Multilingual Speech & FLEURS (Real Human Recordings)
* **Dataset Names**: 
  1. `eQOURSE/multilingual-speech` (Spontaneous native conversational speech across Indian languages)
  2. `google/fleurs` (Open high-fidelity speech corpus across Indian regional dialects)
* **Repositories**:
  * [`eQOURSE/multilingual-speech`](https://huggingface.co/datasets/eQOURSE/multilingual-speech)
  * [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
* **Licenses**:
  * eQOURSE Speech: Open Research License / **CC-BY-4.0**
  * Google FLEURS: Creative Commons Attribution 4.0 International (**CC-BY-4.0**)
* **Languages Extracted**:
  * Hindi (`hi` / `hi_in`)
  * Telugu (`te` / `te_in`)
  * Malayalam (`ml` / `ml_in`)
  * Bengali (`bn` / `bn_in`)
* **Acoustic Characteristics**: Natural human speakers, diverse male/female demographics, varied acoustic environments matching real-world microphones.
* **Target Count**: ~3,000 authentic human audio clips per language (~12,000 clips total).
* **Citation**:
  ```bibtex
  @inproceedings{conneau2023fleurs,
    title={FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech},
    author={Conneau, Alexis and others},
    booktitle={IEEE Spoken Language Technology Workshop (SLT)},
    year={2023}
  }
  ```

---

## 3. Class Balance Summary

| Language | REAL (Bonafide = 0) | FAKE (Synthetic = 1) | Target Total |
| :--- | :--- | :--- | :--- |
| **Hindi** | ~3,000 | ~3,000 (XTTS, VITS, FreeVC24) | 6,000 |
| **Telugu** | ~3,000 | ~3,000 (XTTS, VITS, FreeVC24) | 6,000 |
| **Malayalam** | ~3,000 | ~3,000 (XTTS, VITS, FreeVC24) | 6,000 |
| **Bengali** | ~3,000 | ~3,000 (XTTS, VITS, FreeVC24) | 6,000 |
| **Total Corpus** | **12,000** | **12,000** | **24,000** |
