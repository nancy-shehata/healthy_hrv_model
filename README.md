# Synthetic HRV Generation for ALS Cardiac Autonomic Research

## Project Overview

This project investigates cardiac autonomic dysfunction in Amyotrophic Lateral Sclerosis (ALS) through the analysis of heart rate variability (HRV) signals. ALS is a progressive neurodegenerative disease, and while its effects on motor function are well-documented, its impact on the autonomic nervous system — which regulates heart rhythm — is less understood. HRV, derived from the timing between heartbeats (RR intervals), serves as a non-invasive window into autonomic function.

The core scientific goal is to build a healthy HRV baseline using deep learning, then quantify how ALS patients deviate from that baseline. Critically, the signal of interest is not any single HRV metric but the **deviation profile** — the pattern of how multiple features (SDNN, RMSSD, SD1/SD2, entropy, DFA α1) simultaneously shift in disease. This multi-feature deviation fingerprint is the proposed biomarker.

To build this baseline, we trained two deep learning models on RR interval sequences from 54 healthy subjects in the NSR2DB dataset (PhysioNet). Both models learn cardiac dynamics through a next-beat prediction objective — given a window of past beats, predict the next one — and were validated using nested leave-one-out cross-validation (LOOCV) across all 54 subjects with per-fold standardization to prevent data leakage.

---

## Repository Structure

```
.
└── healthy-hrv-research/
    ├── data/
    │   ├── Normal Sinus Rhythm RR Interval Dataset  # Original dataset
    │   └── NSR Processed/
    │       ├── RR Intervals                         # 54 subject RR interval .txt files (primary training data)
    │       └── patient_info_nsr.csv                 # Population info
    ├── models/
    │   ├── CNN_Cross_Validation.ipynb               # CNN — nested LOOCV, evaluation plots, model saving
    │   └── LSTM_W_Generation.ipynb                # LSTM — nested LOOCV + autoregressive generation
    ├── Notes/
    │   ├── disease_biomarker_data.pdf
    │   ├── Final Presentation.pdf
    │   └── Meeting Notes.pdf
    └── results/
        ├── CNN Results/
        │   ├── cnn_nested_cv_results.pkl            # Per-patient predictions for all 54 subjects (Google Drive only — see note below)
        │   └── RR_CNN_Healthy_Baseline_Weights.pth  # Final CNN model weights (Google Drive only — see note below)
        └── LSTM Results/
            ├── hrv_features_nsr.csv
            ├── loocv_results.csv                    # LSTM LOOCV metrics
            └── results_summary.png
```

**Notebooks:**

| File | Model | Framework | Description |
|------|-------|-----------|-------------|
| `CNN_Cross_Validation.ipynb` | CNN | PyTorch | Full nested LOOCV, final trained model, evaluation plots |
| `LSTM_W_Generation.ipynb` | LSTM | TensorFlow/Keras | Nested LOOCV + autoregressive RR sequence generation |

> **Note on large files:** `RR_CNN_Healthy_Baseline_Weights.pth` and `cnn_nested_cv_results.pkl` exceed GitHub's 100MB file size limit and are stored in Google Drive only. [Access All Information here] [https://drive.google.com/drive/folders/1jI2YyMKGj0UgpXtjC1Ekzm3cdZ29wDx3?usp=share_link]

---

## Dataset

**NSR2DB** — Normal Sinus Rhythm RR Interval Database
- **Source:** PhysioNet ([link](https://physionet.org/content/nsrdb/1.0.0/))
- **Subjects:** 54 healthy adults
- **Recording:** 24-hour Holter, 128 Hz sampling
- **Format:** RR intervals in milliseconds (.txt files)

**Preprocessing decisions:**
- First **1,000 RR intervals** per subject used for training
- **Physiological filter:** keep only intervals between 300ms–2000ms
- Files containing `'24'` in the filename are **dropped** (known annotation artifacts)
- RR values **rounded to integer milliseconds** (consistent with 128 Hz resolution limit of ~7.8ms)
- Per-fold **z-score standardization** fit on training data only, applied to validation — prevents data leakage

---

## Models

### CNN (`CNN_Cross_Validation.ipynb`)

**Architecture:**
```
Conv1d(1→32, k=5) + BatchNorm + ReLU + MaxPool
Conv1d(32→64, k=5) + BatchNorm + ReLU + MaxPool
Conv1d(64→128, k=3) + BatchNorm + ReLU
AdaptiveAvgPool1d(8)
Flatten → FC(128) + ReLU + Dropout(0.3) → FC(1)
```

- **Input:** 50-beat sliding window
- **Output:** next RR interval (ms)
- **Loss:** MSE | **Optimizer:** Adam
- **Batch size:** 1024 | **Final training epochs:** 30

**Validation — Nested LOOCV:**
- Outer loop: hold out 1 patient at a time (54 folds)
- Inner loop: GroupKFold (3 splits) tunes learning rate from {1e-3, 5e-4, 1e-4}
- Final model retrained on full training pool with best LR for 30 epochs
- **Metrics:** MAE (ms), RMSE (ms), R²

**Evaluation outputs:**
- Per-patient MAE / RMSE / R² bar charts
- Predicted vs. actual RR time series per patient
- Bland-Altman agreement plot across all 54 subjects
- Residual distribution histogram

---

### LSTM (`LSTM_W_Generation.ipynb`)

**Architecture:**
```
LSTM(units_1) + Dropout → LSTM(units_2) + Dropout → Dense(16, ReLU) → Dense(1)
```

- **Input:** 50-beat sequence (SEQ_LENGTH=50, STEP_SIZE=10)
- **Output:** next RR interval (ms)
- **Loss:** MSE | **Optimizer:** Adam
- **Generation length:** 1,000 beats per subject

**Validation — Nested LOOCV:**
- Same outer LOOCV structure as CNN
- Inner loop: KFold (5 splits) tunes over lstm_units_1 {32, 64, 128}, lstm_units_2 {16, 32}, dropout_rate {0.1, 0.2, 0.3}

**Generation:**
The LSTM extends beyond prediction to full **autoregressive sequence generation** — the trained model is seeded with real beats and then feeds its own predictions back as input to synthesize a complete 1,000-beat RR sequence. Noise injection (NOISE_STD=0.25) is applied at each generation step to partially recover beat-to-beat variability, which tends to collapse in pure autoregressive rollout.

**HRV validation metrics computed on real vs. synthetic sequences:**
- mean RR, SDNN, RMSSD, SD1, SD2

---

## Known Issues

- **Mean collapse** is a known failure mode in autoregressive generation — the model tends to produce sequences with suppressed short-term variability. RMSSD and SD1 in particular tend to undershoot. Noise injection partially recovers this but does not fully solve it.
- **SDNN and RMSSD alone are insufficient** for validating synthetic sequences. More rigorous validation requires entropy measures, DFA α1, and Wasserstein distance (consistent with Pimentel et al. methodology).
- **128 Hz quantization** introduces a ~7.8ms resolution limit on RR intervals — rounding to integer ms is correct and intentional.
- Free-tier **Colab GPU constraints** were encountered during training. The nested LOOCV is compute-heavy — recommend using an A100 runtime with Google Colab Pro (free for students) or running on Amarel (Rutgers HPC) for full 54-subject runs.

---

## What's Next

> **This section is the most important for the incoming team.**

1. **Obtain ALS HRV data** — Contact has been initiated with the corresponding author of Pimentel et al. (PMC7911551). This is the target dataset for fine-tuning and deviation analysis. If no response, alternative ALS HRV datasets should be explored.

2. **Fine-tune, don't retrain** — When ALS data arrives, fine-tune the existing CNN/LSTM rather than training from scratch. This preserves the healthy cardiac dynamics already learned from the 54-subject cohort.

3. **Autoregressive generation at scale** — Full cohort-level synthetic sequence generation across all 54 subjects has not been fully validated yet. This should be completed and HRV metrics compared systematically.

4. **Upgrade validation metrics** — Move beyond SDNN/RMSSD to include sample entropy, DFA α1, and Wasserstein distance to match Pimentel methodology and enable meaningful real-vs-synthetic comparison.

5. **Build the deviation profile** — The end goal is not binary classification but a **per-feature deviation fingerprint**: for each HRV metric, quantify how far ALS patients fall outside the healthy distribution. The deviation pattern across metrics simultaneously is the scientific signal.

---

## Dependencies

| Library | Version | Used In |
|---------|---------|---------|
| PyTorch | — | CNN model |
| TensorFlow / Keras | — | LSTM model |
| NumPy | — | Both |
| pandas | — | Both |
| scikit-learn | — | Cross-validation, metrics |
| matplotlib / seaborn | — | Evaluation plots |

**Platform:** Google Colab (GPU runtime recommended) + Google Drive for data storage

**To load the saved CNN model:**
```python
from model import RR_CNN
import torch

model = RR_CNN().to(device)
model.load_state_dict(torch.load('RR_CNN_Healthy_Baseline_Weights.pth'))
model.eval()
```

**To load cross-validation results:**
```python
import pickle

with open('cnn_nested_cv_results.pkl', 'rb') as f:
    nested_cv_results = pickle.load(f)
```

---

## Key References

- Task Force of ESC/NASPE (1996). Standards of measurement of HRV. *Circulation*. PMID: 8598068
- Umetani et al. (1998). 24-hour time domain HRV normative values. *JACC*. PMID: 9584889
- Bonnemeier et al. (2003). Circadian profile of HRV in healthy subjects. *J Cardiovasc Electrophysiol*. PMID: 12527537
- Pimentel et al. HRV in ALS. PMC7911551



