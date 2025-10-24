# Tabular MLP Baselines & Training (Breast Cancer **default**, Dry Bean optional)

This repository trains and evaluates a small **PyTorch MLP** on classic tabular datasets with clean baselines, validation, and reproducible artifacts.

- **Default dataset:** **Breast Cancer Wisconsin (Diagnostic)** — binary (benign vs malignant), 30 numeric features.
- **Optional dataset:** Dry Bean — 7‑class, 16 numeric features (kept for continuity).

Colab/Kaggle are recommended platforms (GPU optional; CPU is fine for these models). The course brief explicitly recommends Colab/Kaggle and warns about free‑tier time limits (see brief, p. 1). :contentReference[oaicite:0]{index=0}

---

## Quickstart – Breast Cancer (local)

```bash
python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt

# (Optional) Provide your own CSV:
#   Place wdbc.csv in .cache/breast_cancer/ OR set DATA_DIR=/path/to/folder with wdbc.csv inside.
#   Expected columns: id, diagnosis (B/M), 30 numeric features like radius_mean,...,fractal_dimension_worst.
# If CSV is not found, the code falls back to sklearn’s built-in dataset and caches it as wdbc_from_sklearn.csv.

# Baselines (logreg, kNN, GaussianNB)
bash scripts/run_baselines.sh

# Train MLP (3x256, dropout 0.2, Adam 1e-3, wd 1e-4, patience 8, class weights on)
bash scripts/run_train.sh

# Evaluate on TEST only
LATEST_RUN=$(ls -dt artifacts/breast_cancer/*/ | head -n1 | sed 's:/*$::')
python eval.py --dataset breast_cancer --checkpoint "${LATEST_RUN}/best.pt" --artifacts-dir "${LATEST_RUN}"
```
