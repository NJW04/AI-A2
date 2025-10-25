# README.md

## Wine Quality (White) — 5-Class Classification (PyTorch + Scikit-Learn)

This repo provides a **complete, CPU-friendly** pipeline for the _Wine Quality (white)_ dataset using a fixed **5-class binning**:

- **very_low** := quality ∈ {3,4}
- **low** := quality = 5
- **medium** := quality = 6
- **high** := quality = 7
- **very_high** := quality ∈ {8,9}

### Why Logistic Regression as the baseline?

A multinomial logistic regression baseline is **convex, interpretable, and fast on CPU**. It’s a strong reference for tabular data and provides a clean comparison target for the MLP.

---

## Data placement (no auto-download)

Place the CSV file at:
