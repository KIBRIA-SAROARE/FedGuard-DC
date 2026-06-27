# FedGuard-DC

> **A Federated Framework for Privacy-Preserving Load Forecasting and Cyber-Attack Detection in Data-Center-Rich Power Systems**


## Overview

Hyperscale data centers (DCs) are now among the single largest loads connected to transmission networks, yet their operators guard real-time power demand as commercially confidential. At the same time, the growing measurement infrastructure at these facilities widens the attack surface for **False Data Injection Attacks (FDIA)** that can corrupt state estimation and market settlement.

**FedGuard-DC** addresses both challenges simultaneously within a single federated learning framework:

- **Privacy-preserving load forecasting** — absolute MW demand is never shared; only normalised model updates leave each DC.
- **FDIA detection** — a calibrated, physics-informed anomaly score flags corrupted measurement windows at the local controller before they reach the global model.
- **Robustness** — differential-privacy noise and coordinate-wise trimmed-mean aggregation protect against a compromised local controller.

The framework is validated on measured **Positive Sequence Phasor Domain (PSPD)** data from four DC loads (150–350 MW) embedded in the **IEEE 39-bus New England system**.


## Repository Structure

```
FedGuard-DC/
├── Data/                                   # EMT measurement CSV files (DC1–DC4)
│   ├── DC1.csv                             # DC1 @ Bus 4  (~350 MW, ~120 k samples @ 1 kHz)
│   ├── DC2.csv                             # DC2 @ Bus 23 (~200 MW)
│   ├── DC3.csv                             # DC3 @ Bus 18 (~150 MW)
│   └── DC4.csv                             # DC4 @ Bus 16 (~300 MW)
├── Figures/                                # All paper-ready output figures (PDF/PNG)
├── Modified IEEE 39 Bus with Data Center Simulink/   # PSCAD/EMTDC simulation files
├── Copy_of_FedGuard_DC_NAPS2026.ipynb      # Main Colab notebook (self-contained)
├── copy_of_fedguard_dc_naps2026.py         # Plain Python equivalent of the notebook
└── README.md
```


## Framework at a Glance

Each data center operates as a local FL **client**. The pipeline has five stages:

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL CONTROLLER (DC k)                  │
│                                                                 │
│  Raw PSPD data  ──►  Local normalisation  ──►  Sliding window   │
│                              │                    x_k ∈ ℝ^{W×4} │
│                              ▼                                  │
│          ┌──────────── Shared Encoder ───────────┐             │
│          │         h = ReLU(W₁x + b₁)            │             │
│          │         z = ReLU(W₂h + b₂)            │             │
│          └──────┬─────────────────────┬───────────┘             │
│                 │                     │                         │
│         Forecast Head          Reconstruction Head              │
│         ŷ = wf·z + bf          x̂ = Wd·z + bd                  │
│                 │                     │                         │
│         Load forecast          Anomaly score s                  │
│         (H steps ahead)        → flag if s > τ*                │
└─────────────────────────────────────────────────────────────────┘
                     │  DP-protected Δθ_k only
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                   GLOBAL CONTROLLER (TSO)                       │
│                                                                 │
│  Robust trimmed-mean aggregation  ──►  Global model θ          │
│  Σ normalised forecasts           ──►  Operator load view       │
│  (no raw MW, no scale factors ever received)                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Technical Contributions

| # | Contribution | Description |
|---|---|---|
| 1 | **Dual-head local network** | Shared encoder feeds both a forecast head and a reconstruction head; a single model simultaneously predicts future load and monitors measurement integrity. |
| 2 | **Physics-informed anomaly score** | Fuses z-scored reconstruction error with forecast residual: `s = α·ẽ_rec + β·ẽ_fc`. Auto-tuned (α*, β*, τ*) via F₁-maximising grid search — no manual threshold setting. |
| 3 | **Privacy-by-design** | Each DC normalises by its own peak power `s_k = max P_k(t)` which **never leaves the local controller**. Optional differential-privacy (DP-SGD) adds a formal guarantee. |
| 4 | **Variability-weighted + robust aggregation** | Weights ∝ `n_k · √ℓ_k` so volatile DCs drive more of the global update. Coordinate-wise trimmed-mean neutralises a poisoned client. |


## Test System

The **IEEE 39-bus New England system** (10 machines, 345 kV, 34 lines) is used with four existing constant loads replaced by the DC models below.

| DC | Bus | Rating (MW) | Peak (MW) | CoV | IT % | Cooling % |
|----|-----|-------------|-----------|-----|------|-----------|
| DC1 | 4–5 | 350 | 320.6 | 0.42 | 34 | 40 |
| DC2 | 23–36 | 200 | 180.7 | 0.46 | 35 | 40 |
| DC3 | 18 | 150 | 136.6 | 0.47 | 35 | 40 |
| DC4 | 16–24 | 300 | 274.1 | 0.43 | 34 | 40 |

PSPD simulations were run in **Matlab/Simulink** at 1 kHz for 120 s per DC (~120,474 samples each). The simulation files are in `Modified IEEE 39 Bus with Data Center Simulink/`.


## Results Summary

### Load Forecasting (0.5 s lead time)

| DC | RMSE (pu) | MAE (pu) | MAPE (%) | Persistence RMSE (pu) |
|----|-----------|----------|----------|-----------------------|
| DC1 | 0.038 | 0.034 | 6.27 | 0.315 |
| DC2 | 0.029 | 0.025 | 4.70 | 0.334 |
| DC3 | 0.027 | 0.023 | 4.48 | 0.335 |
| DC4 | 0.023 | 0.018 | 3.17 | 0.322 |
| **Avg.** | **0.029** | **0.025** | **4.66** | **0.327** |

The federated model beats naive persistence by **8–14× in RMSE**.

### FDIA Detection (50% attack mixture, auto-tuned α*=3.97, β*=0.79, τ*=13.22)

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.979** |
| F₁ | **0.930** |
| Accuracy | 0.934 |
| Precision | **0.988** |
| Recall | 0.879 |

**Per-attack-type recall:**

| Scaling | Ramp/Drift | Replay | Spike/Noise | Stealthy |
|---------|------------|--------|-------------|----------|
| 100% | 93% | 88% | 100% | 60% |

### Privacy–Utility Trade-off (DP noise σ)

| σ | Global RMSE (pu) |
|---|-----------------|
| 0.000 | 0.029 |
| 0.005 | 0.122 |
| 0.010 | 0.195 |
| 0.020 | — |
| 0.050 | — |

### Robustness to a Poisoned Controller (×8 amplified update)

| Aggregation | Global RMSE (pu) |
|-------------|-----------------|
| Vanilla weighted avg | 0.042 |
| **Robust trimmed-mean** | **0.035** |
| Clean reference | 0.029 |


## Getting Started

### Requirements

The notebook is **fully self-contained** — no TensorFlow or PyTorch required.

```bash
pip install numpy scikit-learn matplotlib pandas
```

### Running on Google Colab (recommended)

1. Open `Copy_of_FedGuard_DC_NAPS2026.ipynb` in [Google Colab](https://colab.research.google.com).
2. Upload `DC1.csv`, `DC2.csv`, `DC3.csv`, `DC4.csv` from the `Data/` folder when prompted.
3. Run all cells sequentially (Steps 1–9). Total runtime: ~3–5 minutes on a free CPU.

> **For camera-ready figures:** set `DS = 10` and `ROUNDS = 150` (already the defaults). Lower `DS` uses more data; higher `ROUNDS` deepens convergence.

### Running Locally

```bash
git clone https://github.com/KIBRIA-SAROARE/FedGuard-DC.git
cd FedGuard-DC
pip install numpy scikit-learn matplotlib pandas

# Place the CSV files in the working directory, then:
python copy_of_fedguard_dc_naps2026.py
```

### Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `DS` | 10 | Downsample factor (1 ms → 10 ms) |
| `W` | 40 | Input window length (steps) |
| `H` | 25 | Forecast horizon → **0.5 s lead time** |
| `ROUNDS` | 150 | Federated communication rounds |
| `EPOCHS` | 2 | Local epochs per round |
| `HID` | 256 | Encoder hidden width |
| `LAT` | 128 | Latent dimension |
| `LAM` | 0.5 | Reconstruction loss weight λ |
| `LR` | 2e-3 | Adam learning rate |
| `ALPHA` | 3.97 | Anomaly score weight α* (auto-tuned) |
| `BETA` | 0.79 | Anomaly score weight β* (auto-tuned) |
| `DP_CLIP` | 5.0 | DP clipping norm C |
| `DP_SIGMA` | 0.0 | DP noise multiplier σ (0 = FL-only privacy) |
| `ATTACK_FRAC` | 0.5 | Fraction of test windows attacked |

### Output Figures

After a full run, the following PDFs are saved (ready for the paper):

| File | Content |
|------|---------|
| `fig1_load_profiles.pdf` | Normalised load profiles for all four DCs |
| `fig3_convergence.pdf` | FL convergence: proposed vs. vanilla FedAvg |
| `fig4_forecast.pdf` | Actual vs. forecast per DC (400-window segment) |
| `fig5_aggregate_forecast.pdf` | Operator-level aggregate forecast (Σ pu) |
| `fig_DC1_window_attacks.pdf` | Five FDIA scenarios on DC1 windows |
| `fig7_per_attack.pdf` | Per-attack-type detection recall |
| `fig8_privacy_robust.pdf` | Privacy–utility trade-off and poisoning robustness |


# Acknowledgement 
This readme file is created with the help of Generative AI
