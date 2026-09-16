# FedGuard-PT on Jetson Orin Nano: single-site monitor (DC1), ported from the Phase-2 notebook

This package replaces the earlier re-implementation. The model, residuals, attack generators, CSEC,
fusion and metrics are ported from `FedGuard_PT_Phase2_Final.ipynb` (functions keep their notebook names).

- Model: the notebook's depthwise-separable causal TCN, 3,597 parameters (3,376 shared).
- Code: pure NumPy on the board (Ubuntu 24.04 apt packages). No PyTorch or CUDA needed.

## 1. Setup (once)

```bash
cd ~/fgpt_jetson
chmod +x *.sh
./setup_jetson.sh      # numpy, pandas, scipy, scikit-learn, matplotlib, psutil, tk
./get_data.sh 1        # 66 clean CSVs + S02_fault_100ms_DC1_A*.csv + attack_log.csv (sparse git)
sudo nvpmodel -q       # record the power mode; keep it fixed for every reported run
```

A private repo will ask for a GitHub token. You can also copy `FedGuard-DC_v2/dataset/` by USB and run
`ln -sfn /path/to/dataset data/dataset`.

## 2. Run

```bash
./run_all.sh                 # idle baseline -> training -> monitor -> latency sweep -> figures
python3 live_view.py         # live display: DC1, S02 fault + repository attack A1
python3 live_view.py --test T3 --speed 5
```

`./run_all.sh --realtime` paces the monitor replay at wall-clock 100 Hz (5 min per test).

| step | what runs | what is measured |
|---|---|---|
| `measure_idle.py` | nothing (30 s) | idle CPU, RAM, VDD_IN, temperature |
| `train_edge.py` | notebook `run_fold` for **FedGuard-PT**, LOSO fold with **S02 held out**. Calibration on S01. Six sites trained on S01/S03/S04/S05/S06 in one process: 25 rounds, int8 + error feedback, DP clip, trimmed mean. Fusion + CSEC (κ, η) + threshold on the S01 validation mixture (seed 7) | per-site local training time per round, uplink/downlink bytes, parameters, model size, peak RSS |
| `run_monitor.py` | DC1 replays its telemetry sample by sample. Every 40 samples a 1.28 s window is scored | window ROC-AUC/F1/precision/recall/FPR, detection delay, post-fault alarms, feature latency per sample, TCN+fusion latency per window, streaming = batch check |
| `bench_latency.py` | batch-1 forward, NumPy at 1/2/4/6 threads (and PyTorch if installed) | per-window latency |
| `make_figures.py` | waveforms per test, latency, resources, training/communication | `results/figures/*.pdf/png`, `SUMMARY.md` |

Tests in `run_monitor.py` (all DC1, held-out `S02_fault_100ms`):

- `delivered_A1`: the repository file `S02_fault_100ms_DC1_A1.csv` (your attacked dataset)
- `no_attack`: the clean fault record (every alarm is false)
- `T1`, `T2`, `T3a`, `T3`: the notebook's test attacks, regenerated with its seeds (2026 + 1000k + 7·scenario index)

## 3. What matches the notebook, and what does not

| item | status |
|---|---|
| residuals, calibration, T1/T2/T3a/T3 generators, CSEC, fusion, metrics | same code (`fgpt/nb.py`) |
| TCN forward / backward / Adam | NumPy port. Checked against PyTorch with identical weights: forward 1.7e-16, gradients 1.5e-8 (float32 cast), 3 Adam steps 5e-10 |
| parameters, bytes | 3,597 / 3,376 shared. 3,504 B uplink and 13,504 B downlink per site per round, 99.66 KiB/round, 2.433 MiB over 25 rounds, 15 B CSEC descriptor. These match the paper's 99.7 KiB and 2.43 MiB |
| trained weights | not in the notebook, so they are **retrained on the board**. PyTorch initialisation and shuffling RNG streams differ from Colab, so detection numbers are not bit-identical to the paper's LOSO:S02 fold |
| federation | six sites emulated in one process on the board. Per-site time and bytes are measured, but no network link is exercised |
| CSEC | notebook's record-level `csec_signals()` over the six sites' replayed telemetry, computed before the replay. It is not causal (event depth and peers ±2 s), so a live deployment adds up to event length + 2 s delay |
| fusion data | as in the notebook, fusion and threshold use all six sites' S01 validation windows |
| paper latency 2.91 ms | PyTorch on a Colab CPU. `bench_latency.py` reports both backends on the board |

## 4. Outputs

```
results/train_log.json  system_info.json  run_log.txt
results/bundle/DC<k>.npz  calib_DC<k>.json  fusion.json  uplink_message_example.bin
results/<test>/metrics.json  windows.csv  signals.npz  latency_sample_ms.csv  latency_window_ms.csv
results/resources_{idle,train,monitor}.csv  tegrastats_*.log  bench_latency.csv
results/figures/SUMMARY.md  Fig_<test>.pdf/png  Fig_<test>_zoom_40_120s  Fig_latency  Fig_resources  Fig_training_comm
```
