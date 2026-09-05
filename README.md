# FedGuard-DC

Privacy-preserving federated load forecasting and cyber-attack detection for AI data-center loads in transmission systems.

This repository holds the simulation datasets and notebooks for the FedGuard-DC study.

```
FedGuard-DC_v1/            first-generation model and notebooks
FedGuard-DC_v2/dataset/    current dataset  <- use this one
  clean/                   attack-free records
  attacked/                measurement-path FDIA records
  raw/                     1 kHz .mat records
  manifest.csv
  attack_log.csv
```

---

## v2 dataset

Six heterogeneous AI data centers embedded in the IEEE 39-bus New England system, simulated in MATLAB/Simulink R2023b (Simscape Electrical Specialized Power Systems, **phasor** mode, 60 Hz).

| DC | Bus | Rated (MW) | Workload |
|---|---|---|---|
| DC1 | 4 | 350 | training |
| DC2 | 23 | 200 | inference |
| DC3 | 18 | 150 | mixed |
| DC4 | 16 | 300 | training |
| DC5 | 20 | 450 | mixed |
| DC6 | 8 | 250 | inference |

Each site uses its own UPS thresholds, battery size, dc-link parameters, cooling time constants and reactive-power mode, so the six clients are non-IID by construction.

### How it was generated

Each data center is a WECC-style composite load model extended with IT, cooling and auxiliary branches, a synthetic AI workload generator, and an averaged double-conversion UPS (rectifier under dc-link voltage control with current limit, battery with SoC, diesel start, cooling motor stall and restart). The workload trace is synthetic — it reproduces the structure of AI facility power (training iteration cycles, checkpoint dips, inference burstiness, diurnal envelope) rather than replaying a measured trace.

Eleven 300 s scenarios were simulated: normal operation, two three-phase faults, a line trip, a load step, a combined fault + line trip, four control-path attacks, and one case with a genuine fault and an attack in the same run. Signals are logged at 1 kHz and exported at 100 Hz (30 001 samples per file).

Two attack classes are kept separate on purpose:

- **Control-path attacks** are injected inside the model (false UPS transfer, load-altering, cooling setpoint manipulation, voltage-sensor spoofing), so the grid actually responds to them. These are scenarios S07–S11.
- **Measurement-path attacks** (scaling, ramp/drift, replay, spike, stealthy bias) are applied afterwards to the exported telemetry only, since they corrupt what a site reports without changing the physics. Files in `attacked/` carry per-sample `label` and `attack_type` columns.

### Scenarios

| Name | Event |
|---|---|
| `S01_normal` | no disturbance |
| `S02_fault_100ms` | 3φ-G fault, bus 16, 60.00–60.10 s |
| `S03_fault_250ms` | 3φ-G fault, bus 4, 60.00–60.25 s |
| `S04_line_trip` | line trip 90–95 s |
| `S05_load_step` | +200 MW at bus 20, t = 120 s |
| `S06_multi_event` | fault at bus 16 + line trip 150–300 s |
| `S07_atk_false_ups` | false UPS transfer, DC1 and DC4, 140–170 s |
| `S08_atk_load_alter` | load-altering, DC2 and DC5, 100–160 s |
| `S09_atk_sensor_spoof` | voltage-sensor spoofing, DC3, 180–220 s |
| `S10_atk_cooling` | cooling setpoint manipulation, DC6, 200–250 s |
| `S11_fault_plus_atk` | fault at bus 16 + load-altering on DC6, 62–92 s |

`S11` is the discrimination case: a detector should flag DC6 without flagging the fault.

### File naming

```
clean/S<NN>_<event>_DC<k>.csv               k = 1..6
attacked/S<NN>_<event>_DC<k>_A<type>.csv    type = 1..5
raw/S<NN>_<event>.mat                       1 kHz, fields DC1..DC6
```

### Columns

| Column | Unit | Column | Unit |
|---|---|---|---|
| `t` | s | `ups_state` | 0 grid, 1 battery, 2 diesel, 3 IT tripped |
| `Va_pu` `Vb_pu` `Vc_pu` | pu | `soc` | 0–1 |
| `P_meas_MW` | MW | `vdc_pu` | pu |
| `P_total_MW` | MW | `P_batt_MW` | MW |
| `Q_total_Mvar` | Mvar | `cool_stall` | 0/1 |
| `P_it_MW` `P_cool_MW` `P_aux_MW` | MW | `P_ai_pu` | pu |
| | | `P_it_served_MW` | MW |

Attacked files add `label` (0/1) and `attack_type` (0–5).

### Notes for users

- Drop `t < 2 s`. That window is model soft-start, not physics.
- `ups_state`, `soc`, `vdc_pu`, `P_batt_MW` and `P_it_served_MW` are left uncorrupted in the attacked files so they can serve as ground truth. Drop them from the feature set when evaluating measurement-path detectors, or they leak the label.
- The 100 Hz CSVs are decimated by a plain 10:1 stride with no anti-alias filter. Use `raw/*.mat` if you need content above 50 Hz.
- The AI workload is synthetic, not a measured facility trace.
- `S02`, `S06` and `S11` contain a dc-link overvoltage excursion on DC1 and DC6 caused by rectifier current-limit release at cooling-motor restart. Exclude those six files from any experiment that uses `vdc_pu`.

---

## v1

`FedGuard-DC_v1/` is the earlier model and notebook set, kept for reference. Superseded by v2.

---

## Citation

If you use this dataset or find this repository useful, please cite:

> FedGuard-DC: Privacy-Preserving Federated Load Forecasting and Cyber-Attack Detection for Data-Center Loads in Transmission Systems. arXiv:2608.19155. https://arxiv.org/abs/2608.19155

```bibtex
@article{fedguarddc2026,
  title   = {FedGuard-DC: Privacy-Preserving Federated Load Forecasting and
             Cyber-Attack Detection for Data-Center Loads in Transmission Systems},
  author  = {},
  journal = {arXiv preprint arXiv:2608.19155},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.19155}
}
```
