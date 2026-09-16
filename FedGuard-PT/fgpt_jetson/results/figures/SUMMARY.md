# FedGuard-PT on the edge: measured summary

Device: NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super | "Ubuntu 24.04.4 LTS" | power mode: 'NV Power Mode: 25W\n1' | NumPy 2.5.3 | pandas/scipy/sklearn 3.0.5 1.18.1 1.9.1

## Model and communication (FedGuard-PT, LOSO fold, six sites emulated)

| quantity | value |
|---|---|
| parameters total / shared | 3597 / 3376 |
| model file (npz, one site) | 24526 B |
| uplink per site per round (int8 + scales) | 3504 B |
| downlink per site per round (fp32 shared) | 13504 B |
| all sites, per round | 99.66 KiB |
| total to 25 rounds | 2.433 MiB |
| CSEC descriptor | 15 B per event |
| DC1 local training per round | 2.85 s |
| federated training wall clock (6 sites in one process) | 428.7 s |
| fusion + CSEC tuning | 75.8 s |
| peak RSS, training process | 348.1 MiB |

## Board resources

| quantity | value |
|---|---|
| idle system CPU | mean 2.6 %, max 7.4 % |
| idle VDD_IN | mean 3641.5 mW, max 3979.0 mW |
| training VDD_IN | mean 4702.1 mW, max 6080.0 mW |
| monitoring VDD_IN | mean 4288.8 mW, max 4864.0 mW |
| monitoring process CPU (100 % = 1 core) | mean 98.6 %, max 101.5 % |
| monitoring RSS | mean 193.0 MiB, max 269.0 MiB |
| monitoring GPU load | mean 0.9 %, max 40.0 % |

## Detection, DC1, held-out S02_fault_100ms (window level, notebook metrics)

| test | ROC-AUC | F1 | precision | recall | FPR | pos/valid windows | post-fault alarms (59.9-70 s) | TCN+fusion ms/window mean / p99 | features ms/sample | stream vs batch |
|---|---|---|---|---|---|---|---|---|---|---|
| delivered_A1 | 0.9979 | 0.9784 | 0.9577 | 1.0000 | 0.0100 | 136/734 | 0/0 | 1.263 / 1.636 | 0.0576 | 2.5e-06 |
| no_attack | nan | 0.0000 | 0.0000 | 0.0000 | 0.0108 | 0/742 | 2/25 | 1.254 / 1.457 | 0.0577 | 2.5e-06 |
| T1 | 0.9569 | 0.9505 | 0.9412 | 0.9600 | 0.0094 | 100/737 | 1/20 | 1.257 / 1.598 | 0.0574 | 2.5e-06 |
| T2 | 0.9596 | 0.9505 | 0.9412 | 0.9600 | 0.0094 | 100/737 | 1/20 | 1.263 / 1.752 | 0.0577 | 2.5e-06 |
| T3a | 0.7562 | 0.5888 | 0.5526 | 0.6300 | 0.1047 | 100/587 | 14/20 | 1.254 / 1.660 | 0.0573 | 2.5e-06 |
| T3 | 0.8731 | 0.6210 | 0.5714 | 0.6800 | 0.1047 | 100/587 | 14/20 | 1.267 / 1.860 | 0.0577 | 2.5e-06 |

Attack windows in the repository file (decision delay = first alarm decision time - attack start):

- 82.73-112.87 s: detected=True, delay=0.14999999999999147
- 45.12-55.31 s: detected=True, delay=0.1600000000000037
- 54.62-69.81 s: detected=True, delay=0.2600000000000051

Dominant term during alarms (A1): {'|r_bal|': 0.0, '|r_aux|': 0.0, '|r_cool|': 0.0, '|r_ups|': 0.0, '|r_qp|': 0.0, '|r_pv|': 0.0, '|r_reg|': 0.0, 'e_rec': 1.0, 'e_fc': 0.0}
