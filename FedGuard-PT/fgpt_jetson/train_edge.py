#!/usr/bin/env python3
"""Reproduce the notebook's FedGuard-PT training for the LOSO fold that holds out a test scenario
(default S02_fault_100ms), on the Jetson, and export the deployment bundle for one site.

Notebook run_fold() steps, for method "FedGuard-PT":
  1. per-client calibration on S01 (all six clients)                 section 6.1
  2. clean training windows from the attack-free scenarios minus test   section 9
  3. partial federation, 25 rounds, int8 + error feedback, trimmed mean section 11
     (six clients emulated in one process; each client's local time and uplink are measured)
  4. term statistics on S01, fusion on the S01 validation mixture (T1/T2/T3a/T3, seed 7)
     with CSEC gain (kappa, eta) and threshold by grid search          section 13
Outputs: results/bundle/  (DC<k>.npz, calib_DC<k>.json, fusion.json), results/train_log.json
"""
import argparse
import json
import os
import resource
import time

import numpy as np

from fgpt import common, nb, tcn_np
from fgpt.resmon import ResourceMonitor


def window_terms(p, ws, idx, bs=1024):
    W = nb.CFG["WIN"]
    st = ws.starts[idx]
    ab = np.abs(ws.X[:, nb.RESID_IDX]).astype(np.float64)
    c = np.vstack([np.zeros((1, ab.shape[1])), np.cumsum(ab, 0)])
    res = (c[st + W] - c[st]) / W
    er, ef = [], []
    for s in range(0, len(idx), bs):
        xb, yfc, yres, _ = ws.arrays(idx[s:s + bs])
        a, b = tcn_np.errors(p, xb, yfc, yres)
        er.append(a); ef.append(b)
    return np.column_stack([res, np.concatenate(er), np.concatenate(ef)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--results", default="results")
    ap.add_argument("--test-scenario", default="S02_fault_100ms")
    ap.add_argument("--mode", default="partial", choices=["partial", "local"],
                    help="partial = FedGuard-PT (notebook default); local = Local-only baseline")
    args = ap.parse_args()
    nb.DATASET_DIR = args.dataset
    out = os.path.join(args.results, "bundle")
    os.makedirs(out, exist_ok=True)
    mon = ResourceMonitor(os.path.join(args.results, "resources_train.csv"),
                          os.path.join(args.results, "tegrastats_train.log"), 0.5).start()
    T = {}
    clients = list(range(1, 7))
    train_scens = [s for s in nb.CFG["ATTACK_FREE"] if s != args.test_scenario]
    print(f"[fold] LOSO:{args.test_scenario}  train={train_scens}  mode={args.mode}")

    mon.set_phase("calibration")
    t0 = time.perf_counter()
    CAL = {k: nb.calibrate_client(k) for k in clients}
    T["calibration_6_clients_s"] = time.perf_counter() - t0
    for k in clients:
        common.save_json(nb.cal_to_json(CAL[k]), os.path.join(out, f"calib_DC{k}.json"))
    print(f"[calib] {T['calibration_6_clients_s']:.1f} s  s_k(DC1)={CAL[1]['s_k']:.2f} MW")

    mon.set_phase("features")
    t0 = time.perf_counter()
    train_sets = {k: [] for k in clients}
    for s in train_scens:
        for k in clients:
            X = nb.features(nb.telemetry(s, k), k, CAL[k])[0]
            train_sets[k].append(nb.WinSet(X, nb.control_labels(s, k), k, CAL[k]["reg_edges"]))
    T["train_features_s"] = time.perf_counter() - t0
    print(f"[features] {T['train_features_s']:.1f} s")

    mon.set_phase("federated_training")
    models, st = tcn_np.federated_train(train_sets, args.mode, nb.CFG["SEED"], on_phase=mon.set_phase)
    print(f"[train] wall {st['wall_s']:.1f} s, local time per client "
          f"{ {k: round(v, 1) for k, v in st['local_time_per_client_s'].items()} }")

    mon.set_phase("fusion_fit")
    t0 = time.perf_counter()
    cal_sets = {k: nb.WinSet(nb.features(nb.telemetry(nb.CFG["CALIB_SCENARIO"], k), k, CAL[k])[0],
                             nb.control_labels(nb.CFG["CALIB_SCENARIO"], k), k, CAL[k]["reg_edges"]) for k in clients}
    Tc = np.concatenate([window_terms(models[clients[0]], ws, np.flatnonzero(ws.y == 0)) for ws in cal_sets.values()])
    mu = np.median(Tc, 0)
    sd = 1.4826 * np.median(np.abs(Tc - mu), 0)
    sd = np.where(sd <= 0, np.std(Tc, 0) + 1e-9, sd)
    ms = (mu.astype(np.float32), sd.astype(np.float32))
    Zv, yv, cv, uv = [], [], [], []
    scen = nb.CFG["CALIB_SCENARIO"]
    for reg in nb.REGIMES:
        tels, labs = {}, {}
        for k in clients:
            tels[k], labs[k], _ = nb.make_attacked(scen, k, reg, nb.notebook_attack_seed(nb.CFG["ATK_SEED_VAL"], scen, k), CAL)
        corr, unc, _ = nb.csec_signals(tels, CAL)
        for k in clients:
            ws = nb.WinSet(nb.features(tels[k], k, CAL[k])[0], labs[k], k, CAL[k]["reg_edges"])
            idx = np.flatnonzero(ws.y >= 0)
            Zv.append(window_terms(models[k], ws, idx))
            yv.append((ws.y[idx] == 1).astype(int))
            cv.append(nb.csec_window(corr[k], ws.starts[idx]))
            uv.append(nb.csec_window(unc[k], ws.starts[idx]))
    Zv = nb.z_terms(np.concatenate(Zv), ms)
    yv, cv, uv = np.concatenate(yv), np.concatenate(cv), np.concatenate(uv)
    lr = nb.fit_fusion(Zv, yv)
    coef, b0 = lr.coef_.ravel(), float(lr.intercept_[0])
    kap, eta, f1v, thr = nb.tune_csec(coef, b0, Zv, yv, cv, uv)
    T["fusion_fit_s"] = time.perf_counter() - t0
    fusion = dict(coef=coef.tolist(), intercept=b0, ms_mu=ms[0].tolist(), ms_sd=ms[1].tolist(),
                  kappa=kap, eta=eta, threshold=thr, val_f1=f1v, val_windows=int(len(yv)),
                  val_pos=int(yv.sum()), terms=nb.TERMS, test_scenario=args.test_scenario, mode=args.mode)
    common.save_json(fusion, os.path.join(out, "fusion.json"))
    print(f"[fusion] {T['fusion_fit_s']:.1f} s  val windows={len(yv)}  F1={f1v:.4f}  "
          f"kappa={kap} eta={eta} thr={thr:.4f}")

    for k in clients:
        np.savez(os.path.join(out, f"DC{k}.npz"), **models[k])
    msg = st.pop("uplink_message_example", None)
    if msg:
        with open(os.path.join(out, "uplink_message_example.bin"), "wb") as f:
            f.write(msg)
    mon.stop()
    p1 = models[1]
    skeys = tcn_np.shared_keys(p1)
    rounds = max(st["rounds"], 1)
    log = dict(
        system=common.system_info(), timing=T, train_stats=st,
        params_total=tcn_np.n_params(p1), params_shared=int(sum(p1[n].size for n in skeys)),
        npz_bytes_DC1=os.path.getsize(os.path.join(out, "DC1.npz")),
        uplink_bytes_per_client_per_round=(st["bytes_up"] / rounds / len(clients)) if st["bytes_up"] else 0,
        uplink_serialised_bytes=len(msg) if msg else 0,
        downlink_bytes_per_client_per_round=(st["bytes_down"] / rounds / len(clients)) if st["bytes_down"] else 0,
        KiB_per_round_all_clients=(st["bytes_up"] + st["bytes_down"]) / rounds / 1024,
        MiB_to_convergence=(st["bytes_up"] + st["bytes_down"]) / 2**20,
        DC1_local_train_s_per_round=st["local_time_per_client_s"][1] / rounds,
        peak_rss_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        descriptor_bytes=nb.DESCRIPTOR_BYTES)
    common.save_json(log, os.path.join(args.results, "train_log.json"))
    print(json.dumps({k: v for k, v in log.items() if k not in ("system", "train_stats", "timing")}, indent=1, default=float))


if __name__ == "__main__":
    main()
