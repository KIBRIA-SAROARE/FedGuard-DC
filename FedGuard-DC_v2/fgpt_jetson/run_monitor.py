#!/usr/bin/env python3
"""Streaming FedGuard-PT monitor for one site (default DC1) on held-out test records.

Tests (all from the held-out scenario, default S02_fault_100ms):
  delivered_A<n> : the repository file attacked/<scen>_DC<k>_A<n>.csv (inject_attacks.m output)
  no_attack      : the clean record (genuine fault only; every alarm is false)
  T1 T2 T3a T3   : the notebook's test attacks (seed 2026) regenerated for all six sites

The site replays its own telemetry sample by sample (100 Hz). CSEC inputs come from the other five
sites' telemetry for the same scenario, processed with the notebook's csec_signals() before the
replay (record-level, as in the notebook; see README).
Outputs per test: results/<test>/{windows.csv, signals.npz, latency_*.csv, metrics.json}
"""
import argparse
import csv
import os
import resource
import time

import numpy as np

from fgpt import common, nb, stream, tcn_np
from fgpt.resmon import ResourceMonitor

KEYS = ["Va_pu", "P_meas_MW", "P_total_MW", "Q_total_Mvar", "P_it_MW", "P_cool_MW", "P_aux_MW", "ups_state", "P_ai_pu"]


def load_bundle(res, k):
    b = os.path.join(res, "bundle")
    CAL = {j: nb.cal_from_json(common.load_config(os.path.join(b, f"calib_DC{j}.json"))) for j in range(1, 7)}
    p = dict(np.load(os.path.join(b, f"DC{k}.npz")))
    fusion = common.load_config(os.path.join(b, "fusion.json"))
    return CAL, p, fusion


def build_test(name, scen, k, CAL, dataset):
    """Returns site telemetry, per-sample label (0/1/-1), all-site telemetry for CSEC, attack log."""
    if name.startswith("delivered_A"):
        a = int(name.split("A")[1])
        rec, extra = nb.load_record(os.path.join(dataset, "attacked", f"{scen}_DC{k}_A{a}.csv"))
        tel = nb.telemetry_from_record(rec)
        lab = extra["label"].astype(np.int8)
        tels = {j: nb.telemetry(scen, j) for j in range(1, 7)}
        tels[k] = tel
        log = [dict(atype=int(r[0]), t0=r[1], t1=r[2]) for r in _attack_log(dataset, f"{scen}_DC{k}") if int(r[0]) == a]
        return tel, lab, tels, log
    if name == "no_attack":
        tels = {j: nb.telemetry(scen, j) for j in range(1, 7)}
        return tels[k], np.zeros(len(nb.tvec()), np.int8), tels, []
    tels, labs, logs = {}, {}, {}
    for j in range(1, 7):
        tels[j], labs[j], logs[j] = nb.make_attacked(scen, j, name, nb.notebook_attack_seed(nb.CFG["ATK_SEED_TEST"], scen, j), CAL)
    return tels[k], labs[k], tels, logs[k]


def _attack_log(dataset, key):
    out = []
    with open(os.path.join(dataset, "attack_log.csv")) as f:
        f.readline()
        for line in f:
            p = line.strip().split(",")
            if p[0] == key:
                out.append((int(p[1]), float(p[2]), float(p[3])))
    return out


def run_test(name, args, CAL, p, fusion, mon):
    k, scen = args.dc, args.scenario
    tel, lab, tels, alog = build_test(name, scen, k, CAL, args.dataset)
    T = nb.tvec()
    N = len(T)

    mon.set_phase(f"csec:{name}")
    t0 = time.perf_counter()
    corr, unc, clog = nb.csec_signals(tels, CAL)
    t_csec = time.perf_counter() - t0
    n_desc_site = int((clog.dc == k).sum()) if len(clog) else 0

    mon.set_phase(f"monitor:{name}")
    sf = stream.StreamingFeatures(k, CAL[k])
    sc = stream.WindowScorer(p, fusion)
    cols = [tel[c] for c in KEYS]
    lat_s = np.zeros(N)
    rows, lat_w = [], []
    t_run = time.perf_counter()
    for i in range(N):
        if args.realtime:
            dl = t_run + i * nb.CFG["DT"] - time.perf_counter()
            if dl > 0:
                time.sleep(dl)
        a = time.perf_counter_ns()
        x = sf.step(*(float(c[i]) for c in cols))
        b = time.perf_counter_ns()
        r = sc.push(x, corr[k], unc[k])
        e = time.perf_counter_ns()
        lat_s[i] = (b - a) / 1e6
        if r is not None:
            lat_w.append((e - b) / 1e6)
            r["t_end"] = float(T[r["start"] + nb.CFG["WIN"] - 1])
            r["t_decision"] = float(T[i])
            rows.append(r)
    wall = time.perf_counter() - t_run

    starts = np.array([r["start"] for r in rows])
    score = np.array([r["score"] for r in rows])
    alarm = np.array([r["alarm"] for r in rows])
    ywin = nb.window_labels(lab, starts)
    valid = ywin >= 0

    # offline notebook pipeline on the same record -> must reproduce the streaming scores
    X = nb.features(tel, k, CAL[k])[0]
    ws = nb.WinSet(X, lab, k, CAL[k]["reg_edges"])
    from train_edge import window_terms
    Zb = nb.z_terms(window_terms(p, ws, np.arange(len(ws.starts))), (np.asarray(fusion["ms_mu"], np.float32), np.asarray(fusion["ms_sd"], np.float32)))
    sb = nb.fused_score(fusion["coef"], fusion["intercept"], Zb, nb.csec_window(corr[k], ws.starts),
                        nb.csec_window(unc[k], ws.starts), fusion["kappa"], fusion["eta"])
    diff = float(np.abs(sb - score).max()) if len(sb) == len(score) else float("nan")

    y = (ywin[valid] == 1).astype(int)
    thr = fusion["threshold"]
    met = dict(test=name, dc=k, scenario=scen, samples=N, windows=len(rows), windows_valid=int(valid.sum()),
               windows_pos=int(y.sum()), threshold=thr, streaming_vs_batch_max_abs_diff=diff)
    met.update(nb.metrics_from(score[valid], y, thr))
    tw = np.array([r["t_end"] for r in rows])
    met["delay_s_notebook_definition"] = nb.detection_delay(score[valid], y, thr, tw[valid])
    wd = []
    for ev in alog:
        m = alarm & (tw >= ev["t0"]) & (np.array([r["t_decision"] for r in rows]) <= ev["t1"] + nb.CFG["WIN"] * nb.CFG["DT"])
        first = np.flatnonzero(m)
        wd.append(dict(ev, detected=bool(len(first)),
                       decision_delay_s=float(rows[first[0]]["t_decision"] - ev["t0"]) if len(first) else None))
    met["attack_windows"] = wd
    fw = valid & (tw >= 59.9) & (tw <= 70.0) & (ywin == 0)
    met["post_fault_59.9_70s"] = dict(neg_windows=int(fw.sum()), alarms=int(alarm[fw].sum()))
    if alarm.any():
        d = np.array([r["dominant"] for r in rows])[alarm]
        met["dominant_term_share_in_alarms"] = {t: float((d == t).mean()) for t in nb.TERMS}
        dm = np.array([r["domain"] for r in rows])[alarm]
        met["domain_share_in_alarms"] = {t: float((dm == t).mean()) for t in ("single-domain", "cross-domain")}
    lw = np.asarray(lat_w)
    met["latency_ms"] = dict(
        per_sample_features_mean=float(lat_s.mean()), per_sample_features_p99=float(np.percentile(lat_s, 99)),
        per_window_tcn_fusion_mean=float(lw.mean()), per_window_p50=float(np.percentile(lw, 50)),
        per_window_p99=float(np.percentile(lw, 99)), per_window_max=float(lw.max()),
        decision_period_ms=nb.CFG["HOP"] * nb.CFG["DT"] * 1e3,
        cpu_ms_per_second_of_telemetry=float(lat_s.mean() * 100 + lw.mean() * 100 / nb.CFG["HOP"]))
    met["csec"] = dict(record_level_compute_s_all_sites=t_csec, descriptors_this_site=n_desc_site,
                       descriptor_bytes=nb.DESCRIPTOR_BYTES, uplink_bytes_this_site=n_desc_site * nb.DESCRIPTOR_BYTES)
    met["wall_s"], met["realtime_paced"] = wall, args.realtime
    met["peak_rss_MiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    od = os.path.join(args.results, name)
    os.makedirs(od, exist_ok=True)
    common.save_json(met, os.path.join(od, "metrics.json"))
    with open(os.path.join(od, "windows.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_start", "t_end", "t_decision", "label", "score", "alarm", "dominant", "domain", "csec_corr", "csec_unc"] + nb.TERMS)
        for r, yl in zip(rows, ywin):
            w.writerow([f"{T[r['start']]:.2f}", f"{r['t_end']:.2f}", f"{r['t_decision']:.2f}", int(yl), f"{r['score']:.5f}",
                        int(r["alarm"]), r["dominant"], r["domain"], f"{r['csec_corr']:.4f}", f"{r['csec_unc']:.4f}"]
                       + [f"{v:.4g}" for v in r["terms"]])
    np.savez_compressed(os.path.join(od, "signals.npz"), t=T, label=lab, Va=tel["Va_pu"], P_total=tel["P_total_MW"],
                        P_fac=tel["P_it_MW"] + tel["P_cool_MW"] + tel["P_aux_MW"], P_it=tel["P_it_MW"],
                        P_cool=tel["P_cool_MW"], Q=tel["Q_total_Mvar"], corr=corr[k], unc=unc[k], X=X)
    np.savetxt(os.path.join(od, "latency_window_ms.csv"), lw, fmt="%.4f", header="tcn_plus_fusion_ms", comments="")
    np.savetxt(os.path.join(od, "latency_sample_ms.csv"), lat_s, fmt="%.4f", header="features_ms", comments="")
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--results", default="results")
    ap.add_argument("--dc", type=int, default=1)
    ap.add_argument("--scenario", default="S02_fault_100ms")
    ap.add_argument("--tests", default="delivered_A1,no_attack,T1,T2,T3a,T3")
    ap.add_argument("--realtime", action="store_true")
    args = ap.parse_args()
    nb.DATASET_DIR = args.dataset
    CAL, p, fusion = load_bundle(args.results, args.dc)
    assert fusion["test_scenario"] == args.scenario, "bundle was trained with a different held-out scenario"
    mon = ResourceMonitor(os.path.join(args.results, "resources_monitor.csv"),
                          os.path.join(args.results, "tegrastats_monitor.log"), 0.5).start()
    common.save_json(common.system_info(), os.path.join(args.results, "system_info.json"))
    for name in args.tests.split(","):
        print(f"[{name}] DC{args.dc} {args.scenario} ...", flush=True)
        m = run_test(name, args, CAL, p, fusion, mon)
        L = m["latency_ms"]
        print(f"  AUC={m['roc_auc']:.4f} F1={m['f1']:.4f} P={m['precision']:.4f} R={m['recall']:.4f} "
              f"FPR={m['fpr']:.4f}  windows pos/valid={m['windows_pos']}/{m['windows_valid']}  "
              f"post-fault alarms={m['post_fault_59.9_70s']}")
        print(f"  latency: features {L['per_sample_features_mean']:.4f} ms/sample, TCN+fusion "
              f"{L['per_window_tcn_fusion_mean']:.3f} ms/window (p99 {L['per_window_p99']:.3f}); "
              f"CPU {L['cpu_ms_per_second_of_telemetry']:.2f} ms per s; stream-vs-batch diff {m['streaming_vs_batch_max_abs_diff']:.2e}")
    mon.stop()


if __name__ == "__main__":
    main()
