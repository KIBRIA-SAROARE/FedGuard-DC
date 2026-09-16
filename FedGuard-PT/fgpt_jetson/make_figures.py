#!/usr/bin/env python3
"""IEEE-width figures (PDF + 600 dpi PNG) and results/figures/SUMMARY.md from the board's results."""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.grid": True, "grid.alpha": 0.3,
                     "lines.linewidth": 0.8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
W2, W1 = 7.16, 3.5
TESTS = ["delivered_A1", "no_attack", "T1", "T2", "T3a", "T3"]
TITLE = {"delivered_A1": "repository file A1 (scaling FDIA) + fault", "no_attack": "fault only (no attack)",
         "T1": "T1 grid-side FDIA", "T2": "T2 facility-side FDIA", "T3a": "T3a cross-domain", "T3": "T3 cross-domain + V"}


def rcsv(path):
    with open(path) as f:
        r = csv.reader(f); h = next(r); rows = list(r)
    out = {}
    for j, c in enumerate(h):
        col = [x[j] for x in rows]
        try:
            out[c] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[c] = np.array(col)
    return out


def save(fig, od, name):
    fig.savefig(os.path.join(od, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(od, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def spans(ax, t, lab):
    on = np.flatnonzero(np.diff(np.r_[0, (lab == 1).astype(int), 0]))
    for a, b in zip(on[::2], on[1::2]):
        ax.axvspan(t[a], t[b - 1], color="#d62728", alpha=0.12, lw=0)
    ax.axvline(60.0, color="k", ls="--", lw=0.6)


def fig_test(res, od, name, thr, xlim=None, suffix=""):
    S = np.load(os.path.join(res, name, "signals.npz"))
    Wn = rcsv(os.path.join(res, name, "windows.csv"))
    t, lab = S["t"], S["label"]
    fig, ax = plt.subplots(5, 1, figsize=(W2, 6.0), sharex=True)
    ax[0].plot(t, S["Va"], c="k"); ax[0].set_ylabel("$V_a$ (pu)")
    ax[1].plot(t, S["P_total"], c="#1f77b4", label="$P_{total}$ reported")
    ax[1].plot(t, S["P_fac"], c="#ff7f0e", ls=":", label="$P_{it}+P_{cool}+P_{aux}$")
    ax[1].set_ylabel("P (MW)"); ax[1].legend(loc="upper right", ncol=2, frameon=False)
    ax[2].plot(t, S["P_it"], label="$P_{it}$"); ax[2].plot(t, S["P_cool"], label="$P_{cool}$")
    ax[2].set_ylabel("P (MW)"); ax[2].legend(loc="upper right", ncol=2, frameon=False)
    ax[3].step(Wn["t_decision"], Wn["score"], where="post", c="#2ca02c", label="window score")
    ax[3].axhline(thr, c="r", ls="--", lw=0.7, label="threshold")
    ax[3].set_ylabel("score"); ax[3].legend(loc="upper right", ncol=2, frameon=False)
    ax[3].set_yscale("symlog", linthresh=5)
    ax[4].fill_between(t, 0, (lab == 1).astype(float), step="post", color="#d62728", alpha=0.35, label="attack (sample label)")
    ax[4].fill_between(Wn["t_decision"], 1.15, 1.15 + Wn["alarm"], step="post", color="#1f77b4", alpha=0.6, label="alarm")
    ax[4].plot(t, 0.5 * S["corr"] - 0.7, c="#9467bd", lw=0.7, label="CSEC corroboration (x0.5)")
    ax[4].set_yticks([-0.45, 0.5, 1.65]); ax[4].set_yticklabels(["CSEC", "label", "alarm"])
    ax[4].set_xlabel("Time (s)")
    for a in ax:
        spans(a, t, lab)
    if xlim:
        ax[0].set_xlim(*xlim)
    fig.align_ylabels(ax)
    save(fig, od, f"Fig_{name}{suffix}")


def fig_latency(res, od):
    lw = np.loadtxt(os.path.join(res, "delivered_A1", "latency_window_ms.csv"), skiprows=1)
    ls = np.loadtxt(os.path.join(res, "delivered_A1", "latency_sample_ms.csv"), skiprows=1)
    bpath = os.path.join(res, "bench_latency.csv")
    n = 3 if os.path.exists(bpath) else 2
    fig, ax = plt.subplots(1, n, figsize=(W2, 2.0))
    ax[0].hist(ls, bins=80, color="#2ca02c"); ax[0].set_yscale("log")
    ax[0].set_xlabel("features per sample (ms)"); ax[0].set_ylabel("count")
    ax[1].hist(lw, bins=60, color="#ff7f0e"); ax[1].set_yscale("log")
    ax[1].set_xlabel("TCN + fusion per window (ms)")
    if n == 3:
        b = rcsv(bpath)
        for be, c in (("numpy", "#1f77b4"), ("torch", "#d62728")):
            m = b["backend"] == be
            if m.any():
                ax[2].plot(b["threads"][m], b["mean"][m], "o-", c=c, label=be)
        ax[2].set_xlabel("threads"); ax[2].set_ylabel("TCN ms / window (batch 1)"); ax[2].legend(frameon=False)
    fig.tight_layout()
    save(fig, od, "Fig_latency")


def fig_resources(res, od):
    parts, off = [], 0.0
    for tag in ("idle", "train", "monitor"):
        p = os.path.join(res, f"resources_{tag}.csv")
        if os.path.exists(p):
            r = rcsv(p)
            parts.append((tag, r["t_s"] + off, r))
            off = parts[-1][1][-1] + 2
    if not parts:
        return
    keys = [("proc_cpu_pct", "process CPU (%)"), ("proc_rss_MiB", "RSS (MiB)"), ("vdd_in_mW", "VDD_IN (mW)"),
            ("gpu_load_pct", "GPU (%)"), ("tj_temp_C", "$T_j$ (°C)")]
    keys = [k for k in keys if any(np.isfinite(r[k[0]]).any() for _, _, r in parts)]
    fig, ax = plt.subplots(len(keys), 1, figsize=(W2, 1.0 * len(keys) + 0.5), sharex=True, squeeze=False)
    ax = ax[:, 0]
    for tag, tt, r in parts:
        for i, (k, lab) in enumerate(keys):
            ax[i].plot(tt, r[k], c="k"); ax[i].set_ylabel(lab)
        for a in ax:
            a.axvline(tt[0], c="#999", lw=0.5)
        ax[0].text(tt[0], ax[0].get_ylim()[1], tag, fontsize=6, va="top")
    ax[-1].set_xlabel("elapsed time (s; idle, training and monitoring logs concatenated)")
    fig.align_ylabels(ax)
    save(fig, od, "Fig_resources")


def fig_training(res, od):
    L = json.load(open(os.path.join(res, "train_log.json")))
    rl = L["train_stats"]["round_log"]
    if not rl:
        return
    r = [x["round"] for x in rl]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 1.9))
    ax[0].plot(r, L["train_stats"]["convergence"], "o-", ms=2); ax[0].set_xlabel("round"); ax[0].set_ylabel("mean last-batch loss")
    ax[1].bar(r, [x["DC1_local_s"] for x in rl]); ax[1].set_xlabel("round"); ax[1].set_ylabel("DC1 local training (s)")
    ax[2].bar(["uplink\n(int8)", "downlink\n(fp32)"], [L["uplink_bytes_per_client_per_round"], L["downlink_bytes_per_client_per_round"]],
              color=["#2ca02c", "#7f7f7f"])
    ax[2].set_ylabel("bytes / site / round")
    fig.tight_layout()
    save(fig, od, "Fig_training_comm")


def stat(path, key, prefix=None):
    if not os.path.exists(path):
        return None
    r = rcsv(path)
    v = r[key]
    if prefix:
        v = v[np.char.startswith(r["phase"].astype(str), prefix)]
    v = v[np.isfinite(v)]
    return None if not len(v) else (float(v.mean()), float(v.max()))


def summary(res, od, tests):
    L = json.load(open(os.path.join(res, "train_log.json")))
    sysi = L["system"]
    f = lambda x, u: "not measured" if x is None else f"mean {x[0]:.1f} {u}, max {x[1]:.1f} {u}"  # noqa: E731
    ts = L["train_stats"]
    out = ["# FedGuard-PT on the edge: measured summary", "",
           f"Device: {sysi.get('device_model')} | {sysi.get('os')} | power mode: {sysi.get('nvpmodel')!r} | "
           f"NumPy {sysi.get('numpy')} | pandas/scipy/sklearn {sysi.get('pandas_scipy_sklearn')}", "",
           "## Model and communication (FedGuard-PT, LOSO fold, six sites emulated)", "", "| quantity | value |", "|---|---|",
           f"| parameters total / shared | {L['params_total']} / {L['params_shared']} |",
           f"| model file (npz, one site) | {L['npz_bytes_DC1']} B |",
           f"| uplink per site per round (int8 + scales) | {L['uplink_bytes_per_client_per_round']:.0f} B |",
           f"| downlink per site per round (fp32 shared) | {L['downlink_bytes_per_client_per_round']:.0f} B |",
           f"| all sites, per round | {L['KiB_per_round_all_clients']:.2f} KiB |",
           f"| total to {ts['rounds']} rounds | {L['MiB_to_convergence']:.3f} MiB |",
           f"| CSEC descriptor | {L['descriptor_bytes']} B per event |",
           f"| DC1 local training per round | {L['DC1_local_train_s_per_round']:.2f} s |",
           f"| federated training wall clock (6 sites in one process) | {ts['wall_s']:.1f} s |",
           f"| fusion + CSEC tuning | {L['timing']['fusion_fit_s']:.1f} s |",
           f"| peak RSS, training process | {L['peak_rss_MiB']:.1f} MiB |", "",
           "## Board resources", "", "| quantity | value |", "|---|---|",
           f"| idle system CPU | {f(stat(os.path.join(res, 'resources_idle.csv'), 'sys_cpu_pct'), '%')} |",
           f"| idle VDD_IN | {f(stat(os.path.join(res, 'resources_idle.csv'), 'vdd_in_mW'), 'mW')} |",
           f"| training VDD_IN | {f(stat(os.path.join(res, 'resources_train.csv'), 'vdd_in_mW', 'train'), 'mW')} |",
           f"| monitoring VDD_IN | {f(stat(os.path.join(res, 'resources_monitor.csv'), 'vdd_in_mW', 'monitor'), 'mW')} |",
           f"| monitoring process CPU (100 % = 1 core) | {f(stat(os.path.join(res, 'resources_monitor.csv'), 'proc_cpu_pct', 'monitor'), '%')} |",
           f"| monitoring RSS | {f(stat(os.path.join(res, 'resources_monitor.csv'), 'proc_rss_MiB', 'monitor'), 'MiB')} |",
           f"| monitoring GPU load | {f(stat(os.path.join(res, 'resources_monitor.csv'), 'gpu_load_pct', 'monitor'), '%')} |", "",
           "## Detection, DC1, held-out S02_fault_100ms (window level, notebook metrics)", "",
           "| test | ROC-AUC | F1 | precision | recall | FPR | pos/valid windows | post-fault alarms (59.9-70 s) | TCN+fusion ms/window mean / p99 | features ms/sample | stream vs batch |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in tests:
        m = json.load(open(os.path.join(res, n, "metrics.json")))
        lt = m["latency_ms"]
        pf = m["post_fault_59.9_70s"]
        out.append(f"| {n} | {m['roc_auc']:.4f} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['fpr']:.4f} | "
                   f"{m['windows_pos']}/{m['windows_valid']} | {pf['alarms']}/{pf['neg_windows']} | "
                   f"{lt['per_window_tcn_fusion_mean']:.3f} / {lt['per_window_p99']:.3f} | {lt['per_sample_features_mean']:.4f} | "
                   f"{m['streaming_vs_batch_max_abs_diff']:.1e} |")
    m = json.load(open(os.path.join(res, "delivered_A1", "metrics.json")))
    out += ["", "Attack windows in the repository file (decision delay = first alarm decision time - attack start):", ""]
    for w in m["attack_windows"]:
        out.append(f"- {w['t0']:.2f}-{w['t1']:.2f} s: detected={w['detected']}, delay={w['decision_delay_s']}")
    if "dominant_term_share_in_alarms" in m:
        out += ["", f"Dominant term during alarms (A1): {m['dominant_term_share_in_alarms']}"]
    open(os.path.join(od, "SUMMARY.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    od = os.path.join(a.results, "figures")
    os.makedirs(od, exist_ok=True)
    thr = json.load(open(os.path.join(a.results, "bundle", "fusion.json")))["threshold"]
    tests = [t for t in TESTS if os.path.exists(os.path.join(a.results, t, "metrics.json"))]
    for n in tests:
        fig_test(a.results, od, n, thr)
        fig_test(a.results, od, n, thr, xlim=(40, 120), suffix="_zoom_40_120s")
    fig_latency(a.results, od)
    fig_resources(a.results, od)
    fig_training(a.results, od)
    summary(a.results, od, tests)


if __name__ == "__main__":
    main()
