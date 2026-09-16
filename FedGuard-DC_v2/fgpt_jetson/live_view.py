#!/usr/bin/env python3
"""Live on-screen FedGuard-PT site monitor (display only; take timings from run_monitor.py).

Replays a held-out record at 100 Hz through the deployed DC model and shows the last 30 s:
voltage, reported vs facility power, window score vs threshold, alarms and the named term.
  python3 live_view.py                      # repository file S02_fault_100ms_DC1_A1
  python3 live_view.py --test T3 --speed 5  # notebook T3 attack, 5x real time
"""
import argparse
import collections
import os
import time

import matplotlib
matplotlib.use(os.environ.get("MPLBACKEND", "TkAgg"))
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fgpt import nb, stream  # noqa: E402
from run_monitor import KEYS, build_test, load_bundle  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="data/dataset")
ap.add_argument("--results", default="results")
ap.add_argument("--dc", type=int, default=1)
ap.add_argument("--scenario", default="S02_fault_100ms")
ap.add_argument("--test", default="delivered_A1")
ap.add_argument("--speed", type=float, default=1.0)
ap.add_argument("--span", type=float, default=30.0)
a = ap.parse_args()
nb.DATASET_DIR = a.dataset
CAL, p, fusion = load_bundle(a.results, a.dc)
tel, lab, tels, _ = build_test(a.test, a.scenario, a.dc, CAL, a.dataset)
corr, unc, _ = nb.csec_signals(tels, CAL)
T = nb.tvec()
N = len(T)
sf, sc = stream.StreamingFeatures(a.dc, CAL[a.dc]), stream.WindowScorer(p, fusion)
M = int(a.span / nb.CFG["DT"])
tt, vv, pp, ff, ll = (collections.deque(maxlen=M) for _ in range(5))
wt, ws, wa = (collections.deque(maxlen=int(M / nb.CFG["HOP"]) + 2) for _ in range(3))
lat = collections.deque(maxlen=50)
last = None

plt.ion()
fig, ax = plt.subplots(4, 1, figsize=(11, 7), sharex=True)
fig.canvas.manager.set_window_title(f"FedGuard-PT DC{a.dc} monitor - {a.scenario} - {a.test}")
lv, = ax[0].plot([], [], "k"); ax[0].set_ylabel("Va (pu)")
lp, = ax[1].plot([], [], label="P_total reported"); lf, = ax[1].plot([], [], ":", label="P_it+P_cool+P_aux")
ax[1].set_ylabel("MW"); ax[1].legend(loc="upper left")
ls, = ax[2].step([], [], "g", where="post"); ax[2].axhline(fusion["threshold"], c="r", ls="--"); ax[2].set_ylabel("window score")
la, = ax[3].step([], [], "b", where="post", label="alarm")
lb, = ax[3].step([], [], "r", where="post", alpha=0.5, label="attack label")
ax[3].set_ylim(-0.1, 1.2); ax[3].legend(loc="upper left"); ax[3].set_xlabel("time (s)")
txt = fig.text(0.01, 0.005, "", family="monospace")
i, t0 = 0, time.perf_counter()
while i < N and plt.fignum_exists(fig.number):
    due = min(N, int((time.perf_counter() - t0) * a.speed / nb.CFG["DT"]) + 1)
    while i < due:
        x = sf.step(*(float(tel[c][i]) for c in KEYS))
        s0 = time.perf_counter_ns()
        r = sc.push(x, corr[a.dc], unc[a.dc])
        if r is not None:
            lat.append((time.perf_counter_ns() - s0) / 1e6)
            wt.append(T[i]); ws.append(r["score"]); wa.append(float(r["alarm"])); last = r
        tt.append(T[i]); vv.append(tel["Va_pu"][i]); pp.append(tel["P_total_MW"][i])
        ff.append(tel["P_it_MW"][i] + tel["P_cool_MW"][i] + tel["P_aux_MW"][i]); ll.append(float(lab[i] == 1))
        i += 1
    for line, xs, ys in ((lv, tt, vv), (lp, tt, pp), (lf, tt, ff), (lb, tt, ll), (ls, wt, ws), (la, wt, wa)):
        line.set_data(xs, ys)
    for k in (0, 1, 2):
        ax[k].relim(); ax[k].autoscale_view(scalex=False)
    if tt:
        ax[3].set_xlim(max(tt[0], tt[-1] - a.span), tt[-1] + 0.5)
    if last is not None:
        txt.set_text(f"t={tt[-1]:7.2f} s  ALARM={'YES' if last['alarm'] else 'no '}  term={last['dominant'] if last['alarm'] else '-':7s} "
                     f"domain={last['domain'] if last['alarm'] else '-':13s} CSEC c={last['csec_corr']:.2f}  "
                     f"TCN+fusion={np.mean(lat):.3f} ms/window")
        txt.set_color("red" if last["alarm"] else "black")
    plt.pause(0.1)
plt.ioff()
plt.show()
