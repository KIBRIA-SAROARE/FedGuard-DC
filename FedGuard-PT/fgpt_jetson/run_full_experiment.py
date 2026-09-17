#!/usr/bin/env python3
"""Full on-board experiment (resumable). Every stream is replayed at 100 Hz wall clock.

  1. idle baseline (start)
  2. training on the board: 5 LOSO folds (S02..S06 held out) + FULL fold (all attack-free scenarios)
  3. real-time monitor streams for one site
       LOSO fold <S>: no_attack, T1, T2, T3a, T3 (notebook generators), A1..A5 (repository files) on <S>
       FULL fold    : control-path scenarios S07..S11 (never used in training)
  4. idle baseline (end)
Each stage runs in its own process; a finished stage is skipped on restart (delete its folder to redo it).
Run inside tmux or with nohup: the full plan takes several hours.
"""
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOSO = ["S02_fault_100ms", "S03_fault_250ms", "S04_line_trip", "S05_load_step", "S06_multi_event"]
CTRL = ["S07_atk_false_ups", "S08_atk_load_alter", "S09_atk_sensor_spoof", "S10_atk_cooling", "S11_fault_plus_atk"]
LOSO_TESTS = ["no_attack", "T1", "T2", "T3a", "T3", "A1", "A2", "A3", "A4", "A5"]
STREAM_S = 298.0   # 29,801 samples at 10 ms


def plan(name, dc):
    folds, streams = [], []
    if name == "full":
        folds = LOSO + ["FULL"]
        streams = [(f, t, f) for f in LOSO for t in LOSO_TESTS] + [("FULL", "ctrl", s) for s in CTRL]
    elif name == "quick":
        folds = ["S02_fault_100ms", "FULL"]
        streams = [("S02_fault_100ms", t, "S02_fault_100ms") for t in ["no_attack", "T1", "T3", "A1"]] + [("FULL", "ctrl", "S07_atk_false_ups")]
    return folds, streams


def log(msg, fh):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line, flush=True)
    fh.write(line + "\n")
    fh.flush()


def run(cmd, fh, env):
    log("RUN " + " ".join(cmd), fh)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    fh.write(p.stdout)
    fh.flush()
    print(p.stdout[-1500:], flush=True)
    log(f"exit={p.returncode} after {time.time() - t0:.1f} s", fh)
    if p.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="full", choices=["full", "quick"])
    ap.add_argument("--dc", type=int, default=1)
    ap.add_argument("--dataset", default="data/dataset")
    ap.add_argument("--out", default="results_rt")
    ap.add_argument("--idle-s", type=float, default=60)
    ap.add_argument("--cooldown-s", type=float, default=20, help="idle gap between streams")
    ap.add_argument("--net-delay-ms", type=float, default=0.0)
    ap.add_argument("--no-realtime", action="store_true", help="pipeline check only; paper_figures.py rejects these streams")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    folds, streams = plan(a.plan, a.dc)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    fh = open(os.path.join(out, "experiment_log.txt"), "a")
    sys.path.insert(0, HERE)
    from fgpt import common
    common.save_json(dict(plan=a.plan, dc=a.dc, folds=folds, streams=streams, net_delay_ms=a.net_delay_ms,
                          cooldown_s=a.cooldown_s, idle_s=a.idle_s, system=common.system_info(),
                          jetson_clocks=common.sh("jetson_clocks --show 2>&1 | head -20")),
                     os.path.join(out, "experiment_plan.json"))
    todo = sum(not os.path.exists(os.path.join(out, "streams", f, f"{t}_{s}", "metrics.json")) for f, t, s in streams)
    log(f"plan={a.plan}: {len(folds)} trainings, {len(streams)} streams ({todo} to run); "
        f"stream replay alone = {todo * (STREAM_S + a.cooldown_s) / 3600:.2f} h (training time not included)", fh)

    d = os.path.join(out, "idle_start")
    if not os.path.exists(os.path.join(d, "resources_idle.csv")):
        run([sys.executable, "measure_idle.py", "--results", d, "--seconds", str(a.idle_s)], fh, env)

    for f in folds:
        fd = os.path.join(out, "folds", f)
        if os.path.exists(os.path.join(fd, "train_log.json")):
            log(f"skip training {f} (done)", fh)
            continue
        run([sys.executable, "train_edge.py", "--dataset", a.dataset, "--results", fd, "--test-scenario", f], fh, env)
        time.sleep(a.cooldown_s)

    for i, (f, t, s) in enumerate(streams, 1):
        sd = os.path.join(out, "streams", f, f"{t}_{s}")
        if os.path.exists(os.path.join(sd, "metrics.json")):
            log(f"skip stream {i}/{len(streams)} {f}/{t}_{s} (done)", fh)
            continue
        log(f"stream {i}/{len(streams)}: fold={f} test={t} scenario={s}", fh)
        run([sys.executable, "rt_monitor.py", "--dataset", a.dataset, "--fold-dir", os.path.join(out, "folds", f), "--out", sd,
             "--dc", str(a.dc), "--scenario", s, "--test", t, "--net-delay-ms", str(a.net_delay_ms)] + (["--no-realtime"] if a.no_realtime else []), fh, env)
        if not a.no_realtime:
            time.sleep(a.cooldown_s)

    d = os.path.join(out, "idle_end")
    if not os.path.exists(os.path.join(d, "resources_idle.csv")):
        run([sys.executable, "measure_idle.py", "--results", d, "--seconds", str(a.idle_s)], fh, env)
    run([sys.executable, "bench_latency.py", "--results", os.path.join(out, "folds", folds[0]), "--dc", str(a.dc)],
        fh, {k: v for k, v in env.items() if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    log("all measurement stages finished", fh)


if __name__ == "__main__":
    main()
