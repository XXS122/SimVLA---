#!/usr/bin/env python3
"""
Test-Time Scaling analysis for SimVLA on LIBERO.

Consumes the two JSONL logs produced by a sweep:
  - results_*.jsonl  (client, per-episode outcomes; config encoded in filename)
  - diag.jsonl       (server, per-call diagnostics with tts config + metadata)

Produces:
  1. Scaling table: per (N, S, selector) success rate and mean compute
     (action-transformer forwards per call) -> the compute-vs-success curve.
  2. Uncertainty-event alignment: velocity-field variance (uncertainty_v1)
     binned by distance (in policy calls) to the nearest gripper open/close
     event -> tests whether uncertainty peaks at manipulation-critical moments.
  3. Episode-level correlation: mean episode uncertainty vs success.

Usage:
    python analyze_tts.py --results_dir ./tts_sweep --diag ./tts_sweep/diag.jsonl \
        [--out ./tts_sweep/summary.csv] [--plots ./tts_sweep]
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path


RESULTS_RE = re.compile(r"results_(?P<suite>.+)_N(?P<n>\d+)_S(?P<s>\d+)_(?P<sel>\w+)\.jsonl$")


def load_jsonl(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_results(results_dir: Path):
    """Load per-episode outcomes, keyed by (suite, N, S, selector)."""
    by_config = {}
    for path in sorted(results_dir.glob("results_*.jsonl")):
        m = RESULTS_RE.search(path.name)
        if not m:
            print(f"[warn] skipping unrecognized results file: {path.name}")
            continue
        key = (m["suite"], int(m["n"]), int(m["s"]), m["sel"])
        by_config[key] = load_jsonl(path)
    return by_config


def group_diag(diag_records):
    """Group diagnostics by (suite, N, S, selector) and by episode within."""
    by_config = defaultdict(list)
    for r in diag_records:
        suite = r.get("task_suite")
        if suite is None:
            continue
        key = (suite, r["num_samples"], r["steps"], r["selector"])
        by_config[key].append(r)
    return by_config


def episode_key(r):
    return (r.get("task_id"), r.get("episode"))


def pick_unc_field(diag_by_config, requested: str = "auto") -> str:
    """
    Choose the uncertainty field to analyze.

    Preference order: uncertainty_x0hat (noise-cancelled one-step estimate,
    logged by newer servers) > uncertainty_x0 (final-candidate variance,
    present in all logs) > uncertainty_v1 (raw velocity variance at t=1;
    dominated by the noise itself — last resort only).
    """
    if requested != "auto":
        return requested
    for field in ("uncertainty_x0hat", "uncertainty_x0", "uncertainty_v1"):
        for records in diag_by_config.values():
            if any(r.get(field) is not None for r in records):
                return field
    return "uncertainty_v1"


# ---------------------------------------------------------------------------
# 1. Scaling table
# ---------------------------------------------------------------------------
def scaling_table(results_by_config, diag_by_config, out_csv=None):
    rows = []
    for key in sorted(results_by_config):
        suite, n, s, sel = key
        eps = results_by_config[key]
        if not eps:
            continue
        succ = sum(1 for e in eps if e["success"])
        diag = diag_by_config.get(key, [])
        fwd = [d["transformer_forwards"] for d in diag] or [float("nan")]
        active = [d["num_active_samples"] for d in diag] or [float("nan")]
        rows.append({
            "suite": suite, "N": n, "S": s, "selector": sel,
            "episodes": len(eps),
            "success_rate": succ / len(eps),
            "mean_forwards_per_call": sum(fwd) / len(fwd),
            "mean_active_samples": sum(active) / len(active),
        })

    if not rows:
        print("No results found.")
        return rows

    header = f"{'suite':<16} {'N':>3} {'S':>3} {'selector':<11} {'eps':>4} {'succ%':>7} {'fwd/call':>9} {'activeN':>8}"
    print("\n=== Scaling table (compute vs success) ===")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['suite']:<16} {r['N']:>3} {r['S']:>3} {r['selector']:<11} "
              f"{r['episodes']:>4} {100 * r['success_rate']:>6.1f}% "
              f"{r['mean_forwards_per_call']:>9.1f} {r['mean_active_samples']:>8.2f}")

    if out_csv:
        import csv
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved: {out_csv}")
    return rows


# ---------------------------------------------------------------------------
# 2. Uncertainty vs distance-to-gripper-event
# ---------------------------------------------------------------------------
def gripper_events(calls):
    """Indices (into the call sequence) where the commanded gripper flips sign."""
    cmds = [c["gripper_cmd"][0] for c in calls if c.get("gripper_cmd")]
    events = []
    for i in range(1, len(cmds)):
        if (cmds[i] > 0) != (cmds[i - 1] > 0):
            events.append(i)
    return events


def uncertainty_alignment(diag_by_config, max_dist=10, field="uncertainty_x0"):
    print(f"\n=== Uncertainty ({field}) vs distance to nearest gripper event ===")
    printed = False
    for key, records in sorted(diag_by_config.items()):
        suite, n, s, sel = key
        if n <= 1:
            continue  # variance probe needs N > 1

        episodes = defaultdict(list)
        for r in records:
            if r.get(field) is not None:
                episodes[episode_key(r)].append(r)

        bins = defaultdict(list)
        for ep_key, calls in episodes.items():
            if ep_key == (None, None) or len(calls) < 3:
                continue
            calls.sort(key=lambda r: r.get("env_step", r["call_idx"]))
            events = gripper_events(calls)
            if not events:
                continue
            for i, c in enumerate(calls):
                d = min(abs(i - e) for e in events)
                bins[min(d, max_dist)].append(c[field])

        if not bins:
            continue
        printed = True
        print(f"\n[{suite} | N={n} S={s} {sel}]  (distance in policy calls)")
        print(f"{'dist':>6} {'count':>7} {'mean_unc':>12} {'std':>10}")
        for d in sorted(bins):
            vals = bins[d]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            label = f">={max_dist}" if d == max_dist else str(d)
            print(f"{label:>6} {len(vals):>7} {mean:>12.5f} {std:>10.5f}")

    if not printed:
        print("(no N>1 diagnostics with episode metadata found)")


# ---------------------------------------------------------------------------
# 3. Episode-level uncertainty vs success
# ---------------------------------------------------------------------------
def uncertainty_vs_success(results_by_config, diag_by_config, field="uncertainty_x0"):
    print(f"\n=== Episode mean uncertainty ({field}) vs success (point-biserial r) ===")
    printed = False
    for key, eps in sorted(results_by_config.items()):
        diag = diag_by_config.get(key, [])
        if not diag or key[1] <= 1:
            continue

        ep_unc = defaultdict(list)
        for r in diag:
            if r.get(field) is not None:
                ep_unc[episode_key(r)].append(r[field])

        pairs = []
        for e in eps:
            k = (e["task_id"], e["episode"])
            if k in ep_unc:
                pairs.append((sum(ep_unc[k]) / len(ep_unc[k]), 1.0 if e["success"] else 0.0))

        if len(pairs) < 5:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        cov = sum((x - mx) * (y - my) for x, y in pairs)
        vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        vy = math.sqrt(sum((y - my) ** 2 for y in ys))
        if vx == 0 or vy == 0:
            continue
        printed = True
        suite, n, s, sel = key
        print(f"[{suite} | N={n} S={s} {sel}] episodes={len(pairs)}  r={cov / (vx * vy):+.3f}")

    if not printed:
        print("(need N>1 configs with both results and diagnostics)")


# ---------------------------------------------------------------------------
# Optional plots
# ---------------------------------------------------------------------------
def make_plots(rows, plots_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not available, skipping plots")
        return

    by_suite = defaultdict(list)
    for r in rows:
        by_suite[r["suite"]].append(r)

    for suite, rs in by_suite.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        by_s = defaultdict(list)
        for r in rs:
            by_s[(r["S"], r["selector"])].append(r)
        for (s, sel), pts in sorted(by_s.items()):
            pts.sort(key=lambda r: r["mean_forwards_per_call"])
            ax.plot(
                [p["mean_forwards_per_call"] for p in pts],
                [100 * p["success_rate"] for p in pts],
                marker="o", label=f"S={s} ({sel})",
            )
        ax.set_xscale("log")
        ax.set_xlabel("Action-transformer forwards per call (log)")
        ax.set_ylabel("Success rate (%)")
        ax.set_title(f"Test-time compute scaling — {suite}")
        ax.legend()
        ax.grid(alpha=0.3)
        out = Path(plots_dir) / f"tts_scaling_{suite}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser("TTS analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory with results_*.jsonl from run_tts_sweep.sh")
    parser.add_argument("--diag", type=str, default=None,
                        help="Server-side diag.jsonl (enables uncertainty analyses)")
    parser.add_argument("--out", type=str, default=None, help="Optional summary CSV path")
    parser.add_argument("--plots", type=str, default=None, help="Optional directory for PNG plots")
    parser.add_argument("--unc_field", type=str, default="auto",
                        choices=["auto", "uncertainty_x0hat", "uncertainty_x0", "uncertainty_v1"],
                        help="Uncertainty probe field to analyze (auto prefers x0hat > x0 > v1)")
    args = parser.parse_args()

    results = load_results(Path(args.results_dir))
    diag = group_diag(load_jsonl(Path(args.diag))) if args.diag else {}

    rows = scaling_table(results, diag, out_csv=args.out)
    if diag:
        field = pick_unc_field(diag, args.unc_field)
        uncertainty_alignment(diag, field=field)
        uncertainty_vs_success(results, diag, field=field)
    if args.plots and rows:
        make_plots(rows, args.plots)


if __name__ == "__main__":
    main()
