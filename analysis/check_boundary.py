#!/usr/bin/env python3
"""
H1 diagnostic: is the ChunkBoundaryHead's score actually informative at inference?

The boundary head is supposed to fire high at *decisive* moments (contact, grasp,
direction reversal) and low during smooth free-space motion. Our adaptive
re-planning is only useful if that is true at inference time. This script checks
it WITHOUT any new rollouts: run a normal eval with the client's --log_boundary
flag, then point this script at the produced JSONL.

    # 1) record (run the *baseline* so the rollout is a clean, successful one):
    python evaluation/libero/libero_client.py --port 8102 --task_suite libero_spatial \
        --num_trials 5 --log_boundary bd.jsonl
    # 2) analyze:
    python analysis/check_boundary.py bd.jsonl

It uses the gripper-command sign flip (open<->close) as an unambiguous, label-free
proxy for "decisive moment", and reports whether boundary scores are elevated
around those events vs. elsewhere. A near/far ratio comfortably > 1 means the
signal is usable; ~1 (or a near-constant boundary) means H1 is weak and the head
needs to be retrained on an *absolute* change-rate target before adaptive
re-planning can help.
"""

import json
import sys
from statistics import mean, pstdev


def load_episodes(path):
    episodes, cur = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "reset":
                if cur:
                    episodes.append(cur)
                    cur = []
                continue
            cur.append(rec)
    if cur:
        episodes.append(cur)
    return episodes


def main():
    if len(sys.argv) < 2:
        print("usage: python analysis/check_boundary.py <boundary_log.jsonl> [window]")
        sys.exit(1)
    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    episodes = load_episodes(path)
    if not episodes:
        print("No data found (did the run produce any steps?).")
        return

    all_b, near, far = [], [], []
    total_toggles = 0
    for ep in episodes:
        grips = [r.get("grip") for r in ep]
        bs = [r.get("b") for r in ep]
        toggles = [
            i for i in range(1, len(grips))
            if grips[i] is not None and grips[i - 1] is not None
            and (grips[i] > 0) != (grips[i - 1] > 0)
        ]
        total_toggles += len(toggles)
        for i, b in enumerate(bs):
            if b is None:
                continue
            all_b.append(b)
            is_near = any(abs(i - t) <= window for t in toggles)
            (near if is_near else far).append(b)

    if not all_b:
        print("Boundary scores were all None -- the server is not returning 'boundary'.")
        print("Make sure you pulled the updated serve_smolvlm_libero.py and the checkpoint")
        print("was trained with use_adaptive_chunking=true.")
        return

    def fmt(xs):
        return f"mean={mean(xs):.3f} std={pstdev(xs):.3f} n={len(xs)}" if xs else "n=0"

    print(f"episodes={len(episodes)}  executed_steps={len(all_b)}  gripper_toggles={total_toggles}")
    print(f"boundary overall : mean={mean(all_b):.3f} std={pstdev(all_b):.3f} "
          f"min={min(all_b):.3f} max={max(all_b):.3f}")
    print(f"boundary NEAR toggle (|dt|<={window}) : {fmt(near)}")
    print(f"boundary FAR  from toggle            : {fmt(far)}")

    # Verdicts
    if pstdev(all_b) < 0.05:
        print("\n[VERDICT] boundary is ~CONSTANT -> useless for adaptive timing. "
              "H1 FAILS; retrain the head (absolute change-rate target) or pick another signal.")
    elif near and far:
        ratio = mean(near) / max(mean(far), 1e-6)
        print(f"\nnear/far ratio = {ratio:.2f}")
        if ratio > 1.3:
            print("[VERDICT] boundary is clearly ELEVATED at contacts -> H1 SUPPORTED. "
                  "Adaptive re-planning can work; tune beta / max_horizon on the Pareto.")
        elif ratio > 1.1:
            print("[VERDICT] weak signal -> H1 PARTIAL. May need a better target/threshold.")
        else:
            print("[VERDICT] no contact selectivity -> H1 WEAK/FAILS. "
                  "Retrain the head on an absolute change-rate target before relying on it.")
    else:
        print("\n[VERDICT] not enough gripper toggles to judge; try more trials / a "
              "manipulation-heavy suite (e.g. libero_object).")


if __name__ == "__main__":
    main()
