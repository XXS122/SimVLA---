#!/usr/bin/env python3
"""
SimVLA Failure-Signal Evaluation
=================================
Checks whether flow-matching sampling-variance (multiple ODE noise seeds
through the same conditioning) is predictive of LIBERO episode failure.

This is a diagnostic-only script: it does NOT change execution behavior
(no truncation / recovery). It runs the normal fixed-cadence replan loop,
additionally asking the server for `--num_uncertainty_samples` flow-matching
samples per replan call, and logs the resulting per-step uncertainty
alongside the episode's actual success/failure outcome. Requires the server
to be started from a checkpoint (`serve_smolvlm_libero.py`), same as
`libero_client.py`.

Signal: `SmolVLMVLA.generate_actions_with_uncertainty()` (models/modeling_smolvlm_vla.py)
draws K independent noise seeds through the Euler integration and reports
the per-step spread across samples -- unrelated to (and does not require)
the ChunkBoundaryHead / adaptive chunking, which is untested in this repo.

Usage:
  cd evaluation/libero
  python failure_signal_eval.py --port 8102 --task_suite libero_goal \
      --num_trials 20 --checkpoint_tag uniform --out_csv failure_signal.csv

  # After running once per checkpoint (e.g. also --checkpoint_tag tds against
  # a TDS-trained checkpoint's server, appending to the same --out_csv), the
  # AUROC printed per checkpoint_tag directly answers the tds_design.md
  # "cross experiment" question -- does TDS training make the failure
  # signal more discriminative -- with zero code coupling between the two.

  # Re-analyze an existing csv without re-running episodes:
  python failure_signal_eval.py --analyze --out_csv failure_signal.csv --plot unc_vs_failure.png
"""
from __future__ import annotations

import argparse
import collections
import csv as csv_module
import os
from typing import Deque, Dict, List, Optional

import numpy as np
from tqdm import tqdm

try:
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy as ws_client
    HAS_WS_CLIENT = True
except ImportError:
    HAS_WS_CLIENT = False

from libero_client import (
    _quat2axisangle,
    benchmark_dict,
    get_libero_env,
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    MAX_STEPS,
    NUM_STEPS_WAIT,
)


# -----------------------------------------------------------------------------
# Client: same replan cadence as libero_client.WebSocketClient, but requests
# K flow-matching samples per replan call and logs the resulting uncertainty.
# -----------------------------------------------------------------------------
class UncertaintyWebSocketClient:
    """Requires: pip install openpi-client"""

    def __init__(self, host: str, port: int, replan_steps: int = 5,
                 resize_size: int = 224, num_uncertainty_samples: int = 8):
        if not HAS_WS_CLIENT:
            raise ImportError("openpi_client not installed. Run: pip install openpi-client")
        self.client = ws_client.WebsocketClientPolicy(host, port)
        self.replan_steps = replan_steps
        self.resize_size = resize_size
        self.num_uncertainty_samples = num_uncertainty_samples
        self.reset()

    def reset(self) -> None:
        self.action_plan: Deque[np.ndarray] = collections.deque()
        self.uncertainty_log: List[Dict] = []  # one entry per replan (server) call

    def step(self, obs: Dict, goal: str, t: int) -> np.ndarray:
        if not self.action_plan:
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["image"], self.resize_size, self.resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(obs["wrist_image"], self.resize_size, self.resize_size)
            )

            element = {
                "observation/image": img,
                "observation/wrist_image": wrist_img,
                "observation/state": obs["state"],
                "prompt": goal,
                "num_uncertainty_samples": self.num_uncertainty_samples,
            }

            result = self.client.infer(element)
            action_chunk = np.asarray(result["actions"])
            uncertainty = np.asarray(result.get("uncertainty", []), dtype=np.float64)
            gripper_disagreement = np.asarray(result.get("gripper_disagreement", []), dtype=np.float64)

            self.uncertainty_log.append({
                "t": t,
                "uncertainty": uncertainty.tolist(),
                "gripper_disagreement": gripper_disagreement.tolist(),
            })

            for i in range(min(self.replan_steps, len(action_chunk))):
                self.action_plan.append(action_chunk[i])

        return self.action_plan.popleft()


# -----------------------------------------------------------------------------
# Evaluator (adapted from libero_client.eval_libero; does not modify that file)
# -----------------------------------------------------------------------------
def eval_libero_with_uncertainty(
    client: UncertaintyWebSocketClient,
    task_suite_name: str,
    num_trials: int = 20,
    seed: int = 7,
    task_id: Optional[int] = None,
    out_csv: str = "failure_signal.csv",
    checkpoint_tag: str = "default",
) -> None:
    np.random.seed(seed)

    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    max_steps = MAX_STEPS.get(task_suite_name, 400)

    print(f"Task suite: {task_suite_name}")
    print(f"   Tasks: {num_tasks}, Trials per task: {num_trials}")
    print(f"   Checkpoint tag: {checkpoint_tag}")

    rows = []
    task_ids = [task_id] if task_id is not None else range(num_tasks - 1, -1, -1)
    for task_id in tqdm(task_ids, desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, LIBERO_ENV_RESOLUTION, seed)

        for ep in tqdm(range(num_trials), desc=f"{task_description[:30]}...", leave=False):
            env.reset()
            client.reset()
            obs = env.set_init_state(initial_states[ep % len(initial_states)])

            t = 0
            done = False

            while t < max_steps + NUM_STEPS_WAIT:
                try:
                    if t < NUM_STEPS_WAIT:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                    state = np.concatenate([
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ])

                    obs_dict = {"image": img, "wrist_image": wrist_img, "state": state}

                    action = client.step(obs_dict, task_description, t)
                    obs, reward, done, info = env.step(action.tolist())

                    if done:
                        break

                    t += 1

                except Exception as e:
                    print(f"Error in rollout: {e}")
                    break

            log = client.uncertainty_log
            unc_means = [float(np.mean(r["uncertainty"])) for r in log if r["uncertainty"]]
            unc_maxes = [float(np.max(r["uncertainty"])) for r in log if r["uncertainty"]]
            grip_maxes = [float(np.max(r["gripper_disagreement"])) for r in log if r["gripper_disagreement"]]

            rows.append({
                "checkpoint_tag": checkpoint_tag,
                "task_name": f"{task.name}_demo",
                "task_id": task_id,
                "episode": ep,
                "success": int(done),
                "steps": t,
                "unc_mean": float(np.mean(unc_means)) if unc_means else float("nan"),
                "unc_max": float(np.max(unc_maxes)) if unc_maxes else float("nan"),
                "unc_last_chunk_max": unc_maxes[-1] if unc_maxes else float("nan"),
                "gripper_disagree_max": float(np.max(grip_maxes)) if grip_maxes else float("nan"),
            })

            status_icon = "[OK]" if done else "[FAIL]"
            print(f"  {status_icon} Task {task_id} Ep {ep}: "
                  f"{'success' if done else 'failure'} (steps={t})")

        env.close()

    fieldnames = ["checkpoint_tag", "task_name", "task_id", "episode", "success", "steps",
                  "unc_mean", "unc_max", "unc_last_chunk_max", "gripper_disagree_max"]
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} episode rows to {out_csv}")


# -----------------------------------------------------------------------------
# Analysis: does the uncertainty score predict failure?
# -----------------------------------------------------------------------------
def _auroc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AUROC via the rank-sum / Mann-Whitney U identity (avoids adding an
    sklearn dependency; scipy is already a project dependency)."""
    from scipy.stats import rankdata

    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return (ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def analyze(csv_path: str, score_col: str = "unc_max", plot_path: Optional[str] = None) -> None:
    import pandas as pd

    df = pd.read_csv(csv_path).dropna(subset=[score_col])

    for tag, group in df.groupby("checkpoint_tag"):
        labels = 1 - group["success"].to_numpy()  # 1 == failure
        scores = group[score_col].to_numpy()
        n_pos, n = int(labels.sum()), len(labels)

        auroc = _auroc(scores, labels)
        if auroc is None:
            print(f"[{tag}] {score_col}: need both successes and failures to compute AUROC "
                  f"(got {n_pos} failures / {n - n_pos} successes, n={n}) -- skipping")
            continue
        print(f"[{tag}] {score_col}: AUROC(failure) = {auroc:.3f}  "
              f"(n={n}, failures={n_pos}, successes={n - n_pos})")

    if plot_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = sorted(df["checkpoint_tag"].unique())
        fig, axes = plt.subplots(1, len(tags), figsize=(4.5 * len(tags), 4), sharey=True)
        axes = [axes] if len(tags) == 1 else list(axes)
        for ax, tag in zip(axes, tags):
            group = df[df["checkpoint_tag"] == tag]
            data = [group[group.success == 1][score_col], group[group.success == 0][score_col]]
            ax.boxplot(data, labels=["success", "failure"])
            ax.set_title(tag)
            ax.set_ylabel(score_col)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        print(f"Saved plot to {plot_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        "SimVLA failure-signal evaluation (flow-matching sampling-variance uncertainty)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task_suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--num_trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    parser.add_argument("--task_id", type=int, default=None,
                        help="If set, only evaluate this task index (0-based). Omit to run all tasks.")
    parser.add_argument("--num_uncertainty_samples", type=int, default=8,
                        help="K noise seeds per replan call for the sampling-variance signal")
    parser.add_argument("--checkpoint_tag", type=str, default="default",
                        help="Label written into --out_csv (e.g. 'uniform' vs 'tds') so runs "
                             "against different checkpoints can be compared after concatenation")
    parser.add_argument("--out_csv", type=str, default="failure_signal.csv")
    parser.add_argument("--analyze", action="store_true",
                        help="Skip rollout; just analyze an existing --out_csv")
    parser.add_argument("--score_col", type=str, default="unc_max",
                        choices=["unc_mean", "unc_max", "unc_last_chunk_max", "gripper_disagree_max"])
    parser.add_argument("--plot", type=str, default=None,
                        help="If set, save a success-vs-failure score boxplot to this path")

    args = parser.parse_args()

    if args.analyze:
        analyze(args.out_csv, score_col=args.score_col, plot_path=args.plot)
        return

    client = UncertaintyWebSocketClient(
        args.host, args.port,
        replan_steps=args.replan_steps,
        num_uncertainty_samples=args.num_uncertainty_samples,
    )

    eval_libero_with_uncertainty(
        client=client,
        task_suite_name=args.task_suite,
        num_trials=args.num_trials,
        seed=args.seed,
        task_id=args.task_id,
        out_csv=args.out_csv,
        checkpoint_tag=args.checkpoint_tag,
    )

    analyze(args.out_csv, score_col=args.score_col, plot_path=args.plot)


if __name__ == "__main__":
    main()
