#!/usr/bin/env python3
"""
Branching counterfactual experiment (Experiment 5).

Question: when the policy is about to fail (uncertainty spike), is the
failure RECOVERABLE by resampling — do some stochastic continuations from
the exact same simulator state succeed — or is it SYSTEMATIC (all
continuations fail because the policy does not know this state)?

Protocol, per episode:
  1. Roll out normally. Watch the per-chunk uncertainty returned by the
     server. The first time it exceeds --unc_threshold (at or after chunk
     --min_call), snapshot the simulator state ("spike" arm).
  2. If no spike occurs by chunk --control_call, snapshot there instead
     ("control" arm — calm-moment baseline).
  3. From the snapshot, run --branches independent continuations (fresh
     policy noise each time, same remaining step budget) and count
     successes.

Reading the result:
  spike-arm branch success rate >> 0  -> failures recoverable by sampling
                                          (selection bottleneck -> verifier)
  spike-arm branch success rate ~= 0  -> failures systematic
                                          (policy itself wrong -> Flow-DPO)
  control arm calibrates baseline recoverability at calm moments.

Pick the threshold from previous sweep logs, e.g. the 85th percentile of
uncertainty_x0hat for the config you will use for the main rollout:

  python -c "
import json
u = [r['uncertainty_x0hat'] for r in map(json.loads, open('./tts_sweep_10/diag.jsonl'))
     if r.get('uncertainty_x0hat') is not None and r['num_samples'] == 8 and r['steps'] == 5]
u.sort(); n = len(u)
print('P50', u[n//2], 'P85', u[int(0.85*n)], 'P90', u[int(0.9*n)])"

Usage:
  python branch_counterfactual.py --port 8102 --task_suite libero_10 \
      --num_trials 10 --branches 4 --unc_threshold <P85> \
      --log ./branch_10/branches.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from libero_client import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    MAX_STEPS,
    NUM_STEPS_WAIT,
    WebSocketClient,
    _quat2axisangle,
    benchmark_dict,
    get_libero_env,
)


def build_obs(obs) -> dict:
    """Pack a LIBERO observation the same way eval_libero does."""
    return {
        "image": np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]),
        "state": np.concatenate([
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        ]),
    }


def run_segment(env, client, obs, t, max_total, desc, on_new_chunk=None):
    """
    Step the environment with the policy until success or budget.

    on_new_chunk(uncertainty, t) is called right after each fresh chunk is
    fetched and BEFORE the chunk is executed; returning True aborts the
    segment with status "trigger" (env state = pre-chunk decision point).

    Returns (status, obs, t, success) with status in
    {"done", "budget", "trigger"}.
    """
    while t < max_total:
        packed = build_obs(obs)
        prev_fetches = client.num_fetches
        action = client.step(packed, desc)
        if on_new_chunk is not None and client.num_fetches != prev_fetches:
            if on_new_chunk(client.last_uncertainty, t):
                return "trigger", obs, t, False
        obs, reward, done, info = env.step(action.tolist())
        t += 1
        if done:
            return "done", obs, t, True
    return "budget", obs, t, False


def main():
    parser = argparse.ArgumentParser("Branching counterfactual")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    parser.add_argument("--task_suite", type=str, default="libero_10")
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--replan_steps", type=int, default=5)
    # Main-rollout policy config (needs N>1 so the probe is available)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--ode_steps", type=int, default=5)
    # Branch-continuation policy config (plain sampling by default)
    parser.add_argument("--branch_num_samples", type=int, default=1)
    parser.add_argument("--branches", type=int, default=4,
                        help="Independent continuations per snapshot")
    # Trigger
    parser.add_argument("--unc_threshold", type=float, required=True,
                        help="Uncertainty level that counts as a spike")
    parser.add_argument("--min_call", type=int, default=5,
                        help="Earliest chunk index eligible for the spike trigger")
    parser.add_argument("--control_call", type=int, default=15,
                        help="Branch here when no spike occurred (control arm)")
    parser.add_argument("--log", type=str, required=True,
                        help="Output JSONL path")
    # Flow-DPO data collection mode
    parser.add_argument("--save_pairs_dir", type=str, default=None,
                        help="Save snapshot observations + per-branch first "
                             "chunks here (turns the run into a matched-state "
                             "preference-pair collector for Flow-DPO)")
    parser.add_argument("--randomize_control", action="store_true", default=False,
                        help="Draw the control branch point uniformly from "
                             "[min_call, control_call] per episode (state "
                             "diversity for data collection)")
    args = parser.parse_args()

    np.random.seed(args.seed)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    if args.save_pairs_dir:
        Path(args.save_pairs_dir).mkdir(parents=True, exist_ok=True)

    main_extra = {"tts/num_samples": args.num_samples, "tts/ode_steps": args.ode_steps}
    branch_extra = {"tts/num_samples": args.branch_num_samples, "tts/ode_steps": args.ode_steps}

    if args.num_samples < 2:
        parser.error("--num_samples must be >= 2: the spike trigger needs the probe")

    client = WebSocketClient(args.host, args.port, replan_steps=args.replan_steps)

    task_suite = benchmark_dict[args.task_suite]()
    max_total = MAX_STEPS.get(args.task_suite, 400) + NUM_STEPS_WAIT

    arm_stats = {"spike": [], "control": []}
    finished_before_trigger = 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        env, desc = get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for ep in range(args.num_trials):
            print(f"[task {task_id} ep {ep}] rollout ...", flush=True)
            env.reset()
            client.reset()
            client.extra = main_extra
            obs = env.set_init_state(init_states[ep % len(init_states)])
            for _ in range(NUM_STEPS_WAIT):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            t = NUM_STEPS_WAIT

            trigger = {}
            control_at = args.control_call
            if args.randomize_control:
                control_at = int(np.random.randint(args.min_call, args.control_call + 1))

            def on_chunk(unc, t_now, _trigger=trigger, _control_at=control_at):
                calls = client.num_fetches
                if calls >= _control_at:
                    _trigger.update(arm="control", unc=unc, call=calls)
                    return True
                if (
                    unc is not None
                    and calls >= args.min_call
                    and unc >= args.unc_threshold
                ):
                    _trigger.update(arm="spike", unc=unc, call=calls)
                    return True
                return False

            status, obs, t, success = run_segment(
                env, client, obs, t, max_total, desc, on_chunk
            )

            record = {
                "task_suite": args.task_suite,
                "task_id": int(task_id),
                "task_description": desc,
                "episode": int(ep),
                "time": time.time(),
            }

            if status != "trigger":
                # Finished (or exhausted budget) before any trigger fired
                finished_before_trigger += 1
                record.update(arm="none", success=bool(success), env_steps=int(t))
            else:
                print(
                    f"[task {task_id} ep {ep}] trigger: arm={trigger['arm']} "
                    f"call={trigger['call']} unc={trigger['unc']} -> "
                    f"{args.branches} branches ...",
                    flush=True,
                )
                snapshot = env.get_sim_state()
                t_trig = t

                # Data-collection mode: save the snapshot observation once
                snapshot_file = None
                if args.save_pairs_dir:
                    packed = build_obs(obs)
                    snapshot_file = str(
                        Path(args.save_pairs_dir)
                        / f"snap_{args.task_suite}_t{task_id}_e{ep}.npz"
                    )
                    np.savez_compressed(
                        snapshot_file,
                        image=packed["image"].astype(np.uint8),
                        wrist_image=packed["wrist_image"].astype(np.uint8),
                        state=packed["state"].astype(np.float32),
                        prompt=desc,
                    )

                successes = 0
                branch_records = []
                client.extra = branch_extra
                for m in range(args.branches):
                    obs_b = env.set_init_state(snapshot)
                    # set_init_state restores the MuJoCo state but not the
                    # robosuite wrapper's step counter/done flag; they keep
                    # accumulating across branches and step() raises
                    # "executing action in terminated episode" once the
                    # wrapper horizon (1000) is crossed. Clear them so each
                    # branch gets a fresh wrapper budget (our own t budget
                    # remains the binding limit).
                    inner = getattr(env, "env", None)
                    if inner is not None:
                        inner.timestep = 0
                        inner.done = False
                    client.reset()

                    # Grab the first chunk the branch samples at the snapshot
                    # state: that action is the preference-pair candidate
                    first_chunk = {}

                    def grab_chunk(unc, t_now, _fc=first_chunk):
                        if "chunk" not in _fc and client.last_chunk is not None:
                            _fc["chunk"] = client.last_chunk.tolist()
                        return False

                    _, _, _, ok = run_segment(
                        env, client, obs_b, t_trig, max_total, desc,
                        grab_chunk if args.save_pairs_dir else None,
                    )
                    successes += int(ok)
                    if args.save_pairs_dir:
                        branch_records.append({
                            "success": bool(ok),
                            "first_chunk": first_chunk.get("chunk"),
                        })
                    print(
                        f"[task {task_id} ep {ep}]   branch {m + 1}/{args.branches}: "
                        f"{'success' if ok else 'fail'}",
                        flush=True,
                    )
                record.update(
                    arm=trigger["arm"],
                    trigger_call=int(trigger["call"]),
                    trigger_env_step=int(t_trig),
                    unc_at_trigger=trigger["unc"],
                    branches=int(args.branches),
                    branch_successes=int(successes),
                )
                if args.save_pairs_dir:
                    record["snapshot_file"] = snapshot_file
                    record["branch_records"] = branch_records
                arm_stats[trigger["arm"]].append(successes / args.branches)

            with open(args.log, "a") as f:
                f.write(json.dumps(record) + "\n")

            tag = record["arm"]
            extra_info = (
                f"branch_success={record.get('branch_successes')}/{args.branches}"
                if tag in ("spike", "control")
                else f"success={record.get('success')}"
            )
            print(f"[task {task_id} ep {ep}] arm={tag} {extra_info}")

        env.close()

    print("\n=== Branching counterfactual summary ===")
    for arm in ("spike", "control"):
        rates = arm_stats[arm]
        if rates:
            mean = sum(rates) / len(rates)
            zero = sum(1 for r in rates if r == 0)
            full = sum(1 for r in rates if r == 1)
            print(f"{arm:>8}: n={len(rates)}  mean_branch_success={mean:.3f}  "
                  f"all-fail={zero}/{len(rates)}  all-succeed={full}/{len(rates)}")
        else:
            print(f"{arm:>8}: n=0")
    print(f"finished before any trigger: {finished_before_trigger}")


if __name__ == "__main__":
    main()
