"""
Unified pipeline driver: `python -m latent_action.run <stage> [options]`.

Reads paths/GPU config from environment variables (source paths.env first):
  SIMVLA_SMOLVLM_MODEL  SmolVLM backbone path
  LIBERO_DATASETS       LIBERO dataset root
  SIMVLA_CHECKPOINTS    workspace root (metas / lam / z labels / runs)
  SIMVLA_RESUME_CKPT    optional checkpoint to resume training stages from
  WANDB_API_KEY/PROJECT wandb logging (empty key = disabled)
  CUDA_DEVICES          value for CUDA_VISIBLE_DEVICES
  NUM_GPUS              >1 launches training stages via `accelerate launch`
  SIMVLA_Z_DIM          latent dimension (default 16)

Stages
------
  meta        scan $LIBERO_DATASETS, write libero_train.json
  norm-stats  compute norm stats JSON
  splits      demo-level low-data splits (p1 / p10)
  baseline    SimVLA baseline (libero_joint) on a split
  train-lam   latent action model training      [--use_vq for ablation]
  label       write z labels (+ pretraining meta)  [--ego_aug for stress]
  probe       affine probe R^2 + adapter init      [--stress]
  pretrain    flow-expert pretraining on z chunks (action_mode latent_z)
  finetune    downstream fine-tune (libero_z_adapter) [--from_scratch]
  serve       LIBERO evaluation policy server

Unknown options are forwarded to the underlying stage script, so any
train_smolvlm.py flag works, e.g.:
  python -m latent_action.run finetune --split p10 --seed 1 --iters 80000
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Apply GPU selection before torch is imported anywhere.
_cuda = os.environ.get("CUDA_DEVICES", "")
if _cuda:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", _cuda)

from latent_action import config as C  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

SPLIT_ITERS = {"p1": 20000, "p10": 60000, "p100": 200000}


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> None:
    print("+ " + " ".join(str(c) for c in cmd))
    env = dict(os.environ)
    if _cuda:
        env["CUDA_VISIBLE_DEVICES"] = _cuda
    env["SIMVLA_Z_DIM"] = str(C.Z_DIM)
    res = subprocess.run([str(c) for c in cmd], cwd=str(cwd), env=env)
    if res.returncode != 0:
        sys.exit(res.returncode)


def _train_launcher() -> list[str]:
    if C.NUM_GPUS > 1:
        return ["accelerate", "launch", "--num_processes", str(C.NUM_GPUS),
                "--mixed_precision", "no"]
    return [sys.executable]


def _train_common(ws: C.Workspace, output_dir: Path, run_name: str) -> list[str]:
    args = [
        "--output_dir", output_dir,
        "--smolvlm_model_path", C.SMOLVLM_MODEL,
        "--norm_stats_path", ws.norm_stats_path,
        "--num_actions", 10,
        "--image_size", 384,
        "--hidden_size", 768, "--depth", 12, "--num_heads", 12,
        "--learning_rate", 1e-4, "--learning_coef", 0.1,
        "--freeze_steps", 1000, "--warmup_steps", 0,
        "--batch_size", 64, "--num_workers", 4,
        "--max_grad_norm", 1.0,
        "--save_interval", 10000, "--log_interval", 20,
        "--z_dim", C.Z_DIM,
        "--run_name", run_name,
    ]
    if C.RESUME_CKPT:
        args += ["--models", C.RESUME_CKPT, "--resume"]
    return args


# ============================================================
# Stage implementations
# ============================================================
def stage_meta(ws: C.Workspace, extra: list[str], subsets: list[str]):
    _run([sys.executable, "create_libero_meta.py",
          "--data_dir", C.LIBERO_DATASETS,
          "--subsets", *subsets,
          "--output", ws.meta_path("p100")] + extra)


def stage_norm_stats(ws: C.Workspace, extra: list[str], subsets: list[str]):
    _run([sys.executable, "compute_libero_norm_stats.py",
          "--data_dir", C.LIBERO_DATASETS,
          "--subsets", *subsets,
          "--output", ws.norm_stats_path] + extra)


def stage_splits(ws: C.Workspace, extra: list[str]):
    _run([sys.executable, "-m", "latent_action.make_splits",
          "--meta_path", ws.meta_path("p100"),
          "--fractions", 0.01, 0.1] + extra)


def stage_baseline(ws: C.Workspace, extra: list[str], split: str, iters: int | None):
    name = f"baseline_{split}"
    cmd = _train_launcher() + ["train_smolvlm.py"] + _train_common(ws, ws.run_dir(name), name) + [
        "--train_metas_path", ws.meta_path(split),
        "--action_mode", "libero_joint",
        "--iters", iters or SPLIT_ITERS.get(split, 200000),
    ] + extra
    _run(cmd)


def stage_train_lam(ws: C.Workspace, extra: list[str], iters: int | None):
    cmd = [sys.executable, "-m", "latent_action.train_lam",
           "--meta_path", ws.meta_path("p100"),
           "--output_dir", ws.lam_dir]
    if iters:
        cmd += ["--iters", iters]
    _run(cmd + extra)


def stage_label(ws: C.Workspace, extra: list[str], ego_aug: bool):
    cmd = [sys.executable, "-m", "latent_action.label_z",
           "--meta_path", ws.meta_path("p100"),
           "--lam_ckpt", ws.lam_ckpt,
           "--output", ws.z_labels_path(stress=ego_aug)]
    if ego_aug:
        cmd += ["--ego_aug"]
    else:
        cmd += ["--z_meta_out", ws.z_meta_path("p100")]
    _run(cmd + extra)


def stage_probe(ws: C.Workspace, extra: list[str], stress: bool):
    _run([sys.executable, "-m", "latent_action.probe",
          "--meta_path", ws.meta_path("p100"),
          "--z_labels", ws.z_labels_path(stress=stress),
          "--norm_stats_path", ws.norm_stats_path,
          "--output", ws.probe_path(stress=stress)] + extra)


def stage_pretrain(ws: C.Workspace, extra: list[str], iters: int | None):
    name = "pretrain_flow"
    cmd = _train_launcher() + ["train_smolvlm.py"] + _train_common(ws, ws.run_dir(name), name) + [
        "--train_metas_path", ws.z_meta_path("p100"),
        "--action_mode", "latent_z",
        "--iters", iters or 100000,
    ] + extra
    _run(cmd)


def stage_finetune(ws: C.Workspace, extra: list[str], split: str,
                   iters: int | None, from_scratch: bool):
    name = f"finetune_{split}" + ("_scratch" if from_scratch else "_pretrained")
    cmd = _train_launcher() + ["train_smolvlm.py"] + _train_common(ws, ws.run_dir(name), name) + [
        "--train_metas_path", ws.meta_path(split),
        "--action_mode", "libero_z_adapter",
        "--iters", iters or SPLIT_ITERS.get(split, 200000),
    ]
    probe = ws.probe_path()
    if probe.exists():
        cmd += ["--probe_path", probe]
    elif not from_scratch:
        print(f"warning: probe not found at {probe}; adapter will be random-init")
    if not from_scratch:
        ckpt = C.latest_checkpoint(ws.run_dir("pretrain_flow"))
        if ckpt is None:
            sys.exit("no pretraining checkpoint found — run `pretrain` first "
                     "or pass --from_scratch")
        cmd += ["--pretrained_flow_ckpt", ckpt]
    _run(cmd + extra)


def stage_serve(ws: C.Workspace, extra: list[str], ckpt: str | None):
    if ckpt is None:
        sys.exit("serve requires --ckpt <checkpoint dir>")
    _run([sys.executable, "evaluation/libero/serve_smolvlm_libero.py",
          "--checkpoint", ckpt,
          "--norm_stats", ws.norm_stats_path,
          "--smolvlm_model", C.SMOLVLM_MODEL] + extra)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        "latent_action pipeline",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("stage", choices=[
        "meta", "norm-stats", "splits", "baseline", "train-lam",
        "label", "probe", "pretrain", "finetune", "serve",
    ])
    parser.add_argument("--split", type=str, default="p100",
                        help="data split for baseline/finetune (p1|p10|p100)")
    parser.add_argument("--iters", type=int, default=None,
                        help="override default iterations for training stages")
    parser.add_argument("--subsets", type=str, nargs="+",
                        default=C.DEFAULT_SUBSETS,
                        help="LIBERO subsets for meta/norm-stats")
    parser.add_argument("--ego_aug", action="store_true", default=False,
                        help="label: write stress-test z labels")
    parser.add_argument("--stress", action="store_true", default=False,
                        help="probe: evaluate on stress-test z labels")
    parser.add_argument("--from_scratch", action="store_true", default=False,
                        help="finetune: skip loading pretrained flow weights")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="serve: checkpoint dir to serve")
    args, extra = parser.parse_known_args()

    ws = C.Workspace().ensure()
    print(f"workspace: {ws.root}")
    print(f"backbone:  {C.SMOLVLM_MODEL}")
    print(f"data:      {C.LIBERO_DATASETS}")
    print(f"z_dim:     {C.Z_DIM}, gpus: {C.NUM_GPUS} (CUDA_DEVICES='{_cuda}')")

    if args.stage == "meta":
        stage_meta(ws, extra, args.subsets)
    elif args.stage == "norm-stats":
        stage_norm_stats(ws, extra, args.subsets)
    elif args.stage == "splits":
        stage_splits(ws, extra)
    elif args.stage == "baseline":
        stage_baseline(ws, extra, args.split, args.iters)
    elif args.stage == "train-lam":
        stage_train_lam(ws, extra, args.iters)
    elif args.stage == "label":
        stage_label(ws, extra, args.ego_aug)
    elif args.stage == "probe":
        stage_probe(ws, extra, args.stress)
    elif args.stage == "pretrain":
        stage_pretrain(ws, extra, args.iters)
    elif args.stage == "finetune":
        stage_finetune(ws, extra, args.split, args.iters, args.from_scratch)
    elif args.stage == "serve":
        stage_serve(ws, extra, args.ckpt)


if __name__ == "__main__":
    main()
