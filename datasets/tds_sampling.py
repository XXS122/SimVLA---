"""
TDS (Transition-Density Sampling) — streaming integration for SimVLA.

The offline difficulty weights p_k come from `transition_density_stats.py`
(`task_difficulty.csv`).  SimVLA streams training data from an infinite
IterableDataset, so PyTorch's `WeightedRandomSampler` (map-style datasets
only) cannot be used here.  Instead we use two-level sampling: every yield
first draws a task file with probability p_k, then takes the *next* sample
from that task's persistent stream (see `SmolVLMDataReader._iter_tds`).

Because the weighted draw happens once per sample (not once per file or
episode), the fraction of training samples coming from task k converges to
p_k exactly — no per-sample weight spreading (p_k / n_k) is needed, and
tasks with unequal demo counts or episode lengths are handled for free.

Setting all p_k equal recovers the original uniform stream (up to RNG),
so TDS off / uniform csv are both exact baselines.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Sequence


def load_task_weights(csv_path: str) -> Dict[str, float]:
    """Read transition_density_stats.py output -> {task_name: p_k}.

    task_name is the hdf5 file stem (e.g. ``KITCHEN_SCENE3_..._demo``),
    matching `file_stem()` of the meta datalist paths.
    """
    weights: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            weights[row["task_name"]] = float(row["p_k"])
    if not weights:
        raise ValueError(f"No task weights found in {csv_path}")
    return weights


def file_stem(path: str) -> str:
    """`/a/b/KITCHEN_SCENE3_..._demo.hdf5` -> `KITCHEN_SCENE3_..._demo`."""
    return os.path.splitext(os.path.basename(path))[0]


def resolve_stem_weights(stems: Sequence[str],
                         task_pk: Dict[str, float]) -> List[float]:
    """Align p_k with a list of task-file stems, normalized to sum to 1.

    Files missing from the csv (e.g. libero_90 when stats were only computed
    on the 4 eval suites) fall back to the median p_k, so they are sampled
    like an average-difficulty task rather than starved or boosted.
    """
    sorted_pk = sorted(task_pk.values())
    median = sorted_pk[len(sorted_pk) // 2]
    weights, missing = [], []
    for stem in stems:
        pk = task_pk.get(stem)
        if pk is None:
            missing.append(stem)
            pk = median
        weights.append(pk)
    if missing:
        print(f"[TDS] {len(missing)}/{len(stems)} task files have no entry in "
              f"the weights csv; using median p_k for e.g. {missing[:3]}")
    total = sum(weights)
    return [w / total for w in weights]


def mix_with_uniform(task_pk: Dict[str, float], t: float) -> Dict[str, float]:
    """Curriculum ablation helper: linear interpolation uniform -> TDS.

    t=0 gives uniform, t=1 gives the full TDS weights.  For the annealed
    ablation, restart training stages with intermediate csv weights (the
    streaming reader takes a fixed distribution per run).
    """
    n = len(task_pk)
    return {k: (1 - t) / n + t * v for k, v in task_pk.items()}


class SamplingMonitor:
    """Sanity check that TDS is actually in effect during training.

    Sampling bugs are silent: a broken stem mapping degrades to uniform and
    nothing crashes.  This monitor counts draws per task and, after
    `check_at` samples, prints empirical frequency vs target p_k once and
    warns if the largest absolute deviation exceeds `tol`.
    """

    def __init__(self, targets: Dict[str, float], check_at: int = 5000,
                 tol: float = 0.02):
        self.targets = targets
        self.check_at = check_at
        self.tol = tol
        self.counts: Dict[str, int] = {}
        self.n = 0
        self.done = False

    def record(self, stem: str):
        if self.done:
            return
        self.counts[stem] = self.counts.get(stem, 0) + 1
        self.n += 1
        if self.n >= self.check_at:
            self._report()
            self.done = True

    def _report(self):
        rows = []
        for stem, pk in sorted(self.targets.items(), key=lambda kv: -kv[1]):
            emp = self.counts.get(stem, 0) / self.n
            rows.append((stem, pk, emp, abs(pk - emp)))
        max_err = max(r[3] for r in rows)
        print(f"[TDS] sampling check after {self.n} draws "
              f"(max |target - empirical| = {max_err:.4f}):")
        for stem, pk, emp, err in rows[:10]:
            print(f"  {stem[:60]:60s} target={pk:.4f} empirical={emp:.4f}")
        if max_err > self.tol:
            print(f"[TDS][WARNING] empirical sampling deviates from target by "
                  f"{max_err:.4f} > {self.tol} — check the stem->weight "
                  f"mapping in the weights csv vs the meta datalist!")
        else:
            print("[TDS] sampling distribution matches target. OK")


__all__ = [
    "load_task_weights",
    "file_stem",
    "resolve_stem_weights",
    "mix_with_uniform",
    "SamplingMonitor",
]
