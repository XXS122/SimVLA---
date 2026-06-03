"""
Inference-side adaptive computation for SimVLA (no retraining required).

Uses the per-step *boundary scores* produced by the trained ChunkBoundaryHead
(``SmolVLMVLA.generate_actions(..., return_boundary=True)``) to drive
variable-length open-loop execution: commit long open-loop segments during
smooth motion and re-plan (re-run the expensive VLM) only around decisive /
contact moments. Because most of a manipulation trajectory is smooth, this cuts
the number of VLM calls — the dominant inference cost — without losing accuracy.

The controller is framework-agnostic (pure Python): it stores whatever action
array the policy returns and only reads scalar boundary values, so it can be
unit-tested with synthetic scores. Run ``python models/adaptive_inference.py``
for a self-test.
"""

from __future__ import annotations

from typing import Any, Callable, List, Tuple


def _to_float_list(boundary: Any) -> List[float]:
    """Flatten a torch.Tensor / numpy array / sequence of scores to python floats."""
    if hasattr(boundary, "detach"):  # torch.Tensor
        boundary = boundary.detach().to("cpu").reshape(-1).tolist()
    elif hasattr(boundary, "reshape") and not isinstance(boundary, (list, tuple)):  # numpy
        boundary = boundary.reshape(-1).tolist()
    return [float(x) for x in boundary]


class AdaptiveReplanController:
    """Boundary-driven variable-length re-planning.

    At each control step it returns the next action to execute; it only re-runs
    the policy (the expensive VLM forward) when the committed open-loop segment is
    exhausted. The committed length is chosen from the chunk's per-step boundary
    scores: commit up to (but not including) the first *decisive* step.

    Parameters
    ----------
    beta : float
        Decisiveness threshold. Re-plan just before the first step whose boundary
        score exceeds ``beta``. Lower => re-plan more often (safer, slower);
        higher => commit longer (faster, riskier). Sweeping ``beta`` traces the
        success-vs-compute Pareto curve.
    max_horizon : int
        Hard cap on open-loop length; never commit more than this many steps even
        if the chunk looks entirely smooth (guards against a missed decisive
        moment / boundary mis-prediction).
    min_commit : int
        Always execute at least this many steps before re-planning (avoids
        thrashing, i.e. re-planning every single step in a borderline region).
    mode : str
        ``"adaptive"`` (ours) or ``"fixed"`` (baseline: always commit ``fixed_n``).
    fixed_n : int | None
        Commit length for the ``"fixed"`` baseline (defaults to ``max_horizon``).
    """

    def __init__(
        self,
        beta: float = 0.5,
        max_horizon: int = 10,
        min_commit: int = 1,
        mode: str = "adaptive",
        fixed_n: "int | None" = None,
    ) -> None:
        assert mode in ("adaptive", "fixed"), f"unknown mode {mode!r}"
        assert min_commit >= 1, "min_commit must be >= 1"
        assert max_horizon >= min_commit, "max_horizon must be >= min_commit"
        self.beta = float(beta)
        self.max_horizon = int(max_horizon)
        self.min_commit = int(min_commit)
        self.mode = mode
        self.fixed_n = int(fixed_n) if fixed_n is not None else None
        self.reset()

    def reset(self) -> None:
        """Clear state at the start of an episode."""
        self._chunk: Any = None
        self._commit_len: int = 0
        self._ptr: int = 0
        self.num_replans: int = 0           # == number of VLM forwards this episode
        self.committed_lengths: List[int] = []

    def decide_commit_length(self, boundary: Any) -> int:
        """How many steps to execute open-loop given the chunk's boundary scores."""
        b = _to_float_list(boundary)
        T = len(b)
        if T == 0:
            return 1
        cap = max(1, min(self.max_horizon, T))
        if self.mode == "fixed":
            n = self.fixed_n if self.fixed_n is not None else cap
            return max(1, min(n, T))
        # Adaptive: commit up to (but not including) the first decisive step,
        # searching only at indices >= min_commit, capped at max_horizon. If no
        # decisive step is found within the cap, commit the whole capped chunk.
        for i in range(self.min_commit, cap):
            if b[i] > self.beta:
                return i
        return cap

    def act(self, obs: Any, policy_fn: Callable[[Any], Tuple[Any, Any]]) -> Any:
        """Return the next action to execute, re-planning via ``policy_fn`` when needed.

        ``policy_fn(obs)`` must return ``(actions, boundary)`` where ``actions`` is
        indexable along time (``actions[t]``) and ``boundary`` is a length-T
        sequence of per-step scores.
        """
        if self._chunk is None or self._ptr >= self._commit_len:
            actions, boundary = policy_fn(obs)
            self._chunk = actions
            self._commit_len = self.decide_commit_length(boundary)
            self._ptr = 0
            self.num_replans += 1
            self.committed_lengths.append(self._commit_len)
        action = self._chunk[self._ptr]
        self._ptr += 1
        return action


# ---------------------------------------------------------------------------
# Self-test: `python models/adaptive_inference.py`
# Pure-Python, no torch/numpy needed — validates the controller logic and shows
# the VLM-call savings on a scripted "smooth transit + 2 decisive regions" run.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import math

    # --- unit checks on decide_commit_length ---
    c = AdaptiveReplanController(beta=0.5, max_horizon=10, min_commit=1)
    assert c.decide_commit_length([0.1] * 10) == 10              # all smooth -> commit whole
    assert c.decide_commit_length([0.1, 0.1, 0.1, 0.9, 0.2]) == 3  # decisive at idx 3
    assert c.decide_commit_length([0.9, 0.9, 0.9]) == 1          # sustained decisive -> dense re-plan (min_commit)
    assert c.decide_commit_length([0.1] * 20) == 10              # capped by max_horizon
    cf = AdaptiveReplanController(mode="fixed", fixed_n=5, max_horizon=10)
    assert cf.decide_commit_length([0.1] * 10) == 5
    print("[ok] decide_commit_length unit checks passed")

    # --- scripted episode: smooth transit with decisive spikes at grasp & place ---
    H, Tlen = 10, 40
    def true_score(i: int) -> float:
        grasp = math.exp(-((i - 19) ** 2) / 4.0)   # decisive around step ~19
        place = math.exp(-((i - 36) ** 2) / 4.0)   # decisive around step ~36
        return max(grasp, place)
    truth = [true_score(i) for i in range(Tlen + H)]

    def policy_fn(step: int) -> Tuple[list, list]:
        actions = [[float(step + k)] for k in range(H)]          # identifiable dummy actions
        boundary = [truth[step + k] for k in range(H)]           # predicted chunk profile
        return actions, boundary

    def run(controller: AdaptiveReplanController):
        controller.reset()
        step = 0
        while step < Tlen:
            controller.act(step, policy_fn)   # obs == current global step
            step += 1
        return controller.num_replans, list(controller.committed_lengths)

    n_adaptive, lens = run(AdaptiveReplanController(beta=0.4, max_horizon=10, min_commit=1))
    n_fixed1, _ = run(AdaptiveReplanController(mode="fixed", fixed_n=1, max_horizon=10))
    n_fixed10, _ = run(AdaptiveReplanController(mode="fixed", fixed_n=10, max_horizon=10))

    print(f"[sim] VLM calls -- fixed(N=1, re-plan every step): {n_fixed1}")
    print(f"[sim] VLM calls -- fixed(N=10, whole chunk):       {n_fixed10}")
    print(f"[sim] VLM calls -- adaptive (ours, beta=0.4):      {n_adaptive}")
    print(f"[sim] adaptive committed lengths per re-plan:      {lens}")
    assert n_fixed10 <= n_adaptive <= n_fixed1, "adaptive should sit between the two baselines"
    print("[ok] adaptive re-plans far less than every-step, denser than whole-chunk")
    print("[ok] all self-tests passed")
