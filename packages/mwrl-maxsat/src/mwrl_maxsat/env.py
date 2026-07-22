"""MaxSAT environment for the core MWRL runner: the add-a-variable MDP.

Actions are "add variable i" or STOP; the terminal witness is the set of added
variables, and success is the clause-threshold predicate. No auto-terminate, so a
size-indifferent policy can overshoot minimality (fill to the horizon), exactly as in
the paper. Contexts are MaxSAT instances; with a one-instance list this reduces to the
per-instance setting, and with many instances the same runner amortizes over them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mwrl.env import StepResult

from mwrl_maxsat.instance import MaxSatInstance, canonicalize, is_witness


class MaxSatEnv:
    def __init__(self, instances: Sequence[MaxSatInstance], *, d_max: int, horizon: int,
                 allow_remove: bool = False, canonicalize_witness: bool = False,
                 seed: int = 0) -> None:
        self.instances = list(instances)
        self.d_max = d_max
        self.horizon = horizon
        self.allow_remove = allow_remove   # {add, remove, stop} vs the default {add, stop}
        # count valuation's Phi: greedy-prune the terminal witness to a minimal one. The prune order
        # is seeded PER MASK, so Phi is a deterministic function of the proposal S -- the antichain
        # diversity must then come from the POLICY proposing different S, not from the canonicalizer's
        # dice. (A fresh RNG per call would let any policy, PPO included, inherit the coverage.)
        self.canonicalize_witness = canonicalize_witness
        self._seed = seed
        # actions: add 0..d_max-1, [remove d_max..2*d_max-1 if allow_remove], STOP last.
        self.stop_action = 2 * d_max if allow_remove else d_max
        self.num_actions = self.stop_action + 1
        self.num_obs = d_max + 1       # current mask multi-hot + step fraction
        self.current: MaxSatInstance | None = None
        self.mask = 0
        self.step_count = 0

    @property
    def contexts(self) -> Sequence[MaxSatInstance]:
        return self.instances

    def reset(self, context: MaxSatInstance) -> tuple[np.ndarray, np.ndarray]:
        self.current = context
        self.mask = 0
        self.step_count = 0
        return self._obs(), self._action_mask()

    def _obs(self) -> np.ndarray:
        m = np.array([(self.mask >> i) & 1 for i in range(self.d_max)], dtype=np.float32)
        return np.concatenate([m, [self.step_count / max(self.horizon, 1)]]).astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        am = np.zeros(self.num_actions, dtype=bool)
        assert self.current is not None
        for i in range(self.current.n_vars):
            if not ((self.mask >> i) & 1):
                am[i] = True                  # add a variable not yet added
            elif self.allow_remove:
                am[self.d_max + i] = True      # remove an added variable (backtrack)
        am[self.stop_action] = True            # STOP is always legal
        return am

    def step(self, action: int) -> StepResult:
        assert self.current is not None
        stop = action == self.stop_action
        if not stop:
            if action < self.d_max:
                self.mask |= 1 << action                       # add variable
            else:
                self.mask &= ~(1 << (action - self.d_max))     # remove variable
            self.step_count += 1
            if self.step_count >= self.horizon:
                stop = True
        if stop:
            success = is_witness(self.current, self.mask)
            info: dict = {}
            if success and self.canonicalize_witness:      # count valuation: Phi = per-mask prune
                rng = np.random.default_rng([self._seed, self.mask])   # deterministic in the mask
                order = rng.permutation(self.current.n_vars).tolist()
                info["canon_mask"] = canonicalize(self.current, self.mask, order=order)
            return StepResult(self._obs(), self._action_mask(), float(success), True,
                              witness=self.mask, success=success, info=info)
        return StepResult(self._obs(), self._action_mask(), 0.0, False)
