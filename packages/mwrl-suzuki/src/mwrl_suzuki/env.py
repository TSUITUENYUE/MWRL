"""The MWRL outer MDP over condition dimensions (paper Definition 1), amortized over substrates.

The policy walks the subset lattice of condition dimensions with actions {add(d), remove(d),
stop}. The episode starts at the full set S_0 = all allowed dimensions (tuning everything,
which is a witness whenever the task has any measured success), and the policy removes toward
the minimal control sets. On stop the black-box verifier is queried: ``s_exists`` returns the
success bit and the raw opened set (the coverage contract leaves minimality to the policy;
the verifier never prunes). The observation carries the reaction
fingerprint, so one policy is conditioned per substrate. Mirrors ``mwrl_circuits.CircuitEnv``;
the only difference is that the fingerprint is a fixed-width reaction descriptor rather than a
per-component vector, so ``num_obs = n_comp + 1 + fp_dim``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mwrl.env import StepResult

from mwrl_suzuki.verifier import ChemVerifier


def _bits(mask: int, n: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)


class ChemEnv:
    def __init__(self, verifiers: Sequence[ChemVerifier], *, n_comp: int, fp_dim: int, horizon: int) -> None:
        self.verifiers = list(verifiers)
        self.n_comp = n_comp
        self.fp_dim = fp_dim
        self.horizon = horizon
        self.num_actions = 2 * n_comp + 1     # add d (0..n-1), remove d (n..2n-1), STOP (2n)
        self.num_obs = n_comp + 1 + fp_dim    # opened multi-hot + step fraction + reaction fingerprint
        self.current: ChemVerifier | None = None
        self.active = np.zeros(n_comp, dtype=bool)
        self.allowed = np.zeros(n_comp, dtype=bool)
        self.step_count = 0

    @property
    def contexts(self) -> Sequence[ChemVerifier]:
        return self.verifiers

    def reset(self, context: ChemVerifier) -> tuple[np.ndarray, np.ndarray]:
        self.current = context
        self.allowed = _bits(context.allowed, self.n_comp)
        self.active = self.allowed.copy()     # S_0 = tune every allowed dimension
        self.step_count = 0
        return self._obs(), self._action_mask()

    def _obs(self) -> np.ndarray:
        assert self.current is not None
        return np.concatenate(
            [self.active.astype(np.float32), [self.step_count / max(self.horizon, 1)], self.current.attr_features]
        ).astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        return np.concatenate([self.allowed & ~self.active, self.allowed & self.active, [True]])

    def _witness(self) -> int:
        mask = 0
        for i in np.nonzero(self.active & self.allowed)[0]:
            mask |= 1 << int(i)
        return mask

    def step(self, action: int) -> StepResult:
        assert self.current is not None
        stop = action == 2 * self.n_comp
        if not stop:
            if action < self.n_comp:
                self.active[action] = True                    # add(d): open a dimension
            else:
                self.active[action - self.n_comp] = False     # remove(d): close a dimension
            self.step_count += 1
            if self.step_count >= self.horizon:
                stop = True
        if stop:
            witness = self._witness()
            found, canon = self.current.s_exists(witness)
            return StepResult(
                self._obs(), self._action_mask(), float(found), True,
                witness=witness, success=found, info={"canon_mask": canon},
            )
        return StepResult(self._obs(), self._action_mask(), 0.0, False)
