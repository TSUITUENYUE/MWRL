"""The MWRL outer MDP for circuits (paper Definition 1), amortized over behaviors.

The outer policy picks an allowed variable space S (a subset of the components D(c)); the
black-box rollout runs inside S and reports s_c(S), whether a faithful sub-circuit exists there.
Per Definition 1 the actions are {add(d), remove(d), stop}: add opens a component, remove closes
one, stop terminates and calls the black-box verifier on the terminal set S_T. The reward is
r_T = s_c(S_T) (Assumption 1: s_c is monotone in S), and the training objective is the group
coverage objective with the leave-one-out credit, so the coverage weight nu(up S)=2^-|S| is what
pushes the policy to the minimal witnesses M(c); there is no size budget.

At the terminal, s_exists both returns the success s_c(S_T) and canonicalizes S_T to its minimal
witness Phi (the dropout escapes the non-monotone s0 local minima that the policy's single-step
removals get stuck at). The count valuation credits distinct Phi, so the policy is rewarded for
opening sets that canonicalize to different minimal circuits -- the antichain. The episode starts
at the full set S_0 = D(c) (faithful, so the group always has certified witnesses); the policy
removes toward the circuits and add lets it backtrack. H caps the episode length.

A context is a behavior's ``CircuitVerifier``; a list of them is cross-behavior amortization,
and the observation carries the behavior's EAP fingerprint so the amortized policy is conditioned.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from mwrl.env import StepResult

from mwrl_circuits.verifier import CircuitVerifier


def _bits(mask: int, n: int) -> np.ndarray:
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=bool)


class CircuitEnv:
    def __init__(self, verifiers: Sequence[CircuitVerifier], *, n_comp: int, horizon: int) -> None:
        self.verifiers = list(verifiers)
        self.n_comp = n_comp
        self.horizon = horizon                # H: max add/remove steps per episode
        self.num_actions = 2 * n_comp + 1     # add d (0..n-1), remove d (n..2n-1), STOP (2n)
        self.num_obs = 2 * n_comp + 1         # opened multi-hot + step fraction + task attributions
        self.current: CircuitVerifier | None = None
        self.active = np.zeros(n_comp, dtype=bool)
        self.allowed = np.zeros(n_comp, dtype=bool)
        self.step_count = 0
        self.defer_terminal = False   # when set, the terminal step returns S_T unverified so the
        # runner can batch the group's s_exists calls into one search (see Runner._rollout_group)

    @property
    def contexts(self) -> Sequence[CircuitVerifier]:
        return self.verifiers

    def reset(self, context: CircuitVerifier) -> tuple[np.ndarray, np.ndarray]:
        self.current = context
        self.allowed = _bits(context.allowed, self.n_comp)
        self.active = self.allowed.copy()     # S_0 = D(c): the full set, guaranteed faithful
        self.step_count = 0
        return self._obs(), self._action_mask()

    def _obs(self) -> np.ndarray:
        # conditioned on the context c via its EAP attributions (the per-component task
        # fingerprint), so the amortized policy is task-specific, not one policy per behavior.
        return np.concatenate([self.active.astype(np.float32),
                               [self.step_count / max(self.horizon, 1)],
                               self.current.attr_features]).astype(np.float32)

    def _action_mask(self) -> np.ndarray:
        # open an unopened component (add), close an opened one (remove), or STOP.
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
                self.active[action] = True             # add(d): open component d
            else:
                self.active[action - self.n_comp] = False   # remove(d): close component d
            self.step_count += 1
            if self.step_count >= self.horizon or int(self.active.sum()) <= 1:
                stop = True
        if stop:
            # stop calls the black-box verifier on the opened set S_T. s_exists both returns the
            # success s_c(S_T) and canonicalizes S_T to its minimal witness Phi (the dropout escapes
            # the non-monotone s0 traps the policy's single-step removals get stuck at). The count
            # valuation credits distinct Phi, so the recorded witness is the minimal circuit, not S_T.
            witness = self._witness()
            if self.defer_terminal:                        # defer verification: the runner will
                return StepResult(self._obs(), self._action_mask(), 0.0, True,   # batch-verify S_T
                                  witness=witness, success=False, info={"deferred": True})
            found, canon = self.current.s_exists(witness)
            return StepResult(self._obs(), self._action_mask(), float(found), True,
                              witness=witness, success=found, info={"canon_mask": canon})
        return StepResult(self._obs(), self._action_mask(), 0.0, False)
