"""The env interface the amortized MWRL runner drives (rsl_rl-style).

A benchmark brings its own environment with whatever observation and action space it
wants (open/close a variable, add a circuit component, edge-level, hierarchical, ...),
and the runner stays agnostic to it. The runner needs three things from an env:

  * a **context distribution** ``contexts`` to sample ``c ~ D`` from. This is the
    amortization (Def 2.3 / 3.4): one policy ``pi_theta(.|c)`` covers the whole
    distribution.
  * flat ``(obs, action_mask)`` at every step, so the actor can act;
  * at a terminal step, the **witness** ``S`` (a bitmask over that context's variables)
    and its success ``s_c(S)``, the one MWRL-specific hook. From the group's terminal
    witnesses the runner builds the certified up-set ``U_K`` and the leave-one-out credit
    ``A_i = V(U_K) - V(U_{-i})`` (theory section 4.4).

How the env reaches ``S`` (the MDP, the action space, the observation) is the benchmark's
business. The verifier ``s_c`` and any canonicalizer ``Phi`` live inside the env's reward
channel, and no gradient flows through them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class StepResult:
    """One env transition. ``witness`` and ``success`` are meaningful only when ``done``."""

    obs: np.ndarray            # flat observation vector, shape (num_obs,)
    action_mask: np.ndarray    # bool, shape (num_actions,); False = illegal action
    reward: float
    done: bool
    witness: int = 0           # terminal witness S as a bitmask over the context's variables
    success: bool = False      # s_c(S): did the verifier accept the witness?
    info: dict[str, Any] = field(default_factory=dict)  # e.g. canon_mask, diagnostics


@runtime_checkable
class MWRLEnv(Protocol):
    """The contract a benchmark environment satisfies for the MWRL runner."""

    num_obs: int
    num_actions: int
    contexts: Sequence[Any]    # the distribution D; the runner samples c ~ D each update

    def reset(self, context: Any) -> tuple[np.ndarray, np.ndarray]:
        """Start an episode for context ``c``. Returns ``(obs, action_mask)``."""
        ...

    def step(self, action: int) -> StepResult:
        """Advance one step under ``action`` and return the transition."""
        ...
