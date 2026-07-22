"""Minimal-Witness Reinforcement Learning: an amortized-RL library (rsl_rl-style).

The problem is *amortized* minimal-witness identification (theory Def 2.3): learn one
policy ``pi_theta(.|c)`` over a context distribution ``c ~ D`` that, for a new context,
proposes its antichain of minimal sufficient witnesses with no extra search. This
package is the library: the up-set l1o/l2o credit, the amortized grouped
``Runner``, the ``Actor`` module, and the ``MWRLEnv`` interface. A benchmark brings its
own environment (any action/observation space); the runner only needs, at a terminal
step, the witness ``S`` and its success ``s_c(S)``. The scalar baselines (maxrl / grpo /
rloo) ride the same runner as other advantages; the non-amortized baselines (GFlowNet for
MaxSAT, ACDC/EAP for circuits) live with their benchmark. The MaxSAT, circuits, and Suzuki
benchmarks depend only on this package.
"""

from mwrl.advantage import mwrl_group_advantages
from mwrl.baselines import (
    grouped_outcome_advantages,
    grpo_group_advantages,
    maxrl_group_advantages,
    rloo_group_advantages,
)
from mwrl.config import GroupedRLConfig, MWRLConfig
from mwrl.credit import (
    count_advantages,
    coverage_advantages,
    coverage_l2o_advantages,
    nu_upset_layer,
    nu_upset_uniform,
    popcount,
)
from mwrl.env import MWRLEnv, StepResult
from mwrl.modules import Actor
from mwrl.runner import Rollout, Runner

__all__ = [
    # the amortized library: env interface, actor, runner, configs
    "MWRLEnv", "StepResult", "Actor", "Runner", "Rollout", "GroupedRLConfig", "MWRLConfig",
    # the MWRL credit (the method) and the scalar baseline advantages
    "mwrl_group_advantages", "coverage_advantages", "coverage_l2o_advantages",
    "count_advantages", "nu_upset_layer", "nu_upset_uniform", "popcount",
    "maxrl_group_advantages", "grpo_group_advantages", "rloo_group_advantages",
    "grouped_outcome_advantages",
]
