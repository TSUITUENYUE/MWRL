"""Core grouped-RL / MWRL configuration (benchmark-agnostic).

Holds the up-set credit knobs (``witness_*``) shared by every benchmark. Domain
configs (data, reward, chem optimizers, experiment wiring) live with their benchmark,
not here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt, field_validator


class GroupedRLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    updates: PositiveInt = 5
    task_batch_size: PositiveInt = 4
    group_size: PositiveInt = 4
    learning_rate: float = Field(default=8e-4, gt=0.0)
    clip_epsilon: float = Field(default=0.2, gt=0.0)
    clip_ratio_c: float = Field(default=3.0, gt=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=0.5, gt=0.0)
    minibatch_size: PositiveInt = 256
    update_epochs: PositiveInt = 1
    epsilon: float = Field(default=1e-6, gt=0.0)
    occam_beta: float = Field(default=0.15, ge=0.0)
    canonicalize: bool = False
    canonicalize_votes: PositiveInt = 1
    witness_valuation: Literal["coverage", "count"] = "coverage"
    witness_nu: Literal["layer", "uniform", "geometric"] = "layer"
    witness_nu_p: float = Field(default=0.5, gt=0.0, le=1.0)  # geometric measure: nu(|S|)=p**|S|
    witness_center: bool = True
    # Default credit contract: exact leave-two-out (l2o) centering with mean-absolute scaling.
    # The plain group-mean baseline ("mean") is for ABLATIONS ONLY -- never the default.
    witness_baseline: Literal["mean", "rloo", "l2o"] = "l2o"
    witness_normalize: bool = True
    witness_scale: Literal["std", "mean", "size"] = "mean"
    witness_normalization_scope: Literal["group", "update"] = "group"
    witness_size_power: float = Field(default=1.0, ge=0.0)  # "size" reweight is 1/|S|**power
    witness_mc_samples: int = Field(default=20000, ge=0)  # coverage union-measure MC above the exact threshold; 0 forces exact
    witness_discovery: Literal["grpo", "maxrl", "none"] = "none"
    witness_coverage_weight: float = Field(default=1.0, ge=0.0)
    hidden_dim: PositiveInt = 64
    reward_type: Literal["binary"] = "binary"

    @field_validator("witness_baseline", mode="before")
    @classmethod
    def _accept_legacy_l2o_alias(cls, v):
        # The leave-two-out baseline was formerly named "loo2"; older configs and stored
        # run artifacts still carry that spelling. Normalize it so they keep loading.
        return "l2o" if v == "loo2" else v


class MaxRLConfig(GroupedRLConfig):
    pass


class GRPOConfig(GroupedRLConfig):
    pass


class RLOOConfig(GroupedRLConfig):
    pass


class OccamRLConfig(GroupedRLConfig):
    pass


class MWRLConfig(GroupedRLConfig):
    """Minimal-Witness RL: exact lattice leave-one-out credit (amortized theory section 4).

    Defaults to the coverage valuation, which is dense and certificate-free (no
    canonicalization needed); set ``witness_valuation: count`` with ``canonicalize: true``
    for the binary deduplicated-count credit. The non-negative credit is uncentered
    (section 4.5); ``witness_center`` and ``witness_discovery`` add the section 7.1/7.4
    centering / exogenous-exploration terms needed to bootstrap on sparse-reward tasks.
    """

    witness_valuation: Literal["coverage", "count"] = "coverage"
