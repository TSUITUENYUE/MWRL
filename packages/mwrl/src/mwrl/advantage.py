"""Minimal-Witness RL: the per-sample advantage for one context's group of K rollouts.

The whole MWRL-specific learning signal, isolated from the shared grouped-PPO
machinery. It implements the exact lattice leave-one-out credit of the amortized
minimal-witness theory:

    A_i = V(U_K) - V(U_{-i})                                       (theory section 4.4)

where U_K is the certified up-set of the group's successful witnesses. V is either the
coverage measure nu (dense, certificate-free) or the deduplicated minimal-element count.
The lattice math lives in ``credit.py``; this assembles it into a training advantage,
then optionally re-shapes it with affine centering / normalization that preserve its
containment structure (theory section 3.8), plus an optional exploration term.
"""

from __future__ import annotations

import torch

from mwrl.baselines import grouped_outcome_advantages
from mwrl.config import GroupedRLConfig
from mwrl.credit import (
    count_advantages,
    coverage_advantages,
    coverage_l2o_advantages,
    nu_upset_geometric,
    nu_upset_layer,
    nu_upset_uniform,
    popcount,
)


def mwrl_group_advantages(
    *,
    terminal_masks: list[int],
    canon_masks: list[int],
    successes: list[bool],
    rewards: list[float],
    config: GroupedRLConfig,
    device: torch.device,
) -> torch.Tensor:
    """Return the per-sample MWRL advantage for one group (broadcast to steps by the caller)."""
    # 1. exact lattice leave-one-out credit A_i = V(U_K) - V(U_{-i}).
    l2o = config.witness_center and config.witness_baseline == "l2o"
    if config.witness_valuation == "coverage":
        # mc_p is the product-measure inclusion probability the MC estimator samples from;
        # uniform=1/2, geometric=p, layer has no product form so it stays exact (mc_p=None).
        if config.witness_nu == "uniform":
            nu, mc_p = nu_upset_uniform, 0.5
        elif config.witness_nu == "geometric":
            nu, mc_p = nu_upset_geometric(config.witness_nu_p), config.witness_nu_p
        else:
            nu, mc_p = nu_upset_layer, None
        if l2o:  # exact leave-two-out centering, unbiased despite U_K coupling (theory 7.1)
            credit = coverage_l2o_advantages(terminal_masks, successes, nu_upset=nu,
                                              mc_p=mc_p, mc_samples=config.witness_mc_samples)
        else:
            credit = coverage_advantages(terminal_masks, successes, nu_upset=nu)
    else:
        credit = count_advantages(canon_masks, successes)  # binary, on canonicalized witnesses
        l2o = False  # leave-two-out is defined for the coverage measure only
    credit_tensor = torch.as_tensor(credit, dtype=torch.float32, device=device)
    raw_scale = credit_tensor.abs().mean()  # scale of the raw non-negative credit

    # 2. centering (theory 7.1): give failed/dominated samples negative advantage.
    if config.witness_center and not l2o:
        n = credit_tensor.numel()
        centered = credit_tensor - credit_tensor.mean()
        if config.witness_baseline == "rloo" and n > 1:  # leave-one-out mean = (K/(K-1)) * mean-centered
            centered = centered * (float(n) / float(n - 1))
        credit_tensor = centered

    # 3. normalization: rescale the credit VECTOR to a usable step size.
    #    "std"/"mean" are affine (keep the containment structure, theory 3.8). "size" is a
    #    per-sample reweight by 1/|S| BEFORE the scale divide -- a heuristic (NOT the exact
    #    credit) that counteracts the ~1/|S|^2 decay of the layer measure's shrink signal, so
    #    the pressure toward smaller witnesses stays size-independent. Dominated samples are 0,
    #    and 0/|S| = 0, so the support structure survives even though the reweight is non-affine.
    if config.witness_normalize:
        if config.witness_scale == "size":
            sizes = torch.as_tensor([max(popcount(m), 1) for m in terminal_masks],
                                    dtype=torch.float32, device=device)
            credit_tensor = credit_tensor / sizes.pow(config.witness_size_power)
        if config.witness_normalization_scope == "group":
            if config.witness_scale == "size":
                divisor = credit_tensor.std(unbiased=False)
            else:
                divisor = raw_scale if config.witness_scale == "mean" else credit_tensor.std(unbiased=False)
            credit_tensor = credit_tensor / (divisor + config.epsilon)

    # 4. optional exogenous exploration term (theory 7.4); off by default.
    if config.witness_discovery == "none":
        return credit_tensor
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=device)
    discovery = grouped_outcome_advantages(
        reward_tensor, mode=config.witness_discovery, epsilon=config.epsilon
    )
    return discovery + config.witness_coverage_weight * credit_tensor
