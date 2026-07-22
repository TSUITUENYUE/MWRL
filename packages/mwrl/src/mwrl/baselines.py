"""Scalar group-advantage rules for the baseline advantages the Runner can use.

These are the standard-RL (scalar-reward) advantage estimators: MaxRL, GRPO, RLOO, and
the Occam length-weighted reward. Each reads only the scalar reward, so none can see the
containment structure that the MWRL credit in ``mwrl.credit``/``mwrl.advantage`` reads
(theory section 3.8). All of them ride the same ``Runner``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def maxrl_group_advantages(rewards: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    """Official MaxRL group advantage: center by prompt mean, divide by prompt mean."""
    rewards = rewards.to(dtype=torch.float32)
    mean_r = torch.mean(rewards)
    return (rewards - mean_r) / (mean_r + epsilon)


def grpo_group_advantages(rewards: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    """GRPO outcome advantage: group-center and normalize by group std."""
    rewards = rewards.to(dtype=torch.float32)
    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    centered = rewards - torch.mean(rewards)
    std = torch.std(rewards, unbiased=False)
    return centered / (std + epsilon)


def rloo_group_advantages(rewards: torch.Tensor, *, epsilon: float) -> torch.Tensor:
    """RLOO outcome advantage: reward minus leave-one-out group baseline."""
    del epsilon
    rewards = rewards.to(dtype=torch.float32)
    n = rewards.numel()
    if n <= 1:
        return torch.zeros_like(rewards)
    mean_r = torch.mean(rewards)
    return (rewards - mean_r) * (float(n) / float(n - 1))


def occam_weighted_reward(success: float, dimension_cost: float, *, beta: float) -> float:
    return float(success) * float(np.exp(-beta * float(dimension_cost)))


def grouped_outcome_advantages(
    rewards: torch.Tensor,
    *,
    mode: Literal["maxrl", "grpo", "rloo"],
    epsilon: float,
) -> torch.Tensor:
    if mode == "maxrl":
        return maxrl_group_advantages(rewards, epsilon=epsilon)
    if mode == "grpo":
        return grpo_group_advantages(rewards, epsilon=epsilon)
    if mode == "rloo":
        return rloo_group_advantages(rewards, epsilon=epsilon)
    raise ValueError(f"unsupported grouped outcome advantage mode {mode!r}")
