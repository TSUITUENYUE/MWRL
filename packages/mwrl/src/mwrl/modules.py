"""Actor network for the MWRL runner: a masked-categorical MLP over the env's action
space. Like rsl_rl's ActorCritic, it is constructed from ``num_obs``/``num_actions``
that the env reports, and it operates on flat observation tensors plus an action mask;
how observations are encoded is the env's business.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Masked categorical MLP policy over the env's discrete actions."""

    def __init__(self, num_obs: int, num_actions: int, *, hidden_dim: int = 128,
                 activation: str = "tanh") -> None:
        super().__init__()
        act: type[nn.Module] = nn.Tanh if activation == "tanh" else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(num_obs, hidden_dim), act(),
            nn.Linear(hidden_dim, hidden_dim), act(),
            nn.Linear(hidden_dim, num_actions))
        self.num_actions = num_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor, action_mask: torch.Tensor) -> Categorical:
        logits = self.forward(obs).masked_fill(~action_mask.bool(), -1.0e9)
        return Categorical(logits=logits)
