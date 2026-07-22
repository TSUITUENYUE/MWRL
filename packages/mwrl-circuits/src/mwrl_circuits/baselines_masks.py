"""Per-task baseline circuit masks: ACDC-style greedy, Wanda-style scores, HardConcrete.

All three read discovery-side information only (the discovery verifier's prompts, oracle,
and attributions), so their masks are frozen before any held-out prompt is touched.

  acdc_mask          greedy faithfulness pruning from the FULL model against the same s0
                     verifier (every accepted state verified): the ACDC-style classical
                     circuit-discovery baseline. Per-task search, one circuit, own size.
  wanda_scores       activation-aware component importance at component granularity: the
                     mean norm of each component's residual-stream contribution on the
                     task prompts (|W|*|x| in Wanda's spirit, lifted from weights to
                     components). Rank + top-k at matched size.
  hardconcrete_mask  a differentiable mask over components (Louizos et al. L0 gates)
                     trained through the soft ablation path against the same KL
                     faithfulness objective: the white-box learned per-task baseline.
                     Hardened to top-k gates at matched size.
"""

from __future__ import annotations

import math

import torch

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.inner import eap_batched_minimize
from mwrl_circuits.verifier import CircuitVerifier


def acdc_mask(v: CircuitVerifier) -> int:
    """Greedy verified descent from the full component set under the task's s0."""
    full = (1 << v.n_comp) - 1
    return eap_batched_minimize(full, v.s0_batch, v.attrs, v.n_comp)


@torch.no_grad()
def wanda_scores(model: AblatedModel, v: CircuitVerifier) -> torch.Tensor:
    """Mean residual-contribution norm per component on the task's clean prompts."""
    attn_act, mlp_act = model.counterfactual(v.enc)      # records CLEAN activations here
    pad = v.enc["attention_mask"].bool()                 # [B, s]
    scores = torch.zeros(model.n_comp)
    for li in range(model.n_layers):
        w = model._layers[li].self_attn.o_proj.weight    # [hidden, nh*hd]
        xh = attn_act[li]                                # [B, s, nh, hd]
        for h in range(model.n_heads):
            xs = xh[:, :, h, :][pad].float()             # [T, hd] unpadded positions
            block = w[:, h * model.head_dim:(h + 1) * model.head_dim].float()
            scores[model.attn_idx(li, h)] = (xs @ block.T).norm(dim=-1).mean()
        scores[model.mlp_idx(li)] = mlp_act[li][pad].float().norm(dim=-1).mean()
    return scores


def hardconcrete_mask(model: AblatedModel, v: CircuitVerifier, k: int, *, steps: int = 300,
                      lr: float = 0.05, lam: float = 2.0, seed: int = 0,
                      beta: float = 2.0 / 3.0, gamma: float = -0.1,
                      zeta: float = 1.1) -> int:
    """Train HardConcrete gates against KL(P_clean || P_soft) + an L0 target of k
    components, through the differentiable soft-ablation path; return the top-k gates."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    log_alpha = torch.full((model.n_comp,), 2.0, device=model.device, requires_grad=True)
    opt = torch.optim.Adam([log_alpha], lr=lr)
    p_clean = torch.softmax(v.clean_logits.float(), dim=-1).detach()
    shift = beta * math.log(-gamma / zeta)
    target = k / model.n_comp
    with torch.enable_grad():  # callers (bench_eval/bench_holdout) run under no_grad
        for _ in range(steps):
            u = torch.rand(model.n_comp, generator=gen).clamp(1e-6, 1 - 1e-6).to(model.device)
            s = torch.sigmoid((u.log() - (-u).log1p() + log_alpha) / beta)
            z = (s * (zeta - gamma) + gamma).clamp(0.0, 1.0)
            logits = model.soft_logits(v.enc, z, v.cf).float()
            kl = torch.nn.functional.kl_div(torch.log_softmax(logits, dim=-1), p_clean,
                                            reduction="batchmean")
            l0 = torch.sigmoid(log_alpha - shift).mean()
            loss = kl + lam * (l0 - target).abs()
            opt.zero_grad()
            loss.backward()
            opt.step()
    gate = (torch.sigmoid(log_alpha) * (zeta - gamma) + gamma).clamp(0.0, 1.0)
    mask = 0
    for i in torch.topk(gate.detach().cpu(), k).indices.tolist():
        mask |= 1 << i
    return mask
