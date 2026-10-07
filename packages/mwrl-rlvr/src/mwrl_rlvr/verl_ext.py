"""verl advantage estimators ``ds_<arm>``, one per arm of ``credit.ARMS``.

``register`` runs in every Ray worker through the job's runtime_env
(``worker_process_setup_hook: mwrl_rlvr.verl_ext.register``), so the estimators exist in the
TaskRunner process where PPOTrainer computes advantages. Hyperparameters come from the same
runtime_env (``MWRL_SIZE_LAMBDA``, ``MWRL_NU_P``); every call appends the values it used and
the batch's verifier statistics to ``MWRL_TRAIN_LOG``, since verl's own score metrics only
see the packed codes.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict

import numpy as np
import torch

from mwrl_rlvr.credit import ARMS, antichain_sizes, decode, group_advantages, popcounts

_CALLS = {"n": 0}
_LOG = {"fh": None}  # one open handle per process


def _hyper() -> tuple[float, float]:
    return float(os.environ["MWRL_SIZE_LAMBDA"]), float(os.environ["MWRL_NU_P"])


def _groups(index) -> dict:
    g = defaultdict(list)
    for i, uid in enumerate(index):
        g[uid].append(i)
    return g


def _log(arm: str, codes: np.ndarray, groups: dict, lam: float, p: float, lengths: np.ndarray,
         max_len: int) -> None:
    """Every training prompt is fresh (one epoch over generated instances), so these batch
    statistics are on-policy measurements on unseen instances at K = group size."""
    path = os.environ.get("MWRL_TRAIN_LOG")
    if not path:
        return
    masks, parsed, success, minimal = decode(codes)
    anti = antichain_sizes(codes)
    found = {u: len({int(masks[i]) for i in r if minimal[i]}) for u, r in groups.items()}
    row = {
        "call": _CALLS["n"], "time": time.time(), "arm": arm, "size_lambda": lam, "nu_p": p,
        "rollouts": int(codes.size), "groups": len(groups),
        "parsed": float(parsed.mean()), "success": float(success.mean()), "minimal": float(minimal.mean()),
        "soundness": float(minimal.sum() / success.sum()) if success.any() else None,
        "mean_size_success": float(popcounts(masks[success]).mean()) if success.any() else None,
        "distinct_minimal_per_group": float(np.mean(list(found.values()))),
        "recall_at_group": float(np.mean([found[u] / max(int(anti[r[0]]), 1) for u, r in groups.items()])),
        "distinct_success_per_group": float(np.mean([len({int(masks[i]) for i in r if success[i]})
                                                     for r in groups.values()])),
        "mean_response_tokens": float(lengths.mean()),
        "truncated": float((lengths >= max_len).mean()),
    }
    if _LOG["fh"] is None:
        _LOG["fh"] = open(path, "a", buffering=1)
    _LOG["fh"].write(json.dumps(row) + "\n")


def _make(arm: str):
    def estimator(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index, config=None, **_):
        lam, p = _hyper()
        codes = token_level_rewards.sum(dim=-1).round().to(torch.int64).cpu().numpy()
        groups = _groups(index)
        adv = np.zeros(codes.size)
        for rows in groups.values():
            adv[rows] = group_advantages(arm, codes[rows], size_lambda=lam, nu_p=p)
        lengths = response_mask.sum(dim=-1).cpu().numpy()
        _log(arm, codes, groups, lam, p, lengths, response_mask.shape[-1])
        _CALLS["n"] += 1
        out = torch.as_tensor(adv, dtype=torch.float32, device=response_mask.device).unsqueeze(-1)
        out = out * response_mask.to(torch.float32)
        return out, out

    estimator.__name__ = f"ds_{arm}"
    return estimator


def register() -> None:
    """Ray worker setup hook: the ds_<arm> estimators and the ds_answer_stop agent loop."""
    from verl.trainer.ppo import core_algos

    import mwrl_rlvr.verl_agent  # registers the agent loop on import

    for arm in ARMS:
        name = f"ds_{arm}"
        if name not in core_algos.ADV_ESTIMATOR_REGISTRY:
            core_algos.register_adv_est(name)(_make(arm))
