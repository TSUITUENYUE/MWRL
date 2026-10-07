"""verl reward function (``reward.custom_reward_function``) for data sufficiency.

verl's reward loop scores one rollout at a time, and the advantage step only sees each
rollout's score and its prompt group. The score is therefore the integer code of
``credit.encode``: the chosen mask and the parsed / success / minimal flags. The registered
advantage estimators (``verl_ext``) decode it and apply the arm's group credit. The plain
indicators are returned alongside for logging.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from mwrl_rlvr.credit import encode
from mwrl_rlvr.ds import verify


@lru_cache(maxsize=4096)
def _ground_truth(gt: str) -> tuple[int, int, np.ndarray, frozenset[int]]:
    d = json.loads(gt)
    suff = np.frombuffer(d["suff"].encode(), dtype=np.uint8) == ord("1")
    return d["n"], d["x0"], suff, frozenset(d["anti"])


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict:
    n, x0, suff, anti = _ground_truth(ground_truth)
    v = verify(solution_str, n=n, x0=x0, sufficient=suff, antichain=anti)
    return {
        "score": float(encode(v.mask, v.parsed, v.success, v.minimal, len(anti))),
        "parsed": float(v.parsed),
        "success": float(v.success),
        "minimal": float(v.minimal),
        "size": float(v.size),
    }
