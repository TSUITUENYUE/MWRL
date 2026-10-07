"""Build the data-sufficiency train/test sets in verl's parquet format.

    python -m mwrl_rlvr.data --out data/ds --train 50000 --test 512

Each row carries the chat prompt and a JSON ground truth with everything the verifier
needs: n, x0, the 2^n sufficiency bits, and the antichain of minimal sufficient masks.
Train and test come from disjoint seeds, and any test prompt that also occurs in train is
dropped.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

import numpy as np
import pandas as pd

from mwrl_rlvr.ds import Instance, generate_instance, popcount, render_prompt

DATA_SOURCE = "mwrl_rlvr/data_sufficiency"


def ground_truth(inst: Instance) -> str:
    suff = "".join("1" if b else "0" for b in inst.sufficient)
    return json.dumps({"n": inst.n, "x0": inst.x0, "suff": suff, "anti": inst.antichain}, separators=(",", ":"))


def record(inst: Instance, idx: int, split: str) -> dict:
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": render_prompt(inst)}],
        "ability": "math",
        "reward_model": {"style": "rule", "ground_truth": ground_truth(inst)},
        "extra_info": {"split": split, "index": idx, "x0": inst.x0, "y0": inst.y0,
                       "antichain_size": len(inst.antichain)},
    }


def build(out: pathlib.Path, n_train: int, n_test: int, seed: int, **gen) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng_train, rng_test = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    train = [generate_instance(rng_train, **gen) for _ in range(n_train)]
    seen = {render_prompt(i) for i in train}
    test: list[Instance] = []
    dropped = 0
    while len(test) < n_test:
        inst = generate_instance(rng_test, **gen)
        if render_prompt(inst) in seen:
            dropped += 1
            continue
        test.append(inst)
    for name, insts in (("train", train), ("test", test)):
        pd.DataFrame([record(i, k, name) for k, i in enumerate(insts)]).to_parquet(out / f"{name}.parquet")
    stats = {
        "seed": seed, "generator": gen, "train": n_train, "test": n_test, "test_dropped_as_duplicate": dropped,
        "antichain_size": dict(sorted(collections.Counter(len(i.antichain) for i in train).items())),
        "witness_size": dict(sorted(collections.Counter(popcount(m) for i in train for m in i.antichain).items())),
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--train", type=int, default=50_000)
    ap.add_argument("--test", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dom", type=int, default=30)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min-witness", type=int, default=2)
    ap.add_argument("--antichain-min", type=int, default=2)
    ap.add_argument("--antichain-max", type=int, default=8)
    a = ap.parse_args()
    stats = build(a.out, a.train, a.test, a.seed, n_dom=a.n_dom, n=a.n, min_witness=a.min_witness,
                  antichain_range=(a.antichain_min, a.antichain_max))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
