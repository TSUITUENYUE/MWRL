"""Sample B answers per held-out instance with vLLM and report the antichain metrics.

One process per GPU, each on a shard of the test set, then a merge:

    python -m mwrl_rlvr.evaluate run   --model M --data test.parquet --B 64 --shard i --num-shards 8 --out DIR
    python -m mwrl_rlvr.evaluate merge --out DIR

The prompt is built exactly as verl's RLHFDataset builds it: the tokenizer's chat template
over the stored messages with add_generation_prompt=True. Sampling stops at ``</answer>`` as
in training (``verl_agent.AnswerStopAgentLoop``).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import pandas as pd

from mwrl_rlvr.ds import verify
from mwrl_rlvr.metrics import aggregate, instance_metrics
from mwrl_rlvr.reward import _ground_truth

STOP = ["</answer>"]  # the same stop string as verl_agent.STOP (kept verl-free for import)


def _ks(b: int) -> list[int]:
    ks, k = [], 1
    while k <= b:
        ks.append(k)
        k *= 2
    return ks


def run(a) -> None:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(a.data)
    if a.limit:
        df = df.iloc[: a.limit]
    df = df.iloc[a.shard :: a.num_shards]
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    prompts = [tok.apply_chat_template(list(p), add_generation_prompt=True, tokenize=False) for p in df["prompt"]]
    llm = LLM(model=a.model, tokenizer=a.tokenizer or a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_model_len, seed=a.seed, enable_prefix_caching=True)
    params = SamplingParams(n=a.B, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed,
                            stop=STOP, include_stop_str_in_output=True)
    outs = llm.generate(prompts, params)
    rows = []
    for (_, row), out in zip(df.iterrows(), outs):
        n, x0, suff, anti = _ground_truth(row["reward_model"]["ground_truth"])
        verdicts = [verify(o.text, n=n, x0=x0, sufficient=suff, antichain=anti) for o in out.outputs]
        m = instance_metrics(verdicts, sorted(anti), _ks(a.B))
        m["index"] = int(row["extra_info"]["index"])
        m["mean_tokens"] = sum(len(o.token_ids) for o in out.outputs) / len(out.outputs)
        m["truncated"] = sum(o.finish_reason == "length" for o in out.outputs) / len(out.outputs)
        if a.keep_text:
            m["texts"] = [o.text for o in out.outputs]
        rows.append(m)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"shard{a.shard}.json").write_text(json.dumps({"config": {k: str(v) for k, v in vars(a).items()},
                                                             "rows": rows}))
    print(f"shard {a.shard}: {len(rows)} instances written", flush=True)
    os._exit(0)  # vLLM's engine shutdown can hang after the results are safely on disk


def merge(a) -> None:
    shards = sorted(a.out.glob("shard*.json"))
    loaded = [json.loads(s.read_text()) for s in shards]
    rows = [r for d in loaded for r in d["rows"]]
    for r in rows:
        r.pop("texts", None)
    summary = {"config": loaded[0]["config"], "shards": len(shards), **aggregate(rows)}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--tokenizer", default="", help="defaults to the model directory")
    r.add_argument("--data", required=True)
    r.add_argument("--out", type=pathlib.Path, required=True)
    r.add_argument("--B", type=int, default=64)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--shard", type=int, default=0)
    r.add_argument("--num-shards", type=int, default=1)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--max-tokens", type=int, default=4096)
    r.add_argument("--max-model-len", type=int, default=5120)
    r.add_argument("--gpu-mem", type=float, default=0.9)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--keep-text", action="store_true")
    m = sub.add_parser("merge")
    m.add_argument("--out", type=pathlib.Path, required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else merge(a)


if __name__ == "__main__":
    main()
