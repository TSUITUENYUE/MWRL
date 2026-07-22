"""Cluster entry point for amortized minimal-circuit discovery on a frozen LLM.

    python -m mwrl_circuits.run --config configs/qwen3_1p7b.yaml

One policy is trained over a distribution of behaviors (the context set), conditioned on
each behavior's EAP fingerprint, and recovers each behavior's antichain of minimal faithful
circuits with no per-task search. The witness test is the monotone existential closure s_exists
(a faithful sub-circuit exists inside the opened set), realized by the EAP-guided dropout inner
optimizer. ``--holdout N`` keeps N behaviors out of training and evaluates the trained policy on
them, which is the amortization test. A single behavior is the |D|=1 special case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from mwrl.config import MWRLConfig
from mwrl.runner import Runner

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.env import CircuitEnv
from mwrl_circuits.evaluate import evaluate_antichain
from mwrl_circuits.tasks import build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier

# method -> (advantage, optimizer) for the core Runner. mwrl is the method; the rest are the
# scalar baselines expressed as other settings of the same runner.
METHODS = {
    "mwrl": ("mwrl", "reinforce"),
    "maxrl": ("maxrl", "reinforce"),
    "ppo": ("maxrl", "ppo"),
    "grpo": ("grpo", "ppo"),
    "rloo": ("rloo", "reinforce"),
}

def build_config(mwrl_cfg: dict) -> tuple[MWRLConfig, int, int, int]:
    horizon = mwrl_cfg.get("max_steps", 0)   # H: episode cap; 0 -> run.py sets it to n_comp
    seed = mwrl_cfg.get("seed", 0)
    eval_samples = mwrl_cfg.get("eval_samples", 96)
    config = MWRLConfig(
        witness_valuation=mwrl_cfg.get("valuation", "count"), witness_center=True,
        witness_baseline=mwrl_cfg.get("baseline", "mean"), witness_normalize=True,
        witness_scale="mean", witness_nu=mwrl_cfg.get("nu", "layer"),
        witness_nu_p=mwrl_cfg.get("nu_p", 0.5), witness_mc_samples=mwrl_cfg.get("mc_samples", 20000),
        updates=mwrl_cfg.get("updates", 200), task_batch_size=mwrl_cfg.get("groups_per_update", 6),
        group_size=mwrl_cfg.get("k", 10), learning_rate=mwrl_cfg.get("lr", 3e-3),
        entropy_coef=mwrl_cfg.get("entropy_coef", 0.02), hidden_dim=mwrl_cfg.get("hidden", 256),
        update_epochs=1, minibatch_size=256, clip_epsilon=0.2, max_grad_norm=1.0, epsilon=1e-6)
    return config, horizon, seed, eval_samples


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def summarize(report: dict) -> None:
    print(f"\n{'behavior':22s} {'split':>8s} {'sound':>6s} {'#found':>7s} {'faith':>6s} "
          f"{'size':>6s} {'fwd':>8s}")
    print("-" * 68)
    for name, r in report.items():
        if name.startswith("_"):
            continue
        split = "heldout" if r.get("heldout") else "train"
        print(f"{name:22s} {split:>8s} {r['soundness']:6.2f} {r['n_found']:7d} "
              f"{r['mean_faithfulness']:6.2f} {r['median_size']:6.0f} {r['forward_calls']:8d}")


def _component_name(idx: int, model: AblatedModel) -> str:
    if idx < model.n_attn:
        return f"a{idx // model.n_heads}.h{idx % model.n_heads}"
    return f"mlp{idx - model.n_attn}"


def _readable_circuits(report: dict, model: AblatedModel) -> None:
    def walk(obj):
        if isinstance(obj, dict):
            for key, val in obj.items():
                if key == "circuits" and isinstance(val, list):
                    obj[key] = [[_component_name(i, model) for i in range(model.n_comp)
                                 if (m >> i) & 1] for m in val]
                else:
                    walk(val)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(report)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--updates", type=int, default=None, help="override mwrl.updates")
    ap.add_argument("--method", default="mwrl", choices=list(METHODS),
                    help="mwrl (the method) or a baseline: maxrl, ppo, grpo, rloo")
    ap.add_argument("--holdout", type=int, default=0,
                    help="behaviors held out of training, evaluated to test amortization")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.updates is not None:
        cfg["mwrl"]["updates"] = args.updates
    device = cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device

    print(f"[load] {cfg['model_name']} on {device}", flush=True)
    model = AblatedModel(
        cfg["model_name"], device=device, dtype=cfg.get("dtype", "bfloat16"),
        attn_implementation=cfg.get("attn_implementation", "eager"))
    print(f"[model] {model.n_layers} layers, {model.n_heads} heads, "
          f"{model.n_comp} components", flush=True)

    hierarchy = build_hierarchy(cfg["families"])
    tasks = [sub for node in hierarchy.values() for sub in node["subtasks"]]

    config, horizon, seed, eval_samples = build_config(dict(cfg.get("mwrl", {})))
    advantage, optimizer = METHODS[args.method]
    print(f"[resolved-config] valuation={config.witness_valuation} "
          f"baseline={config.witness_baseline} scale={config.witness_scale} "
          f"nu={config.witness_nu} nu_p={config.witness_nu_p} "
          f"k={config.group_size} groups={config.task_batch_size} "
          f"updates={config.updates} lr={config.learning_rate}", flush=True)

    # the context distribution D = the individual behaviors; drop any the model cannot do.
    inner_iters = int(cfg.get("mwrl", {}).get("inner_iters", 8))
    batch_seqs = int(cfg.get("mwrl", {}).get("batch_seqs", 1024))
    print(f"[verifiers] EAP fingerprint for {len(tasks)} behaviors "
          f"(inner_iters={inner_iters}, batch_seqs={batch_seqs})", flush=True)
    verifiers: list[CircuitVerifier] = []
    for t in tasks:
        v = CircuitVerifier(model, t, tau=cfg["tau"], base_seed=seed, inner_iters=inner_iters,
                            batch_seqs=batch_seqs)
        if v.full_metric <= 1e-6:
            print(f"[skip] {t.name}: full-model margin {v.full_metric:.3f} <= 0", flush=True)
            continue
        verifiers.append(v)
    if not verifiers:
        print("[error] no behavior has a positive full-model margin", flush=True)
        return 1

    train_v = verifiers[args.holdout:] if 0 < args.holdout < len(verifiers) else verifiers
    held = [v for v in verifiers if v not in train_v]
    print(f"[amortize] one {args.method} policy over {len(train_v)} behaviors, "
          f"{len(held)} held out", flush=True)

    env = CircuitEnv(train_v, n_comp=model.n_comp, horizon=horizon or model.n_comp)
    print(f"[mdp] add/remove/stop, S_0=full, horizon={horizon or model.n_comp}", flush=True)
    runner = Runner(env, config, advantage=advantage, optimizer=optimizer, device=device, seed=seed)
    t0 = time.time()

    # crash-safe progress: the policy checkpoints every update and the eval writes per
    # behavior, so a wall kill or node failure resumes instead of erasing the run.
    out_dir = Path(args.output) if args.output else Path("runs/circuits/latest")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    partial_path = out_dir / "report_partial.json"
    total_updates = config.updates
    done_updates = 0
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        runner.actor.load_state_dict(ckpt["actor"])
        runner.opt.load_state_dict(ckpt["opt"])
        done_updates = int(ckpt["updates_done"])
        config.updates = max(0, total_updates - done_updates)
        print(f"[resume] {ckpt_path}: {done_updates}/{total_updates} updates done, "
              f"{config.updates} remaining", flush=True)

    def _save_ckpt(n_done: int) -> None:
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"actor": runner.actor.state_dict(), "opt": runner.opt.state_dict(),
                    "updates_done": n_done, "total_updates": total_updates}, tmp)
        tmp.replace(ckpt_path)

    def progress(u: int, sr: float) -> None:
        _save_ckpt(done_updates + u + 1)
        if u % max(1, config.updates // 20) == 0 or u == config.updates - 1:
            queries = sum(v.forward_calls for v in train_v)
            print(f"[update {done_updates + u + 1}/{total_updates}] success={sr:.2f} "
                  f"s0_queries={queries} "
                  f"model_forwards={model.forward_passes} {time.time() - t0:.0f}s", flush=True)

    if config.updates > 0:
        runner.train(on_update=progress)
        _save_ckpt(total_updates)

    # recover each behavior's antichain with the trained (no per-task retraining) policy;
    # each behavior's result lands in report_partial.json as soon as it is evaluated.
    report: dict = json.loads(partial_path.read_text()) if partial_path.exists() else {}
    if report:
        print(f"[resume] eval: {len(report)} behaviors already done", flush=True)
    for v in verifiers:
        if v.name in report:
            continue
        m = evaluate_antichain(runner, v, n_samples=eval_samples)
        m["heldout"] = v in held
        m["forward_calls"] = v.forward_calls
        report[v.name] = m
        partial_path.write_text(json.dumps(report))
    report["_meta"] = {"model_name": cfg["model_name"], "n_comp": model.n_comp, "method": args.method,
                       "tau": cfg["tau"], "behaviors": len(verifiers), "holdout": len(held),
                       "seconds": round(time.time() - t0, 1), "device": device}
    summarize(report)

    _readable_circuits(report, model)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\n[done] {out_dir/'report.json'} ({report['_meta']['seconds']}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
