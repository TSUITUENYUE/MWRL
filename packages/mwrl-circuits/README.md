# mwrl-circuits

Minimal sparse-circuit discovery on frozen large language models — MWRL where the verifier
is stochastic and only approximately monotone.

The elements are the components of a frozen model (attention-head slices and MLP blocks). A
subset is sufficient when resample-ablating its complement keeps the next-token
distribution within a tolerance of the clean model on a behavior's probes. Minimality is
never supplied: the verifier returns the raw opened set and the coverage credit must drive
the policy to minimal, interchangeable circuits. This is the scale and robustness test —
the same objective as MaxSAT and Suzuki, applied to Qwen3-1.7B (the MMLU expert benchmark)
and Qwen3-8B (the hierarchical taxonomy).

## Modules

| File | Role |
| --- | --- |
| `verifier.py` | The monotone existential-closure sufficiency verifier for a behavior on a frozen LLM. |
| `ablation.py` | Frozen-model component ablation (HF transformers, Qwen3/Llama). |
| `inner.py` | The existential-closure inner optimizer: EAP-guided dropout. |
| `env.py` | The MWRL outer MDP over components, amortized over behaviors. |
| `run.py` | Entry point for amortized minimal-circuit discovery on a frozen LLM. |
| `tasks.py`, `bench.py`, `validate.py` | The capability hierarchy, the MMLU/BoolQ probe families, and dense-model leaf validation. |
| `bench_dense.py`, `bench_eval.py`, `bench_holdout.py` | Dense reference accuracy, the retention table and confusion matrix, and held-out generalization. |
| `baselines_masks.py` | Per-task baselines: ACDC-style greedy, Wanda-style scores, HardConcrete gates. |
| `hierarchical.py`, `hierarchical3.py`, `hier_moe_eval.py` | Two- and three-level hierarchical discovery and its leaf-by-circuit evaluation. |
| `router.py`, `route.py` | The task-conditioned sparse-circuit router and the predictive prompt-to-leaf router. |
| `monotonicity.py` | Measures the non-monotonicity of the raw verifier and its realized closure. |

## Configs

Configurations are a matrix of model size × valuation. The valuation suffix is `cov`
(coverage measure, the default), `cnt` (count measure with a canonicalizer), or `hcov`
(hierarchical coverage).

| Config | Purpose |
| --- | --- |
| `qwen3_1p7b_bench.yaml` | The MMLU expert benchmark on Qwen3-1.7B (paper Table 3), seed 0. |
| `qwen3_1p7b_bench_seed1.yaml`, `qwen3_1p7b_bench_seed2.yaml` | Seeds 1 and 2 of the same benchmark. |
| `qwen3_8b_hcov.yaml` | The three-level hierarchical taxonomy on Qwen3-8B. |
| `qwen3_0p6b.yaml`, `qwen3_1p7b.yaml`, `qwen3_4b.yaml`, `qwen3_8b.yaml` | Single-model discovery at each scale. |
| `qwen3_8b_cov.yaml`, `qwen3_8b_cnt.yaml` | Qwen3-8B under the coverage and count valuations. |
| `qwen_cov_smoke.yaml` | A fast smoke configuration for a local sanity check. |

## Reproduce

From the repository root, after `uv sync --all-packages`. The models are downloaded from the Hugging Face
Hub on first use, so a GPU and network (or a warm `HF_HOME` cache) are required.

```bash
# The MMLU expert benchmark on Qwen3-1.7B (paper Table 3).
uv run --package mwrl-circuits python -m mwrl_circuits.run --config configs/qwen3_1p7b_bench.yaml
```
