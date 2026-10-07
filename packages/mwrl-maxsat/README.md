# mwrl-maxsat

Prime-implicant enumeration on monotone MaxSAT — the enumerable benchmark for MWRL with
exact ground truth.

A monotone `k`-CNF instance induces the sufficiency predicate `s(S) = [S satisfies every
clause]`, whose minimal witnesses are the prime implicants of the monotone function
(equivalently the minimal hitting sets of the clauses). Because the full antichain `M` is
enumerable, every learned objective can be scored against the exact target, and exact value
iteration on the subset lattice provides the recovery ceiling. This is the "maze" setting:
the complete target, the planning references, and the failure modes of each objective are
all visible.

## Modules

| File | Role |
| --- | --- |
| `instance.py` | Monotone MaxSAT instances, the sufficiency predicate, and exact minimal witnesses. |
| `env.py` | The add-a-variable subset MDP over the core `mwrl` runner. |
| `vi.py` | Executed value iteration on the subset lattice (the recovery ceiling). |
| `evaluate.py` | Exact-ground-truth scoring: born-minimal rate, antichain recall, witnesses found. |
| `table.py` | The full diagnostic table: every method under both the coverage and count valuations. |
| `run.py` | The headline prime-implicant benchmark entry point. |
| `gflownet.py`, `gflownet_run.py` | The GFlowNet baseline (trajectory balance, with a size-penalized variant) and its driver. |
| `size_penalty_sweep.py` | Size-penalized MaxEnt RL and GFlowNet, each swept over its penalty strength at the table's budget. |
| `query_benchmark.py`, `learned_query_benchmark.py` | Verifier-query-accounted black-box baselines. |
| `scale_ablation.py` | The mean-vs-std normalization ablation. |
| `rare_basin_benchmark.py` | Stress test with one exponentially rare deletion basin. |
| `geometry_phase.py`, `geometry_phase_plot.py` | The antichain-geometry phase diagram. |
| `credit_validation.py`, `gradient_alignment.py` | Validate the shared-sample Monte Carlo credit against exact enumeration. |

## Reproduce

From the repository root, after `uv sync --all-packages`:

```bash
# The diagnostic table (paper Table 1): every method under both valuations.
uv run --package mwrl-maxsat python -m mwrl_maxsat.table

# Defaults reproduce the paper configuration: 8 instances of 14 variables,
# group size 48, geometric measure p = 0.7, 150 updates, seeds 0/1/2.

# The size-penalty sweeps of MaxEnt RL and GFlowNet; their best settings are the
# size-penalized rows of the paper's MaxSAT table.
uv run --package mwrl-maxsat python -m mwrl_maxsat.size_penalty_sweep
```

The table builder reads a precomputed value-iteration ceiling covering the same instances;
regenerate it with `python -m mwrl_maxsat.vi` if you change the instance configuration.
