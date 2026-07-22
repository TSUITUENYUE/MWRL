# mwrl-suzuki

This package runs Minimal-Witness RL on a single self-contained Suzuki reaction CSV.
The boundary is intentionally narrow:

```text
one reaction CSV (passed with --data)  ->  mwrl-suzuki
```

The method imports no generator code, calibration file, hidden mechanistic state, or
generator YAML. It reads one reaction CSV and treats unmeasured combinations as
failures under the closed-world assumption.

## Self-contained CSV contract

Each row is one measured reaction. The CSV must contain:

| column | meaning |
|---|---|
| `task_id` | substrate-pair task |
| `halide_smiles`, `boron_smiles` | coupling partners |
| `product_smiles` | optional product structure |
| one column per condition dimension | the condition value used in this reaction |
| `yield` | measured or generated yield on one consistent scale |
| `success_threshold` | the shared yield threshold, repeated on every row |
| `active_set` | JSON list of dimensions changed from the shared baseline |
| `active_mask` | integer bitmask encoding `active_set` |

The CSV itself identifies the schema. Singleton rows determine the dimension-to-bit order,
the unique zero-mask row for each task determines the shared baseline, and observed values
determine each dimension's candidate set. Loading fails if:

- tasks disagree on the baseline, threshold, candidates, or bit order;
- a task has anything other than one baseline row;
- `active_set`, `active_mask`, and the changed condition columns disagree;
- a dimension lacks singleton coverage, so its bit position cannot be identified.

These checks ensure that the action space seen by the method is exactly the one represented
by the generated measurements.

## Run

From the repository root, with `--data` pointing at the downloaded reaction CSV:

```bash
# Train the fingerprint-conditioned policy (configs/method.yaml, the paper hyperparameters).
uv run --package mwrl-suzuki python -m mwrl_suzuki.run --data /path/to/suzuki_conditions.csv

# Amortization table (paper Table 2): MWRL, Substrate-blind, Scalar RL, and the
# per-substrate ceiling; each seed holds out 205 substrates.
uv run --package mwrl-suzuki python -m mwrl_suzuki.baselines \
  --data /path/to/suzuki_conditions.csv --holdout 205 --seed 0
```

`configs/method.yaml` contains only fingerprint and optimization settings. Passing a
different `--config` never changes the dataset schema. A different dataset is supplied only
through `--data`.

The dataset is one reaction CSV, distributed as a 38 MB gzip on
[Google Drive](https://drive.google.com/file/d/1ybcKhG88jG78QE5T7fS1j3LWOWoHvIhV/view?usp=sharing).
Download it (or `gdown 1ybcKhG88jG78QE5T7fS1j3LWOWoHvIhV`), run
`gunzip suzuki_conditions.csv.gz`, and pass the CSV with `--data`.

## Package map

- `schema.py` infers and validates the condition schema from the CSV.
- `data.py` derives each task's exact closed-world minimal-witness antichain.
- `fingerprint.py` builds the substrate fingerprint that conditions the policy.
- `verifier.py` implements the sufficiency query as an exact closed-world subset test.
- `env.py` defines the condition-dimension MDP.
- `evaluate.py` scores proposals against the exact ground-truth antichain.
- `run.py` trains and evaluates the fingerprint-conditioned policy.
- `baselines.py` runs the matched amortization arms (MWRL, Substrate-blind, Scalar RL, per-substrate).
- `bo.py` is the Bayesian-optimization reference oracle (the wet-lab contract; not used in the closed-world runs).
