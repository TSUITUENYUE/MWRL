# mwrl-rlvr

Minimal-Witness RL as an RLVR objective for LLM post-training, on a math-reasoning task whose
answers form an antichain that is known exactly.

**Task: data sufficiency.** An instance hides integers `(x, y)` in `{1..30}^2` and lists 8
statements that are all true at the hidden point (equations, inequalities, parity,
remainders, divisibility, primality, digit sums). The model answers with a set of statements
and the value of `x` they imply. The verifier accepts when the set determines `x` uniquely on
the domain and the value is right. Adding statements shrinks the solution set, so the
verifier is monotone: every superset of an accepted set is accepted, and the minimal accepted
sets form an antichain. Checking all 256 subsets on the grid gives that antichain exactly.
Instances are filtered to 2 to 8 minimal sets, each using at least two statements.

On one-answer benchmarks the order is trivial: coverage reduces to pass@K and MWRL joins the
pass@k family of objectives. Here the order carries weight, since listing every statement is
always accepted and an objective that only rewards success can drift to supersets.

## Methods

All methods share the model, data, rollouts and PPO update, and differ only in the advantage.
Every rule is the core package's own definition (`mwrl.baselines`, `mwrl.advantage`).

| Arm | Name | Advantage |
| --- | --- | --- |
| `grpo` | GRPO | success, normalized by the group mean and standard deviation |
| `maxrl` | MaxRL | success, `(r - mean) / mean` |
| `grpo_size` | GRPO + size penalty | `success * (1 - 0.1 |S|)`, GRPO |
| `grpo_minimal` | GRPO + minimality oracle | 1 if the set is a minimal accepted set, GRPO |
| `distinct` | Diversity reward | 1 if the set is accepted and no other accepted rollout of the group chose it, rloo-centered, divided by the mean raw credit |
| `distinct_size` | Diversity reward + size penalty | `distinct` times `(1 - 0.1 |S|)` |
| `mwrl` | MWRL | coverage under `nu(up S) = 0.7^|S|`, deletion credit with the l2o baseline, divided by the group's mean absolute credit |

## Modules

| File | Role |
| --- | --- |
| `ds.py` | Statements, instance generator, exact antichain, prompt, answer parser, verifier. |
| `data.py` | Builds the train/test parquet files in verl's format. |
| `credit.py` | Per-group advantages of every method; exact lattice coverage for `mwrl`. |
| `reward.py` | verl reward function; packs each verdict into an integer score. |
| `verl_ext.py` | Registers the `ds_<arm>` advantage estimators in verl and logs per-step statistics. |
| `verl_agent.py` | The `ds_answer_stop` agent loop, which ends a rollout at `</answer>`. |
| `metrics.py` | Soundness, distinct minimal sets found, unbiased recall@k and pass@k. |
| `evaluate.py` | vLLM sampling on held-out instances and the antichain metrics. |

verl's reward loop scores one rollout at a time and hands the advantage step only each
rollout's score and its prompt group. The score is therefore an integer code (chosen set,
parsed, success, minimal, antichain size) that the registered estimator decodes per group.
The estimators are registered in every Ray worker through
`ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=mwrl_rlvr.verl_ext.register`.

## Reproduce

From the repository root. Training and evaluation run on the stack of the reported runs:
torch 2.8.0, vLLM 0.11.0, verl 0.8.0, TransferQueue 0.1.6, `datasets>=3.0`, pyarrow below 21
(the `datasets` release verl imports needs `pa.PyExtensionType`) and flash-attn 2.8.3 for
verl's padding-free attention, installed next to the core and this package.

```bash
# The data: 128,000 training and 512 held-out problems.
python -m mwrl_rlvr.data --out runs/rlvr/data/ds_v1 --train 128000 --test 512
```

The policy is Qwen3-4B-Base. Its `generation_config.json` lists only `<|endoftext|>` as EOS,
so the runs used a local copy (`$MODEL` below) that also lists `<|im_end|>`. Each method then
trains for 100 steps of 256 problems with 16 rollouts each:

```bash
ARM=mwrl  # grpo | maxrl | grpo_size | grpo_minimal | distinct | distinct_size | mwrl
python -m verl.trainer.main_ppo_sync \
  data.train_files=$PWD/runs/rlvr/data/ds_v1/train.parquet data.val_files=$PWD/runs/rlvr/data/ds_v1/test.parquet \
  data.train_batch_size=256 data.max_prompt_length=512 data.max_response_length=4096 \
  actor_rollout_ref.model.path=$MODEL actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.optim.lr=1e-6 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.n=16 actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.agent.default_agent_loop=ds_answer_stop \
  algorithm.adv_estimator=ds_$ARM algorithm.use_kl_in_reward=False \
  reward.custom_reward_function.path=$PWD/packages/mwrl-rlvr/src/mwrl_rlvr/reward.py \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.val_before_train=False trainer.test_freq=-1 \
  trainer.total_training_steps=100 trainer.save_freq=50 trainer.default_local_dir=$PWD/runs/rlvr/ckpt/ds_$ARM \
  +ray_kwargs.ray_init.runtime_env.worker_process_setup_hook=mwrl_rlvr.verl_ext.register \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFER_QUEUE_ENABLE="'1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.MWRL_SIZE_LAMBDA="'0.1'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.MWRL_NU_P="'0.7'" \
  +ray_kwargs.ray_init.runtime_env.env_vars.MWRL_TRAIN_LOG="'$PWD/runs/rlvr/logs/ds_$ARM.credit.jsonl'"
```

On 4 GPUs of 48 GB, the runs also enabled gradient checkpointing, offloaded the FSDP
parameters and optimizer states, capped the update at 8192 tokens per GPU with dynamic
batching, and gave vLLM 55% of GPU memory. `MWRL_TRAIN_LOG` receives one row per step with the
antichain statistics of that step's rollouts, which are fresh problems, so it traces the
training curves.

```bash
# Held-out evaluation: 64 samples for each of the 512 test problems. Shard across GPUs with
# --shard / --num-shards and merge the shards afterwards.
python -m mwrl_rlvr.evaluate run --model runs/rlvr/ckpt/ds_$ARM/global_step_100/actor/huggingface \
  --tokenizer $MODEL --data runs/rlvr/data/ds_v1/test.parquet --B 64 --gpu-mem 0.8 --out runs/rlvr/eval/ds_$ARM
python -m mwrl_rlvr.evaluate merge --out runs/rlvr/eval/ds_$ARM
```

The held-out results of the seven methods are Table 1 of the post
[Correct, minimal, and all](https://tsuituenyue.github.io/blog/correct-minimal-and-all/).
