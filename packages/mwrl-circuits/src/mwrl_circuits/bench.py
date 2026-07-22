"""Benchmark task families: MMLU subjects (and BoolQ) as next-token probes.

Each MMLU subject becomes one behavior (a domain expert for the MoE claim). Answers are
single first tokens (" A".." D", " yes"/" no"), so the KL verifier, the logit-diff
evaluation, and both routers run unchanged. Prompts are length-filtered and arranged so
cyclically adjacent rows have different answers (the ``_task`` counterfactual
requirement), with answers balanced within each subject.

The experiment design is dense-competence-first: sweep the dense model over all 57
subjects (``bench_dense --subjects all``), keep every subject above the accuracy bar, and
discover circuits for all that pass, so the expert set is selected by what the model can
do and never hand-picked.
"""

from __future__ import annotations

import random

from mwrl_circuits.tasks import Task, _task

# The 10 subjects passing the dense-competence bar (acc >= 0.92) on Qwen3-1.7B, from the
# all-57 sweep runs/bench_dense_1p7b.json (cluster job 476567): selected by a stated bar
# on what the dense model can do, never hand-picked.
MMLU_SUBJECTS = [
    "marketing",
    "astronomy", "computer_security", "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "international_law", "jurisprudence", "management",
    "us_foreign_policy",
]
_LETTERS = ["A", "B", "C", "D"]
_MAX_PROMPT_CHARS = 700          # ~<=180 tokens: keeps the batched ablation memory bounded
_MAX_PASSAGE_CHARS = 400

_MMLU_ROWS = None                # the single "all" test split, loaded once
_MMLU_INDEX: dict[str, list[int]] = {}


def _mmlu_rows():
    global _MMLU_ROWS
    if _MMLU_ROWS is None:
        from datasets import load_dataset
        _MMLU_ROWS = load_dataset("cais/mmlu", "all", split="test")
        for i, s in enumerate(_MMLU_ROWS["subject"]):
            _MMLU_INDEX.setdefault(s, []).append(i)
    return _MMLU_ROWS


def all_mmlu_subjects() -> list[str]:
    _mmlu_rows()
    return sorted(_MMLU_INDEX)


def _fmt_mmlu(question: str, choices: list[str]) -> str:
    lines = [question.strip()]
    lines += [f"{letter}. {choice}" for letter, choice in zip(_LETTERS, choices)]
    lines.append("Answer:")
    return "\n".join(lines)


def _select(subject: str, k: int, seed: int,
            exclude: frozenset[int] = frozenset()) -> list[tuple[int, str, str]]:
    """Deterministic balanced selection: (dataset_row_id, prompt, letter) interleaved
    A,B,C,D,... so cyclically adjacent rows always differ. With ``exclude`` empty this is
    byte-identical to the discovery selection that trained runs used (same index list,
    same shuffle, same fill); ``exclude`` removes dataset rows BEFORE sampling, so a
    holdout selection is disjoint from discovery by row id, never merely by seed."""
    rows = _mmlu_rows()
    idx = [i for i in _MMLU_INDEX.get(subject, ()) if i not in exclude]
    if not idx:
        raise ValueError(f"unknown or exhausted MMLU subject {subject!r}")
    random.Random(seed).shuffle(idx)
    pool: dict[str, list[tuple[int, str, str]]] = {letter: [] for letter in _LETTERS}
    want = max(1, k // len(_LETTERS))
    for i in idx:
        r = rows[i]
        letter = _LETTERS[int(r["answer"])]
        if len(pool[letter]) >= want:
            continue
        prompt = _fmt_mmlu(r["question"], list(r["choices"]))
        if len(prompt) > _MAX_PROMPT_CHARS:
            continue
        pool[letter].append((i, prompt, letter))
        if all(len(v) >= want for v in pool.values()):
            break
    per_letter = min(len(v) for v in pool.values())
    if per_letter < 1:
        raise ValueError(f"mmlu/{subject}: not enough short questions per answer letter")
    return [pool[letter][i] for i in range(per_letter) for letter in _LETTERS]


def mmlu_task(subject: str, k: int = 12, seed: int = 0) -> Task:
    """One MMLU subject as a Task (the discovery selection; see ``_select``)."""
    chosen = _select(subject, k, seed)
    return _task(f"mmlu_{subject}", [(p, f" {letter}", "") for _, p, letter in chosen])


def mmlu_discovery_ids(subject: str, k: int = 12, seed: int = 0) -> list[int]:
    """The dataset row ids of the exact discovery questions (for disjointness proofs)."""
    return [i for i, _, _ in _select(subject, k, seed)]


def mmlu_holdout_task(subject: str, k: int = 24, seed: int = 1, *, discovery_k: int = 12,
                      discovery_seed: int = 0) -> tuple[Task, list[int], list[int]]:
    """A held-out Task disjoint from discovery BY DATASET ROW ID: reconstruct the exact
    discovery selection, exclude those rows, then sample. Returns
    (task, holdout_row_ids, discovery_row_ids)."""
    discovery = mmlu_discovery_ids(subject, discovery_k, discovery_seed)
    chosen = _select(subject, k, seed, exclude=frozenset(discovery))
    task = _task(f"mmlu_{subject}", [(p, f" {letter}", "") for _, p, letter in chosen])
    return task, [i for i, _, _ in chosen], discovery


def mmlu_family(subjects: list[str] | None = None, k: int = 12) -> list[Task]:
    return [mmlu_task(s, k=k) for s in (subjects or MMLU_SUBJECTS)]


def boolq_task(k: int = 12, seed: int = 0) -> Task:
    """BoolQ as a Task: short-passage questions, answers alternating yes/no."""
    from datasets import load_dataset

    rows = load_dataset("google/boolq", split="validation")
    rng = random.Random(seed)
    pool: dict[str, list[tuple[str, str, str]]] = {"yes": [], "no": []}
    order = list(range(len(rows)))
    rng.shuffle(order)
    half = max(1, k // 2)
    for i in order:
        r = rows[i]
        if len(r["passage"]) > _MAX_PASSAGE_CHARS:
            continue
        answer = "yes" if r["answer"] else "no"
        if len(pool[answer]) >= half:
            continue
        prompt = (f"{r['passage'].strip()}\n"
                  f"Question: {r['question'].strip()}?\n"
                  f"Answer (yes or no):")
        if len(prompt) > _MAX_PROMPT_CHARS:
            continue
        pool[answer].append((prompt, f" {answer}", ""))
        if all(len(v) >= half for v in pool.values()):
            break
    if not all(len(v) >= half for v in pool.values()):
        raise ValueError("boolq: not enough short passages per answer")
    rows_out = [pool[a][i] for i in range(half) for a in ("yes", "no")]
    return _task("boolq", rows_out)


def boolq_family(k: int = 12) -> list[Task]:
    return [boolq_task(k=k)]
