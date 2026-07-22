"""Self-contained capability hierarchy: families of behaviors, each split into
sub-tasks, every one a next-token logit-diff probe (correct vs foil). No external
data. Each ``Task`` scores a circuit by the mean margin of the correct answer token
over a foil token at the final position; a family task is the union of its sub-tasks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    name: str
    prompts: list[str]  # clean prompts
    corrupt: list[str]  # counterfactual (interchange source) prompts, aligned to prompts
    correct: list[str]  # clean answer strings (leading space), scored by first token
    foil: list[str]     # the counterfactual's answer, scored by first token

    def token_ids(self, tokenizer) -> tuple[list[int], list[int]]:
        def first(answer: str) -> int:
            return tokenizer.encode(answer, add_special_tokens=False)[0]
        return [first(a) for a in self.correct], [first(a) for a in self.foil]


def _task(name, rows: list[tuple[str, str, str]]) -> Task:
    """Rows are ``(clean_prompt, correct_answer, _)``. The counterfactual for row ``i`` is the
    next row (cyclic): its prompt is the interchange source patched in for ablated components and
    its answer is the foil, so the metric is logit(correct) - logit(counterfactual answer).
    Adjacent rows must have different first-token answers (the lists below satisfy this)."""
    prompts = [r[0] for r in rows]
    correct = [r[1] for r in rows]
    n = len(rows)
    corrupt = [prompts[(i + 1) % n] for i in range(n)]
    foil = [correct[(i + 1) % n] for i in range(n)]
    return Task(name, prompts, corrupt, correct, foil)


# -- factual recall ------------------------------------------------------------
_CAPITALS = [("France", "Paris", "Berlin"), ("Japan", "Tokyo", "Beijing"),
             ("Italy", "Rome", "Madrid"), ("Egypt", "Cairo", "Lagos"),
             ("Canada", "Ottawa", "Toronto"), ("Spain", "Madrid", "Lisbon"),
             ("Russia", "Moscow", "Kiev"), ("Brazil", "Brasilia", "Lima"),
             ("Greece", "Athens", "Ankara"), ("Norway", "Oslo", "Helsinki")]
_ELEMENTS = [("gold", "Au", "Ag"), ("iron", "Fe", "Cu"), ("oxygen", "O", "H"),
             ("sodium", "Na", "Cl"), ("carbon", "C", "N"), ("helium", "He", "Li"),
             ("silver", "Ag", "Au"), ("copper", "Cu", "Fe"), ("lead", "Pb", "Sn"),
             ("zinc", "Zn", "Mg")]


def _factual() -> list[Task]:
    cap = _task("country_capital",
                [(f"The capital of {c} is", f" {a}", f" {b}") for c, a, b in _CAPITALS])
    el = _task("element_symbol",
               [(f"The chemical symbol for {c} is", f" {a}", f" {b}") for c, a, b in _ELEMENTS])
    return [cap, el]


# -- arithmetic ----------------------------------------------------------------
# Digits do NOT merge with a leading space under BPE (" 7" -> [space, "7"]), so the
# space goes in the prompt and answers are bare digits, whose first token is the digit.
def _arith() -> list[Task]:
    add = _task("addition", [(f"{a} + {b} = ", f"{a + b}", f"{a + b + 1}")
                             for a, b in [(1, 1), (1, 2), (1, 3), (2, 3), (3, 3),
                                          (3, 4), (4, 4), (4, 5), (3, 5), (2, 5)]])
    sub = _task("subtraction", [(f"{a} - {b} = ", f"{a - b}", f"{a - b + 1}")
                               for a, b in [(9, 2), (7, 3), (8, 1), (6, 4), (5, 2),
                                            (9, 5), (7, 1), (8, 6), (6, 2), (9, 3)]])
    return [add, sub]


# -- linguistic agreement ------------------------------------------------------
def _agreement() -> list[Task]:
    sv = _task("subject_verb", [
        ("The keys to the cabinet", " are", " is"),
        ("The author of the books", " is", " are"),
        ("The dogs near the fence", " are", " is"),
        ("The child with the toys", " is", " are"),
        ("The players on the team", " are", " is"),
        ("The manager of the stores", " is", " are"),
        ("The books on the shelf", " are", " is"),
        ("The woman with the cats", " is", " are"),
        ("The students in the class", " are", " is"),
        ("The leader of the nations", " is", " are")])
    art = _task("article_number", [
        ("I saw a single", " cat", " cats"), ("She bought several", " books", " book"),
        ("He has many", " friends", " friend"), ("There is one", " apple", " apples"),
        ("We need a few", " chairs", " chair"), ("They found one", " key", " keys"),
        ("I ate three", " apples", " apple"), ("She owns a", " car", " cars"),
        ("He met two", " people", " person"), ("We saw a", " bird", " birds")])
    return [sv, art]


def _bench_mmlu() -> list[Task]:
    from mwrl_circuits.bench import mmlu_family   # lazy: needs the ``datasets`` package
    return mmlu_family()


def _bench_boolq() -> list[Task]:
    from mwrl_circuits.bench import boolq_family
    return boolq_family()


FAMILIES = {"factual_recall": _factual, "arithmetic": _arith, "agreement": _agreement,
            "mmlu": _bench_mmlu, "boolq": _bench_boolq}


def build_hierarchy(family_names: list[str]) -> dict[str, dict]:
    """Return {family_name: {"family": Task, "subtasks": [Task, ...]}} (the 2-level form)."""
    out: dict[str, dict] = {}
    for fam in family_names:
        subs = FAMILIES[fam]()
        merged = Task(fam, [p for s in subs for p in s.prompts],
                      [c for s in subs for c in s.corrupt],
                      [c for s in subs for c in s.correct],
                      [f for s in subs for f in s.foil])
        out[fam] = {"family": merged, "subtasks": subs}
    return out


# -- extra leaves for a richer 3-level tree (4 families x 3 subtasks x 2 leaves = 24) ----------
_LANGUAGES = [("France", "French"), ("Japan", "Japanese"), ("Italy", "Italian"),
              ("Spain", "Spanish"), ("Germany", "German"), ("Russia", "Russian"),
              ("Brazil", "Portuguese"), ("Egypt", "Arabic"), ("Greece", "Greek"), ("Korea", "Korean")]
_STATES = [("oxygen", "gas"), ("gold", "solid"), ("mercury", "liquid"), ("iron", "solid"),
           ("helium", "gas"), ("carbon", "solid"), ("nitrogen", "gas"), ("bromine", "liquid")]
_CONTINENTS = [("Egypt", "Africa"), ("Japan", "Asia"), ("France", "Europe"), ("Brazil", "America"),
               ("Kenya", "Africa"), ("India", "Asia"), ("Spain", "Europe"), ("Chile", "America"),
               ("Nigeria", "Africa"), ("China", "Asia")]
_CURRENCIES = [("Japan", "yen"), ("India", "rupee"), ("Russia", "ruble"), ("Mexico", "peso"),
               ("Britain", "pound"), ("Poland", "zloty"), ("Sweden", "krona"), ("Turkey", "lira"),
               ("Thailand", "baht"), ("Israel", "shekel")]
_PAST = [("walk", "walked"), ("play", "played"), ("jump", "jumped"), ("cook", "cooked"),
         ("clean", "cleaned"), ("paint", "painted"), ("call", "called"), ("watch", "watched"),
         ("help", "helped"), ("open", "opened")]
_THIRD = [("walk", "walks"), ("run", "runs"), ("eat", "eats"), ("sleep", "sleeps"), ("read", "reads"),
          ("write", "writes"), ("sing", "sings"), ("jump", "jumps"), ("swim", "swims"), ("cook", "cooks")]
_NOUNS = ["cat", "dog", "book", "tree", "car", "bird", "cup", "hat", "key", "lamp"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
_ALPHA = list("ABCDEFGHIJKLMNOPQRSTU")


def _merge(name: str, leaves: list[Task], cap: int | None = None) -> Task:
    """Union of the child leaves' probes (the subtask/family task). ``cap`` keeps at most that many
    prompts per child: the coarse levels are estimated from a REPRESENTATIVE SAMPLE of each child's
    prompts so the expensive family/subtask searches stay tractable. This subsamples evaluation
    prompts only -- the component search space D(c) is untouched, so the policy still discovers the
    subset over all components (leaves, the reported level, keep every prompt)."""
    def take(attr: str) -> list:
        return [x for t in leaves for x in getattr(t, attr)[:cap]]
    return Task(name, take("prompts"), take("corrupt"), take("correct"), take("foil"))


def _cyc(name: str, items: list[str], template: str, *, step: int = 1, k: int = 10) -> Task:
    """A sequence leaf whose answer is the item ``step`` positions ahead (days/months/letters).
    ``valid`` skips the wrap indices so the correct answer never cycles to a wrong item; distinct
    injective successors then guarantee adjacent rows differ (the cyclic-foil requirement)."""
    n = len(items)
    valid = [i for i in range(n) if 0 <= i + step < n][:k]
    return _task(name, [(template.format(items[i]), f" {items[i + step]}", "") for i in valid])


def build_taxonomy() -> dict[str, dict]:
    """3-level hash-table taxonomy family -> subtask -> subsubtask (leaf), 4 x 3 x 2 = 24 leaves:
    {family: {"family": Task, "subtasks": {subtask: {"subtask": Task, "leaves": [leaf Task, ...]}}}}.
    Leaves are the finest behaviors; subtask and family tasks are the unions of their descendants.
    Every leaf is a deterministic single-first-token probe with a clean cyclic foil; the dense-model
    validation (validate_leaves) drops any leaf the target model can't do before a run commits."""
    cap, sym = _factual()
    add, sub = _arith()
    sv, art = _agreement()
    lang = _task("country_language",
                 [(f"The official language of {c} is", f" {a}", "") for c, a in _LANGUAGES])
    state = _task("element_state",
                  [(f"At room temperature, {c} is a", f" {a}", "") for c, a in _STATES])
    cont = _task("country_continent",
                 [(f"{c} is located on the continent of", f" {a}", "") for c, a in _CONTINENTS])
    curr = _task("country_currency",
                 [(f"The currency of {c} is the", f" {a}", "") for c, a in _CURRENCIES])
    # digits do NOT merge with a leading space under BPE (" 7" -> [space, "7"], so first(" 7") is
    # the space token, identical for correct and foil -> zero margin). Like _arith, put the space in
    # the prompt and answer with a bare digit, whose first token is the digit itself.
    gpairs = [(7, 3), (4, 9), (8, 2), (5, 1), (6, 9), (3, 8), (9, 4), (2, 7), (8, 5), (1, 6)]
    greater = _task("number_greater",
                    [(f"The larger of {a} and {b} is ", f"{max(a, b)}", "") for a, b in gpairs])
    smaller = _task("number_smaller",
                    [(f"The smaller of {a} and {b} is ", f"{min(a, b)}", "") for a, b in gpairs])
    nextn = _task("next_number",
                  [(f"The number after {n} is ", f"{n + 1}", "") for n in [0, 3, 5, 1, 7, 2, 6, 4, 8, 2]])
    prevn = _task("prev_number",
                  [(f"The number before {n} is ", f"{n - 1}", "") for n in [4, 1, 6, 2, 8, 3, 9, 5, 7, 3]])
    past = _task("past_tense", [(f"Today I {v}. Yesterday I", f" {a}", "") for v, a in _PAST])
    third = _task("third_person", [(f"They {v}. He", f" {a}", "") for v, a in _THIRD])
    plural = _task("plural_noun", [(f"One {w}, two", f" {w}s", "") for w in _NOUNS])
    singular = _task("singular_noun", [(f"Two {w}s, one", f" {w}", "") for w in _NOUNS])
    tax = {
        "knowledge": {"geography": [cap, cont], "culture": [lang, curr], "chemistry": [sym, state]},
        "arithmetic": {"basic": [add, sub], "comparison": [greater, smaller],
                       "sequence": [nextn, prevn]},
        "grammar": {"agreement": [sv, art], "tense": [past, third], "number": [plural, singular]},
        "temporal": {"days": [_cyc("day_after", _DAYS, "The day after {} is", step=1),
                              _cyc("day_before", _DAYS, "The day before {} is", step=-1)],
                     "months": [_cyc("month_after", _MONTHS, "The month after {} is", step=1),
                                _cyc("month_before", _MONTHS, "The month before {} is", step=-1)],
                     "alphabet": [_cyc("letter_after", _ALPHA, "The letter after {} is", step=1),
                                  _cyc("letter_before", _ALPHA, "The letter before {} is", step=-1)]},
    }
    out: dict[str, dict] = {}
    for fam, subs in tax.items():
        sub_out = {s: {"subtask": _merge(s, lv), "leaves": lv} for s, lv in subs.items()}
        fam_leaves = [lf for lv in subs.values() for lf in lv]
        out[fam] = {"family": _merge(fam, fam_leaves), "subtasks": sub_out}
    return out
