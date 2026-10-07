"""Data sufficiency: a math-reasoning task whose answers form a partial order.

An instance hides integers (x0, y0) in {1..N}^2 and lists n statements that are all true
at the hidden point. A proposal is a subset S of the statements plus a value for x. It
succeeds when S determines x uniquely on the domain and the stated value is x0.

Sufficiency is monotone: adding statements shrinks the solution set, which always keeps
the hidden point, so a sufficient S stays sufficient under supersets. The minimal
sufficient subsets therefore form an antichain M, and M is computed exactly by checking
all 2^n subsets on the N^2 grid.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np


def popcount(mask: int) -> int:
    return bin(mask).count("1")


def mask_to_list(mask: int) -> list[int]:
    """0-based statement indices in the mask."""
    return [i for i in range(mask.bit_length()) if (mask >> i) & 1]


# ---------------------------------------------------------------------------- statements

@dataclass(frozen=True)
class Statement:
    family: str
    text: str
    truth: np.ndarray = field(repr=False)  # bool, shape (N, N), index [x-1, y-1]


def _grid(n_dom: int) -> tuple[np.ndarray, np.ndarray]:
    v = np.arange(1, n_dom + 1)
    return np.meshgrid(v, v, indexing="ij")


def _is_prime(k: int) -> bool:
    return k >= 2 and all(k % d for d in range(2, int(math.isqrt(k)) + 1))


def _digit_sum(a: np.ndarray) -> np.ndarray:
    return (a // 10) + (a % 10)


def candidate_statements(x0: int, y0: int, n_dom: int, rng: np.random.Generator) -> list[Statement]:
    """Every templated statement that is true at (x0, y0), with randomized parameters."""
    X, Y = _grid(n_dom)
    out: list[Statement] = []

    def add(family: str, text: str, truth: np.ndarray) -> None:
        assert truth[x0 - 1, y0 - 1], (family, text)
        out.append(Statement(family, text, truth))

    # equations in x and y
    add("sum", f"x + y = {x0 + y0}", X + Y == x0 + y0)
    add("diff", f"x - y = {x0 - y0}", X - Y == x0 - y0)
    if x0 != y0:
        add("absdiff", f"x and y differ by {abs(x0 - y0)}", np.abs(X - Y) == abs(x0 - y0))
    add("prod", f"x * y = {x0 * y0}", X * Y == x0 * y0)
    a, b = (int(v) for v in rng.choice([2, 3, 4], size=2, replace=True))
    add("lin", f"{a}x + {b}y = {a * x0 + b * y0}", a * X + b * Y == a * x0 + b * y0)
    add("sqsum", f"x^2 + y^2 = {x0 ** 2 + y0 ** 2}", X ** 2 + Y ** 2 == x0 ** 2 + y0 ** 2)
    add("max", f"the larger of x and y is {max(x0, y0)}", np.maximum(X, Y) == max(x0, y0))
    add("min", f"the smaller of x and y is {min(x0, y0)}", np.minimum(X, Y) == min(x0, y0))

    # order between x and y
    if x0 > y0:
        add("cmp", "x is greater than y", X > Y)
    elif x0 < y0:
        add("cmp", "x is less than y", X < Y)
    else:
        add("cmp", "x is equal to y", X == Y)
    if y0 % x0 == 0 and x0 > 1 and y0 != x0:
        add("divides", "x divides y", Y % X == 0)
    if x0 % y0 == 0 and y0 > 1 and y0 != x0:
        add("divides", "y divides x", X % Y == 0)

    # bounds (one-sided, randomized threshold)
    for var, v0, A in (("x", x0, X), ("y", y0, Y)):
        if v0 < n_dom:
            k = int(rng.integers(v0 + 1, n_dom + 1))
            add(f"ub_{var}", f"{var} is less than {k}", A < k)
        if v0 > 1:
            k = int(rng.integers(1, v0))
            add(f"lb_{var}", f"{var} is greater than {k}", A > k)

    # parity
    for var, v0, A in (("x", x0, X), ("y", y0, Y)):
        add(f"par_{var}", f"{var} is {'even' if v0 % 2 == 0 else 'odd'}", A % 2 == v0 % 2)
    add("par_sum", f"x + y is {'even' if (x0 + y0) % 2 == 0 else 'odd'}", (X + Y) % 2 == (x0 + y0) % 2)

    # remainders and multiples
    for var, v0, A in (("x", x0, X), ("y", y0, Y)):
        m = int(rng.choice([3, 4, 5, 6, 7]))
        add(f"mod_{var}", f"when {var} is divided by {m}, the remainder is {v0 % m}", A % m == v0 % m)
        divs = [d for d in (3, 4, 5, 6, 7, 8, 9, 10) if v0 % d == 0]
        if divs:
            d = int(rng.choice(divs))
            add(f"mult_{var}", f"{var} is a multiple of {d}", A % d == 0)

    # number properties of x and y
    for var, v0, A in (("x", x0, X), ("y", y0, Y)):
        primes = np.vectorize(_is_prime)(A)
        add(f"prime_{var}", f"{var} is {'a' if _is_prime(v0) else 'not a'} prime number",
            primes == _is_prime(v0))
        if v0 >= 10:
            ds = int(_digit_sum(np.array(v0)))  # a one-digit number's digit sum is itself
            add(f"digits_{var}", f"the digits of {var} add up to {ds}", _digit_sum(A) == ds)
    return out


# ---------------------------------------------------------------------------- instances

@dataclass
class Instance:
    x0: int
    y0: int
    n_dom: int
    statements: list[Statement]
    sufficient: np.ndarray        # bool, shape (2^n,), indexed by subset mask
    antichain: list[int]          # minimal sufficient masks

    @property
    def n(self) -> int:
        return len(self.statements)


def sufficiency_table(statements: list[Statement]) -> np.ndarray:
    """suff[mask] = the statements in mask determine x uniquely on the grid."""
    n = len(statements)
    flat = np.stack([s.truth.reshape(-1) for s in statements])  # (n, N*N)
    n_dom = statements[0].truth.shape[0]
    feas = np.ones((1 << n, flat.shape[1]), dtype=bool)
    for mask in range(1, 1 << n):
        low = (mask & -mask).bit_length() - 1
        feas[mask] = feas[mask & (mask - 1)] & flat[low]
    xs_present = feas.reshape(1 << n, n_dom, n_dom).any(axis=2)  # (2^n, N): x value still possible
    return xs_present.sum(axis=1) == 1


def minimal_antichain(suff: np.ndarray, n: int) -> list[int]:
    """Masks that are sufficient while every single-statement removal is not. For a monotone
    table this is exactly the set of minimal elements of the sufficient family."""
    out = []
    for mask in range(1, 1 << n):
        if suff[mask] and all(not suff[mask & ~(1 << i)] for i in mask_to_list(mask)):
            out.append(mask)
    return out


def generate_instance(rng: np.random.Generator, *, n_dom: int = 30, n: int = 8,
                      min_witness: int = 2, antichain_range: tuple[int, int] = (2, 8),
                      max_tries: int = 10_000) -> Instance:
    """Rejection-sample an instance whose full statement set determines x, whose minimal
    sufficient sets all use at least ``min_witness`` statements, and whose antichain size
    lies in ``antichain_range``."""
    for _ in range(max_tries):
        x0, y0 = (int(v) for v in rng.integers(1, n_dom + 1, size=2))
        cands = candidate_statements(x0, y0, n_dom, rng)
        order = rng.permutation(len(cands))
        chosen: list[Statement] = []
        seen: set[bytes] = set()
        for i in order:  # distinct truth tables and at most one statement per family
            s = cands[i]
            key = s.truth.tobytes()
            if key in seen or any(c.family == s.family for c in chosen):
                continue
            seen.add(key)
            chosen.append(s)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            continue
        suff = sufficiency_table(chosen)
        if not suff[(1 << n) - 1]:
            continue
        anti = minimal_antichain(suff, n)
        if min(popcount(m) for m in anti) < min_witness:
            continue
        if not antichain_range[0] <= len(anti) <= antichain_range[1]:
            continue
        return Instance(x0, y0, n_dom, chosen, suff, anti)
    raise RuntimeError("no instance accepted; loosen the filters")


# ---------------------------------------------------------------------------- prompt / answer

PROMPT_TEMPLATE = (
    "x and y are whole numbers from 1 to {n_dom}. The following {n} statements about them are all true:\n"
    "{statements}\n"
    "Choose a set of these statements that together determine the value of x, such that no "
    "statement in your set can be dropped. Several different sets may work; give one.\n"
    "Think step by step inside <think> </think>, then give the statement numbers and the value of x "
    "inside <answer> </answer>, for example: <answer>statements: 2, 5; x = 11</answer>"
)


def render_prompt(inst: Instance) -> str:
    lines = "\n".join(f"({i + 1}) {s.text}" for i, s in enumerate(inst.statements))
    return PROMPT_TEMPLATE.format(n_dom=inst.n_dom, n=inst.n, statements=lines)


_ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_STMTS = re.compile(r"statements?\s*[:=]?\s*(.*?)(?:;|\bx\s*=|$)", re.S | re.I)
_XVAL = re.compile(r"\bx\s*=\s*(-?\d+)", re.I)


@dataclass
class Parsed:
    ok: bool
    mask: int = 0
    x: int | None = None


def parse_answer(text: str, n: int) -> Parsed:
    """Read the last <answer> block: a set of 1-based statement numbers and the value of x."""
    blocks = _ANSWER.findall(text)
    if not blocks:
        return Parsed(False)
    body = blocks[-1]
    m_s, m_x = _STMTS.search(body), _XVAL.search(body)
    if not m_s or not m_x:
        return Parsed(False)
    nums = [int(v) for v in re.findall(r"\d+", m_s.group(1))]
    if not nums or any(v < 1 or v > n for v in nums):
        return Parsed(False)
    mask = 0
    for v in nums:
        mask |= 1 << (v - 1)
    return Parsed(True, mask, int(m_x.group(1)))


@dataclass
class Verdict:
    parsed: bool
    success: bool      # S determines x and the value is right
    minimal: bool      # success and S is a minimal sufficient set
    mask: int
    size: int


def verify(text: str, *, n: int, x0: int, sufficient: np.ndarray, antichain: list[int]) -> Verdict:
    p = parse_answer(text, n)
    if not p.ok:
        return Verdict(False, False, False, 0, 0)
    success = bool(sufficient[p.mask]) and p.x == x0
    return Verdict(True, success, success and p.mask in set(antichain), p.mask, popcount(p.mask))
