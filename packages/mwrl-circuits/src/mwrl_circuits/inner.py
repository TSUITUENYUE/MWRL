"""The monotone existential-closure inner optimizer for circuits: EAP-guided dropout.

This is the inner optimizer that realizes the existential closure inside an opened variable
set. Given the opened component set ``S`` (a bitmask), the base predicate
``s0`` (a real ablation forward: is this exact mask faithful) and cheap EAP attributions
over components, it searches for a faithful sub-circuit inside ``S``. It returns the pruned
minimal faithful circuit; a caller reads ``s_exists(S) = found something`` and takes the
returned mask as the canonical witness ``Phi``. Because every accepted state is verified by
``s0``, positives stay sound (theory section 9.3); the perturbed ranking and the dropout only
change which subsets are proposed, so different seeds land on different minimal circuits (the
antichain). The EAP ranking is only a cheap proposal, so the whole search costs ~log d
forwards rather than one per mask.

The functions are model-agnostic (they act on a bitmask through ``s0``); the attribution
computation lives with the model in ``ablation.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch


def eap_guided_descent(mask: int, s0: Callable[[int], bool], attributions: torch.Tensor,
                       d: int, *, max_passes: int = 6) -> int:
    """Chained descent removing components in EAP order (least |attr| first), every removal
    verified by ``s0`` (theory 9.3). The cheap ranking removes safe components early, so it
    reaches a minimal fixed point in fewer verifier calls than a random order."""
    order = torch.argsort(attributions.abs()).tolist()
    current = mask
    for _ in range(max_passes):
        reduced = current
        for h in order:
            bit = 1 << h
            if (reduced & bit) and s0(reduced & ~bit):
                reduced &= ~bit
        if reduced == current:
            break
        current = reduced
    return current


def eap_batch_descent(mask: int, s0: Callable[[int], bool], attributions: torch.Tensor,
                      d: int, *, max_passes: int = 3) -> int:
    """EAP-guided descent with a batched cut: binary-search the largest prefix of
    least-important components whose removal verifies (~log d forwards), then clean up
    greedily. The scalable inner optimizer, sound because every accepted state is verified."""
    order = [h for h in torch.argsort(attributions.abs()).tolist() if (mask >> h) & 1]
    lo, hi, best = 0, len(order), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = mask
        for h in order[:mid]:
            cand &= ~(1 << h)
        if s0(cand):
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    current = mask
    for h in order[:best]:
        current &= ~(1 << h)
    return eap_guided_descent(current, s0, attributions, d, max_passes=max_passes)


def eap_batched_minimize(mask: int, s0_batch: Callable[[list[int]], list[bool]],
                         attributions: torch.Tensor, d: int) -> int:
    """Minimality cleanup with batched verifier calls, ~2 forwards per pass regardless of how
    non-monotone ``s0`` is. Each pass batch-checks the removal of every kept component in ONE
    forward; if the whole individually-removable set is also jointly faithful it commits it,
    else it commits the longest EAP-ordered prefix of that set that verifies (one more batched
    forward over the cumulative removals). Terminates at a locally minimal faithful circuit (no
    single removal stays faithful), every accepted state verified by ``s0`` (theory 9.3). The
    batched analogue of ``eap_guided_descent``, never falling back to one-forward-per-component."""
    order = [int(h) for h in torch.argsort(attributions.abs()).tolist()]
    current = mask
    while True:
        active = [h for h in order if (current >> h) & 1]
        if not active:
            break
        ok = s0_batch([current & ~(1 << h) for h in active])     # one batched forward
        removable = [h for h, r in zip(active, ok, strict=True) if r]
        if not removable:
            break                                                # minimal: no single removal holds
        trial = current
        for h in removable:
            trial &= ~(1 << h)
        if s0_batch([trial])[0]:                                 # jointly removable -> commit all
            current = trial
            continue
        prefixes, m = [], current                                # cumulative EAP-ordered removals
        for h in removable:
            m &= ~(1 << h)
            prefixes.append(m)
        good = s0_batch(prefixes)                                # one batched forward
        commit = current
        for pm, ok_pm in zip(prefixes, good, strict=True):
            if ok_pm:
                commit = pm                                      # longest verified prefix (>= h0 holds)
        current = commit
    return current


def eap_batched_minimize_group(masks: list[int], s0_batch: Callable[[list[int]], list[bool]],
                               attributions: torch.Tensor, d: int) -> list[int]:
    """``eap_batched_minimize`` run over a group of ``k`` masks in lockstep: every verifier call
    concatenates all still-active masks' candidates into ONE ``s0_batch`` forward. Bit-identical to
    calling ``eap_batched_minimize`` on each mask separately (``s0`` is deterministic)."""
    order = [int(h) for h in torch.argsort(attributions.abs()).tolist()]
    current = list(masks)
    done = [False] * len(masks)
    while not all(done):
        singles: list[int] = []
        spans: list[tuple[int, list[int]]] = []          # (start, active) per not-done mask
        for i in range(len(masks)):
            if done[i]:
                spans.append((len(singles), []))
                continue
            active = [h for h in order if (current[i] >> h) & 1]
            spans.append((len(singles), active))
            singles.extend(current[i] & ~(1 << h) for h in active)
        ok_single = s0_batch(singles) if singles else []
        trials: list[int] = []
        trial_of: list[int] = []
        prefs_of: dict[int, list[int]] = {}
        for i in range(len(masks)):
            if done[i]:
                continue
            start, active = spans[i]
            removable = [active[j] for j in range(len(active)) if ok_single[start + j]]
            if not removable:                             # locally minimal
                done[i] = True
                continue
            trial, prefs, m = current[i], [], current[i]
            for h in removable:
                trial &= ~(1 << h)
                m &= ~(1 << h)
                prefs.append(m)
            trials.append(trial)
            trial_of.append(i)
            prefs_of[i] = prefs
        if not trials:
            continue
        ok_joint = s0_batch(trials)
        need_prefix: list[int] = []
        for t, i in enumerate(trial_of):
            if ok_joint[t]:
                current[i] = trials[t]                     # jointly removable -> commit all
            else:
                need_prefix.append(i)
        if need_prefix:
            pcands: list[int] = []
            pspan: list[tuple[int, int]] = []
            for i in need_prefix:
                pspan.append((len(pcands), len(prefs_of[i])))
                pcands.extend(prefs_of[i])
            ok_pref = s0_batch(pcands)
            for idx, i in enumerate(need_prefix):
                start, length = pspan[idx]
                commit = current[i]
                for j in range(length):
                    if ok_pref[start + j]:
                        commit = prefs_of[i][j]            # longest verified prefix
                current[i] = commit
    return current


def eap_guided_dropout_batch(masks: list[int], s0_batch: Callable[[list[int]], list[bool]],
                             attributions: torch.Tensor, d: int, *, rngs: list[np.random.Generator],
                             iters: int = 80, noise: float = 1.0, max_drop: int = 4) -> list[int]:
    """Batched ``eap_guided_dropout`` over a group of ``k`` masks that share ``s0_batch`` and
    ``attributions`` (same context/behavior). Each mask keeps its own perturbed order and RNG, so
    per-mask it is bit-identical to the sequential ``eap_guided_dropout`` -- but each phase's
    verifier candidates from all ``k`` masks are packed into one ``s0_batch`` forward, so the group
    costs ~one search's worth of model calls instead of ``k``."""
    k = len(masks)
    imp = attributions.abs().cpu().numpy()
    orders = []
    for i in range(k):
        noisy = imp + noise * rngs[i].standard_normal(d) * (imp.std() + 1e-9)
        orders.append([int(h) for h in np.argsort(noisy) if (masks[i] >> int(h)) & 1])

    # Phase 1, bulk cut: k lockstep BINARY searches for each mask's largest removable EAP-ordered
    # prefix. Each step probes every still-searching mask's mid-prefix in ONE s0_batch forward, so
    # the group costs ~log d forwards -- NOT a full scan of all d prefixes per mask, which at
    # batch_masks << d is ~d/batch_masks forwards and was the dominant GPU cost. Matches the
    # sequential binary-search fallback per mask (Phase 2 + minimize handle non-monotone dips).
    def _cut(i: int, mid: int) -> int:
        cand = masks[i]
        for h in orders[i][:mid]:
            cand &= ~(1 << h)
        return cand

    lo = [0] * k
    hi = [len(orders[i]) for i in range(k)]
    best = [0] * k
    while any(lo[i] <= hi[i] for i in range(k)):
        probe, where = [], []
        for i in range(k):
            if lo[i] <= hi[i]:
                mid = (lo[i] + hi[i]) // 2
                probe.append(_cut(i, mid))
                where.append((i, mid))
        oks = s0_batch(probe)
        for (i, mid), ok in zip(where, oks, strict=True):
            if ok:
                best[i], lo[i] = mid, mid + 1
            else:
                hi[i] = mid - 1
    current = [_cut(i, best[i]) for i in range(k)]

    # Phase 2, attribution-weighted dropout: one batched forward per round over all masks' proposals.
    def _propose(i: int) -> int | None:
        active = [h for h in range(d) if (current[i] >> h) & 1]
        if len(active) <= 1:
            return None
        w = 1.0 / (imp[active] + 0.05)
        w = w / w.sum()
        kk = min(int(rngs[i].integers(1, max_drop + 1)), len(active) - 1)
        c = current[i]
        for h in rngs[i].choice(active, size=kk, replace=False, p=w):
            c &= ~(1 << int(h))
        return c

    for _ in range(max(1, iters // 4)):
        cands, spans = [], []
        for i in range(k):
            start = len(cands)
            for _ in range(max(4, iters)):
                c = _propose(i)
                if c is not None:
                    cands.append(c)
            spans.append((start, len(cands) - start))
        if not cands:
            break
        ok = s0_batch(cands)
        for i in range(k):
            start, length = spans[i]
            verified = [cands[start + j] for j in range(length) if ok[start + j]]
            if verified:
                current[i] = min(verified, key=lambda m: bin(m).count("1"))
    return eap_batched_minimize_group(current, s0_batch, attributions, d)


def eap_guided_dropout(mask: int, s0: Callable[[int], bool], attributions: torch.Tensor,
                       d: int, *, rng: np.random.Generator, iters: int = 80, noise: float = 1.0,
                       max_drop: int = 4,
                       s0_batch: Callable[[list[int]], list[bool]] | None = None) -> int:
    """Dropout inside EAP: fast, exploratory, sound, and antichain-covering in one pass.

    Phase 1, perturbed bulk cut: perturb the EAP ranking with per-run noise (antichain
    diversity across seeds) and binary-search the largest prefix of least-important
    components whose removal verifies (~log d forwards). Phase 2, attribution-weighted
    multi-head dropout: repeatedly drop a random small subset of the kept components biased
    toward low |attr|, keeping any reduction that verifies, which escapes non-monotone traps
    while wasting no probes on load-bearing components. Every accepted state is verified by
    ``s0``, so positives stay sound; the noise and dropout only change which subsets are tried.
    """
    imp = attributions.abs().cpu().numpy()
    noisy = imp + noise * rng.standard_normal(d) * (imp.std() + 1e-9)
    order = [int(h) for h in np.argsort(noisy) if (mask >> int(h)) & 1]

    # Phase 1, bulk cut: remove the longest prefix of least-important components that verifies.
    if s0_batch is not None and order:
        prefixes, cand = [], mask
        for h in order:
            cand &= ~(1 << h)
            prefixes.append(cand)
        oks = s0_batch(prefixes)                     # one batched forward over every prefix cut
        best = max((i for i, ok in enumerate(oks, 1) if ok), default=0)
    else:
        lo, hi, best = 0, len(order), 0              # sequential binary search (unbatched fallback)
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = mask
            for h in order[:mid]:
                cand &= ~(1 << h)
            best, lo, hi = (mid, mid + 1, hi) if s0(cand) else (best, lo, mid - 1)
    current = mask
    for h in order[:best]:
        current &= ~(1 << h)

    # Phase 2, attribution-weighted dropout: escape non-monotone traps by dropping random
    # low-|attr| subsets and keeping any reduction that verifies. Batched: a few rounds, each
    # proposing many drops evaluated in one forward, committing the smallest verified reduction.
    def _propose(active):
        w = 1.0 / (imp[active] + 0.05)
        w = w / w.sum()
        k = min(int(rng.integers(1, max_drop + 1)), len(active) - 1)
        c = current
        for h in rng.choice(active, size=k, replace=False, p=w):
            c &= ~(1 << int(h))
        return c

    if s0_batch is not None:
        for _ in range(max(1, iters // 4)):
            active = [h for h in range(d) if (current >> h) & 1]
            if len(active) <= 1:
                break
            cands = [_propose(active) for _ in range(max(4, iters))]
            verified = [c for c, ok in zip(cands, s0_batch(cands), strict=True) if ok]
            if not verified:
                break
            current = min(verified, key=lambda m: bin(m).count("1"))
        return eap_batched_minimize(current, s0_batch, attributions, d)

    for _ in range(iters):
        active = [h for h in range(d) if (current >> h) & 1]
        if len(active) <= 1:
            break
        cand = _propose(active)
        if s0(cand):
            current = cand
    return eap_guided_descent(current, s0, attributions, d, max_passes=1)
