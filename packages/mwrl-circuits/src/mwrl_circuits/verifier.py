"""Monotone existential-closure verifier for a behavior on a frozen LLM.

The base predicate ``s0(S)`` runs
the model with the complement of ``S`` resample-ablated (patched from a counterfactual prompt, the
ACDC/EAP interchange intervention) and asks whether the ablated model reproduces the full model's
next-token DISTRIBUTION within error -- not whether one answer beats one foil. Concretely, with the
interchange signal ``KL_empty = KL(P_clean || P_fully-ablated)`` (how far patching everything from
the corrupt run pulls the distribution, the distributional analogue of the old full margin),

    s0(S) = 1  iff  mean_prompts KL( P_clean(.|x) || P_S(.|x) )  <=  (1 - tau) * KL_empty,

so keeping only ``S`` must recover tau of that distributional signal. ``s0`` is non-monotone, so it
is never the witness test. The witness test is the existential closure

    s_exists(S) = 1  iff  there is a faithful sub-circuit inside S,

computed by the EAP-guided dropout inner optimizer (``inner.py``): it uses cheap component
attributions to propose drops and verifies every accepted state with ``s0``, returning the
pruned minimal faithful circuit ``Phi(S)``. ``s_exists`` is monotone (opening more only helps
the inner search) with the same minimal witnesses, so the credit algebra applies. Each mask is
evaluated once and cached; the per-mask RNG makes it a deterministic function of the mask, so
antichain diversity comes from the policy proposing different S.
"""

from __future__ import annotations

import numpy as np
import torch

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.inner import eap_guided_dropout, eap_guided_dropout_batch
from mwrl_circuits.tasks import Task


class CircuitVerifier:
    def __init__(self, model: AblatedModel, task: Task, *, tau: float = 0.8,
                 allowed: int | None = None, base_seed: int = 0, inner_iters: int = 8,
                 batch_seqs: int = 1024) -> None:
        self.model = model
        self.name = task.name
        self.tau = float(tau)
        self.inner_iters = int(inner_iters)   # EAP-dropout refinement steps per canonicalization
        self.n_comp = model.n_comp
        self.n_components = model.n_comp  # core Verifier protocol
        self.base_seed = int(base_seed)
        # tokenize clean and corrupt together so both are left-padded to one length: the corrupt
        # activations then align position-by-position with the clean batch for interchange patching.
        n = len(task.prompts)
        both = model.tokenize(list(task.prompts) + list(task.corrupt))
        self.enc = {k: v[:n] for k, v in both.items()}          # clean prompts [B, L]
        corrupt_enc = {k: v[n:] for k, v in both.items()}       # corrupt prompts [B, L]
        self.cf = model.counterfactual(corrupt_enc)             # per-component corrupt activations
        # masks per batched forward, bounded by a (masks * prompts) sequence budget for memory.
        self._batch_masks = max(1, batch_seqs // max(1, self.enc["input_ids"].shape[0]))
        cid, fid = task.token_ids(model.tokenizer)
        self.correct = torch.tensor(cid, device=model.device)
        self.foil = torch.tensor(fid, device=model.device)      # the counterfactual's answer token
        self.rows = torch.arange(len(cid), device=model.device)
        self.allowed = allowed if allowed is not None else (1 << self.n_comp) - 1
        # the verifier's faithfulness is distributional (see the module docstring): the reference is
        # the unablated distribution P_clean, and KL_empty = KL(P_clean || P_fully-ablated) is the
        # interchange signal s0 thresholds against (the analogue of the old full margin).
        self.clean_logits = self.model.logits(self.enc, None, self.cf)          # [P, vocab] = P_clean
        empty = torch.zeros(1, self.n_comp, dtype=torch.bool)                   # keep nothing
        self.kl_empty = float(self.model.kl_batch(self.enc, empty, self.clean_logits, self.cf)[0])
        self.full_metric = self.kl_empty       # >0 iff there is an interchange signal to explain
        self.full_margin = self._metric(None)  # correct-foil margin: kept only for eval + attr order
        # one forward + backward: EAP importance ranking for the inner optimizer (a heuristic drop
        # order; soundness is from s0), and the per-component fingerprint the policy is conditioned on.
        self.attrs = model.attributions(self.enc, self.correct, self.foil, self.rows, self.cf)
        self.attr_features = (self.attrs / (self.attrs.abs().max() + 1e-6)).cpu().numpy().astype(np.float32)
        self.forward_calls = 1
        self._s0_cache: dict[int, bool] = {}
        self._exists_cache: dict[int, tuple[bool, int]] = {}

    def _metric(self, keep: torch.Tensor | None) -> float:
        logits = self.model.logits(self.enc, keep, self.cf)
        margin = logits[self.rows, self.correct] - logits[self.rows, self.foil]
        return float(margin.mean())

    def _keep(self, mask: int) -> torch.Tensor:
        return self._keeps([mask])[0]

    def _keeps(self, masks: list[int]) -> torch.Tensor:
        """Bitmask(s) -> [len(masks), n_comp] bool keep tensor, vectorized with numpy.unpackbits. The
        old per-mask Python bit-loop ``[(mask>>i)&1 for i in range(n_comp)]`` was THE inner-optimizer
        bottleneck (single-core CPU-bound): the search probes thousands of candidate masks per behavior
        and rebuilt a 476-element list for each. unpackbits does the whole batch in one C call. Bit-
        identical: byte little-endian + bitorder='little' puts mask bit i at index i."""
        nbytes = (self.n_comp + 7) // 8
        raw = np.frombuffer(b"".join(int(m).to_bytes(nbytes, "little") for m in masks), dtype=np.uint8)
        bits = np.unpackbits(raw.reshape(len(masks), nbytes), axis=1, bitorder="little")[:, :self.n_comp]
        return torch.from_numpy(np.ascontiguousarray(bits)).bool()

    def s0(self, mask: int) -> bool:
        """Base non-monotone predicate: keeping only ``mask`` reproduces the model's next-token
        distribution to within (1 - tau) of the interchange signal KL_empty (distributional
        faithfulness). Non-monotone in ``mask``, so it is the base predicate, not the witness test."""
        return self.s0_batch([mask])[0]

    def s0_batch(self, masks: list[int]) -> list[bool]:
        """Vectorized ``s0``: one chunked KL forward for the whole batch of masks. A mask is faithful
        iff KL(P_clean || P_mask) <= (1 - tau) * KL_empty. Only uncached masks are evaluated."""
        resolved = [m & self.allowed for m in masks]
        todo = [m for m in dict.fromkeys(resolved) if m not in self._s0_cache]
        if todo:
            keeps = self._keeps(todo)                                 # [U, n_comp] vectorized
            kl = self.model.kl_batch(self.enc, keeps, self.clean_logits, self.cf,
                                     chunk=self._batch_masks)          # [U]
            self.forward_calls += len(todo)
            thresh = (1.0 - self.tau) * self.kl_empty
            for mm, val in zip(todo, kl.tolist(), strict=True):
                self._s0_cache[mm] = val <= thresh
        return [self._s0_cache[m] for m in resolved]

    def s_c(self, mask: int) -> bool:
        """The monotone black-box success predicate s_c(S) (MWRL Def 1, Assumption 1): does a
        faithful sub-circuit exist inside the allowed set S. Short-circuits at s0(S) -- if S is
        itself faithful a faithful subset trivially exists (one forward) -- and only runs the inner
        optimizer search when S is not faithful. Monotone: opening more can only help the search."""
        mask &= self.allowed
        if self.s0(mask):
            return True
        return self.s_exists(mask)[0]

    def s_exists(self, mask: int) -> tuple[bool, int]:
        """Existential closure with the canonical minimal witness Phi (for the count valuation and
        evaluation): (found a faithful sub-circuit inside `mask`, the minimal Phi the inner
        optimizer returns). The full inner-optimizer search + minimality descent."""
        mask &= self.allowed
        cached = self._exists_cache.get(mask)
        if cached is not None:
            return cached
        rng = np.random.default_rng([self.base_seed, mask])
        canon = eap_guided_dropout(mask, self.s0, self.attrs, self.n_comp, rng=rng,
                                   iters=self.inner_iters, s0_batch=self.s0_batch)
        found = self.s0(canon)
        result = (found, canon if found else 0)
        self._exists_cache[mask] = result
        return result

    def s_exists_batch(self, masks: list[int]) -> list[tuple[bool, int]]:
        """Vectorized ``s_exists``: run the inner-optimizer searches for a whole group of terminal
        masks in lockstep, so each search phase's verifier candidates from every mask ride one
        ``s0_batch`` forward. Bit-for-bit equal to calling ``s_exists`` per mask (deterministic s0,
        per-mask RNG); only uncached masks are searched."""
        resolved = [m & self.allowed for m in masks]
        todo = [m for m in dict.fromkeys(resolved) if m not in self._exists_cache]
        if todo:
            rngs = [np.random.default_rng([self.base_seed, m]) for m in todo]
            canons = eap_guided_dropout_batch(todo, self.s0_batch, self.attrs, self.n_comp,
                                              rngs=rngs, iters=self.inner_iters)
            founds = self.s0_batch(canons)
            for m, canon, f in zip(todo, canons, founds, strict=True):
                self._exists_cache[m] = (f, canon if f else 0)
        return [self._exists_cache[m] for m in resolved]

    def is_witness(self, mask: int) -> bool:
        return self.s_exists(mask)[0]

    def faithfulness(self, mask: int) -> float:
        """Distributional faithfulness recovered by ``mask``: 1 - KL(P_clean || P_mask) / KL_empty
        (1 = reproduces the full model, 0 = no better than keeping nothing)."""
        if self.kl_empty <= 0:
            return 1.0
        kl = float(self.model.kl_batch(self.enc, self._keep(mask).unsqueeze(0),
                                       self.clean_logits, self.cf)[0])
        return 1.0 - kl / self.kl_empty
