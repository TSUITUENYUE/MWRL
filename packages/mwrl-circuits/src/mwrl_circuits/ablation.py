"""Frozen-LLM component ablation for circuit discovery (HF transformers, Qwen3/Llama).

A *component* is one attention head or one MLP sublayer. Global indices:

    attn head (layer l, head h) -> l * n_heads + h            in [0, n_layers*n_heads)
    mlp (layer l)               -> n_layers*n_heads + l        in [.., n_comp)

``keep`` is a boolean array over the ``n_comp`` components (True = kept active). A
kept component passes through untouched; a component that is *not* kept is replaced
by its activation on a counterfactual ("corrupt") prompt (resample / interchange
ablation, the ACDC/EAP standard), which stays on-distribution and yields sparse
faithful circuits (mean ablation over many components pushes the residual stream far
off-distribution, so it does not). The corrupt activations are per behavior and per
(row, position), aligned to the clean batch. Attention heads are ablated at the input
to ``o_proj`` (the concatenated per-head outputs, shape ``[batch, seq,
n_heads*head_dim]``); Qwen3 decouples ``head_dim`` from ``hidden_size``, so the reshape
uses ``head_dim`` explicitly. MLP sublayers are ablated at the module output.

Hooks are installed once and gated by ``self._keep``; ``None`` means pass-through, so
the same model computes clean logits, records corrupt activations, and ablated logits.
"""

from __future__ import annotations

import torch




def attn_index(layer: int, head: int, n_heads: int) -> int:
    return layer * n_heads + head


def mlp_index(layer: int, n_layers: int, n_heads: int) -> int:
    return n_layers * n_heads + layer


class AblatedModel:
    """Wraps a HF causal LM with per-component mean-ablation hooks."""

    def __init__(self, model_name: str, *, device: str | None = None,
                 dtype: str = "bfloat16", attn_implementation: str = "eager") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if self.device != "cpu" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype,
            attn_implementation=attn_implementation)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # last position = answer for every row

        cfg = self.model.config
        self.n_layers = int(cfg.num_hidden_layers)
        self.n_heads = int(cfg.num_attention_heads)
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads))
        self.n_attn = self.n_layers * self.n_heads
        self.n_comp = self.n_attn + self.n_layers
        self._layers = self.model.model.layers

        self._keep: torch.Tensor | None = None   # None = pass-through
        self._soft: torch.Tensor | None = None   # float [n_comp] gates for the differentiable path
        self._grab: dict | None = None           # set while recording corrupt activations
        self._capture: dict | None = None        # set during EAP attribution (grad capture)
        self._cf_attn: list | None = None        # per-behavior corrupt attn acts, base [B, s, nh, hd]
        self._cf_mlp: list | None = None         # per-behavior corrupt mlp acts, base [B, s, hidden]
        self.forward_passes = 0                  # model() calls (the wall-clock cost)
        self._install_hooks()

    # -- component index helpers -------------------------------------------------
    def attn_idx(self, layer: int, head: int) -> int:
        return layer * self.n_heads + head

    def mlp_idx(self, layer: int) -> int:
        return self.n_attn + layer

    # -- hooks -------------------------------------------------------------------
    def _install_hooks(self) -> None:
        for li, layer in enumerate(self._layers):
            layer.self_attn.o_proj.register_forward_pre_hook(self._attn_pre_hook(li))
            layer.mlp.register_forward_hook(self._mlp_hook(li))

    def _attn_pre_hook(self, li: int):
        def hook(_module, args):
            x = args[0]  # [b, s, n_heads*head_dim]
            b, s, _ = x.shape
            xh = x.view(b, s, self.n_heads, self.head_dim)
            if self._grab is not None:  # record corrupt activation (interchange source)
                self._grab["attn"][li] = xh.detach()
                return None
            if self._capture is not None:  # grad-enabled copy for EAP attribution
                z = xh.detach().requires_grad_(True)
                self._capture["attn"][li] = z
                return (z.view(b, s, -1),) + args[1:]
            if self._soft is not None:                         # differentiable convex blend
                base = li * self.n_heads
                cf = self._cf_attn[li]
                reps = b // cf.shape[0]
                if reps > 1:
                    cf = cf.repeat(reps, 1, 1, 1)
                m = self._soft[base:base + self.n_heads].to(xh.dtype)[None, None, :, None]
                xh = m * xh + (1 - m) * cf.to(xh.dtype)
                return (xh.view(b, s, -1),) + args[1:]
            if self._keep is None:
                return None
            base = li * self.n_heads
            cf = self._cf_attn[li]                             # [B, s, nh, hd] corrupt activation
            reps = b // cf.shape[0]
            if reps > 1:                                       # tile to the (mask-major) batch
                cf = cf.repeat(reps, 1, 1, 1)
            cf = cf.to(xh.dtype)
            if self._keep.dim() == 1:                          # one mask over all rows
                drop = ~self._keep[base:base + self.n_heads]   # [nh]
                if drop.any():
                    xh = torch.where(drop[None, None, :, None], cf, xh)
            else:                                              # per-row keep [batch, n_comp]
                drop = ~self._keep[:, base:base + self.n_heads]   # [b, nh]
                if drop.any():
                    xh = torch.where(drop[:, None, :, None], cf, xh)
            return (xh.view(b, s, -1),) + args[1:]
        return hook

    def _mlp_hook(self, li: int):
        def hook(_module, _args, output):
            if self._grab is not None:  # record corrupt activation (interchange source)
                self._grab["mlp"][li] = output.detach()
                return None
            if self._capture is not None:  # grad-enabled copy for EAP attribution
                z = output.detach().requires_grad_(True)
                self._capture["mlp"][li] = z
                return z
            if self._soft is not None:                         # differentiable convex blend
                cf = self._cf_mlp[li]
                reps = output.shape[0] // cf.shape[0]
                if reps > 1:
                    cf = cf.repeat(reps, 1, 1)
                m = self._soft[self.mlp_idx(li)].to(output.dtype)
                return m * output + (1 - m) * cf.to(output.dtype)
            if self._keep is None:
                return None
            cf = self._cf_mlp[li]                              # [B, s, hidden] corrupt activation
            reps = output.shape[0] // cf.shape[0]
            if reps > 1:
                cf = cf.repeat(reps, 1, 1)
            cf = cf.to(output.dtype)
            if self._keep.dim() == 1:
                if not bool(self._keep[self.mlp_idx(li)]):
                    return cf
                return None
            drop = ~self._keep[:, self.mlp_idx(li)]   # [b] per-row keep
            if drop.any():
                return torch.where(drop[:, None, None], cf, output)
            return None
        return hook

    # -- forward passes ----------------------------------------------------------
    def tokenize(self, prompts: list[str]) -> dict:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True)
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def counterfactual(self, corrupt_enc: dict) -> tuple[list, list]:
        """Record every component's activation on the corrupt prompts (the interchange source).
        Returns base activations ``attn_cf[li] : [B, s, nh, hd]`` and ``mlp_cf[li] : [B, s, hidden]``,
        aligned position-by-position with a clean batch tokenized to the same length."""
        attn_cf: list = [None] * self.n_layers
        mlp_cf: list = [None] * self.n_layers
        self._grab = {"attn": attn_cf, "mlp": mlp_cf}
        try:
            self.model(**corrupt_enc)
        finally:
            self._grab = None
        return attn_cf, mlp_cf

    def soft_logits(self, enc: dict, soft: torch.Tensor, cf: tuple[list, list]) -> torch.Tensor:
        """Differentiable final-position logits: every component's activation is the convex
        blend ``soft[c] * clean + (1 - soft[c]) * corrupt``, so gradients reach a mask over
        components while the weights stay frozen (the HardConcrete baseline trains through
        this path). Grad-enabled on purpose; callers manage their own autograd context."""
        self._cf_attn, self._cf_mlp = cf
        self._soft = soft
        try:
            out = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
        finally:
            self._soft = None
        self.forward_passes += 1
        return out

    @torch.no_grad()
    def logits(self, enc: dict, keep: torch.Tensor | None, cf: tuple[list, list]) -> torch.Tensor:
        """Final-position logits with the complement of ``keep`` patched from the corrupt run ``cf``."""
        self._cf_attn, self._cf_mlp = cf
        self._keep = None if keep is None else keep.to(self.device)
        try:
            # logits_to_keep=1: the lm_head runs on the final position only, not the whole [B, seq,
            # vocab] tensor -- with a ~152k vocab that all-position logits tensor is the memory wall
            # (a big mask-major batch OOMs on it), and every position but the last is discarded here.
            out = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
        finally:
            self._keep = None
        self.forward_passes += 1
        return out

    @torch.no_grad()
    def margins_batch(self, enc: dict, keeps: torch.Tensor, correct: torch.Tensor,
                      foil: torch.Tensor, rows: torch.Tensor, cf: tuple[list, list],
                      chunk: int = 32) -> torch.Tensor:
        """Correct-minus-foil margin (mean over prompts) for many masks at once, reduced PER CHUNK
        so the full ``[M, P, vocab]`` logits are never materialized -- a 400-mask probe would be
        ~1 GB of vocab logits in one tensor and OOM a 24 GB card. ``keeps`` is ``[M, n_comp]`` bool;
        the per-row keep in the hooks evaluates each mask over all ``P`` prompts and tiles the
        corrupt activations ``cf`` to the mask-major batch. Returns ``[M]``."""
        self._cf_attn, self._cf_mlp = cf
        ids, attn = enc["input_ids"], enc["attention_mask"]      # [P, seq]
        p, m = ids.shape[0], keeps.shape[0]
        keeps = keeps.to(self.device).bool()
        out = torch.empty(m, device=self.device)
        for i in range(0, m, chunk):
            ck = keeps[i:i + chunk]                              # [c, n_comp]
            c = ck.shape[0]
            self._keep = ck.repeat_interleave(p, dim=0)          # [c*P, n_comp]; row j*P+k -> mask j
            try:
                logit = self.model(input_ids=ids.repeat(c, 1),   # mask-major tiling of the prompts
                                   attention_mask=attn.repeat(c, 1),
                                   logits_to_keep=1).logits[:, -1, :].view(c, p, -1)
            finally:
                self._keep = None
            out[i:i + c] = (logit[:, rows, correct] - logit[:, rows, foil]).mean(dim=1)
            self.forward_passes += 1
        return out                                               # [M]

    @torch.no_grad()
    def kl_batch(self, enc: dict, keeps: torch.Tensor, clean_logits: torch.Tensor,
                 cf: tuple[list, list], chunk: int = 32) -> torch.Tensor:
        """Mean KL(P_clean || P_S) over prompts, for many masks S at once. ``clean_logits`` [P, vocab]
        is the unablated final-position logits (the reference distribution); each mask keeps only its
        components (rest resample-ablated) and we measure how far its final-position distribution has
        moved from clean. This is the distributional faithfulness the verifier's ``s0`` thresholds --
        the circuit must reproduce the model's whole next-token distribution, not win a two-way
        margin. Reduced per chunk so the [M, P, vocab] logits are never materialized. Returns [M]
        (nats). KL is computed in fp32 for a stable log-softmax over the ~152k vocabulary."""
        self._cf_attn, self._cf_mlp = cf
        ids, attn = enc["input_ids"], enc["attention_mask"]      # [P, seq]
        p, m = ids.shape[0], keeps.shape[0]
        keeps = keeps.to(self.device).bool()
        logp_clean = torch.log_softmax(clean_logits.float(), dim=-1)     # [P, vocab]
        p_clean = logp_clean.exp()                                       # [P, vocab]
        neg_ent = (p_clean * logp_clean).sum(-1)                         # [P]  (= -H(P_clean))
        out = torch.empty(m, device=self.device)
        for i in range(0, m, chunk):
            ck = keeps[i:i + chunk]                              # [c, n_comp]
            c = ck.shape[0]
            self._keep = ck.repeat_interleave(p, dim=0)          # [c*P, n_comp]; row j*P+k -> mask j
            try:
                logit = self.model(input_ids=ids.repeat(c, 1),   # mask-major tiling of the prompts
                                   attention_mask=attn.repeat(c, 1),
                                   logits_to_keep=1).logits[:, -1, :].view(c, p, -1).float()
            finally:
                self._keep = None
            logp_s = torch.log_softmax(logit, dim=-1)            # [c, P, vocab]
            kl = neg_ent[None, :] - (p_clean[None] * logp_s).sum(-1)     # [c, P] = KL(clean||S)
            out[i:i + c] = kl.mean(dim=1)
            self.forward_passes += 1
        return out                                               # [M]

    @torch.no_grad()
    def generate(self, enc: dict, keep: torch.Tensor | None, cf: tuple[list, list],
                 max_new_tokens: int = 4) -> torch.Tensor:
        """Greedy-decode up to ``max_new_tokens`` tokens under the ablation ``keep`` (None = dense;
        a [n_comp] mask; or a [B, n_comp] per-row mask). The sequence is re-run in full each step (no
        KV cache) so the per-component hooks fire cleanly over the whole sequence; the corrupt patch
        ``cf`` covers the prompt, and generated positions are patched with the last prompt position
        tiled forward, so the circuit ablation stays defined as the answer is decoded. This is the
        real-performance metric: does the sparse circuit actually produce the answer string, not just
        win a two-way logit margin. Returns the generated token ids ``[B, max_new_tokens]``."""
        ids, attn = enc["input_ids"].clone(), enc["attention_mask"].clone()
        cf_attn, cf_mlp = cf
        outs = []
        for _ in range(max_new_tokens):
            s = ids.shape[1]

            def _pad(t: torch.Tensor) -> torch.Tensor:   # right-pad to seq len s, tiling last pos
                if t.shape[1] >= s:
                    return t
                tail = t[:, -1:].expand(-1, s - t.shape[1], *t.shape[2:])
                return torch.cat([t, tail], dim=1)

            cf_s = ([_pad(a) for a in cf_attn], [_pad(m) for m in cf_mlp])
            nxt = self.logits({"input_ids": ids, "attention_mask": attn}, keep, cf_s).argmax(-1, keepdim=True)
            outs.append(nxt)
            ids = torch.cat([ids, nxt], dim=1)
            attn = torch.cat([attn, torch.ones_like(nxt)], dim=1)
        return torch.cat(outs, dim=1)                            # [B, max_new_tokens]

    def attributions(self, enc: dict, correct: torch.Tensor, foil: torch.Tensor,
                     rows: torch.Tensor, cf: tuple[list, list]) -> torch.Tensor:
        """EAP with the corrupt reference: one clean forward + backward gives every component's
        first-order attribution of the correct-minus-foil margin under the resample patch,
        ``attr(c) = ((cf_c - z_c) * d margin / d z_c).sum``. Only a cheap importance ranking
        (|attr|) for the inner optimizer; soundness comes from ``s0``."""
        cf_attn, cf_mlp = cf
        self._capture = {"attn": {}, "mlp": {}}
        try:
            with torch.enable_grad():
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :]
                margin = (logits[rows, correct] - logits[rows, foil]).mean()
                attn_z = [self._capture["attn"][li] for li in range(self.n_layers)]
                mlp_z = [self._capture["mlp"][li] for li in range(self.n_layers)]
                grads = torch.autograd.grad(margin, attn_z + mlp_z)
        finally:
            self._capture = None
        attrs = torch.zeros(self.n_comp, device=self.device)
        for li in range(self.n_layers):
            z, g = attn_z[li], grads[li]                       # [b, s, n_heads, head_dim]
            attr = ((cf_attn[li].to(z.dtype) - z) * g).sum(dim=(0, 1, 3)).float()
            attrs[li * self.n_heads:(li + 1) * self.n_heads] = attr
        for li in range(self.n_layers):
            z, g = mlp_z[li], grads[self.n_layers + li]        # [b, s, hidden]
            attrs[self.mlp_idx(li)] = ((cf_mlp[li].to(z.dtype) - z) * g).sum().float()
        return attrs.detach()
