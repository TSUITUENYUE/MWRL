"""Amortized minimal-circuit discovery on modern LLMs (Qwen3), on the MWRL core.

Components are attention heads and MLP sublayers; the witness test is the monotone existential
closure (a faithful sub-circuit exists inside the opened set), realized by an EAP-guided dropout
inner optimizer. One policy, conditioned on each behavior's EAP fingerprint, is amortized over a
distribution of behaviors and recovers each one's antichain of minimal faithful circuits. See
``run.py`` for the entry point.
"""

from mwrl_circuits.ablation import AblatedModel
from mwrl_circuits.env import CircuitEnv
from mwrl_circuits.evaluate import evaluate_antichain
from mwrl_circuits.tasks import Task, build_hierarchy
from mwrl_circuits.verifier import CircuitVerifier

__all__ = ["AblatedModel", "CircuitVerifier", "CircuitEnv", "evaluate_antichain",
           "Task", "build_hierarchy"]
