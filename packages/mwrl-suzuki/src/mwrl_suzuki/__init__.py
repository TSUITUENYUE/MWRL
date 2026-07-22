"""Minimal-Witness RL on a real Suzuki dataset.

The dataset is the environment. The policy proposes a set of condition dimensions to tune
(a subspace ``S`` over the schema's dimensions); the closed-world verifier ``s_c(S)`` returns
1 iff a real measured reaction that deviated from the baseline only within ``S`` cleared the
yield threshold. Per substrate pair we recover the antichain of minimal condition sets, and
one policy conditioned on the reaction fingerprint amortizes across substrates. See README.md.
"""

from mwrl_suzuki.data import ChemTask, load_dataset
from mwrl_suzuki.env import ChemEnv
from mwrl_suzuki.schema import ConditionSchema
from mwrl_suzuki.verifier import ChemVerifier

__all__ = ["ChemTask", "load_dataset", "ChemEnv", "ConditionSchema", "ChemVerifier"]
