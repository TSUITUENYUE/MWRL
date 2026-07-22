"""The substrate/reaction fingerprint that conditions the amortized policy.

The reported results use interpretable substrate descriptors computed from the halide and
boron SMILES (``descriptors``). The differential reaction fingerprint over the reaction
SMILES (``drfp``) is also supported.
"""

from __future__ import annotations

import numpy as np


def _reaction_smiles(halide: str, boron: str, product: str | None) -> str:
    return f"{halide}.{boron}>>{product or ''}"


_SMARTS = {
    "aryl_cl": "[c][Cl]",
    "aryl_br": "[c][Br]",
    "aryl_i": "[c][I]",
    "triflate": "OS(=O)(=O)C(F)(F)F",
    "boronic_acid": "[B]([OX2H])[OX2H]",
    "pinacol": "B1OC(C)(C)C(C)(C)O1",
    "mida": "B1OCC(=O)N(C)C(=O)CO1",
    "trifluoroborate": "[B-](F)(F)F",
    "aromatic_nh": "[nH]",
    "ring_n": "[n]",
    "ewg": "[c]-[$([N+](=O)[O-]),$(C#N),$(C(F)(F)F),$(C=O)]",
    "edg": "[c]-[$([OX2][CH3]),$([NX3]([CH3])[CH3]),$([OX2H])]",
}


def _substrate_descriptors(halide: str, boron: str) -> np.ndarray:
    """Interpretable substrate descriptors: leaving-group class, boron class,
    heteroaryl and NH content, ortho load at the coupling carbons, and simple
    electronic/size proxies. Everything derives from the SMILES alone."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    patterns = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}

    def ortho_load(mol, ipso_smarts: str) -> float:
        pat = Chem.MolFromSmarts(ipso_smarts)
        count = 0.0
        for match in mol.GetSubstructMatches(pat):
            ipso = mol.GetAtomWithIdx(match[0])
            for nbr in ipso.GetNeighbors():
                if not nbr.GetIsAromatic():
                    continue
                heavy = [a for a in nbr.GetNeighbors()
                         if a.GetIdx() != ipso.GetIdx() and not a.GetIsAromatic()
                         and a.GetAtomicNum() > 1]
                count += len(heavy)
        return count

    def side(smiles: str, ipso_patterns: list[str]) -> list[float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * 12
        hits = {k: float(len(mol.GetSubstructMatches(p)))
                for k, p in patterns.items()}
        ipso_ortho = max((ortho_load(mol, s) for s in ipso_patterns), default=0.0)
        return [
            min(hits["ring_n"], 4.0),
            min(hits["aromatic_nh"], 2.0),
            min(hits["ewg"], 3.0),
            min(hits["edg"], 3.0),
            min(ipso_ortho, 4.0),
            float(rdMolDescriptors.CalcNumAromaticRings(mol)),
            float(rdMolDescriptors.CalcTPSA(mol)) / 100.0,
            float(Descriptors.MolWt(mol)) / 300.0,
            float(Descriptors.MolLogP(mol)) / 5.0,
            float(rdMolDescriptors.CalcNumRotatableBonds(mol)) / 5.0,
            float(sum(a.GetAtomicNum() == 16 for a in mol.GetAtoms())),
            float(sum(a.GetAtomicNum() == 8 for a in mol.GetAtoms())) / 4.0,
        ]

    hmol = Chem.MolFromSmiles(halide)
    bmol = Chem.MolFromSmiles(boron)
    lg = [0.0, 0.0, 0.0, 0.0]
    if hmol is not None:
        for i, key in enumerate(("aryl_cl", "aryl_br", "aryl_i", "triflate")):
            lg[i] = float(bool(hmol.GetSubstructMatches(patterns[key])))
    bc = [0.0, 0.0, 0.0, 0.0]
    if bmol is not None:
        for i, key in enumerate(("boronic_acid", "pinacol", "mida", "trifluoroborate")):
            bc[i] = float(bool(bmol.GetSubstructMatches(patterns[key])))
    if sum(bc) == 0.0:
        bc.append(1.0)          # other boron class
    else:
        bc.append(0.0)
    vec = (
        lg
        + side(halide, ["[c][Cl]", "[c][Br]", "[c][I]"])
        + bc
        + side(boron, ["[c][B]", "[C][B]"])
    )
    return np.asarray(vec, dtype=np.float32)


_CELLFEAT_CACHE: dict[str, np.ndarray] | None = None


def _cellfeat(context_id: str) -> np.ndarray:
    """Precomputed per-context feature vector (e.g. PCA of cell-line expression),
    read from the JSON sidecar named by $MWRL_CONTEXT_SIDECAR and keyed by the
    identity column. The conditioning signal for non-chemistry benchmarks."""
    global _CELLFEAT_CACHE
    if _CELLFEAT_CACHE is None:
        import json
        import os

        path = os.environ["MWRL_CONTEXT_SIDECAR"]
        _CELLFEAT_CACHE = {
            k: np.asarray(v, dtype=np.float32) for k, v in json.load(open(path)).items()
        }
    return _CELLFEAT_CACHE[context_id]


def reaction_fingerprint(
    halide: str, boron: str, product: str | None = None, *, n_bits: int = 2048, method: str = "drfp"
) -> np.ndarray:
    """Fixed-length fingerprint of the substrate pair (and product, if given)."""
    if method == "cellfeat":
        return _cellfeat(halide)          # halide column carries the context id
    rxn = _reaction_smiles(halide, boron, product)
    if method == "drfp":
        from drfp import DrfpEncoder

        fp = DrfpEncoder.encode(rxn, n_folded_length=n_bits)[0]
        return np.asarray(fp, dtype=np.float32)
    if method == "descriptors":
        return _substrate_descriptors(halide, boron)
    raise ValueError(
        f"unknown fingerprint method {method!r} (use 'drfp' or 'descriptors')"
    )


def drfp_available() -> bool:
    try:
        import drfp  # noqa: F401

        return True
    except Exception:
        return False
