"""Rigid-motion invariant chemical and local contact descriptors (40 channels)."""

import sys
import numpy as np
from .common import ROOT, CODE_ROOT, ligand, residues, xyz

sys.path.insert(0, str(CODE_ROOT / "scripts"))
from design_utils import mol_features

AA = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]


def encode(protein, ligand_path, utility=0.0, remaining=1.0):
    from scipy.spatial import cKDTree
    from rdkit import Chem

    mol = ligand(ligand_path)
    r = residues(protein)
    coords = []
    names = []
    for atoms in r.values():
        names.append(next(iter(atoms.values()))[17:20])
        coords.extend(xyz(l) for l in atoms.values())
    p = np.array(coords)
    x = mol.GetConformer().GetPositions()
    distance = cKDTree(p).query(x)[0]
    hist = np.histogram(distance, bins=[0, 1.5, 2.5, 3.5, 4.5, 6, 8, 10, np.inf])[
        0
    ] / max(len(x), 1)
    counts = np.array([names.count(n) for n in AA]) / max(len(names), 1)
    result = np.r_[
        mol_features(mol),
        counts,
        hist,
        float(utility) / 20,
        float(remaining),
        Chem.GetFormalCharge(mol) / 2,
    ].astype(np.float32)
    if result.shape != (40,) or not np.isfinite(result).all():
        raise ValueError("Invalid policy features")
    return result


def chemical_mask(paths, reference, cfg):
    from rdkit import Chem
    from rdkit.Chem import QED
    from .common import similarity

    ref = ligand(reference)
    mask = []
    for p in paths:
        try:
            m = ligand(p)
            mask.append(
                cfg["atom_ratio_min"] * ref.GetNumHeavyAtoms()
                <= m.GetNumHeavyAtoms()
                <= cfg["atom_ratio_max"] * ref.GetNumHeavyAtoms()
                and abs(Chem.GetFormalCharge(m)) <= cfg["charge_abs_max"]
                and QED.qed(m) >= cfg["qed_min"]
                and similarity(m, ref) >= cfg["min_reference_tanimoto"]
            )
        except Exception:
            mask.append(False)
    return np.array(mask, dtype=bool)
