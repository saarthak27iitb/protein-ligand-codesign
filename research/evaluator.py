"""Cached, receptor-consistent oracle with explicit failures and accounting."""

from collections import OrderedDict
from pathlib import Path
import json, subprocess, time, sys
import numpy as np
from .common import (
    ROOT,
    CODE_ROOT,
    atomic_json,
    append_json,
    identity,
    sha,
    ligand,
    similarity,
    residues,
    xyz,
)

sys.path.insert(0, str(CODE_ROOT / "scripts"))
from score_pair import geometry


class Evaluator:
    def __init__(self, root, config):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.cfg = config
        self.calls = 0
        self.cache_hits = 0
        self.errors = 0
        self.receptors = OrderedDict()
        self.ledger = self.root / "requests.jsonl"
        if self.ledger.exists():
            for line in self.ledger.read_text().splitlines():
                event = json.loads(line)
                self.calls += 1
                self.cache_hits += int(event["cache_hit"])
                self.errors += int(not event["ok"])

    def evaluate(self, protein, ligand_path, target, seed=42, dock=False):
        self.calls += 1
        key = identity(
            {
                "protein": sha(protein),
                "ligand": sha(ligand_path),
                "config": self.cfg["constraints"],
                "reward": self.cfg["reward"],
                "reference": sha(target["ligand"]),
                "dock": dock,
                "seed": seed if dock else None,
                "oracle_version": 1,
            }
        )
        folder = self.root / key
        file = folder / "score.json"
        if file.exists():
            self.cache_hits += 1
            cached = json.loads(file.read_text())
            append_json(
                self.ledger, {"key": key, "cache_hit": True, "ok": cached["ok"]}
            )
            return cached
        folder.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        result = {
            "ok": False,
            "feasible": False,
            "utility": None,
            "protein": str(protein),
            "ligand": str(ligand_path),
        }
        try:
            from rdkit import Chem

            mol = ligand(ligand_path)
            ref = ligand(target["ligand"])
            metrics = geometry(protein, mol)
            c = self.cfg["constraints"]
            sim = similarity(mol, ref)
            charge = Chem.GetFormalCharge(mol)
            metrics.update(reference_tanimoto=sim, formal_charge=charge)
            feasible = (
                metrics["connected"]
                and metrics["mmff_minimization_converged"]
                and metrics["mmff94s_strain_kcal_mol"] is not None
                and -0.01 <= metrics["mmff94s_strain_kcal_mol"] <= c["strain_max"]
                and metrics["clashing_atom_pairs_0_4A"] <= c["clash_pairs_max"]
                and c["atom_ratio_min"] * ref.GetNumHeavyAtoms()
                <= mol.GetNumHeavyAtoms()
                <= c["atom_ratio_max"] * ref.GetNumHeavyAtoms()
                and abs(charge) <= c["charge_abs_max"]
                and metrics["qed"] >= c["qed_min"]
                and sim >= c["min_reference_tanimoto"]
            )
            # Score valid molecules even if geometrically infeasible; report why utility is unavailable.
            from meeko import MoleculePreparation, PDBQTWriterLegacy
            from vina import Vina

            setup = MoleculePreparation().prepare(Chem.AddHs(mol, addCoords=True))
            if len(setup) != 1:
                raise ValueError("Ambiguous ligand preparation")
            qt, ok, error = PDBQTWriterLegacy.write_string(setup[0])
            if not ok:
                raise ValueError(error)
            lqt = folder / "ligand.pdbqt"
            lqt.write_text(qt)
            pkey = sha(protein)
            rec = self.root / "receptors" / pkey
            rec.mkdir(parents=True, exist_ok=True)
            rqt = rec / "protein.pdbqt"
            if not rqt.exists():
                clean = rec / "protein.pdb"
                clean.write_text(
                    "\n".join(
                        l
                        for l in Path(protein).read_text().splitlines()
                        if l.startswith(("ATOM  ", "TER"))
                    )
                    + "\nEND\n"
                )
                with (rec / "preparation.log").open("w") as log:
                    subprocess.run(
                        [
                            str(ROOT / "envs/flowr/bin/obabel"),
                            str(clean),
                            "-O",
                            str(rqt),
                            "-xr",
                            "-h",
                            "--partialcharge",
                            "gasteiger",
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=120,
                    )
                if not rqt.exists() or rqt.stat().st_size == 0:
                    raise ValueError("Empty prepared receptor")
            if pkey not in self.receptors:
                v = Vina(sf_name="vina", cpu=1, seed=seed, verbosity=0)
                v.set_receptor(str(rqt))
                coords = ref.GetConformer().GetPositions()
                v.compute_vina_maps(
                    center=coords.mean(0).tolist(),
                    box_size=np.maximum(np.ptp(coords, axis=0) + 16, 24).tolist(),
                )
                self.receptors[pkey] = v
                if len(self.receptors) > 2:
                    self.receptors.popitem(last=False)
            v = self.receptors[pkey]
            v.set_ligand_from_file(str(lqt))
            metrics["vina_score_input_pose"] = float(v.score()[0])
            if dock:
                v.dock(
                    exhaustiveness=self.cfg["evaluation"]["dock_exhaustiveness"],
                    n_poses=5,
                )
                metrics["vina_score_docking"] = float(v.energies(n_poses=1)[0, 0])
                v.write_poses(str(folder / "docked.pdbqt"), n_poses=1, overwrite=True)
                # UFF uses a different intramolecular potential than the optimization strain term.
                from rdkit.Chem import AllChem

                mh = Chem.AddHs(Chem.Mol(mol), addCoords=True)
                if AllChem.UFFHasAllMoleculeParams(mh):
                    ff = AllChem.UFFGetMoleculeForceField(mh)
                    before = ff.CalcEnergy()
                    converged = ff.Minimize(maxIts=1000)
                    metrics["uff_strain_kcal_mol"] = float(before - ff.CalcEnergy())
                    metrics["uff_converged"] = converged == 0
            native = residues(target["protein"])
            now = residues(protein)
            if native.keys() != now.keys():
                raise ValueError("Scaffold residue identifiers changed")
            mutations = sum(
                next(iter(native[k].values()))[17:20]
                != next(iter(now[k].values()))[17:20]
                for k in native
            )
            feasible = feasible and mutations <= c["max_mutations"]
            metrics["mutations"] = mutations
            value = (
                -metrics["vina_score_input_pose"]
                - self.cfg["reward"]["strain_weight"]
                * metrics["mmff94s_strain_kcal_mol"]
                - self.cfg["reward"]["mutation_weight"] * mutations
                if feasible
                else None
            )
            result.update(
                ok=True, feasible=bool(feasible), utility=value, metrics=metrics
            )
        except Exception as e:
            self.errors += 1
            result["error"] = repr(e)
        result["seconds"] = time.monotonic() - start
        atomic_json(file, result)
        append_json(self.ledger, {"key": key, "cache_hit": False, "ok": result["ok"]})
        return result
