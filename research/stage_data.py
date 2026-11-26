"""Two-stage data preparation for clusters whose compute nodes lack internet."""

import argparse, json, pickle
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .common import ROOT, atomic_json, identity, sha
from .prepare import full_chain


def export(config, out):
    from flowr.util.pocket import PocketComplex
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    file = out / "candidates.json"
    if file.exists():
        return
    cfg = json.loads(config.read_text())
    out.mkdir(parents=True, exist_ok=True)
    splits = {}
    excluded = []
    with ZipFile(ROOT / "downloads/smol_data.zip") as z:
        for split in ["test", "val", "train"]:
            print("Exporting " + split, flush=True)
            with z.open(f"smol_data/{split}.smol") as f:
                systems = pickle.load(f)
            order = np.random.default_rng(cfg["data"]["selection_seed"]).permutation(
                len(systems)
            )
            rows = []
            for ix in order:
                if split == "train" and len(rows) >= 512:
                    break
                try:
                    c = PocketComplex.from_bytes(systems[int(ix)], remove_hs=True)
                    if c is None or not c.system_id:
                        raise ValueError("Invalid complex")
                    m = c.ligand.to_rdkit()
                    Chem.SanitizeMol(m)
                    if len(Chem.GetMolFrags(m)) != 1:
                        raise ValueError("Disconnected reference")
                    if (
                        not cfg["data"]["min_ligand_atoms"]
                        <= m.GetNumHeavyAtoms()
                        <= cfg["data"]["max_ligand_atoms"]
                    ):
                        raise ValueError("Reference size outside protocol")
                    if (
                        abs(Chem.GetFormalCharge(m))
                        > cfg["constraints"]["charge_abs_max"]
                    ):
                        raise ValueError("Reference charge outside protocol")
                    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
                    if not scaffold:
                        raise ValueError("No ring scaffold")
                    folder = out / split / c.system_id
                    folder.mkdir(parents=True, exist_ok=True)
                    c.holo.write_pdb(folder / "prepared_pocket.pdb")
                    with Chem.SDWriter(str(folder / "ligand.sdf")) as w:
                        w.write(m)
                    rows.append(
                        dict(
                            id=c.system_id,
                            split=split,
                            source_index=int(ix),
                            protein=str(folder / "protein.pdb"),
                            ligand=str(folder / "ligand.sdf"),
                            scaffold=scaffold,
                            reference_smiles=Chem.MolToSmiles(m),
                            serialized_sha256=__import__("hashlib")
                            .sha256(systems[int(ix)])
                            .hexdigest(),
                        )
                    )
                except Exception as e:
                    excluded.append(
                        {"split": split, "index": int(ix), "reason": repr(e)}
                    )
            splits[split] = rows
            del systems
            print(f"Exported {len(rows)} eligible {split} candidates", flush=True)
    atomic_json(
        file, {"config_sha256": sha(config), "splits": splits, "exclusions": excluded}
    )
    atomic_json(
        out / "downloads.json",
        {"ids": sorted({r["id"][:4] for rows in splits.values() for r in rows})},
    )


def finalize(config, out):
    cfg = json.loads(config.read_text())
    candidates = json.loads((out / "candidates.json").read_text())
    if candidates["config_sha256"] != sha(config):
        raise ValueError("Candidate configuration mismatch")
    result = {
        "config_sha256": sha(config),
        "source": "official SPINDR splits + aligned full RCSB crystal chains",
        "selection_seed": cfg["data"]["selection_seed"],
        "splits": {},
        "exclusions": list(candidates["exclusions"]),
        "limitations": [
            "Exact protein-sequence and scaffold exclusion across controller splits; not a protein-family homology guarantee.",
            "FLOWR pretrained on SPINDR train; PocketGen overlap not ruled out.",
            "Full resolved single chains are used; other chains/cofactors and equilibrium populations are not modeled.",
        ],
    }
    used_seq = set()
    used_scaff = set()
    used_ids = set()
    for split in ["val", "test", "train"]:
        selected = []
        pool = list(candidates["splits"][split])
        if split == "val":
            from collections import Counter

            test_scaffolds = Counter(
                r["scaffold"] for r in candidates["splits"]["test"]
            )
            pool.sort(key=lambda r: test_scaffolds[r["scaffold"]])
        for row in pool:
            if len(selected) >= cfg["data"][split]:
                break
            try:
                folder = Path(row["protein"]).parent
                source = out / "rcsb" / f'{row["id"][:4]}.pdb'
                if not source.exists():
                    source = out / "rcsb" / f'{row["id"][:4]}.cif'
                if not source.exists():
                    raise ValueError("RCSB cache entry missing")
                structure = full_chain(
                    folder / "prepared_pocket.pdb",
                    source,
                    Path(row["protein"]),
                    cfg["data"],
                )
                if (
                    structure["sequence_sha256"] in used_seq
                    or row["scaffold"] in used_scaff
                    or row["id"] in used_ids
                ):
                    raise ValueError("Cross-split identity/scaffold overlap")
                item = dict(row, **structure)
                selected.append(item)
                used_ids.add(row["id"])
                atomic_json(folder / "target.json", item)
            except Exception as e:
                result["exclusions"].append(
                    {"split": split, "id": row["id"], "reason": repr(e)}
                )
        result["splits"][split] = selected
        used_seq.update(r["sequence_sha256"] for r in selected)
        used_scaff.update(r["scaffold"] for r in selected)
        atomic_json(out / "selection_progress.json", result)
        if len(selected) != cfg["data"][split]:
            raise RuntimeError(
                f'{split}: {len(selected)} eligible targets; expected {cfg["data"][split]}'
            )
    result["stage"] = "complete"
    atomic_json(out / "manifest.json", result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["export", "finalize"], required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    (export if a.stage == "export" else finalize)(
        a.config.resolve(), a.output.resolve()
    )


if __name__ == "__main__":
    main()
