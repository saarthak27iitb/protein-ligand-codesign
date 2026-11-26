"""Prepare controller splits from official SPINDR and aligned full RCSB chains.

Filtering is structural and predefined, never based on a method's reward.
Full chains are required: cropped SPINDR pockets must not be simulated as proteins.
"""

import argparse, json, pickle, gzip, urllib.request, time
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from .common import ROOT, atomic_json, identity, sha, residues, xyz, kabsch, ligand

AA = {
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
}


def fetch(pdbid, cache):
    p = cache / f"{pdbid}.cif"
    cache.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"https://files.rcsb.org/download/{pdbid.upper()}.cif",
                    headers={"User-Agent": "DH307-academic-research/1.0"},
                )
                with urllib.request.urlopen(req, timeout=60) as f:
                    data = f.read()
                if b"ATOM" not in data:
                    raise ValueError("RCSB returned no atoms")
                p.write_bytes(data)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
    return p


def full_chain(pocket, source, out, cfg):
    # Match author residue IDs and atom identities, selecting the best crystal chain.
    ref = residues(pocket)
    chains = {}
    if source.suffix == ".cif":
        import io
        from biotite.structure.io import pdbx, pdb

        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(source), model=1, use_author_fields=False
        )
        for chain in sorted(set(atoms.chain_id)):
            array = atoms[
                (atoms.chain_id == chain) & (~atoms.hetero) & (atoms.element != "H")
            ].copy()
            if len(array) == 0:
                continue
            array.chain_id = np.full(len(array), "A")
            file = pdb.PDBFile()
            pdb.set_structure(file, array)
            stream = io.StringIO()
            file.write(stream)
            chains[chain] = [
                l for l in stream.getvalue().splitlines() if l.startswith("ATOM  ")
            ]
    else:
        for l in source.read_text().splitlines():
            if l.startswith("ENDMDL"):
                break
            if l.startswith("ATOM  ") and l[16] in (" ", "A"):
                chains.setdefault(l[21], []).append(l[:16] + " " + l[17:])
    candidates = []
    for chain, lines in chains.items():
        rr = {}
        for l in lines:
            rr.setdefault((l[22:26], l[26]), {})[l[12:16].strip()] = l
        # Prepared structures can use label-sequence rather than author residue
        # numbers. Search constant numbering offsets, then verify geometric fit.
        ref_ca = [(k, v) for k, v in ref.items() if "CA" in v]
        offsets = {0}
        for key, atoms in ref_ca[:3]:
            for full_key, full_atoms in rr.items():
                if "CA" in full_atoms and atoms["CA"][17:20] == full_atoms["CA"][17:20]:
                    offsets.add(int(full_key[0]) - int(key[1]))
        for offset in offsets:
            pp = []
            qq = []
            for k, atoms in ref_ca:
                v = rr.get((f"{int(k[1])+offset:4d}", k[2]), {})
                if "CA" in v and atoms["CA"][17:20] == v["CA"][17:20]:
                    pp.append(xyz(v["CA"]))
                    qq.append(xyz(atoms["CA"]))
            if len(pp) >= max(10, int(0.8 * len(ref_ca))):
                r, t, err = kabsch(pp, qq)
                candidates.append((err, chain, lines, rr, r, t))
    if not candidates:
        raise ValueError("No full chain matches prepared pocket residue IDs")
    err, chain, lines, rr, r, t = min(candidates, key=lambda x: x[0])
    if err > cfg["max_alignment_rmsd_A"]:
        raise ValueError("Full chain/pocket alignment exceeds limit")
    if not 30 <= len(rr) <= cfg["max_residues"]:
        raise ValueError("Protein length outside protocol")
    if any(
        next(iter(v.values()))[17:20] not in AA or not {"N", "CA", "C", "O"} <= v.keys()
        for v in rr.values()
    ):
        raise ValueError("Nonstandard or incomplete backbone")
    vals = list(rr.values())
    if any(
        np.linalg.norm(xyz(a["C"]) - xyz(b["N"])) > 2.0 for a, b in zip(vals, vals[1:])
    ):
        raise ValueError("Full chain contains unresolved internal break")
    # Normalize chain to A for both generators, retaining author residue numbers.
    output = []
    for l in lines:
        v = xyz(l) @ r + t
        output.append(
            l[:21] + "A" + l[22:30] + "".join(f"{x:8.3f}" for x in v) + l[54:]
        )
    out.write_text("\n".join(output) + "\nTER\nEND\n")
    seq = "-".join(v["CA"][17:20] for v in vals)
    return {
        "rcsb_chain": chain,
        "alignment_rmsd_A": err,
        "residues": len(rr),
        "sequence_sha256": identity(seq),
        "rcsb_sha256": sha(source),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        if old["config_sha256"] != sha(args.config):
            raise ValueError("Config changed; use a new campaign directory")
        print("Prepared manifest already complete")
        return
    from flowr.util.pocket import PocketComplex
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    result = {
        "config_sha256": sha(args.config),
        "source": "official SPINDR splits + full RCSB crystal chains",
        "selection_seed": cfg["data"]["selection_seed"],
        "splits": {},
        "exclusions": [],
        "limitations": [
            "FLOWR was pretrained on SPINDR train; PocketGen overlap with target complexes has not been ruled out.",
            "Controller splits exclude exact protein sequences and ligand scaffolds across splits; this is not a protein-family homology guarantee.",
        ],
    }
    used_seq = set()
    used_scaffold = set()
    used_system = set()
    # Select test first so that training cannot consume its identities/scaffolds.
    with ZipFile(ROOT / "downloads/smol_data.zip") as z:
        for split in ["test", "val", "train"]:
            print("Loading " + split, flush=True)
            with z.open(f"smol_data/{split}.smol") as f:
                systems = pickle.load(f)
            order = np.random.default_rng(cfg["data"]["selection_seed"]).permutation(
                len(systems)
            )
            selected = []
            for idx in order:
                if len(selected) == cfg["data"][split]:
                    break
                target = None
                try:
                    c = PocketComplex.from_bytes(systems[int(idx)], remove_hs=True)
                    if c is None:
                        raise ValueError("Invalid serialized complex")
                    target = c.system_id
                    if not target or target in used_system:
                        raise ValueError("Duplicate/missing system ID")
                    mol = c.ligand.to_rdkit()
                    Chem.SanitizeMol(mol)
                    if len(Chem.GetMolFrags(mol)) != 1:
                        raise ValueError("Disconnected reference")
                    if (
                        not cfg["data"]["min_ligand_atoms"]
                        <= mol.GetNumHeavyAtoms()
                        <= cfg["data"]["max_ligand_atoms"]
                    ):
                        raise ValueError("Reference size outside protocol")
                    if (
                        abs(Chem.GetFormalCharge(mol))
                        > cfg["constraints"]["charge_abs_max"]
                    ):
                        raise ValueError("Reference charge outside protocol")
                    scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
                    if not scaffold:
                        raise ValueError("No reference ring scaffold")
                    folder = out / split / target
                    folder.mkdir(parents=True, exist_ok=True)
                    c.holo.write_pdb(folder / "prepared_pocket.pdb")
                    source = fetch(target[:4], out / "rcsb")
                    structural = full_chain(
                        folder / "prepared_pocket.pdb",
                        source,
                        folder / "protein.pdb",
                        cfg["data"],
                    )
                    # Cross-split exclusions only, not within-split deduplication of unique systems.
                    if (
                        structural["sequence_sha256"] in used_seq
                        or scaffold in used_scaffold
                    ):
                        raise ValueError("Cross-split sequence/scaffold overlap")
                    with Chem.SDWriter(str(folder / "ligand.sdf")) as w:
                        w.write(mol)
                    ligand(folder / "ligand.sdf")
                    row = dict(
                        id=target,
                        split=split,
                        source_index=int(idx),
                        protein=str(folder / "protein.pdb"),
                        ligand=str(folder / "ligand.sdf"),
                        scaffold=scaffold,
                        reference_smiles=Chem.MolToSmiles(mol),
                        serialized_sha256=__import__("hashlib")
                        .sha256(systems[int(idx)])
                        .hexdigest(),
                        **structural,
                    )
                    atomic_json(folder / "target.json", row)
                    selected.append(row)
                    used_system.add(target)
                    print(
                        f'{split}: {len(selected)}/{cfg["data"][split]} {target}',
                        flush=True,
                    )
                except Exception as exc:
                    result["exclusions"].append(
                        {
                            "split": split,
                            "source_index": int(idx),
                            "id": target,
                            "reason": str(exc),
                        }
                    )
                atomic_json(
                    out / "preparation_progress.json",
                    dict(
                        result,
                        current_split=split,
                        selected=len(selected),
                        required=cfg["data"][split],
                    ),
                )
            if len(selected) != cfg["data"][split]:
                raise RuntimeError(
                    f'Insufficient structurally eligible {split} targets: {len(selected)}; requested {cfg["data"][split]}. No silent reduction.'
                )
            result["splits"][split] = selected
            used_seq.update(r["sequence_sha256"] for r in selected)
            used_scaffold.update(r["scaffold"] for r in selected)
            del systems
    result["stage"] = "complete"
    atomic_json(manifest, result)


if __name__ == "__main__":
    main()
