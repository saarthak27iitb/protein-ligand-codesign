"""Long-lived frozen model workers; environments stay isolated.

PocketGen preprocessing is adapted from its MIT-licensed inference example.
Input/output use JSON lines prefixed @@DH307@@; upstream log output is retained.
"""

import argparse, copy, json, sys, traceback
from pathlib import Path
import numpy as np
from .common import ROOT, CODE_ROOT, atomic_json, ligand

PREFIX = "@@DH307@@"


class FlowWorker:
    def __init__(self, config):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("A GPU allocation is required")
        from flowr.gen import generate_from_pdb as g

        self.g = g
        sys.argv = [
            "worker",
            "--arch",
            "pocket",
            "--pocket_type",
            "holo",
            "--dataset",
            "spindr",
            "--pdb_file",
            "unused.pdb",
            "--ligand_file",
            "unused.sdf",
            "--data_path",
            str(ROOT),
            "--save_dir",
            str(ROOT),
            "--ckpt_path",
            str(ROOT / "checkpoints/flowr_noHs.ckpt"),
            "--cut_pocket",
            "--pocket_cutoff",
            "6",
            "--gpus",
            "1",
            "--num_workers",
            "0",
            "--categorical_strategy",
            "uniform-sample",
        ]
        self.args = g.get_args()
        self.model, self.hp, self.vocab, _, _ = g.load_model(self.args)
        self.model = self.model.cuda().eval()
        self.config = config

    def __call__(self, r):
        import torch, lightning as L
        from rdkit import Chem
        from flowr.scriptutil import generate_ligands_per_target

        L.seed_everything(r["seed"], workers=True)
        a = copy.deepcopy(self.args)
        out = Path(r["output"])
        out.mkdir(parents=True, exist_ok=True)
        a.pdb_file = r["protein"]
        a.ligand_file = r["ligand"]
        a.save_dir = str(out)
        a.data_path = str(out)
        a.sample_n_molecules_per_target = r["samples"]
        a.batch_cost = r["samples"]
        a.integration_steps = self.config["generation"]["flow_steps"]
        data, transform, interp = self.g.load_data(a, self.hp, self.vocab)
        dataset = self.g.get_dataset(data, transform, self.vocab, interp, a, self.hp)
        mols = []
        with torch.no_grad():
            for i, batch in enumerate(
                self.g.get_dataloader(dataset, self.vocab, interp, a, self.hp)
            ):
                mols.extend(
                    generate_ligands_per_target(
                        a, self.hp, self.model, batch, save_traj=False, iter=str(i)
                    )
                )
        # Raw None/invalid entries are never omitted from the denominator.
        torch.save(mols, out / "raw_molecules.pt")
        rows = []
        for i, m in enumerate(mols):
            row = {"index": i, "valid": False}
            try:
                if m is None:
                    raise ValueError("No molecule generated")
                Chem.SanitizeMol(m)
                if len(Chem.GetMolFrags(m)) != 1:
                    raise ValueError("Disconnected output")
                p = out / f"ligand_{i}.sdf"
                with Chem.SDWriter(str(p)) as w:
                    w.write(m)
                ligand(p)
                row.update(
                    valid=True, path=str(p), smiles=Chem.MolToSmiles(Chem.RemoveHs(m))
                )
            except Exception as e:
                row["error"] = str(e)
            rows.append(row)
        result = {
            "rows": rows,
            "requested": r["samples"],
            "returned": len(rows),
            "valid": sum(x["valid"] for x in rows),
        }
        atomic_json(out / "generation.json", result)
        return result


class PocketWorker:
    def __init__(self, config):
        import torch, esm
        from models.PD import Pocket_Design_new
        from utils.transforms import FeaturizeProteinAtom, FeaturizeLigandAtom
        from utils.misc import load_config

        if not torch.cuda.is_available():
            raise RuntimeError("GPU allocation required")
        self.pf, self.lf = FeaturizeProteinAtom(), FeaturizeLigandAtom()
        c = load_config(str(ROOT / "vendor/PocketGen/configs/train_model.yml"))
        self.model = Pocket_Design_new(
            c.model,
            protein_atom_feature_dim=self.pf.feature_dim,
            ligand_atom_feature_dim=self.lf.feature_dim,
            device="cuda:0",
        ).cuda()
        ckpt = torch.load(ROOT / "checkpoints/pocketgen.pt", map_location="cpu")
        self.model.load_state_dict(ckpt["model"], strict=True)
        del ckpt
        self.model.eval()
        self.model.write_pdb = True
        # Export only protein coordinates: never reconstruct the ligand graph/charge.
        self.model.write_whole_pdb = True
        self.converter = esm.data.Alphabet.from_architecture(
            "ESM-1b"
        ).get_batch_converter()
        # Native sidechains/backbone are restored/projected later; avoid global relaxation.
        import models.PD as pd

        pd.openmm_relax = lambda path, *a, **kw: path
        pd.to_sdf = lambda *a, **kw: None
        pd.to_pdb = lambda *a, **kw: a[4] + 1

    def __call__(self, r):
        import torch
        from torch_geometric.transforms import Compose
        from utils.protein_ligand import PDBProtein, parse_sdf_file
        from utils.data import torchify_dict, collate_mols_block
        from utils.misc import seed_all

        seed_all(r["seed"])
        out = Path(r["output"])
        out.mkdir(parents=True, exist_ok=True)
        protein = PDBProtein(Path(r["protein"]).read_text())
        seq = "".join(protein.to_dict_residue()["seq"])
        lig = parse_sdf_file(r["ligand"], feat=False)
        ridx, rres = protein.query_residues_ligand(
            lig, radius=10, selected_residue=None, return_mask=False
        )
        full, _ = protein.query_residues_ligand(
            lig, radius=3.5, selected_residue=rres, return_mask=False
        )
        pp = out / "pocket.pdb"
        pp.write_text(protein.residues_to_pdb_block(rres))
        p = PDBProtein(pp.read_text())
        res = p.to_dict_residue()
        _, res["protein_edit_residue"] = p.query_residues_ligand(lig)
        if len(full) == 0:
            raise ValueError("No editable pocket residues")
        data = {"protein_" + k: v for k, v in torchify_dict(p.to_dict_atom()).items()}
        data.update({"ligand_" + k: v for k, v in torchify_dict(lig).items()})
        data.update(torchify_dict(res))
        data.update(
            seq=seq,
            full_seq_idx=torch.tensor(sorted(full)),
            r10_idx=torch.tensor(sorted(ridx)),
            protein_filename=str(pp),
            ligand_filename=r["ligand"],
            whole_protein_name=r["protein"],
        )
        data = Compose([self.pf, self.lf])(data)
        self.model.generate_id1 = 0
        self.model.generate_id = 0
        rows = []
        for i in range(r["samples"]):
            batch = collate_mols_block(
                [copy.deepcopy(data)], batch_converter=self.converter
            )
            batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.no_grad():
                aar, _, _ = self.model.generate(batch, str(out))
            path = out / f"{i}_whole.pdb"
            if not path.exists():
                raise RuntimeError("PocketGen protein output missing")
            rows.append({"path": str(path), "aar": float(aar)})
        result = {
            "rows": rows,
            "ligand_chemistry_preserved_by": "ligand export disabled; caller retains original SDF",
        }
        atomic_json(out / "generation.json", result)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["flowr", "pocketgen"], required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    worker = (FlowWorker if a.kind == "flowr" else PocketWorker)(
        json.loads(a.config.read_text())
    )
    print(PREFIX + json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("close"):
                return
            result = worker(request)
            response = {"ok": True, "result": result}
        except Exception as e:
            traceback.print_exc()
            response = {"ok": False, "error": repr(e)}
        print(PREFIX + json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
