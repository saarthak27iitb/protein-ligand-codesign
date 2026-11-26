"""Fixed-sequence independent-replica restrained MD, aligned to crystal frame."""

import argparse, json
from pathlib import Path
from .common import ROOT, atomic_json, sha, identity, lock, conformer_align


def build(protein, out, cfg):
    from pdbfixer import PDBFixer
    from openmm import (
        app,
        unit,
        Platform,
        LangevinMiddleIntegrator,
        CustomExternalForce,
    )
    import numpy as np

    protein = Path(protein)
    out = Path(out)
    key = identity({"protein": sha(protein), "md": cfg})
    with lock(out.with_suffix(".lock")):
        manifest = out / "ensemble.json"
        if manifest.exists():
            result = json.loads(manifest.read_text())
            if result["key"] != key:
                raise ValueError("Ensemble cache mismatch")
            return result
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for replica in range(cfg["replicas"]):
            seed = cfg["seed"] + replica
            fixer = PDBFixer(filename=str(protein))
            fixer.missingResidues = {}
            fixer.findMissingAtoms()
            fixer.addMissingAtoms(seed=seed)
            ff = app.ForceField("amber14-all.xml", "implicit/obc2.xml")
            mod = app.Modeller(fixer.topology, fixer.positions)
            mod.addHydrogens(ff, pH=7.0, platform=Platform.getPlatformByName("CPU"))
            system = ff.createSystem(
                mod.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
            )
            force = CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
            force.addGlobalParameter("k", cfg["backbone_restraint_kJ_mol_nm2"])
            for n in ["x0", "y0", "z0"]:
                force.addPerParticleParameter(n)
            for atom in mod.topology.atoms():
                if atom.name in ["N", "CA", "C", "O"]:
                    force.addParticle(
                        atom.index,
                        mod.positions[atom.index].value_in_unit(unit.nanometer),
                    )
            system.addForce(force)
            integrator = LangevinMiddleIntegrator(
                300 * unit.kelvin,
                1 / unit.picosecond,
                cfg["timestep_ps"] * unit.picoseconds,
            )
            integrator.setRandomNumberSeed(seed)
            sim = app.Simulation(
                mod.topology,
                system,
                integrator,
                Platform.getPlatformByName("CUDA"),
                {"Precision": "mixed"},
            )
            sim.context.setPositions(mod.positions)
            sim.minimizeEnergy(maxIterations=1000)
            sim.context.setVelocitiesToTemperature(300 * unit.kelvin, seed)
            sim.step(cfg["equilibration_steps"])
            for i in range(cfg["frames"]):
                sim.step(cfg["steps_per_frame"])
                state = sim.context.getState(getPositions=True, getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                if not np.isfinite(energy):
                    raise RuntimeError("Nonfinite MD energy")
                raw = out / f"replica{replica}_frame{i}.raw.pdb"
                dest = out / f"replica{replica}_frame{i}.pdb"
                with raw.open("w") as f:
                    app.PDBFile.writeFile(
                        sim.topology, state.getPositions(), f, keepIds=True
                    )
                rmsd = conformer_align(raw, protein, dest)
                row = {
                    "path": str(dest),
                    "replica": replica,
                    "frame": i,
                    "seed": seed,
                    "split": "train" if replica < cfg["train_replicas"] else "heldout",
                    "ca_rmsd_A": rmsd,
                    "potential_kJ_mol": float(energy),
                    "eligible": rmsd <= cfg["max_backbone_rmsd_A"],
                }
                rows.append(row)
                atomic_json(out / "progress.json", {"rows": rows, "key": key})
            del sim, integrator, system
        for split in ["train", "heldout"]:
            if sum(r["eligible"] and r["split"] == split for r in rows) < 2:
                raise RuntimeError("Insufficient plausible MD states in " + split)
        result = {
            "key": key,
            "rows": rows,
            "config": cfg,
            "protein_sha256": sha(protein),
            "sequence_fixed": True,
            "interpretation": "Independent local restrained implicit-solvent trajectories, not equilibrium populations or complete biological conformations.",
        }
        atomic_json(manifest, result)
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protein", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    build(a.protein, a.output, json.loads(a.config.read_text())["conformations"])


if __name__ == "__main__":
    main()
