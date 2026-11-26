"""Fresh online proposal support; no fixed ligand/pocket payoff bank."""

import os, sys, json, subprocess, time, copy, queue, threading
from pathlib import Path
import numpy as np
from .common import (
    ROOT,
    CODE_ROOT,
    atomic_json,
    identity,
    sha,
    ligand,
    residues,
    sequence,
    xyz,
)


class Worker:
    def __init__(self, kind, config, logdir, timeout):
        self.kind = kind
        self.config = config
        self.logdir = Path(logdir)
        self.timeout = timeout
        self.p = None

    def start(self):
        self.logdir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(
            PYTHONPATH=str(CODE_ROOT)
            + os.pathsep
            + str(
                CODE_ROOT
                / "vendor"
                / ("flowr" if self.kind == "flowr" else "PocketGen")
            ),
            TORCH_HOME=str(ROOT / "data/torch-cache"),
            FLOWR_NUM_WORKERS="0",
            OPENMM_CPU_THREADS="4",
            OMP_NUM_THREADS="4",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            WANDB_MODE="disabled",
        )
        self.log = (self.logdir / f"{self.kind}.log").open("a")
        self.p = subprocess.Popen(
            [
                str(ROOT / "envs" / self.kind / "bin/python"),
                "-u",
                "-m",
                "research.workers",
                "--kind",
                self.kind,
                "--config",
                str(self.config),
            ],
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()

        def read_stdout():
            for line in self.p.stdout:
                self.log.write(line)
                self.log.flush()
                if "@@DH307@@" in line:
                    self.lines.put(line[line.index("@@DH307@@") :])
            self.lines.put(None)

        self.reader = threading.Thread(target=read_stdout, daemon=True)
        self.reader.start()
        ready = self.receive()
        if not ready.get("ready"):
            raise RuntimeError("Worker did not initialize")

    def receive(self):
        try:
            line = self.lines.get(timeout=self.timeout)
        except queue.Empty:
            self.close()
            raise TimeoutError(self.kind + " worker timeout")
        if line is None:
            raise RuntimeError(f"{self.kind} worker exited {self.p.poll()}")
        return json.loads(line[len("@@DH307@@") :])

    def call(self, request):
        if self.p is None:
            self.start()
        self.p.stdin.write(json.dumps(request) + "\n")
        self.p.stdin.flush()
        response = self.receive()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Worker failure"))
        return response["result"]

    def close(self):
        if self.p is not None:
            if self.p.poll() is None:
                self.p.terminate()
                try:
                    self.p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    self.p.kill()
                    self.p.wait()
            self.reader.join(timeout=10)
            self.p = None
            self.log.close()


def protein_clashes(path):
    from scipy.spatial import cKDTree

    r = residues(path)
    coords = []
    keys = []
    for k, v in r.items():
        for line in v.values():
            coords.append(xyz(line))
            keys.append(k)
    return sum(keys[i] != keys[j] for i, j in cKDTree(coords).query_pairs(1.0))


def project_pocket(current, proposed, native, dest, max_mutations, seed):
    # Transfer selected sidechains onto the current exact backbone; preserve every unselected atom.
    sys.path.insert(0, str(CODE_ROOT / "scripts"))
    from mutation_lattice import substitute, residues as legacy_residues

    a, b, ref = residues(current), residues(proposed), residues(native)
    if a.keys() != b.keys() or a.keys() != ref.keys():
        raise ValueError("Pocket proposal changed scaffold identifiers")
    changed = [
        k
        for k in a
        if next(iter(a[k].values()))[17:20] != next(iter(b[k].values()))[17:20]
    ]
    if not changed:
        raise ValueError("No proposed residue change")
    rng = np.random.default_rng(seed)
    rng.shuffle(changed)
    # Try largest bounded subset first; reject rather than silently exceed cumulative native budget.
    selected = set()
    for k in changed:
        trial = selected | {k}
        count = sum(
            (next(iter(b[x].values())) if x in trial else next(iter(a[x].values())))[
                17:20
            ]
            != next(iter(ref[x].values()))[17:20]
            for x in a
        )
        if count <= max_mutations:
            selected = trial
    if not selected:
        raise ValueError("No edits within cumulative mutation budget")
    substitute(legacy_residues(current), legacy_residues(proposed), selected, dest)
    new = residues(dest)
    for k, v in a.items():
        for name, line in v.items():
            if k not in selected or name in {"N", "CA", "C", "O", "OXT"}:
                if name not in new[k] or not np.array_equal(
                    xyz(line), xyz(new[k][name])
                ):
                    raise ValueError("Fixed backbone/scaffold moved")
    return dest


class Proposals:
    def __init__(self, config_path, config, out):
        self.cfg = config
        self.config_path = Path(config_path).resolve()
        self.root = Path(out)
        self.counter = 0
        self.generated = 0
        self.valid = 0
        self.pocket_generated = 0
        self.failures = 0
        existing = [
            int(p.name) for p in (self.root / "proposals").glob("*") if p.name.isdigit()
        ]
        self.counter = max([0] + [x + 1 for x in existing])
        for file in (self.root / "proposals").glob("*/flowr/generation.json"):
            old = json.loads(file.read_text())
            self.generated += old["returned"]
            self.valid += old["valid"]
        for file in (self.root / "proposals").glob("*/pocketgen/generation.json"):
            self.pocket_generated += len(json.loads(file.read_text())["rows"])
        self.flow = Worker(
            "flowr",
            self.config_path,
            self.root / "workers",
            config["generation"]["timeout_seconds"],
        )
        self.pocket = Worker(
            "pocketgen",
            self.config_path,
            self.root / "workers",
            config["generation"]["timeout_seconds"],
        )

    def generate(self, protein, ligand_path, target, seed, cooperative=True):
        folder = self.root / "proposals" / f"{self.counter:07d}"
        self.counter += 1
        folder.mkdir(parents=True, exist_ok=False)
        ligands = [str(ligand_path)]
        pockets = [str(protein)]
        errors = []
        try:
            result = self.flow.call(
                dict(
                    protein=str(protein),
                    ligand=str(ligand_path),
                    seed=seed,
                    samples=self.cfg["generation"]["ligands"],
                    output=str(folder / "flowr"),
                )
            )
            self.generated += result["returned"]
            self.valid += result["valid"]
            ligands.extend(x["path"] for x in result["rows"] if x["valid"])
        except Exception as e:
            self.failures += 1
            errors.append({"flowr": repr(e)})
        if cooperative:
            try:
                result = self.pocket.call(
                    dict(
                        protein=str(protein),
                        ligand=str(ligand_path),
                        seed=seed,
                        samples=self.cfg["generation"]["pockets"],
                        output=str(folder / "pocketgen"),
                    )
                )
                self.pocket_generated += len(result["rows"])
                for i, row in enumerate(result["rows"]):
                    try:
                        path = folder / f"pocket_{i}.pdb"
                        project_pocket(
                            protein,
                            row["path"],
                            target["protein"],
                            path,
                            self.cfg["constraints"]["max_mutations"],
                            seed + i,
                        )
                        if (
                            protein_clashes(path)
                            > protein_clashes(target["protein"])
                            + self.cfg["constraints"]["protein_clash_increase_max"]
                        ):
                            raise ValueError(
                                "Protein internal clashes increased beyond limit"
                            )
                        pockets.append(str(path))
                    except Exception as e:
                        errors.append({"projection": repr(e)})
            except Exception as e:
                self.failures += 1
                errors.append({"pocketgen": repr(e)})
        atomic_json(
            folder / "manifest.json",
            dict(ligands=ligands, pockets=pockets, errors=errors, seed=seed),
        )
        return ligands, pockets

    def close(self):
        self.flow.close()
        self.pocket.close()
