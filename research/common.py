"""Atomic artifacts, source identity, file locks, and geometry invariants."""

from pathlib import Path
from contextlib import contextmanager
import hashlib, json, os, random, time, fcntl
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("DH307_ROOT", str(CODE_ROOT))).resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(obj, allow_nan=False) + "\n")
        f.flush()


def residues(path):
    result = {}
    for l in Path(path).read_text().splitlines():
        if l.startswith("ATOM  ") and l[16] in (" ", "A"):
            if (l[76:78].strip() or l[12:16].strip()[0]).upper() == "H":
                continue
            key = (l[21], l[22:26], l[26])
            result.setdefault(key, {})[l[12:16].strip()] = l
    return result


def xyz(line):
    return np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])


def sequence(path):
    return tuple((k, next(iter(v.values()))[17:20]) for k, v in residues(path).items())


def kabsch(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    u, _, vt = np.linalg.svd((p - p.mean(0)).T @ (q - q.mean(0)))
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    r = u @ vt
    t = q.mean(0) - p.mean(0) @ r
    return r, t, float(np.sqrt(np.mean(np.sum((p @ r + t - q) ** 2, axis=1))))


def transform_pdb(source, dest, r, t):
    out = []
    for l in Path(source).read_text().splitlines():
        if l.startswith(("ATOM  ", "HETATM")):
            v = xyz(l) @ r + t
            l = l[:30] + "".join(f"{x:8.3f}" for x in v) + l[54:]
        out.append(l)
    Path(dest).write_text("\n".join(out) + "\n")


def conformer_align(source, native, dest):
    a, b = residues(source), residues(native)
    if sequence(source) != sequence(native):
        raise ValueError("MD changed residue identity or identifiers")
    keys = [k for k in a if "CA" in a[k] and "CA" in b[k]]
    r, t, rmsd = kabsch(
        [xyz(a[k]["CA"]) for k in keys], [xyz(b[k]["CA"]) for k in keys]
    )
    transform_pdb(source, dest, r, t)
    return rmsd


def ligand(path):
    from rdkit import Chem

    mol = Chem.SDMolSupplier(str(path), removeHs=True)[0]
    if mol is None:
        raise ValueError("Invalid ligand")
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Disconnected ligand")
    if (
        mol.GetNumConformers() != 1
        or not np.isfinite(mol.GetConformer().GetPositions()).all()
    ):
        raise ValueError("Invalid ligand geometry")
    return mol


def fingerprint(mol):
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    ).GetFingerprint(mol)


def similarity(a, b):
    from rdkit import DataStructs

    return float(DataStructs.TanimotoSimilarity(fingerprint(a), fingerprint(b)))


def seed_all(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Policy code stays on CPU; only worker processes allocate CUDA.
