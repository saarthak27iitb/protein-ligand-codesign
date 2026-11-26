"""Freeze runnable source and submit at most eight jobs, with an auditable ledger.

Does not cancel any job. Existing ledger entries are never duplicated. A failed
run is resumed by explicitly requeueing its recorded job, not by silently creating
another copy. Source is frozen before submission and imported from the snapshot.
"""

import argparse, datetime, json, os, shutil, subprocess, tarfile
from pathlib import Path
from .common import ROOT, atomic_json, sha, lock


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config", type=Path, default=ROOT / "experiments/full_v1/config.json"
    )
    p.add_argument("--finalize-job")
    p.add_argument("--freeze-only", action="store_true")
    a = p.parse_args()
    folder = ROOT / "experiments/full_v1"
    ledger = folder / "jobs.json"
    snapshot = folder / "snapshot"
    with lock(folder / "submission.lock"):
        jobs = (
            json.loads(ledger.read_text())
            if ledger.exists()
            else {
                "jobs": [],
                "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ["research", "scripts", "configs", "tests"]:
                shutil.copytree(
                    ROOT / name,
                    snapshot / name,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
                )
            for name in ["flowr", "PocketGen"]:
                shutil.copytree(
                    ROOT / "vendor" / name,
                    snapshot / "vendor" / name,
                    ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
                )
            (snapshot / "experiments/full_v1").mkdir(parents=True)
            shutil.copy2(a.config, snapshot / "experiments/full_v1/config.json")
            for script in folder.glob("*.slurm"):
                shutil.copy2(script, snapshot / "experiments/full_v1" / script.name)
            for x in ["README_RESEARCH.md", "pyproject.toml", "EXPERIMENTS.md"]:
                if (ROOT / x).exists():
                    shutil.copy2(ROOT / x, snapshot / x)
            hashes = {
                str(f.relative_to(snapshot)): sha(f)
                for f in snapshot.rglob("*")
                if f.is_file()
            }
            atomic_json(snapshot / "source_hashes.json", hashes)
            with tarfile.open(folder / "submission_source.tar.gz", "w:gz") as t:
                t.add(snapshot, arcname="dh307-molecular-games")
        if sha(a.config) != sha(snapshot / "experiments/full_v1/config.json"):
            raise ValueError("Frozen config changed; create a versioned campaign")
        jobs["snapshot"] = str(snapshot)
        jobs["source_archive"] = str(folder / "submission_source.tar.gz")
        jobs["config_sha256"] = sha(a.config)
        atomic_json(ledger, jobs)
        if a.freeze_only:
            print(json.dumps(jobs, indent=2))
            return
        active = subprocess.check_output(
            ["squeue", "-h", "-r", "-u", str(os.getuid()), "-o", "%i"], text=True
        ).splitlines()
        need = (
            6
            - len([x for x in jobs["jobs"] if x["kind"] == "training"])
            + int(not any(x["kind"] == "report" for x in jobs["jobs"]))
        )
        if len(active) + need > 8:
            raise RuntimeError(
                f"{len(active)} active jobs + {need} new exceeds eight. Resolve obsolete dependencies first."
            )
        dependency = a.finalize_job
        if not dependency and not (ROOT / "data/full_v1/manifest.json").exists():
            raise ValueError("Supply --finalize-job or a completed manifest")

        def submit(kind, label, setting=None, seed=None, deps=None):
            existing = next((x for x in jobs["jobs"] if x["label"] == label), None)
            if existing:
                return existing["job_id"]
            env = f"ALL,DH307_ROOT={ROOT},DH307_CODE_ROOT={snapshot}"
            if setting:
                env += f",SETTING={setting},POLICY_SEED={seed}"
            command = [
                "sbatch",
                "--parsable",
                f"--chdir={ROOT}",
                f"--job-name={label}",
                f"--export={env}",
            ]
            if deps:
                command += ["--dependency=afterok:" + ":".join(deps)]
            script = folder / ("train.slurm" if kind == "training" else "report.slurm")
            command.append(str(script))
            job = (
                subprocess.check_output(command, text=True, cwd=ROOT)
                .strip()
                .split(";")[0]
            )
            if not job.isdigit():
                raise RuntimeError("Unexpected scheduler response " + job)
            row = {
                "kind": kind,
                "label": label,
                "job_id": job,
                "setting": setting,
                "seed": seed,
                "dependencies": deps or [],
                "command": command,
                "submitted_utc": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
            jobs["jobs"].append(row)
            atomic_json(ledger, jobs)
            print(json.dumps(row), flush=True)
            return job

        ids = []
        for seed in json.loads(a.config.read_text())["seeds"]:
            for setting in ["cooperative", "competitive"]:
                ids.append(
                    submit(
                        "training",
                        f"dh307-{setting[:4]}-{seed}",
                        setting,
                        seed,
                        [dependency] if dependency else None,
                    )
                )
        submit("report", "dh307-full-report", deps=ids)


if __name__ == "__main__":
    main()
