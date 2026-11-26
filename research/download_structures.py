"""Stage only RCSB PDB files from an internet-enabled login host."""

import argparse, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from .prepare import fetch
from .common import atomic_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    a = p.parse_args()
    ids = json.loads(a.plan.read_text())["ids"]
    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(fetch, i, a.cache): i for i in ids}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as e:
                errors.append({"id": jobs[job], "error": repr(e)})
            done += 1
            if done % 25 == 0:
                print(f"RCSB: {done}/{len(ids)}; failures={len(errors)}", flush=True)
    atomic_json(
        a.cache / "download_report.json", {"requested": len(ids), "errors": errors}
    )
    print(
        f"Finished {len(ids)} downloads, {len(errors)} unavailable entries", flush=True
    )


if __name__ == "__main__":
    main()
