"""Paired, protein-cluster bootstrap summaries from all seeds and baselines."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from .common import atomic_json


def bootstrap(values, reps, seed=307):
    x = np.asarray(values, float)
    if len(x) == 0:
        return None
    rng = np.random.default_rng(seed)
    samples = [np.mean(rng.choice(x, len(x), replace=True)) for _ in range(reps)]
    return {
        "mean": float(x.mean()),
        "ci95": [float(v) for v in np.quantile(samples, [0.025, 0.975])],
        "protein_clusters": len(x),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    manifest = json.loads(a.manifest.read_text())
    clusters = {x["id"]: x["sequence_sha256"] for x in manifest["splits"]["test"]}
    summary = {}
    table = []
    for setting in ["cooperative", "competitive"]:
        by = {}
        statuses = []
        for seed in cfg["seeds"]:
            folder = a.root / setting / f"seed_{seed}"
            file = folder / "status.json"
            if not file.exists():
                raise RuntimeError(f"Missing run: {folder}")
            status = json.loads(file.read_text())
            statuses.append(status)
            if status["stage"] != "complete":
                raise RuntimeError(f"Incomplete run: {folder}")
            rows = json.loads((folder / "evaluation/test/results.json").read_text())
            common_queries = {}
            for row in rows:
                target = row["target"]
                common_queries[target] = min(
                    common_queries.get(target, 64), row.get("optimization_queries", 0)
                )
            for row in rows:
                score = row.get("heldout_worst", row.get("best_utility", -20.0))
                if setting == "cooperative":
                    # Compare at the same realized oracle count, not merely the
                    # same cap when some baselines have fewer legal actions.
                    candidates = [row.get("initial_utility")]
                    candidates.extend(
                        x.get("utility")
                        for x in row.get("observations", [])[
                            : common_queries[row["target"]]
                        ]
                    )
                    feasible = [float(x) for x in candidates if x is not None]
                    score = max(feasible) if feasible else -20.0
                by.setdefault(row["method"], {}).setdefault(row["target"], []).append(
                    score
                )
                table.append(
                    dict(
                        setting=setting,
                        seed=seed,
                        method=row["method"],
                        target=row["target"],
                        ok=row["ok"],
                        score=score,
                        full_budget_score=row.get("best_utility", -20.0),
                        matched_query_budget=common_queries[row["target"]],
                        optimization_queries=row.get("optimization_queries", 0),
                        generated_ligands=row["costs"]["generated_ligands"],
                    )
                )
        means = {
            m: {t: float(np.mean(v)) for t, v in targets.items()}
            for m, targets in by.items()
        }
        comparisons = {}
        for m in means:
            values = {}
            for target, value in means[m].items():
                values.setdefault(clusters[target], []).append(value)
            method_summary = bootstrap(
                [np.mean(v) for v in values.values()],
                cfg["evaluation"]["bootstrap_samples"],
            )
            if m != "learned":
                delta = {}
                for t, value in means["learned"].items():
                    if t in means[m]:
                        delta.setdefault(clusters[t], []).append(value - means[m][t])
                method_summary["learned_minus_baseline"] = bootstrap(
                    [np.mean(v) for v in delta.values()],
                    cfg["evaluation"]["bootstrap_samples"],
                )
            comparisons[m] = method_summary
        summary[setting] = {
            "methods": comparisons,
            "runs": statuses,
            "unit": "protein sequence clusters, averaging policy seeds and systems within cluster",
            "metric": (
                "heldout local-ensemble worst utility"
                if setting == "competitive"
                else "best feasible utility at matched realized oracle count"
            ),
            "interpretation": "Computational proxy comparisons; no experimental binding or foldability validation.",
        }
    a.root.mkdir(parents=True, exist_ok=True)
    atomic_json(a.root / "summary.json", summary)
    with (a.root / "per_target.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=table[0].keys())
        w.writeheader()
        w.writerows(table)
    print(json.dumps({k: v["methods"] for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
