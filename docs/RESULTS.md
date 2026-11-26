# Exploratory results and current limitations

Snapshot checked against saved artifacts on 27 November 2026. Status is not a live
scheduler query. Both settings trained three policy seeds for 512 episodes each.
Cooperative evaluation completed for all three; competitive evaluation is partial.

## Cooperative comparison

| Method | Mean utility |
|---|---:|
| Learned cooperative policy | 6.350307 |
| Random joint | 6.300863 |
| Greedy alternating | 6.226692 |
| Ligand-only | 5.902440 |

The three policy seeds evaluate the same systems, not independent new test sets.
The exploratory paired cluster-bootstrap interval for learned minus random
includes zero. Full-cap mean differences are not a compute-matched superiority
claim. Different action eligibility leads to different actual evaluator usage.
Ligand-only has no eligible new candidate on 48 of 64 systems. Final feasible designs
are 61 of 64 per method per seed. Utility is a docking/strain/mutation proxy, not an assay.

The detailed three-seed summary is
`reports/full_v1/cooperative_completed_seeds_42_123_2026.json`. The included
`analysis/summarize_shared_results.py` independently recomputes headline means
from `results/per_target_evaluations.json`.

## Competitive evaluation

Saved entries per seed: 42 → 8, 123 → 20, 2026 → 176, each of an expected 256 target–method
entries. They include errors; these are not molecule counts. No complete
competitive performance comparison is claimed. Errors are retained in the compact
records and status snapshot.

An audited uniform-ensemble case for target 7vjt__1__1.A__1.C (seed 2026) generated
64 molecules: 62 were valid and connected, but all 62 failed reference Tanimoto ≥ 0.25 (maximum
0.19753). They were rejected before conformation scoring. This diagnoses one
case, not all errors. Another evaluation encountered an all-masked action set.
Interrupted jobs and candidate-level errors are separate issues; signal 9 does not
by itself establish an out-of-memory cause. The dependent report job was cancelled;
this export does not attempt to recreate or submit it.

## What remains unresolved

- De novo proposals are filtered as near-reference analogues; align task and sampling.
- Effective ligand/pocket action availability may be imbalanced; measure contributions.
- Lightweight observations can lose local chemistry and geometry.
- Restrained MD covers local flexibility, not full equilibrium conformational states.
- Binding, foldability, expression and catalysis have not been experimentally validated.
- Distillation into selectors followed by PPO and balanced training are proposed work.
