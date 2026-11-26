# Protein–Ligand Design

**Cooperative and Conformation-Aware Search**
Saarthak Krishan · 22B3959 · DH307 · Supervisor: Prof. Debabrata Maiti

Exploratory research on learned candidate selection over pretrained molecular
generators. Two lightweight PPO actor–critic policies choose ligand/pocket edits
in a cooperative setting, or ligand/protein-conformation combinations in a
competitive robustness setting. FLOWR and PocketGen remain frozen. PocketGen is
the implementation substitute for the PocketFlow-inspired pocket-design direction.
This is not a single jointly trained bidirectional molecular generator.

## Current evidence

Three cooperative policy seeds completed evaluation. Higher utility is better:

| Method | Mean utility |
|---|---:|
| Learned cooperative policy | 6.350 |
| Random joint search | 6.301 |
| Greedy alternating | 6.227 |
| Ligand-only | 5.902 |

These results **do not establish a reliable RL advantage**. The paired uncertainty
interval for learned versus random includes zero. Actual scoring usage differs,
and ligand-only often has no eligible new candidate. Competitive evaluation is
partial and contains candidate-filter failures. See [results notes](docs/RESULTS.md).
There is no experimental binding or protein-foldability result. Distillation and
balancing are proposed next experiments, not implemented results.

## Repository map

| Path | Contents |
|---|---|
| `research/` | Current PPO policies, feature encoding, generators' workers, scoring, MD, data preparation and evaluation |
| `scripts/` | Supporting helpers plus historical pilots; see `docs/PILOT_HISTORY.md` |
| `configs/` | Separate FLOWR and PocketGen environment specifications |
| `experiments/full_v1/` | Exact experiment configuration and cluster Slurm templates |
| `tests/` | Policy and structural-invariant checks |
| `patches/`, `provenance/` | Pinned upstream commits, compatibility patches, package inventories and artifact sources |
| `results/`, `reports/full_v1/` | Compact per-target evidence, aggregate summaries and original analysis scripts |
| `presentation/final/` | Final presentation PDF, Beamer source and speaker notes |
| `presentation/progress/` | Short progress presentation |

## Setup and reproduction

This is a source-checkout workflow for Linux/CUDA. It is not a one-command,
self-contained package: model weights, upstream code, datasets and full MD/run
artifacts are external. Start with [REPRODUCING.md](docs/REPRODUCING.md) and
[ARTIFACTS.md](docs/ARTIFACTS.md).

```bash
python3 scripts/setup_upstreams.py
# Create the two environments and stage external assets as documented.
export DH307_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/vendor/flowr"
OMP_NUM_THREADS=1 envs/flowr/bin/python tests/test_research.py
python3 analysis/summarize_shared_results.py
```

The scientific pipeline in `research/` is byte-for-byte equal to the submitted
runtime source. See `provenance/source_comparison.json`. Export-specific changes
are documentation, sharing utilities and removal of a server-path fallback from
a Slurm template; no training behavior was changed. Installation and a complete
GPU reproduction on a fresh machine have not been validated.

The Slurm templates include the original cluster partition/account/resource
settings; adapt them before submitting. `research.submit` can submit six GPU
training jobs plus a dependent report. Old pilot submission scripts are retained
for provenance and should not be used for the current campaign.

## Scientific details and limitations

- [Implementation and protocol](README_RESEARCH.md)
- [Current results and limitations](docs/RESULTS.md)
- [Papers, upstream code and attribution](THIRD_PARTY_NOTICES.md)
- [Final presentation](presentation/final/DH307_RL_PPO_explained.pdf)

The repository omits environments, datasets, large checkpoints, generated molecules,
MD trajectories, raw logs, duplicate snapshots and paper PDFs. Original papers
and downloadable assets are linked instead. Export prepared 27 November 2026;
status reflects saved artifacts, not a live scheduler query.

## Licensing

An original-code reuse license has not yet been selected for this research
snapshot. Upstream projects and downloaded assets retain their own terms;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Their licenses are not
replaced by this repository's documentation.
