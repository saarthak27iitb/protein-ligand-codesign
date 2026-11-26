# Reproducing the experiment

Use the checkout as the working directory. Keep the molecular environments separate.
The scientific code and fixed configuration match the submitted campaign; the
export is not a freshly validated installer or a complete data archive.

## Upstream code

```bash
python3 scripts/setup_upstreams.py
python3 scripts/setup_upstreams.py --check-only
```

This downloads the exact two commits in `provenance/upstreams.json` and applies
our recorded compatibility patches. Existing modified checkouts are not replaced.
Upstream code lives under ignored `vendor/`; only pins and patches belong in Git.

## Environments

With Conda or Mamba available, create environments at the expected paths:

```bash
conda env create -p "$PWD/envs/flowr" -f configs/flowr-env.yml
envs/flowr/bin/python -m pip install -r configs/flowr-pip.txt
conda env create -p "$PWD/envs/pocketgen" -f configs/pocketgen-env.yml
envs/pocketgen/bin/python -m pip install -r configs/pocketgen-pip.txt
envs/pocketgen/bin/python -m pip install --only-binary=:all: \
  torch_scatter torch_cluster torch_sparse torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
```

The PyG wheel URL corresponds to the original Torch2.1/CUDA11.8 environment.
Different CUDA/PyTorch combinations require corresponding wheels. The package
inventories in `provenance/` record observed versions; they are not portable lock
files. Conda-supplied libraries are required in addition to pip dependencies.
The old `scripts/setup_environments.sh` expects a separately installed
`bin/micromamba`; that executable is not distributed here.

`pyproject.toml` defines research CLI entry points, but does not install the two
complete molecular stacks or package all the checkout-relative helper scripts.
Use the source checkout and separate environments above rather than expecting a
standalone pip installation to run generation.

## Weights and data

See `ARTIFACTS.md`. Place model files at the paths used by `research/workers.py`.
The SPINDR archive should be `downloads/smol_data.zip`. Set the project root:

```bash
export DH307_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/vendor/flowr"
envs/flowr/bin/python -m research.stage_data --stage export \
  --config experiments/full_v1/config.json --output data/full_v1
# Full-chain structures are staged from an internet-enabled host:
envs/flowr/bin/python -m research.download_structures \
  --plan data/full_v1/downloads.json --cache data/full_v1/rcsb
envs/flowr/bin/python -m research.stage_data --stage finalize \
  --config experiments/full_v1/config.json --output data/full_v1
```

`provenance/cohort_ids.json` records the original selected IDs and sequence hashes.
Compare a regenerated manifest with these IDs. RCSB contents or external asset
versions can change; exact historical recreation is not guaranteed without the
original staged data. Manifests contain machine-local paths and must be generated
or relocated for the new checkout; do not copy a server manifest blindly.

## Checks and execution

```bash
OMP_NUM_THREADS=1 envs/flowr/bin/python tests/test_research.py
python3 analysis/summarize_shared_results.py
envs/flowr/bin/python -m research.run --help
```

The structure-projection test uses an optional real historical PocketGen artifact
and is skipped when that excluded artifact is absent. Other checks test alignment,
masks, all-masked rejection and PPO learning preference, not scientific performance.

For GPU runs, adapt Slurm templates to your cluster first. Ensure upstream code,
weights, ESM cache and prepared data are staged before scheduling:

```bash
mkdir -p logs
envs/flowr/bin/python -m research.submit --freeze-only
# The following command submits jobs; run only when ready for GPU work:
envs/flowr/bin/python -m research.submit
```

There are no old jobs or snapshot directories in this export. The submission
ledger is created locally and ignored by Git. A fresh full campaign requires
substantial compute; the source-only checks above do not establish end-to-end
reproduction. Checkpoint outputs include optimizer/RNG state for resuming runs.

## Results analysis

`analysis/summarize_shared_results.py` recomputes the headline means using only
included compact records. The original `reports/full_v1/analyze_three_seeds.py`
requires the original prepared references, manifest and raw evaluation results
at their documented paths; these large inputs are external.
