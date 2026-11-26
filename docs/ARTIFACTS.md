# External artifacts (not committed to Git)

Exact released file URLs, sizes and published checksums where available are in
`provenance/artifact_sources.json`. Download original assets from their authors
rather than mirroring pretrained weights or datasets in this source repository.

| Artifact | Expected local path | Source |
|---|---|---|
| FLOWR without-hydrogens checkpoint | `checkpoints/flowr_noHs.ckpt` | https://zenodo.org/records/15737419 |
| Prepared SPINDR archive | `downloads/smol_data.zip` | https://zenodo.org/records/15257565 |
| PocketGen released checkpoint | `checkpoints/pocketgen.pt` | https://github.com/zaixizhang/PocketGen#model-checkpoints |
| ESM2 650M checkpoint and contact-regression weights | Cache beneath `data/torch-cache/` | PocketGen/ESM download mechanism; original staged files also exist under the server's checkpoints directory |
| Full protein chains | `data/full_v1/rcsb/` | RCSB PDB, staged with `research.download_structures` |

The workers set `TORCH_HOME` to `data/torch-cache`. Stage any ESM dependencies
before using a compute node without internet. Model loading depends on the
upstream pinned code's expected cache layout.

The source export omits trained controller weights, generated molecules, MD
frames, raw oracle caches and full episode logs. These remain on the research
server under `runs/full_v1/`. Share them separately as a versioned scientific
artifact if needed, with protocol, provenance and intended scope documented.
Git LFS or a suitable artifact repository may be used for selected future assets;
ordinary source Git history should not contain the full working directory.

The included compact evaluation records retain IDs, methods, scores, feasibility,
errors and search histories without machine-specific file paths. They are sufficient
to check reported means but do not recreate molecular generation or structural
validation without the external structures and model files.
