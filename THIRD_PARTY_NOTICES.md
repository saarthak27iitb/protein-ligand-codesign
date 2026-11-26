# Attribution and upstream materials

This repository contributes the controller/orchestration implementation and local
compatibility patches. It does not claim authorship of the molecular generators.

| Work | Role | Primary sources |
|---|---|---|
| FLOWR | Frozen ligand generator | https://arxiv.org/abs/2504.10564 ; https://github.com/jule-c/flowr |
| PocketGen | Frozen pocket proposal model | https://doi.org/10.1038/s42256-024-00920-9 ; https://github.com/zaixizhang/PocketGen |
| PocketFlow | Original pocket-direction motivation; not implemented here | https://arxiv.org/abs/2409.19520 |
| PPO | Selection-policy optimization | https://arxiv.org/abs/1707.06347 |
| FlexSBDD | Protein-flexibility precedent | https://arxiv.org/abs/2409.19645 |

Pinned commits and local patch checksums are in `provenance/upstreams.json`.
Upstream source is downloaded separately into ignored `vendor/`; it is not bundled
or re-licensed here. PocketGen includes an MIT license. The pinned FLOWR checkout
had no top-level LICENSE file; no reuse rights for that source are inferred from
the PocketGen license or from a dataset/checkpoint license. Consult each upstream
project's current terms for its materials. Paper PDFs are linked, not copied.

Dataset and checkpoint records include their own terms. The downloaded FLOWR
records identify CC-BY4.0 for those released assets; this does not establish the
license of every source-code dependency or other downloaded artifact.

No original-code reuse license has been selected for this snapshot. Choosing a
license for project-authored code is a separate decision from preserving upstream
attribution and terms.
