# Local upstream compatibility fixes

PocketGen commit: see repository-commits.json.

1. utils/relax.py uses unit.kilojoules_per_mole but did not import unit.
   Added `from openmm import unit`.
2. models/PD.py called the local-pocket PDB writer with the full-protein path.
   The collator explicitly stores the full structure in `protein_filename` and
   the local structure in `pocket_filename`. Pass `pocket_filename` to the two
   local-pocket writers. Leave the full-protein writer unchanged.

The pilot labels the upstream RMSD-like scalar as nonstandard: its formula is
not conventional coordinate RMSD. It must not be reported as standard RMSD.

These changes need verification in the GPU inference pilot before scientific use.

3. utils/transforms.py imported subgraph from a removed PyG module path.
   Use the supported public `torch_geometric.utils` import (PyG 2.5.3).

FLOWR environment: biotite 1.0.1 resolves to biotraj 1.2.2, which requires
SciPy >=1.13. Keep NumPy 1.26.2 and use SciPy 1.13.1 instead of upstream 1.11.4.
This is an explicit dependency deviation to record when comparing reproduction results.

FLOWR's datamodule hardcoded eight data-loading workers, irrespective of the
allocated CPUs and generation CLI. Respect `FLOWR_NUM_WORKERS`, defaulting to
the smaller of eight and available CPU affinity. Pilots set zero workers.

Use NumPy 1.26.4 to satisfy open-source PyMOL 3.1.0 (instead of upstream
1.26.2), and pin setuptools 80.9.0 because ProLIF 2.0.3 uses pkg_resources,
removed by newer setuptools. Upstream 65.6.3 conflicts with current SIP >=69.5.

Load GenBench3D's notebook-only MolConfViewer lazily so headless inference does
not require nglview/Jupyter. Accessing that viewer still imports its real dependencies.

Load FLOWR checkpoints onto CPU explicitly before moving the constructed model
to GPU. Published tensors carry CUDA device tags; this enables CPU preflight and
avoids allocating checkpoint tensors on a training machine's original GPU index.

Job 454949 exposed a generation-postprocessing failure: the diversity filter
fingerprinted a None molecule. The bulk filter now excludes None, empty and
unsanitizable molecules before computing fingerprints, while keeping optional
pocket associations aligned. Empty paired results return a tuple of empty lists.
Valid input molecules are copied rather than modified in place.
Generation now saves raw proposals before postprocessing, records filtered
proposals separately, and fails explicitly if no valid molecules remain.
The raw proposals remain available for unbiased validity accounting.
