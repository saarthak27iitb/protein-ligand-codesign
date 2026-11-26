> Original full-v1 implementation protocol. For current results and sharing limitations, see `README.md` and `docs/RESULTS.md`.

# Protein–Ligand Design: Online Cooperative and Conformation-Adversarial Policies

Saarthak Krishan, 22B3959. DH307 research implementation.

This package implements two **online learned decision-policy experiments**, using
released, frozen molecular generators. It does not fine-tune FLOWR/PocketGen's
molecular weights, implement PocketFlow, or establish experimental binding.
The old scripts under `scripts/` remain as pilot records. Use `research/` and
`experiments/full_v1/` for this campaign.

## Algorithms

**Cooperation.** A ligand actor selects among fresh FLOWR proposals conditioned
on the current pocket. A pocket actor, conditioned on the selected ligand, selects
among fresh PocketGen-proposed edits. Both receive the same feasible pair utility
improvement minus action cost. Feasible worsening intermediate moves are allowed
in training; the controllers are not confined to greedy acceptance. The two
variable-action actors have separate value baselines and a common PPO optimizer.
They are trained across protein targets, not fitted to one cached payoff table.

**Competition.** A ligand actor selects among fresh proposals generated across
fixed-sequence protein conformations. An adversary observes the selected ligand
and chooses a plausible training conformation. Rewards are opposite: the ligand
maximizes utility, the conformation actor minimizes it. Local implicit-solvent MD
uses independent replicas; the last replica is withheld from all training rewards,
selection and checkpoint decisions. This is a sequential, ligand-aware robustness
opponent, not a claim that proteins deliberately compete or that equilibrium
populations are known. Molecular generators remain frozen.

Both are hierarchical proposal-selection methods. They learn **which designs to
select**, not atom-by-atom molecular synthesis or a single bidirectional generator.

## Fixed protocol

See `experiments/full_v1/config.json` for authoritative values.

- 64 controller-training systems, 16 validation systems, 64 test systems from the
  official SPINDR splits; three policy seeds 42, 123, 2026 for each setting.
- 512 training episodes/seed,4 decisions/episode;8 ligand and 3 pocket proposals per
  cooperative decision;100 FLOWR integration steps.
- PPO updates every 8 episodes, 4 optimization passes,128/64 hidden actor layers,
  learning rate 3e-4, ratio clip 0.2, entropy coefficient 0.01, gradient norm clip 1.
- A 40-channel invariant descriptor observation:9 ligand descriptors,20 amino-acid
  composition channels,8 local contact-distance histogram bins,utility,budget and
  ligand charge. This is a lightweight descriptor policy, not a full equivariant
  learned protein encoder or a sufficient-state guarantee.
- Cooperative test baselines: ligand-only, greedy alternating, random joint.
  All receive the same proposal-count and oracle-budget caps. The report also
  compares at matched realized oracle counts when an action space is smaller. Learned selection
  ranks joint pairs by the factorized actor probabilities. Every method uses the
  same feasible best-so-far selection during evaluation.
- Competitive baselines: rigid-context generation, uniform ensemble generation,
  and ensemble generation with cheap clash-geometry ranking. All methods generate
  the same number of proposals, select up to 8 candidates for exact screening across
  up to 8 training states (64 oracle requests), and select by training worst case.
  Thus “rigid” includes a common final robust screen; it is not an entirely
  conformation-blind end-to-end protocol. Held-out replica scores are never used
  to pick the ligand. The learned adversary economizes training feedback; exact
  training-ensemble minimization is used for evaluation.
- Checkpoint selection uses validation systems only. Test evaluation begins after
  loading the frozen validation-selected policy. Default validation occurs every
  64 episodes. All failures and feasibility rates remain in the reports.

## Data and conformation preparation

SPINDR's serialized protein objects are **cropped pockets**. They must not be
simulated as complete proteins. The export stage extracts chemistry and prepared
pocket coordinates. Public RCSB full protein chains are matched using residue
identities/C-alpha coordinates, rigidly aligned to the prepared pocket, and checked
for standard residues, complete backbone, length and resolved chain continuity.
Prepared and source residue numbering can differ; constant offsets are tested
and accepted only with residue-identity and geometric alignment checks. Validation
selection prioritizes scaffolds with lower test overlap before selecting test and
training systems. Selection uses these structural criteria, not method performance.

Exact protein sequences and Murcko scaffolds are excluded across controller
splits. This is not a guarantee of family-level novelty. FLOWR's training includes
SPINDR train; PocketGen's pretraining overlap has not been ruled out. The manifest
records every exclusion and source index; insufficient targets cause an error,
not an unnoticed reduction to a handful of examples.

Each selected full chain has 3 independent restrained MD replicas: 300 K, 2 fs steps,
100 ps equilibration and approximately 1 ns production per replica, 6 saved frames.
Backbone restraints are 418.4 kJ/mol/nm². Frames are aligned back to the crystal frame
and require C-alpha RMSD at most 2 Å. Two replicas supply training states; one supplies
held-out states. These are local, restrained, implicit-solvent perturbations—not
long-timescale conformational transitions, equilibrium state populations, or a
substitute for experimental alternate conformations. Other chains, ions and
cofactors are not modeled in this protocol.

## Chemistry and structural contracts

The PocketGen ligand writer rebuilds molecules without all formal-charge metadata.
Our worker disables ligand export and uses only protein proposals. Selected
sidechains are transferred onto the **unchanged current backbone**, preserving
all unselected scaffold atoms. The cumulative mutation cap is 4 relative to the
starting chain. Proposed internal protein clashes are constrained. This is
fixed-backbone pocket sequence/sidechain design; generated backbone changes are
not accepted silently.

Ligands must be connected, finite 3D structures, within 20% of reference heavy-atom
count, with absolute formal charge at most 2, QED at least 0.2 and reference Morgan
Tanimoto similarity at least 0.25. Retention of an exact chemical core is not
claimed. MMFF minimization must converge; ligand strain must be at most 100 kcal/mol
and protein–ligand overlap count at most 5 under the specified 0.4 Å criterion.

Utility is minus input-pose Vina score minus 0.01 times MMFF94s ligand strain
minus 0.05 times the mutation count. Full-docking Vina and UFF ligand strain are
reported for selected test designs. Full docking is a more extensive search with
the **same Vina scoring family**, not an independent experimental affinity oracle.
Protein foldability/expression/catalysis are not evaluated. The current filters
must not be presented as full biological validation or the full PoseBusters suite.

## Reproduce

Use the isolated environments in `configs/`. FLOWR/controller/MD code uses
`envs/flowr`; the PocketGen service uses `envs/pocketgen`. Do not combine their
PyTorch environments. Vendor commit IDs, local patches and downloaded checkpoint
records are under `reports/` and `downloads/`. Vendor licenses remain authoritative
for vendor code. Original project distribution licensing should be decided before
public release.

From the DH307 directory:

```bash
export DH307_ROOT="$PWD"
export PYTHONPATH="$PWD:$PWD/vendor/flowr"
envs/flowr/bin/python -m research.stage_data --stage export \
  --config experiments/full_v1/config.json --output data/full_v1
# On an internet-enabled host; downloads only public RCSB files:
envs/flowr/bin/python -m research.download_structures \
  --plan data/full_v1/downloads.json --cache data/full_v1/rcsb
envs/flowr/bin/python -m research.stage_data --stage finalize \
  --config experiments/full_v1/config.json --output data/full_v1
OMP_NUM_THREADS=1 envs/flowr/bin/python tests/test_research.py
# Freezes the package, vendor code/configs, and a source archive:
envs/flowr/bin/python -m research.submit --freeze-only
# With a completed manifest, submits six GPU runs and the dependent report:
envs/flowr/bin/python -m research.submit
```

On this cluster, heavy preparation runs on CPU Slurm nodes. External downloads
run on the login host because compute nodes lack external DNS. The submission
ledger is `experiments/full_v1/jobs.json`. Training imports the frozen source from
`experiments/full_v1/snapshot/`. The portable review bundle is
`experiments/full_v1/submission_source.tar.gz`; weights and datasets are separate.

Slurm allows 4 running GPU jobs and 8 submitted GPU jobs for this account. The full
campaign submits 6 GPU jobs and 1 dependent CPU reporting job; data finalization may
occupy the eighth active slot. Pending failed-dependency pilots must be removed
before filling these slots. Do not rerun the old pilot submission script.

## Resume, audit and outputs

Checkpoints include both actors/critics, optimizer state, pending PPO transitions,
episode index and random states. Each completed episode checkpoints atomically.
Resume with the same setting,seed,config,manifest and output directory; old raw
proposal directories are never overwritten. The submission script is idempotent.
An existing failed job is not silently duplicated: requeue that job, or explicitly
submit the same run with its original output directory after diagnosing the error.
At most the unfinished episode is replayed after interruption.

Every run has `status.json`,`checkpoint.pt`,`best_policy.pt`,`training.jsonl`,
`updates.jsonl`,`proposals/`,`oracle/`,worker logs and per-target evaluation JSON.
Unavailable actions, invalid molecules and exceptions are retained. A circuit
breaker stops training with no feasible nontrivial transitions rather than calling
failure penalties a useful optimization result. Evaluation errors are retained as
failed outcomes, not dropped from the denominator.

The report job produces `runs/full_v1/summary.json` and `per_target.csv`. It averages
policy seeds and systems within identical protein-sequence clusters before paired
bootstrap comparisons, avoiding treating generated molecules as independent test
proteins. No accuracy, novelty or significance claim is made until those outputs
exist and have been audited.

## Reading and attribution

- FLOWR: https://arxiv.org/abs/2504.10564 ; https://github.com/jule-c/flowr
- PocketGen: https://doi.org/10.1038/s42256-024-00920-9
- PocketFlow (conceptual motivation; not implemented here): https://arxiv.org/abs/2409.19520
- FlexSBDD (flexibility precedent): https://arxiv.org/abs/2409.19645
- PocketX (RL pocket-alignment precedent): https://aiproteindesign.org/papers/pocketx/

The neural-network checkpoint distributions and receptor ensembles are modeling
assumptions, not proof of optimal or experimentally binding recognition pairs.
