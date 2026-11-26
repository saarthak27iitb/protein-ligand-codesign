# DH307 protein–ligand research campaign

## Scope and status

Initial setup, not completed research implementations. No improvement claim is assumed.
The official PocketFlow repository has no released implementation/checkpoint as inspected
on 2026-11-04. The user approved PocketGen as the initial substitute. FLOWR and
PocketGen are separate conditional models; a unified bidirectional model is an objective,
not an existing capability. Ligand-to-pocket requires a scaffold; protein-to-ligand
requires a binding-site specification. This is not ligand-to-full-protein generation.

Verified GPU partition limits: 4 running jobs/user, 8 submitted jobs/user.
Use at most 8 slots and throttle arrays to 4. Do not submit placeholder research jobs.

## Eight intended experiment slots

0. Independent FLOWR ligand-generation baseline.
1. Independent PocketGen pocket-design baseline.
2. Cooperative alternating pair optimization with both conditional models.
3. Fixed-sequence pocket-conformation adversary and ligand designer.
4. Joint conditional compatibility training; compare against both independent and
   alternating baselines. Do not enforce exact sequence/ligand reconstruction for
   this many-to-many problem.
5. RL fragment-selection controller using frozen FLOWR proposals.
6. Evaluation-budget allocation controller (compare bandit and fixed funnels).
7. Mutation-budget pocket-edit controller using PocketGen proposals.

These slots describe research arms, not eight currently runnable programs.
First establish inference, shared data preparation, structural validity, and scoring.
Smoke checks use validation data only and are not evidence of scientific performance.

## Evaluation contract

- Keep generation, scoring, and final evaluation budgets explicit and matched.
- Retain provenance, model commits, checkpoints, seeds, raw outputs, failures and timing.
- Split by protein family and ligand scaffold where feasible; audit pretrained-model
  overlap before calling any test target unseen.
- Use training/validation for controller fitting and decisions; freeze held-out tests.
- Report molecular validity, clashes, strain, diversity and interaction preservation.
- Evaluate protein edits in the scaffold context; sequence confidence is not stability.
- Use a held-out scoring method; docking reward is not measured binding affinity.
- Conformation adversaries must preserve sequence and obey physically motivated
  constraints. Binding every state is not necessarily a desired biological behavior.
- Compare random, greedy and search baselines before attributing value to RL.
- Report runtime including setup/training; a pilot on one seed is not a benchmark.

## Repositories

- https://github.com/jule-c/flowr
- https://github.com/zaixizhang/PocketFlow (unreleased implementation)
- https://github.com/zaixizhang/PocketGen (approved substitute)

Environment and data setup are isolated under DH307; existing research environments
are not modified. Environment deviations from upstream: omit notebooks and commercial
PyMOL; use open-source PyMOL. Pin core numerical and model libraries to upstream versions.
