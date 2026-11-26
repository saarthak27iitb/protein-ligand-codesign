# DH 307 — revised presentation notes

Ten slides; approximately 6–8 minutes. Saved evidence checked 25 November 2026.
This deck is an exploratory progress report. No live scheduler state is asserted.
Dataset sizes and train/validation/test split counts are omitted from the slides.

## 1 — Title
“My work explores learned selection for protein–ligand design in two settings:
cooperative co-design and conformation-aware ligand search. We have implemented
and trained both settings, completed cooperative evaluation, and identified
specific limitations that motivate the next experiments.”

## 2 — Conditional directions and settings
“q_theta(L | P) is a distribution over ligands given a pocket. q_phi(P | L, S)
is a distribution over pocket designs given a ligand and the surrounding scaffold.
L denotes ligand chemistry and pose; P denotes the pocket; S is the rest of the
protein. These are many-to-many directions, not unique inverse functions.”

This is explanatory task notation, not a claim that the models have identical
variables or form one consistent joint distribution. The original PocketFlow task
can update ligand coordinates while holding its chemistry supplied. FLOWR provides
the ligand direction; PocketGen is the user-approved implementation substitute
for the pocket direction. PocketGen and PocketFlow are different architectures.

“In cooperation, both ligand chemistry and permitted pocket residues change, so
we study engineered pairs. In the conformation setting the protein sequence is
fixed and the ligand must cope with local structural variation.”

## 3 — Cooperative search
“Two selection policies share a pair-improvement reward. The ligand actor selects
a proposal, and the pocket actor evaluates its choices using features of the
selected ligand. Infeasible moves receive a penalty. Pocket changes preserve the
backbone and obey a four-mutation cumulative cap.”

The formula is the feasible-transition reward. c is a decision cost (0.02).
Training allows feasible intermediate worsening moves; evaluation retains the
best feasible pair found. Pocket proposals are generated before selection, using
the current ligand; the pocket selector sees the newly chosen ligand, but the
pocket generator is not rerun for every chosen molecule. This is a coordination
limitation worth testing later.

## 4 — Competitive search
“The opponent chooses a difficult conformation from a prepared local restrained-MD
ensemble. The ligand gets utility; the opponent gets its negative. The sequence
is unchanged. A separate MD replica supplies held-out conformations.”

The opponent cannot invent arbitrary protein distortions. The ensemble represents
local flexibility, not comprehensive biological states or equilibrium populations.
During evaluation, candidate ranking is followed by exact training-ensemble
screening shared across methods; the adversary is not the final evaluation oracle.
Worst-case robustness is a modeling choice and may be inappropriate when
state-selective binding is the goal. No equilibrium-convergence claim is made.

## 5 — RL connection
“The frozen generators supply proposals. Small PPO policies select among them;
an evaluator returns rewards. We train actors and critics, not the molecular
generator weights.”

Observations combine ligand properties, amino-acid composition, contact-distance
histograms, utility, remaining budget and charge. Candidate features and current
state features enter a small action-scoring network. Eligibility masks remove
unavailable choices. This lightweight representation can miss detailed contact
chemistry. It is not a full geometric policy encoder.

## 6 — Utility definition
“U is a score we want to maximize. V is the Vina input-pose score, which is better
when more negative, hence the minus sign. E is the ligand's internal strain:
energy at the supplied geometry minus energy after MMFF94s minimization. N is the
number of amino-acid mutations relative to the starting protein. We penalize
strain and mutations using fixed heuristic weights.”

Example: V = −8, E = 20, N = 2 gives U = 8 − 0.2 − 0.1 = 7.7.
The terms are combined into a proxy utility, not a thermodynamically calibrated
binding free energy. Mutation count is not a protein-foldability metric.

Feasibility checks include chemistry, connectedness, ligand size, charge, QED,
reference similarity, MMFF convergence, strain, clashes and mutation budget.
A valid molecular graph is not necessarily geometrically feasible or a binder.

Full docking is a separate final assessment with the same Vina scoring family.
It is not independent experimental binding validation.

## 7 — Results and evaluation counts
“The cooperative evaluation is complete across three seeds. The learned policy
is numerically slightly ahead of random joint search, but the difference is
uncertain. Greedy search uses fewer evaluations, and ligand-only is constrained
by candidate eligibility. We have not established a reliable RL advantage.”

Mean utilities: learned 6.350, random joint 6.301, greedy alternating 6.227,
ligand-only 5.902. Mean pair evaluations per system: 29.3, 29.2, 14.1 and 5.7.

**Pair evaluations per system** means the number of requests to the pair evaluator
inside the optimization search, averaged over systems and seeds. Each request
checks/scores a selected protein–ligand pair. Requests include repeated pairs
served from cache and requests whose outcome is infeasible or an error. Thus
this is a search-query count, not the count of unique docking calculations,
valid molecules, wall-clock time, total compute or successful designs.

It excludes proposal-generation work, initial reference evaluation, and final
full docking. In competitive search, evaluating one ligand against several
conformations uses one request per ligand–conformation pair. Averages are
fractional because they aggregate integer request counts over runs.

The exploratory paired protein-cluster bootstrap averages policy seeds before
resampling protein-sequence clusters. The full-cap learned-minus-random interval
is approximately [−0.003, +0.126]; the pairwise query-matched interval also includes
zero. Raw table means differ slightly from cluster-weighted estimates. At the
common actual query count across all methods, learned utility is 5.965 and
greedy is 5.969; this comparison is limited because ligand-only often makes no
eligible move. Generation costs also differ across methods, so query matching
alone is not a full computational-cost comparison.

## 8 — Imbalance and eligibility
“We suspect an imbalance in the effective strength of the two sides. Pocket edits
are often available while eligible ligand changes are sparse. The search can
therefore progress largely through pocket changes, making it harder to learn
meaningful coordination. This does not yet establish that one pretrained model
is intrinsically better than the other.”

The generators solve different tasks and use different constraints, so raw
accuracy or rewards are not directly comparable. Measure eligible-proposal yield,
accepted changes, marginal utility contribution, and update/gradient statistics
for each side. Use ligand-only, pocket-only and joint ablations to determine
which side actually drives improvements. Feature quality and filtering can create
an apparent capability imbalance even when the generators are competent.

In competition, an opponent may obtain a strong penalty by selecting a state
where the ligand is infeasible. That can make the game uninformative; it is a
hypothesis to audit, not a proven diagnosis in the current results.

The audited competitive example: 64 generated, 62 valid and connected, zero
eligible. All 62 failed the reference Tanimoto cutoff (maximum 0.19753, required
0.25) despite passing size, charge and QED checks. This is one diagnosed case,
not proof that every failure has that cause. Most recorded evaluation errors say
no eligible candidates; a separate all-masked case says no legal actions.
Competitive evaluation remains incomplete, so successes alone cannot estimate
comparative performance.

## 9 — Distillation and balancing: proposed work
“First align proposal generation with the intended task. Then use a stronger
search procedure as a teacher to initialize the selection policies, and refine
those policies with PPO. At the same time we will diagnose and balance the two
sides, then compare against simple search under matched budgets.”

The policies are already trained with RL. The precise proposal is **distillation
or imitation into the policy networks, followed by RL fine-tuning**. It does not
mean we have already distilled either molecular generator.

A possible teacher scores feasible candidates or joint actions more extensively.
A soft target over candidates could be pi_T(a | s) proportional to exp(U_T(s,a)/T),
with ineligible actions masked. T is temperature and U_T is the teacher's
look-ahead or ensemble-screened value. Train the selector with cross-entropy or
KL(pi_T || pi_theta), then continue PPO. For cooperation, derive consistent ligand
marginal and pocket-conditional targets from joint teacher scores. For competition,
teacher values can use worst-case utility over plausible training conformations.

Collect teacher examples only from training systems. Use validation for choices;
retain untouched evaluation systems. Teacher search costs must be reported. If
we distill a poor teacher, the student can inherit its errors, so teacher quality
must be verified first. Distillation cannot create eligible molecules that the
proposal generators never produce.

To address imbalance, first measure it. Then test more updates for the weaker
selector, temporarily hold the stronger selector fixed, or supply a curriculum
with feasible states before harder states. If the weakness is proposal coverage,
change conditioning or sampling rather than only increasing policy updates.
These interventions are experiments, not demonstrated fixes. Blindly rescaling
only one player's competitive reward can also alter the intended game.

Compare imitation alone, PPO alone, and imitation followed by PPO using the same
candidate support and comparable budgets. Also retain random and greedy baselines.

## 10 — Thank you
Invite discussion about the task choice, the imbalance hypothesis, teacher quality,
and what evidence would justify the RL layer.

## Sources and provenance

- FLOWR: https://arxiv.org/abs/2504.10564
- PocketFlow: https://arxiv.org/abs/2409.19520
- PocketGen: https://doi.org/10.1038/s42256-024-00920-9
- Implementation: ../research/ and ../experiments/full_v1/snapshot/
- Analysis: ../reports/full_v1/analyze_three_seeds.py
- Evidence: data/completed_cooperative_summary.json and data/status_snapshot.json

No research code, policies, run outputs or jobs were changed for this revision.
