# Speaker notes — Protein–Ligand Design

Suggested duration: 20–25 minutes. Slides 1–20 explain the implementation;
slide21 is thank-you; slide22 is the backup reference/code map. Findings remain
exploratory. No dataset/split sizes appear in the slides.

## 1 — Title
“This deck explains what our decision policies actually learn and how PPO trains
them. It concerns selection over pretrained proposals, rather than training a
new molecular generator.”

## 2 — Generation versus selection
q is a conditional generative distribution; pi is an action-selection policy.
FLOWR supplies the ligand direction. PocketGen is the approved pocket-generation
substitute for the original PocketFlow motivation, not a PocketFlow implementation.
The generators' weights are frozen. S is the surrounding scaffold, P the pocket,
and L ligand chemistry plus pose. These directions are many-to-many and need not
be conditionals of one consistent learned joint distribution.

## 3 — One decision step
The wrapper queries models using a configured proposal budget. RL neither chooses
this budget nor directly queries a model as an action. It chooses candidate indices.
The selected pair is evaluated afterward. Candidate observations do not contain
its as-yet-uncomputed Vina score; the utility feature is the current state's utility.
Past outcomes train the selectors to recognize useful candidate features.

Eight fresh ligand and three pocket proposals are requested per cooperative
training step, before validity/projection can reduce the available sets. Existing
ligand/pocket options are also retained. Pocket proposals are generated using the
current ligand before selection. The pocket selector sees the chosen ligand,
but PocketGen is not rerun separately after every ligand choice.

## 4 — Observation
The 40 channels are nine ligand properties, twenty amino-acid fractions, eight
nearest-protein-atom distance bins, scaled utility, remaining budget and charge.
The nine ligand descriptors are heavy-atom count, molecular weight, logP, TPSA,
rotatable bonds, hydrogen-bond donors, hydrogen-bond acceptors, ring count and QED.
Composition is computed over the protein file supplied to the encoder, not a
learned residue-specific map. A histogram discards contact direction and atom-pair
identity. Distinct complexes can therefore share similar observations. Call this
an observation, not a proven fully Markov state.

## 5 — Actions and masks
Each candidate gets a 40-channel encoding. The actor processes current-state and
candidate vectors together. A masked softmax defines a categorical distribution.
Training samples an index. Invalid actions receive an effectively zero probability
(the implementation uses a −1e9 masked logit). Training forces the retained ligand
option into the ligand mask; evaluation does not consistently use the same fallback.
Chemical masking is not the same as full feasibility: detailed strain and clash
checks occur in the evaluator. An all-masked evaluation case raises “No legal
actions.” We propose handling that explicitly, not silently ignoring the failure.

## 6 — Architecture
There are two instances of Agent, each with an actor and critic. Actor:
80→128→64→1, Tanh after hidden layers. Critic: 40→128→64→1, same activations.
The actor network is reused across candidates within that player; parameters are
not shared across players or between actor and critic. They use one Adam optimizer.
A separate critic estimates each player's return from its state summary. This is
not a joint-action critic, recurrent policy, transformer policy or raw-atom GNN.
The critic symbol V_psi is distinct from Vina score V_pose.

## 7 — Cooperative reward
Factorization is pi_L(a_L | o) pi_P(a_P | o,a_L). Each actor sees candidate features;
the second actor's encoding incorporates the selected ligand. Both receive delta
utility minus0.02 for a feasible move; infeasible reward is−2. Feasible worsening
moves update state during training, which allows non-greedy trajectories. At most
four residue identities may differ cumulatively from the reference. Backbone and
unselected atoms are preserved. If no pocket edit proposal exists, the retained
pocket is used and no fictitious pocket-policy training record is created.
Cooperative sequence changes make this an engineered-pair task, not optimization
against an unchanged natural target.

## 8 — Competitive reward
The conformation opponent chooses among precomputed plausible fixed-sequence states.
It does not move atoms or simulate MD as its action. Rewards are U and−U for a
feasible pair; an infeasible pair gives−20 and+20. This creates a possible shortcut
where the opponent wins on infeasibility rather than informative binding variation.
That is an audit hypothesis, not a demonstrated diagnosis of every run.
Training contexts for ligand generation cycle through available conformations;
the opponent then selects its challenge after observing the ligand. The protein
reference used in the ligand observation remains fixed, while partner action
features describe each candidate state. The ensemble is local restrained MD,
not an equilibrium biological state distribution. Do not claim Nash convergence.

## 9 — Utility
V_pose is Vina's score at the supplied ligand pose. It is not full redocking at
every decision. E is the MMFF94s energy reduction of a hydrogenated ligand copy
after local minimization (up to500iterations, convergence required). Minimization
is ligand-only; the proposed geometry used for Vina is not replaced by that copy.
N counts residue identities different from the starting protein, not edit steps.
N=0 in fixed-sequence competition. The coefficients are heuristic.
Example: V_pose=−8, E=30 and N=3 gives U=7.55. This is not a measured binding
free energy. Feasibility additionally checks connectedness, size, charge, QED,
reference similarity, MMFF convergence/strain, clashes and mutation budget.

## 10 — Episode
Reset to a sampled training reference; run up to four joint decision steps;
store experience; reset the molecular design at the next episode. The policy
weights persist. A system may recur in multiple episodes. A proposal failure can
terminate early. The saved no-valid-generated-molecules trace records−2, but the
code breaks before appending a decision record; do not imply this is necessarily
an extra PPO transition. One episode is not one pass over the dataset, one new
protein, or one flow trajectory. FLOWR sampling itself uses100flow steps, separate
from the RL horizon.

## 11 — Returns and advantages
For each player separately, compute returns backward, with gamma=1 and terminal
return0. There is no GAE or value bootstrapping. Raw advantage is G minus the old
critic estimate. For rewards [0.2,−0.1,0.4,0.1], returns are [0.6,0.4,0.5,0.1].
If the first critic predicted0.3, the first raw advantage is0.3. Before optimization,
the code normalizes advantages using the combined buffer from both players.
Normalization can shift the zero point, so the raw-advantage intuition is a useful
starting explanation, not a guarantee about every normalized sample's sign.
A potential ablation is player-wise normalization, particularly under imbalance.

## 12 — Policy gradient
Use the log-derivative identity to move from the gradient of expected return to
an expectation containing grad log pi times return. An action-independent value
baseline reduces variance. In our sampled update, advantages are fixed targets.
The scalar reward need not be differentiable: Vina supplies feedback, and the
policy gradient passes through log probabilities, not through molecular coordinates
or the generators. This cannot optimize candidates outside the proposal support.
The two-agent environment changes as both policies learn, so ordinary single-agent
convergence arguments do not automatically apply.

## 13 — PPO ratio and clipping
Save the old log probability during rollout. Recompute the new log probability
from the same observation, candidates and mask during optimization. The ratio is
exp(new_logp−old_logp). The clipped surrogate limits incentive for favorable-direction
probability changes beyond the clip range. It does not force every ratio or every
probability to stay in a hard interval. Approximate KL is logged; it is not an
explicit KL penalty or an early-stop rule in the current implementation.

## 14 — Worked clipping example
For positive advantage+1, old probability0.2→new0.3 gives ratio1.5. The two terms
are1.5 and1.2; the minimum is1.2. Further increasing the ratio gives no extra
surrogate benefit for that sample. For negative advantage−1, old0.2→new0.1 gives
ratio0.5. Terms are−0.5 and−0.8; minimum−0.8 removes further incentive in that
favored direction. If a change moves in the harmful direction, the objective
continues penalizing it. Other samples, entropy and value losses can still move
parameters. These probabilities are illustrative, not measured from our runs.

## 15 — Total loss
Minimize negative clipped surrogate plus0.5times critic squared error minus0.01
entropy. Critic regression targets are the player's returns. Entropy encourages
a broader action distribution; it does not ensure chemical diversity among similar
proposals. The implementation forms one mean loss over buffered decision records
from both players, performs one backward pass per PPO epoch, clips the global
gradient norm to1, and steps the shared optimizer. Missing partner records are
skipped. Critics do not backpropagate through the returns or old predictions.

## 16 — Schedule
512episodes/seed, at most4steps/episode, update after8episodes,4PPOepochs per update,
Adam3e−4. There are64scheduled update cycles; some rollouts can be shorter. Each
PPO epoch uses the full buffer, not shuffled minibatches. The update buffer is
cleared afterward, so this is not off-policy replay. A final update call is made
but is empty after an exactly divisible schedule. Policies are selected using
validation performance. Molecular models are not trained for512epochs.

## 17 — Evaluation differs from training
Cooperation ranks pairs by the sum of ligand and conditional pocket log probabilities,
then evaluates leading pairs within the budget. All methods retain the best feasible
pair. Random joint search changes pair ordering; greedy alternating restricts
which component can change; ligand-only keeps the pocket fixed.
Competition ranks candidate ligands, explicitly screens them across training states,
and chooses by worst-case training utility. Held-out conformations assess the
selected ligand afterward. The learned opponent is not used as the final scoring
oracle. Rigid-context generation also receives the shared robust final screen.
This distinction means a useful training opponent may or may not improve final
performance; measure its contribution against exact or random state selection.

## 18 — Results
Three completed cooperative seeds give mean utilities6.350learned,6.301random,
6.227greedy and5.902ligand-only. These are the same target systems across seeds.
The exploratory paired protein-cluster uncertainty interval includes zero for
learned minus random, and query-matched comparisons do not establish reliable
learning superiority. Candidate eligibility limits ligand-only, and computational
usage differs across methods. Utility is a docking/strain/mutation proxy.
Competitive evaluation is incomplete; reporting only its successful subset would
bias interpretation. The deck does not claim improved affinity or foldability.

## 19 — Bottlenecks
The audited competitive case had64generated,62validconnected,0eligible because
all62failed the reference-similarity cutoff. That is rejection before conformation
scoring, not evidence of poor binding. Pocket-versus-ligand action availability
suggests imbalance, but does not prove intrinsic superiority of either pretrained
model. Separate proposal support, observation quality, reward scaling, credit
assignment and nonstationarity when diagnosing learning. Longer training alone
cannot repair absent eligible actions.

## 20 — Proposed distillation
A stronger search teacher can supply candidate distributions or preferred choices
on training systems. Supervise the same selection networks with cross-entropy/KL,
then fine-tune with PPO. For example teacher probabilities may be proportional to
exp(teacher_value/temperature), with feasibility masks. A cooperative teacher can
score joint choices; a competitive teacher can use explicit ensemble worst-case
values. The teacher must be better under the chosen objective; imitation can
inherit its errors. Include teacher cost in comparisons. Do not use evaluation
systems for teacher training. Distillation does not create missing proposal support.
Balance only after diagnosing whether the weaker side lacks good proposals or
merely needs better selection. Potential tests include extra selector updates or
a feasible-to-difficult curriculum, without silently changing the intended game.

## 21 — Thank you
Invite questions about what is learned, why PPO is justified, and which controlled
experiment would establish an advantage over search.

## 22 — Backup references and code
PPO: https://arxiv.org/abs/1707.06347
FLOWR: https://arxiv.org/abs/2504.10564
PocketGen: https://doi.org/10.1038/s42256-024-00920-9
PocketFlow motivation: https://arxiv.org/abs/2409.19520

The implementation is authoritative for what ran. Source mapping is on the slide.
This is a technical explanation of our code, not a claim that every standard PPO
option (GAE, minibatches, KL stopping, recurrent state) is implemented.
