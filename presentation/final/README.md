# Protein–Ligand Design: Cooperative and Conformation-Aware Search

Saarthak Krishan, 22B3959 · Supervisor: Prof. Debabrata Maiti

A separate 22-slide Beamer deck for approximately 20–25 minutes, including title,
thank-you and one backup reference slide. The concise progress deck is unchanged.

The slides explain the actual implementation: frozen molecular generators,
variable-action selection, 40-channel observations, separate MLP actors/critics,
cooperative and competitive rewards, episodic returns, policy gradients, PPO
ratios/clipping, value/entropy losses, update schedule and evaluation differences.
A clipping example uses illustrative values. Completed cooperative results and
proposed distillation/balancing are clearly separated.

- `DH307_RL_PPO_explained.pdf`: compiled presentation.
- `slides.tex`: editable LaTeX.
- `SPEAKER_NOTES.md`: narration, implementation details and anticipated questions.
- `build_slides.sh`: build using Tectonic, latexmk or pdflatex.

Run `bash build_slides.sh`. Standard Beamer/Latin Modern packages are required;
no external figures are needed. Evidence reflects saved artifacts inspected on
25 November 2026; no live scheduler state is asserted. Dataset sizes and split
counts are deliberately omitted. Training hyperparameters are included.

Code audited: `DH307/research/{models,features,run,evaluator}.py` and
`DH307/experiments/full_v1/config.json`. Three-seed results are documented in
`DH307/reports/full_v1/cooperative_completed_seeds_42_123_2026.json`.
No training implementation, checkpoint, run artifact or scheduler job was changed.
