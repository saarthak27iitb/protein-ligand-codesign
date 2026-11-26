# DH 307 — exploratory progress presentation

**Protein–Ligand Design: Cooperative and Conformation-Aware Search**
Saarthak Krishan · 22B3959 · Supervisor: Prof. Debabrata Maiti

Ten slides, including title and thank-you, for a 6–8 minute presentation.
The deck covers the two settings, RL's role in each, work completed, exploratory
cooperative results, current bottlenecks and proposed next steps. It uses a white
background with navy and teal accents, minimal mathematics and no source footers.

- `DH307_progress_update.pdf`: ready to present.
- `slides.tex`: editable Beamer source.
- `SPEAKER_NOTES.md`: narration, result caveats and likely questions.
- `DH307_progress_update_source.zip`: source, notes, build script and aggregate evidence.
- `data/`: three-seed cooperative summary and saved-artifact status snapshot.

Evidence checked 25 November 2026. All three cooperative seeds have completed;
competitive evaluation remains partial. This is an exploratory progress report,
not a final performance claim. No live scheduler status is asserted.

Rebuild with `bash build_slides.sh` or compile `slides.tex` using Tectonic,
latexmk or pdflatex. No external images are needed. The ZIP contains everything
required for the deck, except a LaTeX installation. Compatibility filenames
`DH307_research_meeting.pdf` and `DH307_beamer_source.zip` contain the same update.
The previous deck is preserved in `archive/progress_2026_11_14/`.

Research jobs, code, policies and run outputs were not modified.

This revision adds conditional-generation notation and the utility definition,
removes dataset/split counts from the slides, clarifies pair-evaluation requests,
and proposes teacher distillation followed by PPO and tests of side imbalance.
Imbalance is a working hypothesis; distillation has not been implemented.

Layout refinement: larger body text, roomier table rows, and compact process
diagrams on both method slides; the deck remains ten slides.
