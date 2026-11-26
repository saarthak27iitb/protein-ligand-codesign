# Publishing this prepared source export

Intended public destination: `saarthak27iitb/protein-ligand-codesign`.
This directory contains the selected source export, not the entire server workspace.

With an authenticated GitHub CLI available:

```bash
git init -b main
git add .
git status --short
git commit -m "Share exploratory protein-ligand co-design research"
gh repo create saarthak27iitb/protein-ligand-codesign --public --source=. --remote=origin --push
```

If Git was already initialized and committed, skip those steps. Set your intended
commit identity if Git requests it. The CLI requires your GitHub authentication;
do not put tokens into tracked files or remote URLs.

Alternatively, create an empty public repository on GitHub's website (no generated
README), then use its HTTPS/SSH remote:

```bash
git remote add origin https://github.com/saarthak27iitb/protein-ligand-codesign.git
git push -u origin main
```

If origin already exists, inspect it instead of adding another. Do not force-push
over an existing repository. The export does not contain credentials or a token.
The project-authored code currently has no selected reuse license; attribution and
external-material notes are included in `THIRD_PARTY_NOTICES.md`.
