---
name: Ai4MatLectures project
description: New public repo ECLIPSE-Lab/Ai4MatLectures — PyTorch dataset wrappers + teaching notebooks for MFML, MLPC, MG lecture triad
type: project
---

Public repo `ECLIPSE-Lab/Ai4MatLectures` at `/home/philipp/Insync/braunphil@gmail.com/gdrive/erlangen/lecture/SS26/Ai4MatLectures/`.

**Why:** Sandfeld's mdsdata package has numpy/pandas API only. All three lectures (MFML, MLPC, MG) need PyTorch-first notebooks from week 1 onward.

**How to apply:** When working on lecture notebooks or slide decks, reference this repo for dataset imports (`from ai4mat.datasets import IsingDataset` etc.). When adding notebook callouts to _public_presentations slides, link to eclipse-lab.github.io/Ai4MatLectures.

**Key facts:**
- Package name: `ai4mat`, 7 Dataset classes wrapping mdsdata
- ~20 notebooks across MFML (6), MLPC (5), MG (4) + MFML week03 fully written
- Quarto website on GitHub Pages (GitHub Actions deploy)
- Colab badge on every notebook (points to committed .ipynb in repo)
- No build coupling to _public_presentations — slide decks link out
- GitHub repo is public: `https://github.com/ECLIPSE-Lab/Ai4MatLectures`
- Default branch is `main` so Colab links and Pages workflow are aligned

Status:
- Claude co-author trailers were removed from published commit history before first push.
- Repository was published publicly under `ECLIPSE-Lab`.
- Default branch was migrated from `master` to `main` after publish so GitHub Pages and notebook links work.

Spec: `../docs/superpowers/specs/2026-03-23-ai4matlectures-pytorch-design.md`
Plan: `../docs/superpowers/plans/2026-03-23-ai4matlectures.md`
