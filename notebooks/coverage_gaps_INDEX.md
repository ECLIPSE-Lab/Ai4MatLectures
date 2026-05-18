# Triad exercise coverage audit — consolidated index (Weeks 6–14)

**Audited 2026-05-17.** Each braided weekly exercise (`weekN_*.py` main + `weekN_homework.py`)
was checked against its three triad lectures (MFML + ML-PC + MG) after the mid-semester
restructure. Per-week detail with prioritized to-dos:
[w6](week6_coverage_gaps_TODO.md) ·
[w7](week7_coverage_gaps_TODO.md) ·
[w8](week8_coverage_gaps_TODO.md) ·
[w9](week9_coverage_gaps_TODO.md) ·
[w10](week10_coverage_gaps_TODO.md) ·
[w11](week11_coverage_gaps_TODO.md) ·
[w12](week12_coverage_gaps_TODO.md) ·
[w13](week13_coverage_gaps_TODO.md) ·
[w14](week14_coverage_gaps_TODO.md)

Priority key: **[P1]** core lecture topic absent/cosmetic · **[P2]** core but partial · **[P3]** support.

## Verdict matrix

| Wk | Exercise | MFML | ML-PC | MG | P1 | Dominant issue |
|----|----------|------|-------|-----|----|----------------|
| 6  | optimization_and_finetuning | STRONG | STRONG | WEAK | ~6 | MG = toy GNN only; no PBC/RBF/ranking metrics |
| 7  | generalization_and_ensembles | STRONG | PARTIAL | PARTIAL | 7 | W7 lectures **cancelled** (Pfingstdienstag); exercise = W8 preview, models treated as black boxes |
| 8  | uncertainty_and_robustness | STRONG | PARTIAL→STRONG | PARTIAL | 6 | Whole CV/HPO + robustness half absent (Fronleichnam live slot cancelled); MG deck title promises MLIPs the body lacks |
| 9  | latent_geometry | STRONG | STRONG | WEAK | 6 | Notebook braids the **wrong MG lecture** (MG U9/W10 content, not true-W9 regression-generalization) |
| 10 | attention_and_transfer | STRONG | STRONG | WEAK | 2 | Notebook braids **next week's** MG lecture; ML-PC = 09b-transformers (defensible, not W10-automation) |
| 11 | generative_inverse_design | STRONG | PARTIAL/WEAK | WEAK | 7 | ML-PC braids calendar-W9 inverse-problems not true-W11 automation; MG wrong content-half |
| 12 | uncertainty_and_discovery | STRONG | STRONG | SPLIT/severe | 8 | Notebook declares a **deleted** MG lecture; true-W12 MG generative has **zero** coverage |
| 13 | physics_and_gp | STRONG | PARTIAL | WEAK | 6 | Exercise teaches GP math, not MG discovery loop (E-hull/acquisition); MFML Lagaris hard-BC never coded |
| 14 | explainability_and_limits | STRONG | PARTIAL→STRONG | WEAK | 5 | MG constraint-enforcement / conformal / system-OOD unexercised |

## Systemic findings

1. **MFML axis is solid every week.** Optimizer / probabilistic / latent / attention /
   generative / uncertainty / physics / XAI *mechanics* are genuinely exercised. MFML gaps
   are minor support-tier items, with two notable [P1] exceptions: W13 Lagaris hard-BC
   (deck's "guaranteed exam derivation") is never implemented; W11 diffusion mechanics
   (forward marginal, ε-prediction loop, classifier-free guidance) are narrated not coded.

2. **MG is structurally under-integrated and mis-synced.** The Ai4MatLectures notebooks
   were not re-synced after the 2026-05-13 MG realignment + the two cancelled sessions, so
   several braid the *adjacent* or a *deleted* MG lecture rather than the calendar one
   (W9, W10, W11, W12; W12 worst). Where the slug is correct (W13, W14) the exercise still
   sits at the wrong abstraction level (GP math instead of the discovery loop;
   diagnose-only symmetry audit instead of constraint enforcement).

3. **ML-PC's data-discipline half is the recurring content gap.** Leakage-safe CV /
   group-K-fold, augmentation-as-invariance, three-set protocol, robustness/OOD and
   process-window material are repeatedly absent or prose-only.

4. **ML-PC deck filenames are non-canonical** in later units (`12_uncertainty_gp.qmd`,
   `13_pinns.qmd`, `14_reflection.qmd`, `09_inverse_problems.qmd`, …) — handled in the
   audit but worth normalizing to `01_intro.qmd` separately.

## Cross-week action list (do in this order)

### A. Structural decisions — ✅ RESOLVED & APPLIED 2026-05-17 (except W7/W8)
- [x] **W12 generative gap.** Decision: *add* generative coverage to the existing
      `week12_uncertainty_and_discovery` (no re-date/file move). DONE — new Block 6.5
      (conditional CVAE + classifier-free guidance + inverse-design/S.U.N. discovery
      funnel + uncertainty triage), header corrected from the deleted-lecture
      mis-declaration to MFML U12 + ML-PC U11 + MG U12. [w12]
- [x] **MG braid re-sync (W9–W12).** Each notebook's MG block rewritten to the true
      calendar-week MG lecture: W9→MG U8 (regression/generalization, split design +
      baseline ladder + residual diagnostics + reporting checklist); W10→MG U9 (SchNet/
      CGCNN property nets, permutation-invariance check, extensive/intensive readout,
      Magpie ceiling, foundation-model recipe); W11→MG U10 (embedding diagnostics:
      linear-probe vs random-init vs Magpie, retrieval, "pretty t-SNE/dead downstream");
      W12→MG U12 (above). All run end-to-end. [w9,w10,w11,w12]
- [x] **W11 ML-PC rebraid.** Added Block 5b self-driving-lab loop (acquire→model→decide,
      active-learning acquisition beats random, conformal automate-vs-escalate) braiding
      true-W11 ML-PC U10 Automation. [w11]
- [x] **W7/W8 cancelled-session intent.** Lecturer-confirmed 2026-05-17 *as designed*:
      W7 exercise = W8 preview; Fronleichnam-cancelled W8 live slot owned by self-study.
      Recorded in the w7/w8 TODO headers. Not a gap. [w7,w8]

> **Follow-up flagged:** `week10_homework.py` still carries the old MG mislabel
> ("Representation learning…") — only the W10 *main* notebook was re-synced. Homework
> notebooks for W9–W12 were out of this pass's scope.

### B. Highest-leverage single additions (one block fixes multiple [P1]) — ✅ DONE 2026-05-17
- [x] **W14:** confounder block (Block 2b, `furnace_id` back-door) on the tensile MLP —
      closes the top [P1] for *both* MFML §9 and ML-PC process-chain causality
      (SHAP vs permutation vs `do()` intervention). [w14]
- [x] **W6:** Block 6b — PBC minimum-image crystal graphs + Gaussian RBF & smooth-cutoff
      (hard-cutoff artifact demo) + ranking metrics (ρ/τ/top-k) + sum-vs-mean readout;
      converts MG from WEAK to covered. [w6]
- [x] **W13:** Block 1b Lagaris hard-BC (exact BC, no λ_BC) + Block 5b discovery loop
      (E-hull objective + EI/UCB/Thompson acquisition). [w13]
- [x] **W8:** Block 1b leakage-safe K-fold/group-K-fold + three-set + KS shift diagnosis;
      Block 6b noise-injection + MSE-vs-Huber outlier robustness. [w8]

> All four implemented additively (no existing cells/decks altered), `py_compile` clean,
> paired `.ipynb` regenerated, each notebook executed end-to-end. Per-week TODO files
> retain the remaining [P2]/[P3] items.

### C. Recurring per-course content to thread wherever the week allows
- [ ] ML-PC: leakage-safe CV / group-K-fold, augmentation-as-invariance, OOD/robustness.
- [ ] MG: invariance/equivariance tests, PBC/cutoff audits, ranking/discovery metrics,
      constraint-enforcement (soft vs hard vs architectural).
- [ ] MFML: down-weighted — only W11 diffusion-loop and W13 Lagaris are [P1].

---

## New MG datasets added to `ai4mat` (2026-05-18)

Four static-download (no API key) `torch` datasets added under `ai4mat/datasets/`,
exported from `ai4mat.datasets`, each with a contract test. They unlock MG-side
content for the exercises (notebook integration is a follow-up, not yet done):

| Class | `ai4mat.datasets` | Unlocks (MG units / weeks) | Notes |
|---|---|---|---|
| `MatBenchDataset` | `task=` selector (jdft2d…mp_gap) | U7–U9 property prediction; U8 generalization/splits | composition/structure → property; surrogate 5-fold CV (official folds need the `matbench` pkg) |
| `RMD17Dataset` | `molecule=` (10), `n_samples≤1000` | U4–U6 MLIP/dynamics (was dataset-less) | energy+forces; 67–175 MB/molecule one-time cache; >1000-sample warning |
| `CDVAEMaterialsDataset` | `subset=perov_5\|carbon_24\|mp_20` | U12 generative + U13 discovery | CIF + formation energy/e_above_hull; official splits |
| `QM9Dataset` | `target=` (19 props) | U9 molecular property prediction | 134k molecules, CC0; crude SMILES feature baseline |

Real-download verified for rMD17/CDVAE/QM9; MatBench parse-path verified against a
real cached file (its figshare mirror 403'd inside the sandbox only). Core paths need
no heavy deps (pymatgen/ase/rdkit optional); raw CIF/SMILES kept accessible per item.

## Exercise integration status (2026-05-18, plan executed)

Per `docs/superpowers/plans/2026-05-18-mg-dataset-exercise-integration.md`
(subagent-driven, two-stage reviewed, committed on `main`):

| Wk | Change | Dataset | Commit |
|----|--------|---------|--------|
| W9 | MG-U8 ladder+splits → real data; GNN tier kept as labelled synthetic appendix | `MatBenchDataset` | `2e0d619`,`5018e76` |
| W12 | added real-crystal discovery-funnel stage; 2-D CVAE toy core preserved | `CDVAEMaterialsDataset` | `c0fe1d3` |
| W8 | added rMD17 MLIP energy+force self-study block (correlated-sample lesson) | `RMD17Dataset` | `4429563` |
| W10 | added real composition-ceiling block + QM9 contrast; GNN lesson preserved | `MatBenchDataset`,`QM9Dataset` | `11dfa46` |
| W13 | discovery loop on real mp_20 hull pool | `CDVAEMaterialsDataset` | `2829945` (done by P. Pelz directly, not this loop) |

Cross-cutting: `d7e882a` fixed the MatBench downloader (browser User-Agent —
materialsproject.org 403s the default urllib UA, would have broken it for
students). `.gitignore` extended so the new dataset caches
(`data/{matbench,rmd17,cdvae,qm9}/`) are not committed (mirrors the existing
`data/NEU-DET/`,`data/estm/` convention). Fast test suite: 67 passed, 9
slow-download tests deselected. Strategy: swap data only in data-agnostic
blocks; ADD real-data blocks where a rewrite would break the graph/CVAE lesson.

---
*Audit changes: the `weekN_coverage_gaps_TODO.md` files and this index were
created/edited. Dataset work added `ai4mat/datasets/{matbench,rmd17,cdvae_materials,qm9}.py`
+ tests + the `__init__.py` exports (committed `ac306f3`). Exercise-integration
commits are listed above. Unrelated in-progress working changes (week6 files,
data/week3_mystery.npz) were left untouched.*
