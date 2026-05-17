# Week 7 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 7 (lecture 26.05.2026 — public holiday, see caveats)** ·
braided exercise = `week7_generalization_and_ensembles.py` (main) + `week7_homework.py`.

> **Lecturer-confirmed 2026-05-17 (as designed):** the W7 exercise is officially the
> Week-8 *preview* — the 26.05 W7 lecture slot is cancelled (Pfingstdienstag) and the
> Fronleichnam-cancelled W8 live content is owned by self-study. The cancelled-session
> structure is intended, not a gap; the [P*] items below are still valid content gaps
> against what the exercise *does* braid.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-7 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/08_tree_ensembles_tabular/01_intro.qmd` (YAML title "Unit 8: Tree Ensembles for Tabular Learning"; delivered as the MFML W7 pairing, one-slide bias–variance/regularization recap then trees) | Generalization/bias–variance recap + **tree ensembles** (RF, GBM, XGBoost, CatBoost, TabPFN) |
| ML-PC | `ml_for_characterization_and_processing/unit07_generalization_robustness/01_intro.qmd` (YAML "Unit 7: Generalization, Robustness, and Process Windows") | **W7 (26.05) CANCELLED — Pfingstdienstag.** Per AGENT_INSTRUCTIONS this unit is *delivered W8 (02.06)* — a one-week calendar slip. The exercise bridges into it (TensileTestDataset across $T$). |
| MG | `materials_genomics/06_local_atomic_envs/01_intro.qmd` (YAML "Unit 6: Local Atomic Environments & Universal MLIPs") | **W7 (26.05) CANCELLED — public holiday.** Exercise previews MG Week 8 (02.06): SOAP / local atomic environments + grouped CV. Slug authoritative; the REALIGNMENT_2026-05-13 table labels this folder "07" but the on-disk prev/next chain (05_monte_carlo → 06_local_atomic_envs → 07_graph_based_rep) and the YAML "Unit 6" are authoritative — numeric prefix is off by one in the realignment doc. |

**Caveats.** Two of the three triad lectures (ML-PC, MG) for the 26.05.2026 slot
do **not happen** — that Tuesday is a public holiday. The braided exercise is
therefore *not* synchronised with a same-week lecture for those two courses; it
is a **forward preview** of the ML-PC and MG Week-8 (02.06.2026) decks. Only the
MFML strand is a true same-week pairing. Coverage below is judged against what
the exercise *claims to braid* (its own header: bias-variance + tree ensembles;
generalization/robustness/group-CV; SOAP + grouped CV). The ML-PC deck's
robustness/process-window/HPO halves are out of the exercise's stated scope but
are flagged where a cheap add would strengthen the W8 preview. The MFML
homework Part E (TabPFN) and Part C feed the in-class blocks directly.

## Verdict

- **MFML (tree ensembles + generalization):** STRONG. Bias-variance-as-measurement
  (homework A, Block 3), RF-plateau vs GBM-dip (homework C), trees-vs-MLP
  inductive bias and off-support extrapolation (Block 5), regularization
  (Block 4), TabPFN (homework E) all exercised. Genuine gaps are the
  *internal mechanism* slides: the bagging variance / ρ-ceiling formula, OOB,
  honest feature importance, CatBoost, and the functional-gradient view — all
  [CORE] exam statements, none exercised in code.
- **ML-PC (generalization/robustness — delivered W8):** PARTIAL. The
  generalization spine (bias-variance, the U-curve, in-distribution
  regularizers failing under shift) and the [CORE] **group-CV** lesson are
  exercised well (Block 3, 4, 6). The deck's other four sections —
  robustness/noise/outliers/distribution-shift *taxonomy*, CV mechanics
  (stratified-K, three-set discipline), HPO, process windows, sensitivity —
  are absent or only prose. Acceptable as a *preview*, but the distribution-shift
  framing the exercise leans on is never named with the deck's vocabulary.
- **MG (local atomic environments / SOAP — delivered W8):** PARTIAL. Block 6
  delivers real `dscribe` SOAP, periodic structures, mean-pooling, and the
  *headline* grouped-vs-random-CV leakage demo cleanly — the single most
  important MG transfer. But the deck's descriptor-discipline core (invariance
  test, PBC/cutoff audit, hard-vs-smooth cutoff, sum-vs-mean pooling,
  composition baseline, polymorph aliasing) is asserted in prose, not exercised.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before 26.05 · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Tree ensembles for tabular learning

- [ ] **[P1] The bagging variance formula & the ρ-correlation ceiling.** [CORE] —
  exam statement #4/#5, the "intellectual center of the whole bagging/RF story"
  and a deck interactive. RF is *used* (Block 1/5, homework C) but the
  $\mathrm{Var}=\rho\sigma^2+\frac{1-\rho}{B}\sigma^2$ floor and *why feature
  subsampling lowers ρ* is never measured. Add a short block: bag fully-grown
  trees vs RF (feature-subsampled) on the SOAP or tensile data, measure mean
  pairwise prediction correlation ρ and ensemble variance vs B, show the floor.
  Natural extension of the existing Block 3 bootstrap-ensemble code.
- [ ] **[P1] Out-of-bag (OOB) error.** [CORE] — exam statement #6, "one of the
  most under-used diagnostics," and the deck's answer to scarce-data CV. The
  notebooks use only `train_test_split` / `GroupKFold`; `oob_score=True` is
  never set. Add an OOB-vs-K-fold comparison in Block 1 or homework Part B
  (RF already there): show OOB ≈ CV at a fraction of the cost.
- [ ] **[P1] Honest feature importance: permutation / TreeSHAP vs impurity.**
  [CORE] — exam statement #7, the slide that "prevents a real, common
  scientific error." Currently only a *stretch* exercise (Block 7 Ex 4) and it
  uses `permutation_importance` on a leaky `xgb_random` model without the
  impurity-vs-permutation contrast or the "never claim causality" point.
  Promote to a core block: impurity importance vs permutation importance on the
  same RF, show the high-cardinality bias, state the causality caveat.
- [ ] **[P2] Gradient boosting as gradient descent in function space (pseudo-residuals).**
  [CORE] — exam statements #8/#9, the section's "aha" slide; explicitly
  braided with MFML's optimizer unit (η is the same η). XGBoost is *called*
  (Block 1/5/6, homework C) but the residual-fitting mechanism is never made
  visible. Add a ~15-line hand-rolled GBM-of-stumps on the 1-D tensile data
  (fit tree to residual, add η·tree, repeat) with train/val curves — mirrors
  the deck's boosting interactive and ties to the homework C XGB sweep.
- [ ] **[P2] Shrinkage η × early stopping is the boosting recipe.** [CORE] —
  exam statement #10, "early stopping is non-negotiable." Homework C sweeps
  `n_estimators` and shows XGB rising at deep depth, but never sweeps the
  *learning rate* nor uses XGBoost early-stopping (`early_stopping_rounds`).
  Block 4 has MLP early-stopping but not GBM. Add an η ∈ {0.3, 0.1, 0.03}
  × early-stopping sweep to homework Part C so the "small η + many trees +
  early stop" rule is measured, not just stated.
- [ ] **[P2] CatBoost — the materials-tabular default.** [CORE] — exam
  statement #11, the deck's explicit "change your default modeling choice"
  recommendation for categorical-heavy materials data, and named in the unit
  summary. Completely absent (only RF/XGBoost/TabPFN appear). Add CatBoost
  with default settings alongside XGBoost in Block 5 or homework Part C /
  Part E; one extra import, makes the deck's headline actionable.
- [ ] **[P3] Single decision tree: greedy impurity splits, piecewise-constant,
  high-variance instability.** [CORE] foundation (exam statements #1–#3) — the
  notebooks jump straight to *ensembles*; a single `DecisionTreeRegressor` is
  never fit. Add a depth-sweep of one tree (underfit → memorize, resample
  instability) before the ensemble blocks, mirroring the deck's first
  interactive. Cheap, closes the conceptual on-ramp.
- [ ] **[P3] Extremely randomized trees / RF-in-practice knobs.** [SUPPORT] —
  ExtraTrees ("decorrelation lever to its extreme") and the
  matters/doesn't-matter hyperparameter asymmetry (B, min-leaf, max-features
  vs depth/criterion) are deck content with no code touch. One-line
  `ExtraTreesRegressor` swap + a max-features sweep would land it.
- [ ] **[P3] Trees ≠ extrapolation as a *materials-discovery* hazard.**
  [SUPPORT/CORE pitfall] — Block 5 shows bounded off-support failure as a
  *virtue* (graceful), but the deck's pitfall framing ("confident, flat,
  wrong answers" when discovering alloys in unexplored composition space) is
  the opposite reading. Add one markdown cell making the discovery-hazard
  framing explicit next to the Block 5 "honest caveat."

## ML-PC — Generalization, robustness & process windows (delivered W8)

- [ ] **[P1] Distribution-shift taxonomy named with the deck's vocabulary.**
  [CORE] — covariate / label / prior shift (slide 16) is the deck's
  precise name for *exactly* the across-$T$ failure the whole exercise is
  built on. Block 2/3 prose says "process drift" but never classifies it.
  Add one markdown cell in Block 3 stating the $T$-shift is **covariate +
  concept shift** ($p(x)$ and $p(y\mid x)$ both move with dislocation
  mobility) and naming the detection tools (KS / MMD) the deck lists — the
  exercise already has the data to do an input-statistic drift check.
- [ ] **[P2] Group-CV stated as "the materials killer" with mean ± std.**
  [CORE] — slide 24, "if students take one practical lesson from today."
  Block 6 *does* implement `GroupKFold` by prototype (excellent) and prints
  per-fold R², but never frames it as the leakage rule the deck hammers, and
  homework Part B uses plain `KFold`/LOOCV only (no group structure). Add the
  one-sentence "5 specimens × 200 patches ≈ 5 independent samples" framing to
  Block 6 and report grouped CV as **mean ± std** (deck slide 22 protocol),
  not just pooled R².
- [ ] **[P2] In-distribution regularizers do not fix shift — make the Bayesian-prior
  framing explicit.** [CORE] — Block 4 measures L2/dropout/early-stopping
  failing under $T$-shift (strong, matches deck slide 11–12) and Exercise 2
  asks *why*. But the deck's [CORE] "every regularizer is a prior" sentence
  and the "$\lambda$ cannot be set on the training set" rule are never stated
  in the exercise. Add the prior-correspondence table (L2↔Gaussian,
  L1↔Laplace) to the Block 4 take-away.
- [ ] **[P3] Robustness vs generalization distinction; outliers & robust losses.**
  [CORE] of §2, but genuinely outside the exercise's stated braid. The
  generalization-vs-robustness definition (slide 13) and the
  MSE/MAE/Huber outlier table (slide 17) are not touched. Optional: add a
  one-cell aside in Block 2 contrasting "same distribution, new sample"
  (generalization) vs "perturbed distribution" (robustness) since the
  across-$T$ demo is literally the latter.
- [ ] **[P3] Permutation importance / sensitivity as a shortcut detector.**
  [CORE] of §5. Overlaps the MFML permutation-importance [P1] above — a
  single shared block (permutation importance on the SOAP model, framed as
  ML-PC's "physical signal up, nuisance noise down" audit) closes both
  cheaply. Block 6's stretch Ex 4 is the natural host.
- [ ] **[P3] Stratified-K / three-set "drawn once, used once" discipline; HPO;
  process windows.** [CORE] of §3–§4 but out of scope for a 90-min
  tree/generalization exercise and not previewed by the notebook header.
  Note for the W8 ML-PC exercise, not this one.

## MG — Local atomic environments & universal MLIPs (delivered W8)

- [ ] **[P1] Invariance test — rotate a structure, descriptor unchanged.**
  [CORE] — must-know #3-adjacent, slide 16/49 checklist item #4, and the
  deck's repeated "test that catches more bugs than any unit test."
  Block 6 builds real SOAP but never verifies $\|\phi(R\mathbf{x})-\phi(\mathbf{x})\|<10^{-10}$.
  Add ~5 lines: take one `bulk` structure, apply a random rotation, recompute
  SOAP, assert invariance (and contrast with raw Cartesian coords drifting).
  Single biggest MG discipline gap and nearly free given the existing SOAP code.
- [ ] **[P1] Periodic-image / cutoff audit (boundary vs interior coordination).**
  [CORE] — must-know #3 ("periodic images are not optional; first audit"),
  slide 15/43 checklist item #1. `SOAP(periodic=True)` is set (correct) but
  the deck's *audit* — coordination of boundary atoms vs a 2×2×2 supercell —
  is never run. Add the supercell-consistency check on one prototype; this is
  the canonical MG data-quality discipline the deck demands every time.
- [ ] **[P1] $r_c$ as a scientific hyperparameter + sensitivity scan; hard vs
  smooth cutoff.** [CORE] — must-know #2, slides 14/19/45 (a full failure-mode
  slide). Block 6 hard-codes `r_cut=4.0` with a one-line "covers nearest
  neighbours" comment and never scans it. Add an $r_c \in \{3,4,5,6\}$ Å
  MAE-sensitivity scan (deck slide 45 diagnostic: flat = robust, swings =
  brittle) and a sentence on why MLIPs need the *smooth* cutoff $f_c$.
- [ ] **[P2] Composition-only (Magpie/matminer) baseline reported first.**
  [CORE] — must-know #1, checklist item #8, repeated "every time," and the
  deck's single loudest reproducibility message. Block 6 goes straight to
  SOAP with no tier-1 baseline; the toy target is even *designed* to correlate
  with prototype identity, so a composition baseline is exactly the right
  control. Add a Magpie/`matminer` (or even a plain prototype-mean) baseline
  before the SOAP model and report the gap.
- [ ] **[P2] Sum (extensive) vs mean (intensive) pooling.** [CORE] — slides
  37–39, the "unsung hero of MLIP work" ($E=\sum_i\epsilon_i$). Block 6 uses
  `desc.mean(axis=0)` only and the toy target is loosely norm-dependent. Add
  the sum-vs-mean contrast with the extensive/intensive reasoning (total vs
  per-atom energy), and the size-invariance diagnostic (64- vs 128-atom
  supercell mean must match).
- [ ] **[P2] Histogram pooling for rare-motif / polymorph signal.** [CORE] —
  slide 40, the explicit answer to mean-pooling's "average crystal"
  pathology and to polymorph aliasing (slide 47). Only mean-pool exists.
  Add histogram pooling as a third pooling variant in Block 6 and show it
  separates two prototypes that mean-pool blurs.
- [ ] **[P3] Polymorph / framework aliasing diagnostic (SOAP-kernel similarity).**
  [CORE] failure mode (slides 46–47), explicitly the deck's preview of the
  *next* MG unit's split design — directly braids with the Block 6
  grouped-vs-random-CV story. Add the SOAP-kernel cosine-similarity matrix
  across prototypes; flag high-similarity / different-label pairs as the
  aliasing pathology. Strong cross-course tie-in, but heavier than the [P1]s.
- [ ] **[P3] ACSF vs SOAP; descriptor ladder; species-resolved pooling;
  long-range-physics limits.** [SUPPORT/CORE-context] — the descriptor-ladder
  spine (Magpie→RDF→ACSF/SOAP→GNN), the ACSF construction, species-resolved
  pooling (matminer `SiteStatsFingerprint`), and the "when locality is the
  wrong tool" boundary are deck-central but conceptual. At minimum frame
  Block 6's SOAP explicitly as "rung 4 of the descriptor ladder; rung 1
  baseline owed (see [P2] above)" in one markdown cell.

---

## Cross-cutting note

The exercise is genuinely well-braided on the **generalization ↔ tree-ensemble
↔ distribution-shift** axis (MFML × the ML-PC generalization spine): bias-variance
as a *measurement procedure* (homework A → Block 3), the textbook decomposition's
missing shift term, in-distribution regularizers failing under shift, and
trees-vs-MLP off-support inductive bias are all exercised in code, not just
asserted — this is the strongest braid of the three.

Two structural facts shape the priorities. First, **the 26.05.2026 ML-PC and MG
lectures do not occur** (Pfingstdienstag); this exercise is their *forward
preview* into the 02.06 Week-8 decks, so "missing" ML-PC robustness/HPO/window
and MG ACSF/ladder content is partly by design — judged here only against what
the notebook header claims to braid. Second, the gaps that *do* matter are the
**internal-mechanism** slides each course makes [CORE] but the exercise treats
as black boxes: MFML's bagging-variance/ρ-ceiling, OOB, honest importance, and
functional-gradient view; MG's invariance test, PBC/cutoff audit, and the
composition baseline. The exercise *uses* RF/XGBoost/SOAP as tools but never
opens them — closing the MFML [P1]/[P2] mechanism items and the MG [P1] audit
trio (invariance test, supercell-coordination check, $r_c$ scan) is what would
turn this from "calls the right libraries" into a true three-lecture exercise.
