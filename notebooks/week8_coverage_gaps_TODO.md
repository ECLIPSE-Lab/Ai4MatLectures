# Week 8 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 8 (calendar W8; MFML probabilistic lecture
delivered this week)** · braided exercise =
`week8_uncertainty_and_robustness.py` (main) +
`week8_uncertainty_and_robustness_homework.py`.

> **Lecturer-confirmed 2026-05-17 (as designed):** the Fronleichnam-cancelled W8 live
> exercise slot is officially owned by self-study (W7 exercise = W8 preview). The
> cancelled-session structure is intended; the [P*] items below remain valid content
> gaps against what the exercise braids.

**Triad lectures audited (post-restructure, all three verified):**

| Course | Deck | Week-8 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/07_probabilistic_view_of_learning/01_intro.qmd` (folder `07_*`; deck self-titles "Unit 7") | Probabilistic view of learning: aleatory/epistemic, Gaussian, MLE=MSE, Bayes, MAP=ridge, predictive distribution, calibration, **split conformal + CQR** |
| ML-PC | `ml_for_characterization_and_processing/unit07_generalization_robustness/01_intro.qmd` (Unit 7 folder; **delivered calendar W8** per `AGENT_INSTRUCTIONS.md`) | Generalization & bias-variance, robustness/noise/shift, CV & HPO, **process windows**, sensitivity & feature importance |
| MG | `materials_genomics/06_local_atomic_envs/01_intro.qmd` (slug `06_*` authoritative; deck self-titles "Unit 6"; MG **Week 8, 02.06.2026** per `REALIGNMENT_2026-05-13.md`) | Local atomic environments: descriptor ladder, PBC neighbour lists, invariance, ACSF/**SOAP**, pooling, failure-mode audits; universal MLIPs named in title |

**Caveats.**

- **Exercise-cancelled / self-study character.** The live ML-PC exercise slot
  on 04.06.2026 is cancelled (Fronleichnam); the main notebook's Blocks 6–7
  explicitly double as ML-PC self-study. Week 8 therefore carries a higher
  self-contained burden than a normal braided week — gaps below matter more.
- **Numeric-prefix vs slug mismatch confirmed (not a triad error).** All three
  decks are off-by-one between folder slug and self-titled unit number. The
  schedule authorities (`AGENT_INSTRUCTIONS.md`, `REALIGNMENT_2026-05-13.md`)
  resolve the calendar-W8 triad exactly as hypothesised. The notebook header's
  own narrative ("MG Week 8 — Local atomic environments + universal MLIPs",
  "older Regression-and-generalisation MG slot has moved to Week 9") matches.
- **MG deck under-delivers its own title.** The deck is titled
  "Local Atomic Environments & **Universal MLIPs**" and the realignment doc
  promises "+ universal MLIPs (MACE-MP-0, M3GNet, CHGNet)", but the 52-slide
  body contains **no dedicated universal-MLIP section** — MACE/M3GNet/CHGNet
  appear only as one-line forward mentions inside the ACSF/SOAP equivariance
  notes. The notebook's MACE-MP-0 / M3GNet content is therefore *ahead of*
  the deck on that sub-topic; audit treats the deck's actual CORE content
  (descriptor ladder, PBC, invariance, SOAP, pooling, failure audits) as the
  benchmark, with the universal-MLIP strand flagged as a deck gap, not a
  notebook gap.
- **MG split across two notebooks.** Homework Part F (PBC list, RBF,
  message-passing/over-smoothing, M3GNet stretch) carries most MG mechanics;
  main-notebook Block 8 is a deliberately thin SOAP+MACE bridge. The deeper
  standalone MG walkthrough is referenced as `notebooks/MG/week06_soap_and_mace.qmd`
  (not audited here — out of scope, not the braided target).

## Verdict

- **MFML (probabilistic view):** STRONG. The CORE spine — MLE=MSE,
  aleatoric/epistemic decomposition (closed-form *and* ensemble), MAP=ridge
  with λ=σ²/τ², predictive distribution, calibration, split conformal — is all
  exercised with numerical verification. Gaps are on the support tier
  (Student-t, KL, MDN, the conformal *exchangeability-failure* demonstration).
- **ML-PC (generalization & robustness):** PARTIAL→STRONG. Bias-variance
  U-curve, leakage/group-split, sensitivity, and the process-window deliverable
  are well covered (homework B/C + Blocks 6–7). **The entire CV/HPO §3 and the
  bias-variance-as-regularizer-knob framing are absent**, and robustness is
  only tested via input drift — noise-injection, distribution-shift, and the
  outlier/robust-loss half of §2 are prose-only. Given the cancelled live slot,
  these omissions are costlier than usual.
- **MG (local atomic environments):** PARTIAL. PBC neighbour list, Gaussian-RBF
  edges, message passing + over-smoothing, and a SOAP fingerprint + MACE-MP-0
  single-point are all implemented (homework F + Block 8). But the deck's CORE
  organising spine — **the descriptor ladder and the mandatory composition-only
  (Magpie) baseline**, the **rotation-invariance numerical test**, **pooling as
  a scientific choice (mean vs sum vs histogram)**, and the **descriptor-aliasing
  failure modes** — is barely touched in code.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should
add before the session · **[P2]** core but partially covered — extend ·
**[P3]** support/nice-to-have.

---

## MFML — Probabilistic view of learning

- [ ] **[P2] Conformal under broken exchangeability — measured, not just stated.**
  Homework Part E does split-conformal and *prose-discusses* the
  calibrate-at-T=600 / deploy-at-T=0 coverage collapse, but never runs it. The
  deck makes the exchangeability-failure slide [CORE] ("the single most common
  student error") and ML-PC depends on this. Add a short block: reuse the Part-E
  conformal predictor, evaluate empirical coverage on an *out-of-condition*
  test slice (calibrate T=600, test T=0), print coverage ≪ 0.9 — closes the
  most load-bearing conformal caveat with one numeric line.
- [ ] **[P2] Plug-in vs predictive interval — the over-confidence mechanism.**
  The deck's predictive-distribution slide is [CORE] ("plug-in intervals are
  systematically over-confident, especially at small N / extrapolation").
  Block 2 builds the Bayesian predictive band but never contrasts it against a
  point-estimate ±σ̂ "plug-in" band. Add the overlay (OLS ŷ ± z·σ̂_MLE vs the
  Block-2 predictive ±2σ) and measure the coverage gap in the sparse / extrapolated
  strain region — this is the slide's whole point and currently only Block 2's
  epistemic-band picture gestures at it.
- [ ] **[P3] Student-t robust likelihood — promote from homework G.2 to a braided block.**
  Robustness-to-outliers is a [CORE] must-know statement (#9). It lives only in
  homework Part G.2 (a standalone synthetic-contamination demo) and stretch
  Exercise 4 mentions heteroscedastic NLL. Neither is wired to the tensile
  process story. Optional: a short main-notebook block injecting a few bad
  tensile measurements and refitting OLS vs Student-t, tying the heavy tail to
  the calibration plot in Block 5.
- [ ] **[P3] KL divergence between Gaussians.** [CORE] must-know statement #11,
  flagged as the highest-leverage forward investment (VAE regularizer, Week 11).
  Implemented only in homework Part G.1 as an isolated 3-row table with no link
  to the predictive/Bayesian machinery used in the main notebook. Optional:
  one line computing KL(posterior ‖ prior) for the Block-2/3 Bayesian model so
  the symbol is anchored in *this* week's model, not a detached appendix.
- [ ] **[P3] Maximum-entropy / CLT justification of the Gaussian noise model.**
  [CORE] (must-know #2). The entire exercise *assumes* Gaussian residuals
  (Part A overlay is the closest touch) but never states or checks the
  maximum-entropy / CLT reason the Gaussian is the right default. A two-line
  markdown callback plus a residual normality glance would close the
  "why this likelihood at all" gap the deck spends three slides on.
- [ ] **[P3] Conditioning / precision-weighted-average reading of the posterior.**
  The deck's Bayesian-update interactive and "posterior = precision-weighted
  average" framing is [CORE] pedagogy. Block 2/3 compute S_N, μ_N in matrix
  form but never expose the precision-weighting intuition (prior precision +
  data precision) or the σ₀²→∞ ⇒ MLE / N→∞ ⇒ MLE sanity limits. One print
  block showing μ_N → OLS as τ²→∞ would land the deck's reconciliation message.

## ML-PC — Generalization, robustness & process windows

- [ ] **[P1] Cross-validation: k-fold + group-K-fold with mean ± std.**
  ML-PC §3 (CV/HPO, ~18 min, two of seven learning outcomes, the "single
  biggest mistake in published materials-ML" slide on group splits) is
  **entirely absent**. Homework Part C does a *single* leave-condition-out
  split; there is no k-fold, no GroupKFold, no mean±std reporting anywhere.
  With the live ML-PC slot cancelled this is the costliest gap. Add a block:
  5-fold CV on T=600 *and* a group-wise split keyed by temperature condition
  (the specimen/instrument analogue the deck stresses), reporting mean ± std,
  contrasted with the single-split number from homework C.
- [ ] **[P1] Three-set discipline made explicit (train / val / test, touch-once).**
  [CORE] — the deck's "test set is sacred" slide and the three-set slide are
  core, and the homework even has a 3-way 60/20/20 split in Part E but never
  *names* the protocol or enforces touch-once. Add an explicit markdown +
  code framing in the CV block: train fits θ, val tunes (e.g. ridge α / degree
  / trust threshold), sealed test reported exactly once. Currently every block
  silently reuses one seeded split.
- [ ] **[P1] Noise-injection robustness (aleatory-tolerance) — the diagnostic test.**
  ML-PC §2 slide 14 gives an explicit [CORE] diagnostic: add n detector-noise
  realisations to one input, measure prediction spread vs across-input
  variation. Block 6 does *sensitivity* (∂ŷ/∂x) and Exercise 2 does a single
  ±10 °C / ±1 °C drift, but nobody samples a noise envelope and measures spread.
  Add a block: take fixed (ε, T) points, draw N Gaussian noise samples at a
  realistic strain/T jitter, plot ensemble-prediction spread, compare to
  model RMSE — the deck's literal "are we at the noise floor?" test.
- [ ] **[P2] Distribution shift as a named, measured failure (not just leakage).**
  [CORE] §2 covariate/label/prior shift slide. Homework Part C demonstrates
  the *effect* (leave-T-out RMSE blow-up) but never frames it as covariate
  shift nor measures input-statistic drift (KS / mean-shift between T
  conditions). Extend the CV block with a one-line train-vs-test feature-stat
  comparison labelling it covariate shift — connects the leakage gap to the
  deck's distribution-shift taxonomy and to the conformal-exchangeability
  failure (MFML P2 above).
- [ ] **[P2] Outlier robustness / robust-loss contrast (MSE vs MAE/Huber).**
  [CORE] §2 slides 17–18 (the loss-sensitivity table, the "outlier or
  discovery?" think-pair-share). Completely absent from both notebooks on the
  tensile data (homework G.2's Student-t is the MFML-flavour cousin, on
  synthetic residuals only). Add a short block: inject one gross tensile
  outlier, refit OLS vs Huber/MAE, show the line move vs not-move — the deck's
  chalkboard demo, on real data, and a clean braid with MFML's Student-t.
- [ ] **[P2] Probabilistic process window with a *calibrated* threshold.**
  Block 7 builds a genuine process window (in-spec ∧ trustworthy) — strong, and
  the single best-braided ML-PC deliverable. But the trust threshold is an ad-hoc
  `0.10·(max−min)`; the deck's §4 slide 35 makes the *calibrated* probability
  contour (95% safe / uncertain band) [CORE]. Extend Block 7: derive the trust
  threshold from the Block-5 calibration result (or the conformal q̂), and draw
  the deck's three-zone map (qualified / collect-more-data / do-not-operate)
  rather than a binary mask.
- [ ] **[P3] Permutation importance / global sensitivity.** §5 slide 45 makes
  permutation importance [CORE] ("the workhorse", model-agnostic). Block 6
  gives only local finite-difference gradients. Add a cheap permutation-importance
  pass over the two inputs (shuffle strain, shuffle T_norm, measure RMSE drop)
  — complements the local gradient with the global picture the deck pairs it with.
- [ ] **[P3] Lipschitz / physical-continuity smell test; HPO (random vs grid).**
  Slide 21 (physical continuity as a robustness requirement) and §3's
  random-vs-grid / Bayesian-opt HPO content are [CORE]/[SUPPORT] but heavy for
  a 90-min slot. At minimum: one markdown note framing the Block-6 ∂(stress)/∂T
  field as an empirical Lipschitz check, and a one-line pointer that ridge α /
  ensemble M are hyperparameters that must be val-tuned (ties to the three-set
  P1 item).

## MG — Local atomic environments

- [x] **[P2] MLIP energy+force on a real interatomic dataset + the correlated-sample
  trap.** *(Closed 2026-05-18 — Block 8d, self-study.)* rMD17 (benzene) had no
  host block in this notebook (SOAP/MLIP force-matching moved to MG Week 6;
  Week 8's MG content is graph reps). Added a self-contained Block 8d: tiny
  invariant-feature (pairwise-distance) MLP energy regressor with honest
  disjoint train/held-out error, the rMD17 `n_samples>1000` `UserWarning`,
  a measured energy-autocorrelation diagnostic (showing the shipped release is
  already decorrelated), a clearly-labelled *constructed* pseudo-trajectory
  demonstrating the naive-random-vs-trajectory-block optimism gap, and a
  force-norm magnitude diagnostic. This is the MLIP-modality instance of the
  cross-cutting "your test point looked in-distribution but wasn't" braid and
  ties the correlated-split leakage to Week 8's generalisation/robustness theme
  (same failure as the Block 1b leave-condition-out gap, one modality out).
- [ ] **[P1] The descriptor ladder + mandatory composition-only (Magpie) baseline.**
  This is *the* organising spine of the MG deck — must-know statement #1,
  repeated ~6×: "every materials-ML project reports a composition-only Magpie
  baseline before any structure-aware model." **Absent from both notebooks.**
  The notebooks jump straight to per-atom graph/SOAP machinery with no tier-1
  reference point. Add a block (homework Part F or Block 8): Magpie/matminer
  composition features → random forest on a tiny formation-energy subset, then
  a SOAP/structure model, and *report the gap* — the deck's central
  reproducibility lesson, currently entirely unexercised.
- [ ] **[P1] Rotation-invariance numerical test on the descriptor.**
  [CORE] — the deck's slide 16/17 invariance-discipline content and the explicit
  diagnostic "rotate by random R, check ‖φ(rot) − φ(orig)‖ < 1e-10; this catches
  more bugs than any unit test." Homework F builds a PBC list and RBF edges but
  never tests invariance; Block 8 computes SOAP but never rotates the structure.
  Add ~10 lines: rotate one Block-8 prototype by a random SO(3) matrix,
  recompute the SOAP fingerprint, assert numerical invariance — the single
  highest-yield MG audit per the deck.
- [ ] **[P1] Pooling as a scientific choice: mean vs sum vs histogram.**
  [CORE] §E (slides 37–41) and must-know #4; the deck explicitly flags
  "mean-pool a tier-3 descriptor → throws away the minority-motif information
  you climbed the ladder for" as an anti-pattern. Block 8 does
  `_soap.create(atoms).mean(axis=0)` — exactly the flagged anti-pattern, with
  no contrast. Add: compute mean-pool *and* sum-pool *and* a coarse histogram
  pool of the per-atom SOAP for the three prototypes; show they differ and
  state the extensive/intensive (sum vs mean) reasoning the deck makes core.
- [ ] **[P2] Descriptor aliasing failure mode (polymorph / framework).**
  [CORE] §F slides 46–47 and must-know #5 ("local similarity does not imply
  property similarity"); the TiO₂ rutile-vs-anatase example is the deck's
  recurring teaching case. Nothing in either notebook demonstrates aliasing.
  Add (can reuse Block 8 SOAP): build two same-composition, different-structure
  cells (e.g. an fcc vs hcp metal, or two cubic polymorphs), show high SOAP
  kernel similarity despite the structures being distinct — the deck's
  headline cautionary diagnostic.
- [ ] **[P2] Hard-cutoff discontinuity vs smooth cutoff envelope (artifact demo).**
  [CORE] slide 14: a hard cutoff is non-differentiable in positions and "fatal
  for forces"; MLIPs *always* use a smooth window f_c. Homework F.1's
  `periodic_neighbours` uses a pure hard `d <= r_cut`; F.2's RBF has no cutoff
  envelope. Add: slide one neighbour across r_cut and plot the resulting
  feature with hard cutoff (step) vs cosine envelope (smooth) — the deck's
  differentiability / reproducibility [CORE] point, currently only prose.
- [ ] **[P2] PBC correctness audit (boundary vs supercell coordination).**
  [CORE] slide 15/43: "the first audit on any local-descriptor pipeline" —
  compare coordination of boundary atoms vs a 2×2×2 supercell; must match to
  precision. Homework F.1 has a correct `periodic_neighbours` and an
  `image_range` pitfall note, but never runs the supercell-equivalence audit
  the deck makes the canonical sanity check. Add ~6 lines: build the FCC Cu
  cell and its 2×2×2 supercell, assert per-atom coordination is identical.
- [ ] **[P3] Tier-2 RDF / aggregated-coordination rung.** [SUPPORT]→[CORE]
  bridge: §A slides 9–10 (RDF, structure-aggregated coordination) are the
  middle of the ladder and the deck notes RDF sometimes beats tier-3+mean-pool
  for polymorph aliasing. Neither notebook touches a globally-pooled structural
  descriptor. Optional: add a partial-RDF or coordination-moment featuriser as
  the tier-2 baseline between the (new) Magpie tier-1 and the SOAP tier-3.
- [ ] **[P3] Universal-MLIP strand framed honestly (deck under-delivers its title).**
  The notebook's MACE-MP-0 (Block 8) / M3GNet (homework F.4) single-points are
  actually *ahead of* the MG deck, which despite its title has no universal-MLIP
  section. This is a deck gap, not a notebook gap — but the notebook should not
  imply the lecture covered MACE/M3GNet mechanics. Add one markdown line in
  Block 8 stating the universal-MLIP material is title-only in the W8 deck and
  the standalone walkthrough (`notebooks/MG/week06_soap_and_mace.qmd`) is the
  real treatment; optionally flag for the deck author that slides are missing.

---

## Cross-cutting note

The exercise is **excellently braided on the uncertainty axis**
(MFML × ML-PC): the aleatoric/epistemic decomposition, calibration, and the
process-window deliverable form one genuine three-step story, and Block 7 is a
real ML-PC deliverable, not a token. That spine is the strongest of any audited
week.

The two structural weaknesses are: (1) **the cancelled live ML-PC slot is not
compensated** — the missing CV/HPO + three-set + noise-robustness content
(ML-PC [P1]s) is exactly the §1–§3 material students now have *no* lecture for;
and (2) **the MG lecture is only nominally braided** — the notebooks exercise
*graph/SOAP mechanics* but almost none of the MG deck's actual CORE spine
(descriptor ladder + Magpie baseline, invariance test, pooling-as-choice,
aliasing audits). Closing the four ML-PC/MG [P1] items is what would make this
a true three-lecture exercise rather than an MFML-led one with MG appendices.
One clean braid worth engineering: the conformal-exchangeability-failure
(MFML P2) ≡ distribution-shift (ML-PC P2) ≡ descriptor-aliasing /
out-of-distribution split (MG P2) are the *same* phenomenon at three layers —
a single shared "your test point looked in-distribution but wasn't" block
would tie all three lectures together.
