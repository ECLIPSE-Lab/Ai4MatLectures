# Week 12 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 12 (lecture 30.06.2026)** · braided exercise =
`week12_uncertainty_and_discovery.py` (main) + `week12_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-12 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/12_uncertainty_in_predictions/01_intro.qmd` | Uncertainty in predictions — Bayesian predictive distribution, evidence, GPs, MC dropout, ensembles, MDNs, calibration |
| ML-PC | `ml_for_characterization_and_processing/unit11_uncertainty_gp/12_uncertainty_gp.qmd` (deck file is `12_uncertainty_gp.qmd`, **not** the canonical `01_intro.qmd`; YAML title = "Unit 11: Uncertainty-aware regression & Gaussian Processes") | Materials UQ case studies: 21CrMoV5-7 GP for hardness, MC-dropout SEM segmentation, L-PBF active learning, conformal/CQR, calibration hygiene |
| MG | **MISALIGNED — see caveat.** True calendar W12 = `materials_genomics/12_generative_models_and_inverse_design/01_intro.qmd` (Unit 12: Generative Models & Inverse Design). The notebook instead braids the *dropped* "folder 11 clustering vs discovery" lecture; its actual content maps to MG **Unit 13** (`materials_genomics/13_uncertainty_aware_discovery_and_gaussian_processes/01_intro.qmd`, lecture 07.07.2026). |

### ⚠️ MG misalignment (read before using this audit)

This is the recurring off-by-one MG braid, and this week it is the worst case
seen so far — a **two-way** miss:

1. **The notebook self-describes the wrong lecture.** Its header says it braids
   *"MG (curriculum) → slides folder 11: Clustering vs discovery in materials
   spaces — top-k retrieval, per-cluster acquisition budgets, the
   discovery-vs-labeling distinction."* Per
   `materials_genomics/REALIGNMENT_2026-05-13.md`: "**Clustering as standalone
   discovery dropped**. … cluster-based triage is folded into the Week 13
   active-learning session," and old folder `11_clustering_vs_discovery…` was
   **repurposed/renamed** to `12_generative_models_and_inverse_design`. The
   lecture the notebook claims to braid **no longer exists**.
2. **What the notebook actually exercises** (Block 5 per-cluster GPs +
   novelty/discovery flags, Block 6 acquisition-budget allocation, Blocks 2–4
   GP active learning / acquisition functions / cost-aware AL) maps cleanly
   onto MG **Unit 13** "Uncertainty-Aware Discovery and Gaussian Processes"
   (§B ranking under uncertainty, §C GPs, §D acquisition / cost-aware /
   multi-fidelity / batch, §F discovery-loop cases). That is **next week's**
   MG lecture (07.07.2026), not Week 12's.
3. **The true calendar Week-12 MG lecture is Unit 12: Generative Models &
   Inverse Design** (CDVAE → DiffCSP/DiffCSP++ → MatterGen → FlowMM →
   CrystaLLM, the discovery funnel, the S.U.N. metric, classifier-free
   guidance, conditional inverse design, downstream DFT/MLIP filtering, the
   GNoME story). **None of this content appears anywhere in either notebook.**

Consequence: the audit below treats the MG axis two ways — (a) it scores the
notebook against what it *actually braids* (MG U13 discovery/AL material), which
is largely well-covered and only needs polish; and (b) it flags the
**true-W12 MG (generative / inverse design) as entirely absent** — a structural
gap, not a missing exercise. The lecturer must decide whether Week 12's
exercise is meant to braid generative models (then most of the notebook is
the wrong week) or whether the notebook should be re-dated/re-labelled to
Week 13 and a separate generative-models exercise authored for Week 12.

## Verdict

- **MFML (uncertainty in predictions):** STRONG. GP posterior from scratch,
  ML hyperparameter learning, the three predictive-UQ estimators
  (GP / MC dropout / deep ensemble), calibration / reliability diagram, the
  aleatory-vs-epistemic / misspecification distinction, and BO acquisition
  functions are all genuinely exercised. Gaps are the *evidence-framework*
  half of the deck (marginal likelihood as Occam's razor, Bayes factors,
  effective parameters, MDNs) which are prose-only or absent.
- **ML-PC (materials UQ case studies):** STRONG. The 21CrMoV5-7-style GP,
  AM/L-PBF active learning, cost-aware AL, and CQR are all implemented on
  realistic lab datasets. Gaps are the MC-dropout-on-microscopy case study
  (segmentation entropy maps), per-tool temperature-scaling recalibration,
  and OOD detection (Mahalanobis / ensemble disagreement) — all [CORE] ML-PC
  slides, none touched.
- **MG:** SPLIT VERDICT. Against the **braided** lecture (U13 discovery/AL):
  MODERATE-to-STRONG — per-cluster GPs, discovery-vs-labeling, acquisition
  budgets, EI/UCB/PI, cost-aware and multi-fidelity AL are all present,
  missing only convex-hull-aware acquisition, ranking metrics, and
  Thompson/batch acquisition. Against the **true calendar W12 lecture**
  (U12 generative / inverse design): **ABSENT** — no generative model, no
  inverse-design loop, no discovery funnel, no S.U.N. metric anywhere.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before 30.06 · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Uncertainty in predictions

- [ ] **[P2] Variance decomposition (aleatory + epistemic), measured.** [CORE] — the deck calls it "the conceptual keystone of the entire unit" and the highest-probability exam derivation. Homework Part D *discusses* epistemic-vs-misspecification in prose; nothing splits a predictive variance into the two terms numerically. Add a short block: on the deep ensemble, estimate `Var_θ[μ]` (between-member) vs `E_θ[σ²]` (within-member / noise) and show only the first shrinks as the training set grows.
- [ ] **[P2] Evidence / log marginal likelihood as Occam's razor.** [CORE], a whole deck section (marginal likelihood, Bayes factor, model comparison). Homework Part A *optimises* the NLL and plots a multi-start bar chart but never frames it as model selection. Add a block: fit GPs with short vs long `length_scale` priors (or RBF vs linear kernel), compare log-marginal-likelihood, show the evidence prefers the simpler explanation at small N — the deck's $N{=}5$ degree-1 story.
- [ ] **[P1] Mixture-Density Networks (multi-modal predictive distribution).** [CORE] deck section, the "Direct prediction" row of the UQ taxonomy — **completely absent** from both notebooks. Add a compact block: a 2-component MDN (softplus σ, NLL of a Gaussian mixture) on a deliberately one-to-many toy (one input → phase A or phase B) showing a unimodal regressor putting its mean in the empty valley between modes.
- [ ] **[P2] Effective number of parameters γ = Σ λ/(λ+α).** [SUPPORT]/[CORE]-adjacent; pure prose in the deck. Optional small block computing γ for a Bayesian-linear / GP-marginal model on the tensile data and contrasting it with the raw parameter count.
- [ ] **[P3] Matérn ν=5/2 vs RBF kernel contrast.** [SUPPORT] but the ML-PC deck explicitly recommends Matérn for metallurgical regime-change responses, and the notebook only ever uses RBF. Add a one-cell RBF-vs-Matérn fit on `TensileTestDataset(T=600)` showing the differentiability/extrapolation difference (also closes a real ML-PC recommendation).
- [ ] **[P3] Sequential Bayesian learning visual (prior → 1 → 2 → 20 obs).** [SUPPORT]. Homework Part A shows prior samples and a posterior at fixed N; the deck's signature 4-row "bundle collapses as data arrives" picture is not reproduced. Cheap add to Part A: posterior at N ∈ {1,2,8,20}.

## ML-PC — Materials UQ case studies

- [ ] **[P1] MC-dropout on microscopy segmentation (per-pixel entropy map).** [CORE] — Case study B is a large fraction of the ML-PC deck and is **completely absent**. Both notebooks only run MC dropout on a 1-D toy MLP. Add a block: a tiny U-Net-ish CNN with dropout on a synthetic SEM-like image (or the Cahn–Hilliard / Ising fields used earlier in the semester), keep dropout on at inference, plot the per-pixel predictive-entropy map and a "flag pixels with H > τ for review" operating curve.
- [ ] **[P1] OOD detection (Mahalanobis / ensemble disagreement / GP-variance).** [CORE] deck section "OOD detection — when the model sees something new" — not implemented. Exercise 2 *touches* GP-variance off-support but never frames it as an OOD detector with a reject threshold. Add a block: fit a Mahalanobis detector on `TensileTestDataset(T=600)` features, score `T=0` as the shift set, plot a detection ROC, and route high-score points to "refuse to predict."
- [ ] **[P1] Per-tool / per-condition temperature-scaling recalibration.** [CORE] — "Tool-shift calibration" is a named deck slide and the deck's stated #1 deployment-hygiene rule. The homework reliability diagram is single-condition only. Add a block: show miscalibration on `T=0` after calibrating on `T=600`, then fit a single scalar recalibration factor on a small `T=0` calibration slice and re-plot the reliability diagram (cheap, post-hoc, the deck's exact recipe).
- [ ] **[P2] Reliability diagram on the *deep ensemble / MC dropout*, not just the GP.** [CORE] — the deck insists "always run a reliability diagram on any deployed model"; homework Part C calibrates only the GP. Extend Part C / Block 1 to add the ensemble and MC-dropout reliability curves on the same `T=600` test set so the "approximate UQ loses calibration guarantees" claim (must-know statement #3) is *shown*, not asserted.
- [ ] **[P3] Constrained Bayesian optimisation (safety / no-go zones).** [SUPPORT]/[CORE] — Case study C explicitly covers hard constraints (P/v keyhole, no-go zones) via constrained BO. Block 4 is cost-aware but unconstrained. Optional extension: add a forbidden region to the AL pool and a rejection-sampling or constraint-GP filter on the acquisition.
- [ ] **[P3] TabPFN as the small-tabular GP alternative.** [SUPPORT] — a full deck slide ("Modern small-tabular alternative: TabPFN"). At minimum frame `make_gp` explicitly as "the classical baseline TabPFN now competes with," or add an optional TabPFN-vs-GP LOO-RMSE comparison on the tensile data as a reading/stretch task.

## MG — *(against what the notebook actually braids: U13 discovery / AL)*

- [ ] **[P2] Ranking / screening-decision metrics under uncertainty.** [CORE] in MG U13 (§13 "Ranking Under Uncertainty", §14 "Screening-Decision Economics"). Block 5/6 report MSE and a discovery *count* only. Add Spearman ρ / Kendall τ and top-k recall on the per-cluster GP predictions, plus a simple screening-economics table (cost of a false-accept vs a missed discovery) — the metrics that actually matter for the discovery framing.
- [ ] **[P2] Convex-hull-aware / energy-above-hull acquisition.** [CORE] MG U13 (§08, §29). The notebook's discovery flag is a generic novelty+variance heuristic; the MG lecture frames discoverability via energy-above-hull and a hull-aware acquisition. At minimum add a markdown bridge naming the analogy, or (better) a synthetic "distance-to-hull" target so the acquisition optimises a discovery-relevant quantity rather than raw predictive variance.
- [ ] **[P3] Thompson sampling and batch acquisition.** [CORE]/[SUPPORT] MG U13 (§27, §32). Block 3 covers UCB/EI/PI/max-std; Thompson sampling (posterior-sample argmax) and batch (query-q-at-once) acquisition are in the deck but unexercised. One extra acquisition variant + a batch-of-5 vs sequential comparison.
- [ ] **[P3] Closed-loop / autonomous-lab framing.** [SUPPORT] MG U13 §F case studies (closed-loop perovskite, BO for alloys, autonomous labs). The notebook runs the AL loop but never frames it as the self-driving-lab prototype. A short closing markdown cell tying Blocks 2–6 to the autonomous-lab loop would braid the MG narrative properly (currently only the homework red-thread mentions it).

## MG — *(against the TRUE calendar Week-12 lecture: U12 Generative Models & Inverse Design)*

> These are **structural** gaps: if Week 12's exercise is genuinely meant to
> braid the Week-12 MG lecture, the entire generative/inverse-design axis is
> missing. If the lecturer instead intends this notebook for Week 13, these
> become out-of-scope and the [P1]s above stand alone. **Resolve the
> scheduling question first** — do not silently add generative blocks to a
> notebook that may be re-dated.

- [ ] **[P1] DECISION REQUIRED: which week is this exercise?** Confirm whether `week12_*` should braid MG U12 (generative) or be renamed/re-dated to Week 13 (uncertainty-aware discovery). Every item in this section is contingent on that answer. Recommended: this notebook is a near-perfect MG-U13 braid and a poor MG-U12 braid — re-label to Week 13 and author a separate generative-models exercise for the true Week 12.
- [ ] **[P1] Generative model for crystals/structures (the entire MG U12 spine).** [CORE] and **absent**. If kept as Week 12: add a minimal generative block — even a 1-D/2-D toy diffusion or flow-matching sampler (forward-noise → learned reverse → sample) to make "sample from p(x | y*)" concrete, framed against the deck's CDVAE→DiffCSP→MatterGen→FlowMM lineage.
- [ ] **[P1] Conditional / inverse-design loop (sample for a target property).** [CORE] MG U12 ("Conditional vs Unconditional Generation", "Targeting a Property", classifier-free guidance). Absent. A toy conditional sampler (generate candidates near a target y*, then filter) would braid directly with the MFML UQ surrogate and the ML-PC acquisition story.
- [ ] **[P2] The discovery funnel + S.U.N. (Stable/Unique/Novel) evaluation.** [CORE] MG U12. The notebook's Block 5 "discovery flag" is a *clustering*-era heuristic; the true-W12 deck evaluates generated candidates via validity/novelty/uniqueness/stability rates and a multi-stage funnel (generate → pre-filter → MLIP relax → DFT → uncertainty triage → synthesise). Add a funnel-style multi-stage filter with a reported S.U.N.-style pass rate, with the UQ surrogate doing the "uncertainty triage" stage — this is the natural braid point with MFML/ML-PC.
- [ ] **[P3] Uncertainty-aware filtering as the generative↔UQ bridge.** [SUPPORT] MG U12 ("Uncertainty-Aware Filtering", "Active Learning + Generative Loop", the GNoME story). If a generative block is added, close the loop: use the GP/ensemble predictive variance from the MFML half to triage generated candidates — this is the single cleanest three-lecture braid available and would make Week 12 a true triad exercise.

---

## Cross-cutting note

On the **MFML × ML-PC** axis the exercise is genuinely well-braided: the GP
posterior, the three predictive-UQ estimators, calibration/CQR, and the
cost-aware active-learning loop on realistic lab datasets form a coherent
end-to-end Bayesian-optimisation story. The MFML *evidence-framework* half and
the ML-PC *microscopy / OOD / recalibration* case studies are the main
intra-axis gaps and are individually addable.

The **MG axis is broken at the scheduling level, not just the content level.**
The notebook braids a lecture that was *deleted* in the 2026-05-13 realignment
(standalone clustering/discovery), accidentally covering most of MG **Week 13**
while covering **none** of MG **Week 12** (generative models & inverse design).
This is a more serious instance of the recurring off-by-one MG braid: it is
off by a *repurposed* slot, so the notebook looks internally consistent and
the miss is invisible without the realignment doc. Closing the MG-U13 [P2]/[P3]
items polishes the braid the notebook *has*; the MG-U12 section is a
go/no-go decision the lecturer must make before 30.06 — the recommended
resolution is to re-date this notebook to Week 13 and commission a separate
generative-models exercise for the true Week 12.
