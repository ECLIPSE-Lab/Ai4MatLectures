# Week 14 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 14 (lecture 17.05.2026 slot; calendar W14)** ·
braided exercise = `week14_explainability_and_limits.py` (main) +
`week14_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-14 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/14_explainability_limits_trust/01_intro.qmd` | Explainability, Limits, and Trust (E1–E6, sensitivity, SHAP, IG, mech-interp, counterfactuals, causality, OOD, fairness, retrospective) |
| ML-PC | `ml_for_characterization_and_processing/unit13_reflection/14_reflection.qmd` (filename non-canonical `14_reflection.qmd`, NOT `01_intro.qmd`; YAML still reads "Unit 13" — delivered calendar W14 per `AGENT_INSTRUCTIONS.md`) | Integration, limits, reflection (XAI for experimental ML, causality in the process chain, ontologies, limits/ethics, expert-in-2030) |
| MG | `materials_genomics/14_constraints_trust_and_integration_outlook/01_intro.qmd` (slug authoritative; folder index = calendar week per `REALIGNMENT_2026-05-13.md`; YAML reads "Unit 14") | Physical constraint enforcement, trust under distribution shift, the autonomous-lab loop, FAIR/repro, 2026 outlook |

### Caveats / misalignment flags

- **MG content misalignment — FLAGGED.** The folder/slug *is* the true
  calendar-W14 MG deck (`14_constraints_trust_and_integration_outlook`,
  "Constraints, Trust, and Integration Outlook") — there is **no
  off-by-one in the deck selection**. However, the **notebook's
  characterisation of MG W14 is wrong**: both notebook headers describe
  MG W14 as "symmetry constraints; what ML can and cannot discover; how
  the pipeline integrates with experimental workflows." That is a sliver
  of a 50-slide deck whose CORE (§B, 8 slides; §D, 8 slides) is
  **physical-constraint enforcement mechanisms** (softmax/simplex heads,
  charge-balance projection, hard-projection vs soft-penalty vs
  architectural prior) and **trust as a system property** (conformal
  prediction, multi-signal OOD gating, silent-extrapolation, audit
  trail), plus the autonomous-lab loop. The notebook braids MG **only**
  via Block 5 (a rotation/flip symmetry audit + TTA on the Ising CNN) —
  one narrow corner of one constraint family. This is the recurring
  "braids the wrong MG content" issue, manifesting as a *scope* mismatch
  rather than a week off-by-one. Audited below against what the deck
  actually contains.
- **ML-PC deck file is `14_reflection.qmd`** (not the canonical
  `01_intro.qmd`); its YAML title still says "Unit 13". This is the
  expected post-realignment state (unit13_reflection → delivered W14),
  not an error.
- **MG §C is an explicit MFML-W13 PINN recap** (slides 14–17) and is
  out of scope for a W14 exercise audit except where it states the
  soft-vs-hard constraint duality; not counted as a W14 gap.

## Verdict

- **MFML (explainability, limits, trust):** STRONG. The applied XAI
  spine — sensitivity → SHAP → IG → mechanistic interp (SAE) →
  counterfactuals → OOD → symmetry/inductive-bias — is exercised
  end-to-end and with from-scratch implementations. Gaps are in the
  *framework/limits* half: the E1–E6 ladder, causality vs correlation,
  the data-manifold/extrapolation detectors as a toolbox, fairness, and
  the "when NOT to trust" checklist are prose-only.
- **ML-PC (integration & reflection):** PARTIAL→STRONG on the technical
  half (SHAP, IG, SAE all land on materials data: tensile MLP, Ising
  CNN). The reflective/limits half that defines this deck — causality
  in the *process chain*, materials ontologies, data/success bias,
  physical hallucinations, the expert's role — is entirely prose (the
  homework Part D reflection is the only nod).
- **MG (constraints, trust, integration):** WEAK. Only Block 5's
  symmetry audit + TTA touches the deck, and even that is *diagnose-only*
  (it never enforces the constraint architecturally). The deck's CORE —
  constraint enforcement mechanisms (softmax/simplex, charge balance,
  hard projection vs soft penalty), conformal prediction, multi-signal
  OOD as a *system* property, the discovery-loop framing — is unexercised.
  Block 6's AE-reconstruction OOD detector is good but is the MFML
  single-score view, not MG's combine-multiple-signals doctrine.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic —
should add before 17.05 · **[P2]** core but partially covered — extend ·
**[P3]** support/nice-to-have.

---

## MFML — Explainability, Limits, and Trust

- [ ] **[P1] E1–E6 explainability ladder + audience-matching.** [CORE]
  — the organising skeleton of the entire MFML unit (six slides + the
  audience table), explicitly flagged "everything for the next 20
  minutes hangs on these six hooks." Nowhere in the notebooks. Add a
  short retrospective/framing cell after Block 4 that labels each
  method already built by its level (sensitivity→E3, SHAP/IG/CF→E5,
  the Block 7 chart→E6) and prints the operator/data-scientist/
  regulator/scientist → levels table. Cheap, high pedagogical payoff.
- [ ] **[P1] Causality vs correlation: the confounder trap +
  detection-vs-prediction.** [CORE], the "conceptual climax" of the
  MFML unit and "the highest-probability conceptual exam question."
  Completely absent. Add a block on the tensile MLP: inject a
  confounded feature (e.g. a synthetic "furnace_id"-style column
  correlated with `T` but causally inert), show SHAP attributes to it,
  then show an intervention/permutation breaks the spurious link —
  directly demonstrates "SHAP is faithful-to-model, not true-of-world."
- [ ] **[P2] Sensitivity analysis from scratch (perturbation S_j,
  tornado plot, global vs local).** [CORE] — the *one formula students
  must write from memory* and the only examinable computation. The
  notebooks jump straight to KernelSHAP; the cheaper screening
  primitive it is contrasted against is never built. Add a ±1σ
  one-at-a-time perturbation scan on the 6-feature tensile MLP, a
  tornado bar chart, and a global-vs-local contrast — then frame SHAP
  as the interaction-aware upgrade.
- [ ] **[P2] Data-manifold / extrapolation as a toolbox (4 detectors).**
  [CORE] — the deck assembles latent density, reconstruction error, GP
  variance, ensemble disagreement as four proxies for one question.
  Block 6 builds exactly *one* (AE reconstruction error). Extend Block
  6 to also compute ensemble disagreement on the Ising CNN (or latent
  k-NN distance) on the same OOD slabs and show the detectors agree
  in-dist and *disagree on the interesting cases* — the deck's point.
- [ ] **[P2] "When models should NOT be trusted" pre-flight checklist.**
  [CORE] — "photograph this slide," the highest-density summary of the
  unit (extrapolation / confounding / insufficient data / missing
  physics / poor calibration). Add a closing markdown checklist cell in
  Block 7 mapping each item to the concrete artefact in the notebook
  that would catch it (Block 6 → extrapolation, the new causality block
  → confounding, etc.).
- [ ] **[P3] Fairness / equalized odds.** [SUPPORT] for a materials
  cohort but explicitly defended in the deck as "the same mathematical
  object as the §9 confounder." Optional: a short note in the causality
  block reframing the confounder demo as a bias-on-a-protected-proxy
  instance; no full ROC-polytope implementation needed.
- [ ] **[P3] Ontology consistency check as automated falsification.**
  [SUPPORT]. The deck's deductive "ontology says grain-size matters but
  SHAP says 0 → data or model is broken" check. Could be a one-cell
  assertion harness on the tensile SHAP output ("known-relevant feature
  must have |φ| above a floor"); low priority for a 90-min slot.
- [ ] **[P3] Genealogy: vanilla-gradient → IG saturation failure,
  stated as the motivating defect.** Homework builds both, but the
  *why IG exists* (saturated activation ⇒ zero gradient ⇒ important
  pixel gets zero credit) is asserted in prose, not demonstrated. A
  tiny synthetic-saturation demo would make the axiom concrete; nice-
  to-have.

## ML-PC — Integration, limits, and reflection

- [ ] **[P1] Causality in the process chain (composition → processing
  → microstructure → properties; prediction vs detection).** [CORE] of
  this deck and a shared spine with MFML §9. Absent as code. The
  confounder block proposed for MFML above should be framed
  *explicitly* in the materials process-chain vocabulary (furnace /
  lot / campaign confounder), satisfying both decks with one block —
  see cross-cutting note.
- [ ] **[P2] XAI *for experimental ML*: SHAP/IG on an image model, not
  just a tabular MLP.** [CORE] — the deck's framing is CAMs/SHAP for
  defect-segmentation / micrograph CNNs. SHAP (Block 2) runs only on
  the 6-feature tensile MLP; IG (homework Part B) does run on the Ising
  CNN, so this is PARTIAL. Add a short KernelSHAP-on-superpixels (or
  occlusion-SHAP) pass on a couple of Ising images so the
  "per-pixel/segment attribution for experimental imaging" claim is
  exercised, not only the tabular case.
- [ ] **[P2] Limits of AI in materials: data bias / success bias /
  physical hallucinations.** [CORE] reflective content; only the
  homework Part D free-text reflection gestures at it. Add a small
  demonstrable instance: train the tensile MLP on a biased slice (only
  T∈{0,400}), show confident-wrong extrapolation at T=600, and label it
  as the "success-bias / unseen-regime" failure the deck names.
- [ ] **[P3] Materials ontologies / digitizing meaning.** [SUPPORT].
  Hard to exercise in code; fold a one-paragraph markdown tie-in into
  the MFML ontology-check P3 item rather than duplicating.
- [ ] **[P3] The expert's role in 2030 / ethical cost / efficiency
  (PINN greener than brute-force NN).** [SUPPORT], pure reflection.
  The Block 7 retrospective is the natural home — add two bullets; no
  code.

## MG — Constraints, Trust, and Integration Outlook

- [ ] **[P1] Physical-constraint enforcement: softmax/simplex head +
  the soft-penalty vs hard-projection vs architectural-prior trichotomy.**
  [CORE] — this is §B, the single largest block of the MG deck, and
  **completely absent**. The notebook only ever *diagnoses* a symmetry
  gap (Block 5). Add a block on the tensile/composition setting: take an
  unconstrained head that emits a fraction vector summing to ≠1, show
  the post-hoc-rescale bias, then a softmax/simplex head that is on the
  simplex by construction at zero cost — the deck's "cheapest win in
  materials ML" and its canonical exam stem. This is the single biggest
  MG gap.
- [ ] **[P1] Symmetry as an *enforced* constraint, not just an audit.**
  [CORE]. Block 5 measures the rotation-equivariance gap and patches it
  with TTA — but the deck's point is the trichotomy: TTA = inference
  patch, train-time augmentation = soft, group-equivariant head =
  architectural prior (guaranteed by construction). Exercise 3 already
  asks for train-time augmentation; promote it into the main Block 5
  narrative and add the explicit "TTA vs aug vs equivariant-arch"
  framing with the trade-off (guarantee vs expressivity/effort) the
  deck makes [CORE].
- [ ] **[P2] Conformal prediction wrapper (distribution-free,
  finite-sample coverage).** [CORE] — "one of the most important slides
  in the unit," the 2026 best-practice trust stack
  (surrogate→conformal→OOD→acquisition). Nothing in the notebooks
  produces a calibrated interval. Add a split-conformal wrapper on the
  tensile MLP (hold-out calibration residuals, (1−α)-quantile band) and
  show empirical coverage ≈ 1−α; ~15 lines, high exam relevance.
- [ ] **[P2] OOD as a *system* property: combine ≥2 independent signals;
  silent-extrapolation failure.** [CORE]. Block 6 builds one OOD score
  (AE reconstruction). The MG doctrine is explicitly "trust is a system
  property, not a model property — combine signals" and the
  silent-extrapolation trap (low surrogate variance + OOD = refuse).
  Extend Block 6: add a Mahalanobis-in-feature-space score on the
  tensile features, construct a "low conformal width BUT high OOD"
  candidate, and show the AND/OR refusal gate — directly braids with
  the conformal item above.
- [ ] **[P2] Charge-balance / multi-constraint projection (joint
  constraint via reparameterisation or hyperplane projection).** [CORE]
  (§B slide 08, Case Study 1 MoS₂, Case Study 2 ternary alloy). The
  simplex item above covers Σx=1; this adds the *coupled* constraint
  (e.g. fixed-ratio 1:2, or Σ cᵢzᵢ = 0) enforced by reparameterising
  only the free DOF. Could be a short add-on to the [P1] constraint
  block on a synthetic 2-/3-component composition target.
- [ ] **[P3] Conformal as an acquisition / refusal gate + audit
  trail.** [SUPPORT/CORE-adjacent] — interval-width-as-decision-gate
  and the per-decision audit-trail log. Once the conformal + combined-
  OOD items exist, add a small "synthesise iff width<δ and not OOD,
  else escalate" decision function and print a one-line audit record;
  stretch.
- [ ] **[P3] The autonomous-lab loop framing / discovery-loop
  economics.** [SUPPORT] for an exercise (it is systems-level, hard to
  code in 90 min). Add a markdown cell in Block 7 placing the
  notebook's surrogate+XAI+OOD machinery as the "select/trust" steps of
  a 6-step closing-the-loop diagram; no code.
- [ ] **[P3] Sim–experiment gap / calibration drift across chemistry
  families.** [SUPPORT]. The biased-slice extrapolation demo proposed
  for ML-PC doubles as a "calibrate-per-family, re-calibrate on a new
  family" illustration if the conformal wrapper is computed per
  T-slice; fold in rather than add a separate block.

---

## Cross-cutting note

The exercise is genuinely strong on the **applied-XAI** axis
(MFML×ML-PC technical half): from-scratch saliency → IG → KernelSHAP →
sparse-autoencoder mechanistic interpretability → counterfactuals →
AE-based OOD, every primitive built on real materials data and explicitly
framed as "the Unit-5 autoencoder plus one term," etc. That spine is
exemplary and needs no work.

The two structural gaps that would make this a true three-lecture
exercise:

1. **One causality/confounder block serves two decks at once.** MFML §9
   (confounder trap, faithful-but-wrong, detection-vs-prediction) and
   ML-PC's "causality in the process chain" are the same idea in
   different vocabulary. A single block — inject a furnace/lot-style
   confounder into the tensile MLP, show SHAP attributes to it, break it
   by intervention — closes the top [P1] item for *both* MFML and ML-PC.
   Highest leverage single addition.

2. **The MG lecture is only nominally braided.** The notebook *diagnoses*
   a symmetry gap but never *enforces* a physical constraint, never wraps
   a surrogate in conformal coverage, and uses a single OOD score where
   MG's entire doctrine is "combine independent signals; trust is a
   *system* property." Closing the MG [P1] items (simplex/architectural
   constraint enforcement; symmetry as enforced-not-audited) plus the
   [P2] conformal + combined-OOD pair is what converts Block 5/6 from "an
   MFML OOD demo with a symmetry side-quest" into the MG W14 lecture the
   students actually see — physical correctness and operational trust as
   first-class, enforced, multi-signal machinery.
