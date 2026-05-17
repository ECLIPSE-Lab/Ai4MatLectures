# Week 13 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 13 (lecture 07.07.2026 MG / per-course W13)** ·
braided exercise = `week13_physics_and_gp.py` (main) + `week13_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-13 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/13_physics_informed_learning/01_intro.qmd` | Physics-Informed Learning (50 slides + full `::: {.notes}` spine) |
| ML-PC | `ml_for_characterization_and_processing/unit12_pinns/13_pinns.qmd` (title says "Unit 12: Physics-constrained ML patterns for the lab"; file is `13_pinns.qmd` — non-canonical name, this IS the deck) | Physics-constrained ML patterns for the lab |
| MG | `materials_genomics/13_uncertainty_aware_discovery_and_gaussian_processes/01_intro.qmd` | Uncertainty-aware discovery: GPs, deep ensembles, active learning |

### Triad caveats — MG alignment checked explicitly

- **No off-by-one this week.** Per `materials_genomics/REALIGNMENT_2026-05-13.md`,
  true calendar W13 MG = folder `13_uncertainty_aware_discovery_and_gaussian_processes`
  = "Uncertainty-aware discovery: GPs, deep ensembles, active learning". The notebook
  headers braid exactly that ("MG Unit 13 — Uncertainty-aware discovery and Gaussian
  Processes"). The recurring off-by-one trap (braiding W12 generative `12_generative_models_and_inverse_design`
  or W14 `14_constraints_trust_and_integration_outlook`) **did not occur**. The slug is correct.
- **BUT a content-level mismatch exists (flagged prominently).** The notebook braids a
  *GP-as-regression-math* lecture. The true MG W13 deck is a **materials-discovery-loop /
  Bayesian-optimisation** lecture: GP theory is explicitly delegated upstream ("MFML W12
  derived the marginal likelihood … today we treat all of those as black boxes"). The
  examinable MG spine is the **discovery loop** (database → predict → screen → synthesise
  → refine), **convex hull / E-hull as the objective**, **acquisition functions
  (EI / UCB / Thompson, hull-aware, cost-aware)**, and **calibration discipline**.
  The notebook implements a from-scratch RBF GP and a variance-greedy active-learning
  loop on a *physics oscillator* — it touches the §C "read a GP posterior" sliver and a
  thin slice of §D, but the entire §A (databases, formation energy, convex hull, E-hull),
  most of §D (EI/UCB/Thompson, hull-aware acquisition), and all of §E (deep ensembles,
  conformal, calibration / reliability diagrams) are absent. This is the dominant gap.
- ML-PC "Unit 12" vs file `13_pinns.qmd`: numbering noise only; the deck content
  (PINN for melt-pool, soft-constrained CNN, symmetry, monotonicity) is the right W13 lecture.

## Verdict

- **MFML (physics-informed learning):** STRONG on the PINN core. The homework + Blocks 1-7
  exercise the PINN loss decomposition, auto-diff residual, collocation, λ-balancing,
  inverse PINN (Ex 4), and the "wrong physics → confidently wrong" failure mode (Block 7) —
  the deck's highest-weight exam items. **Gap: the Lagaris hard-BC substitution** —
  the deck's single most-emphasised "guaranteed exam derivation" (§9, exam statement #6) —
  is *never* implemented; BCs are only ever soft penalties. Data-enrichment half (§3-5)
  and Occam/MDL are prose-only (acceptable: deck marks them lecture-not-exercise).
- **ML-PC (physics-constrained lab patterns):** PARTIAL. Block 6 (tensile σ(0)=0 +
  elastic monotonicity soft penalty) is a clean hit on Pattern D and the "kinematic
  constraints" checklist. Block 7 + the gradient-pathology prose touch the failure-mode
  material. **But 3 of the 4 named lab patterns are absent**: Pattern A (soft-constrained
  inverse problem with two-axis reporting), Pattern C (symmetry / equivariance), and the
  structural-hard-constraint *vs* soft contrast (monotone-by-construction via cumulative
  softplus increments). Two-axis reporting (data-fit + violation-rate) is never done.
- **MG (uncertainty-aware discovery):** WEAK as a *discovery* exercise. The from-scratch
  GP (Block 2), PINN-vs-GP uncertainty (Block 3), and variance-greedy active learning
  (Block 5) are solid pedagogy but braid the *wrong abstraction level* — they teach GP
  regression mechanics (which the deck explicitly outsources to MFML W12) and miss the
  examinable MG spine: convex hull / E-hull objective, EI/UCB/Thompson acquisition,
  hull-aware acquisition, and calibration / reliability diagrams. The MG lecture's
  exam-weight outcomes (1, 2, 3, 5) are essentially untouched.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before the session ·
**[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Physics-Informed Learning

- [ ] **[P1] Lagaris hard-BC substitution (the missing exam derivation).** The deck spends
  §9 (4 slides + checkpoints) on $f(x)=A(x)+B(x)\,\mathrm{NN}(x)$ and the notes call it
  "the single most elegant idea in the unit and a guaranteed exam derivation" (exam
  statement #6). The notebooks enforce *every* BC/IC as a soft penalty (homework Part C,
  Blocks 1/4/7 all add `loss_bc`). Nowhere is a BC satisfied by construction. Add a block:
  re-solve the damped oscillator with the trial form $x(t)=1 + t\,\mathrm{NN}(t)$ (which
  gives $x(0)=1$ exactly) — or for both ICs $x(0)=1,\dot x(0)=0$ use
  $x(t)=1 + t^2\,\mathrm{NN}(t)$ — show `loss_bc` is identically 0, drop the BC term and
  $\lambda_{bc}$, and compare convergence vs the soft-BC PINN. This is the highest-value
  single addition in the whole audit.
- [ ] **[P2] Make the data↔physics information-budget trade-off measurable.** The deck's
  conceptual spine (statistical-balance / Occam / MDL, exam statements #8-9) is asserted
  in homework Part D prose but never *shown*. Block 1 already has the machinery: sweep
  $N_{obs} \in \{3,5,7,15\}$ for vanilla MLP vs PINN, plot grid-RMSE vs N — demonstrate
  the PINN curve is nearly flat (physics buys down data) while the MLP curve is steep.
  Closes "PINNs work with tiny datasets *because* physics supplies the information."
- [ ] **[P2] λ failure modes shown deliberately (the deck's stated Part-C learning goal).**
  Deck "Choosing λ" + "Exercise setup" notes explicitly want students to *cause* the
  under/over-constraining failures. Notebooks fix `LAMBDA_PHYS = 1.0` everywhere. Add a
  λ sweep on the oscillator PINN ({0, 0.01, 1, 100}) showing λ→0 collapses to the vanilla
  overfit and λ→∞ ignores the noisy data — the "interpolation dial" made tactile.
- [ ] **[P3] Data-enrichment taxonomy (FFT / derivative / dimensionless features).**
  Deck §3-5 is a large block but explicitly marked lecture-not-exercise; the deck's own
  "lecture-essential vs exercise" slide excludes it. Optional: a 10-line cell feeding
  $\dot x$ as an enriched feature to the vanilla MLP to show the cheap end of the spectrum.
- [ ] **[P3] Spectral bias / smooth-activation requirement.** Deck flags ReLU's zero
  second derivative as "a classic, fatal student bug". The notebook MLP correctly uses
  `nn.Tanh()` but never *demonstrates* why. Optional: one contrast cell (Tanh vs ReLU
  PINN) showing the ReLU residual is garbage — cheap, high pedagogical payoff.
- [ ] **[P3] Equivariant / Hamiltonian "physics in the architecture" framing.** Deck §12
  treats HNN/equivariant nets as the (c) end of the taxonomy. Out of scope for a 90-min
  slot; at minimum frame Exercise 5's PINO explicitly as taxonomy slot (b→c) so the
  spectrum the deck builds is closed in the notebook narrative.

## ML-PC — Physics-constrained ML patterns for the lab

- [ ] **[P1] Two-axis result reporting (data error + constraint-violation rate).** The
  deck states this twice as "the only honest way to compare constrained methods" (Pattern
  A callout + Wrap point 3). Block 6 reports only the visual fit — no violation-rate
  metric. Add: for the tensile constrained vs unconstrained fits, report (i) data RMSE
  *and* (ii) a physical-violation rate (fraction of grid points with $d\sigma/d\varepsilon<0$
  in the elastic regime, plus $|\sigma(0)|$). Make the "5% lower RMSE but 10× violations
  is worse" point quantitatively.
- [ ] **[P1] Structural (hard) constraint vs soft penalty contrast.** Pattern D's headline
  is monotonicity-by-construction via cumulative non-negative increments
  ($\Delta\sigma=\mathrm{softplus}(\mathrm{NN}),\ \sigma=\sigma_0+\sum\Delta\sigma$),
  contrasted against the soft slope-penalty. Block 6 implements *only* the soft penalty.
  Add the structural variant on the same tensile data; show monotonicity is now exact
  (zero violations by construction) and discuss the "behaves badly near yield" soft caveat.
  Directly mirrors the MFML Lagaris soft-vs-hard story — a clean cross-course braid.
- [ ] **[P2] Pattern A — soft-constrained inverse problem with a known forward operator.**
  The deck's first full pattern (XRD phase-fraction inversion: simplex constraint +
  out-of-library penalty + forward-consistency term $\|R(f_\theta)-\text{pattern}\|^2$).
  Nothing in the notebooks does an *inverse problem with a forward-operator regulariser*.
  Add a compact analogue: recover a sparse "phase-fraction"-style vector from a synthetic
  linear forward operator, with simplex + out-of-support penalty, reporting both axes.
- [ ] **[P2] Gradient pathology — shown, not just narrated.** Deck §03 + §07 (Wang 2021)
  is a load-bearing lab lesson: $\nabla J_{\text{phys}}$ dominating $\nabla J_{\text{data}}$.
  Notebooks only state it in prose (Block 7 commentary). Add a diagnostic to any oscillator
  PINN: log $\|\nabla_\theta J_{\text{data}}\|$ vs $\|\nabla_\theta J_{\text{phys}}\|$
  per epoch, show the orders-of-magnitude imbalance, then demonstrate one fix (loss
  log-transform or curriculum: data-only warmup then ramp $\lambda$).
- [ ] **[P3] Constraint-conflict diagnosis ("don't suppress, diagnose").** Deck §07 makes
  this a named lab discipline (conflict ⇒ measurement error / wrong physics / leakage).
  Block 7 (wrong γ) is the natural host: frame the large data-MSE-vs-noise mismatch
  explicitly as the *diagnostic signal*, not just a curiosity — small prose+assert add.
- [ ] **[P3] Symmetry / equivariance (Pattern C).** Augmentation-vs-equivariant-net
  trade-off for the elastic tensor. Heavier lift; reasonable to leave as lecture-only,
  but note the gap so the notebook does not implicitly claim Pattern C coverage.

## MG — Uncertainty-aware discovery (GPs, ensembles, active learning)

- [ ] **[P1] Convex hull / energy-above-hull as the discovery objective.** §A (slides
  6-11) and exam statements #1-2 make E-hull *the* MG spine; "energy-above-hull is the
  discoverability signal; raw formation energy is not". The notebook optimises a physics
  oscillator — *no* hull, no E-hull, no composition objective anywhere. Add a block on a
  small synthetic (or `pymatgen`-derived) binary/ternary: build the lower convex hull from
  formation energies, compute $E_{\text{hull}}$, and use it as the target a surrogate
  predicts. This is the single biggest MG gap — without it the exercise is not a
  *discovery* exercise.
- [ ] **[P1] Acquisition functions: EI / UCB / Thompson on a candidate set.** §D (slides
  24-27) and exam statement #4 ("EI is the default; UCB more aggressive; Thompson batches")
  are core examinable MG content. Block 5 uses only raw `argmax σ` (pure-exploration
  variance-greedy), which the deck explicitly frames as the *naïve* baseline. Add EI
  ($\sigma[z\Phi(z)+\phi(z)]$) and UCB ($\mu+\beta\sigma$); compare regret/RMSE curves of
  argmax-σ vs EI vs UCB over the acquisition budget — the deck's core §D experiment.
- [ ] **[P1] Hull-aware acquisition vs raw-objective acquisition.** Slide 29 calls this
  "the materials-specific punchline of §D" and "the most pedagogically useful experiment
  in the unit": raw-$E_f$ BO re-discovers known compounds; hull-aware EI proposes novel
  candidates. Entirely absent. Add the contrast on the hull dataset from the P1 above —
  this is the experiment the MG deck most wants the exercise to deliver.
- [ ] **[P2] Calibration / reliability diagram on a held-out OOD slice.** §E (slide 38) +
  exam statement #5: "calibration on held-out data is non-negotiable; GPs well-calibrated
  in-distribution, miscalibrated under shift", and exercise task 4 is explicitly "the
  pedagogical core". The notebook plots ±2σ bands but never *checks* coverage. Add a
  reliability-diagram cell: bin held-out points by predicted σ, plot predicted vs
  empirical coverage, and report over/under-confidence on an extrapolation slice.
- [ ] **[P2] Deep ensembles vs GP as the UQ tool + the decision table.** §E (slides 34-37)
  and learning-outcome 6: when *not* to use a GP. Block 3 already builds a 5-network PINN
  ensemble for uncertainty — extend it into an explicit GP-vs-deep-ensemble UQ comparison
  ($\sigma$ shape, calibration, cost) and state the $n$-regime decision rule. Cheap given
  the ensemble already exists.
- [ ] **[P2] Aleatoric vs epistemic, made explicit.** §B slide 15 + exam statement #3:
  "GP gives epistemic for free via posterior variance; aleatoric is the likelihood term".
  The notebook's `SIGMA_N2 = NOISE_SIGMA**2` *is* the aleatoric term and the posterior σ
  *is* epistemic, but this is never named or decomposed. Add a short cell/markdown
  decomposing predictive variance = posterior (epistemic) + $\sigma_n^2$ (aleatoric),
  showing epistemic shrinks with active-learning queries while aleatoric does not.
- [ ] **[P3] GP $O(n^3)$ scaling / kernel-choice (Matérn-5/2) framing.** Deck slides
  19-22: Matérn-5/2 is "the materials-ML default" (RBF is too smooth); $O(n^3)$ wall.
  The notebook hard-codes an RBF kernel with hand-tuned ℓ. Optional: a one-line Matérn
  variant + a sentence on why RBF's infinite smoothness is unphysical for materials
  targets, and a timing note on the cubic cost.
- [ ] **[P3] Marginal-likelihood hyperparameter fitting.** Deck slide 19/20: ℓ and
  $\sigma_f^2$ "optimised by maximising the marginal likelihood — robust on small data".
  Notebook fixes `ELL = 1.6` by hand and Exercise 1 only does a manual sweep. Optional:
  add a true marginal-likelihood maximisation so the deck's "robust on small data" claim
  is exercised, not just hand-waved.

---

## Cross-cutting note

The exercise is genuinely strong on the **MFML×ML-PC PINN axis**: the homework→Thursday
arc on the damped oscillator exercises the PINN loss, collocation, inverse PINN, the
wrong-physics failure mode, and (in Ex 5) the PINO/operator extension that all three
decks reference. The **soft-vs-hard constraint** thread, however, is only half-built:
both MFML (Lagaris) and ML-PC (structural monotone increments) make *hard, by-construction*
enforcement the headline, and the notebooks implement *only soft penalties everywhere* —
closing the two [P1] hard-constraint items would braid that thread cleanly across both
courses with one shared idea.

The **MG lecture is braided at the wrong abstraction level**. The notebook teaches GP
regression *mathematics* (which the MG deck explicitly delegates upstream to MFML W12)
on a physics oscillator, while the actual examinable MG W13 spine — the **discovery loop,
convex hull / E-hull objective, EI/UCB/hull-aware acquisition, and calibration discipline**
— is essentially untouched. There is *no* off-by-one in the lecture *slug* (the notebook
correctly targets `13_uncertainty_aware_discovery_and_gaussian_processes`), but the
braided *content* is closer to MFML W12's GP material than to MG W13's discovery material.
Closing the three MG [P1] items (hull/E-hull objective, EI/UCB acquisition, hull-aware
vs raw acquisition) is what would make this a true three-lecture exercise rather than a
two-lecture (MFML×ML-PC) exercise with a GP appendix.
