# Week 11 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 11 (lecture 23.06.2026)** · braided exercise =
`week11_generative_inverse_design.py` (main) + `week11_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-11 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/11_generative_vae_diffusion/01_intro.qmd` | Generative Models — VAE & Diffusion (flow matching, consistency) |
| ML-PC | `ml_for_characterization_and_processing/unit08_inverse_problems/09_inverse_problems.qmd` (filename non-canonical; **delivered W9, not W11** — see caveat) | Inverse Problems & Process Maps |
| MG | `materials_genomics/10_representation_learning_and_feature_discovery/01_intro.qmd` (slug authoritative; "Unit 10 delivered as W11" label is the *true* W11 per realignment) | Representation learning + latent spaces (merged) |

**Caveats — read before the verdict:**

- **ML-PC misalignment (calendar-vs-topic).** The notebook header self-declares it
  braids "ML-PC Unit 11 (`unit08_inverse_problems`)". That parenthetical is internally
  contradictory and the calendar resolution is the bad one: per
  `ml_for_characterization_and_processing/AGENT_INSTRUCTIONS.md`, the **true
  calendar ML-PC Week-11 lecture is `unit10_automation`** (Automation in
  microscopy — control theory, RL, autofocus/beam alignment, self-driving labs,
  conformal classification), delivered W11. `unit08_inverse_problems` is the
  **W9** lecture (the unit-folder→week slip: "delivered W9"). The notebook
  braids the *topically correct* inverse-design lecture (Unit 8) and **ignores
  the lecture ML-PC students actually have on 23.06**. The audit below scores
  ML-PC against Unit 8 (what is braided) and flags what the true-W11 automation
  lecture would additionally demand at the end of the ML-PC section.
- **MG label is off-by-one in *number* but correct in *slug*.** The notebook
  says "MG Unit 10 (delivered as W11): latent spaces of materials —
  composition–structure–property maps, latent-space arithmetic for property
  targeting". Per `materials_genomics/REALIGNMENT_2026-05-13.md`, the true W11
  MG lecture *is* folder `10_representation_learning_and_feature_discovery`
  ("Representation learning + latent spaces (merged)", which absorbs
  `11_latent_spaces_of_materials`). So the **slug/folder the notebook points at
  is the correct W11 deck** — unlike earlier weeks, there is no folder
  off-by-one here. **But the *content* the notebook braids is the wrong half**:
  the notebook's Block 5 frames MG-W11 as *latent-space arithmetic for property
  targeting / inverse design* and cites MG §"Latent-space arithmetic for
  property targeting" and §"Composition–structure–property maps". The true W11
  deck explicitly defers generation/inverse-design to **MG U12** (slide 49,
  "Bridge to Unit 12 — Generative Models for Discovery") and slide 48
  ("Interpreting the Latent Space") is *interpretation* (what do the axes mean),
  **not** property-targeted generation. The notebook is effectively braiding
  MG-U12 content under the W11 label. The genuine W11 MG payload —
  crystal-vs-image priors, SSL pretext tasks, contrastive positive-pair physics,
  the foundation-model menu, and above all the **embedding-diagnostics stack
  (linear probe, random-init baseline, retrieval, "pretty t-SNE / dead
  downstream")** — is almost entirely unexercised. This is the dominant gap of
  the week and is flagged P1 below.

## Verdict

- **MFML (VAE & diffusion / flow matching):** STRONG. The load-bearing spine —
  VAE/ELBO/reparameterisation (homework), conditional generation, the
  DDPM→flow-matching pivot with a real trained ODE-velocity U-Net and 10-step
  Heun sampler, consistency distillation as a stretch — is genuinely exercised
  in code, not prose. Gaps are in the diffusion *mechanics* (forward marginal,
  ε-prediction, classifier-free guidance, schedule) that the deck makes [CORE]
  but the notebook only narrates.
- **ML-PC (inverse problems — the braided Unit 8):** PARTIAL. The
  generative-inverse-design half of Unit 8 (slides 43–47: flow matching,
  consistency models, conditional sampling, the averaging trap) is well
  covered. The *rest* of Unit 8 — ill-posedness/Hadamard, non-uniqueness as the
  reason standard NNs fail, regularisation as prior, Bayesian posterior,
  process maps / safe operating windows — is absent (out of scope for a
  generative notebook, noted but not penalised heavily). **Against the true
  calendar W11 (`unit10_automation`): WEAK→ABSENT** — none of control/RL/
  autofocus/self-driving-labs/conformal-classification is touched.
- **MG (representation learning + latent spaces — true W11):** WEAK. Block 5 is
  a single hand-rolled TinyCGNN PCA + linear stability-axis walk. It touches
  *latent-space arithmetic* (which the true deck assigns to U12) but exercises
  essentially **none** of the actual W11 [CORE] content: crystal-specific
  priors, SSL pretext tasks, contrastive positive/negative-pair physics,
  foundation-model selection, and the entire embedding-diagnostics discipline
  (linear probe vs random-init baseline, retrieval, the "pretty t-SNE" failure
  mode). Block 5 is nominally MG-flavoured but lecture-orthogonal.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before 23.06 · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Generative models: VAE & diffusion

- [ ] **[P1] Diffusion forward process: closed-form marginal & noising, in code.** The
  deck makes $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\epsilon$ and the
  variance-preserving $\sqrt{1-\beta_t}$ scaling a protected [CORE] derivation
  ("the slide that makes diffusion trainable"). Block 4 jumps straight to flow
  matching and keeps DDPM as a 2-line prose anchor. Add a short block: pick a
  linear (or cosine) $\bar\alpha_t$ schedule, visualise the forward noising of
  one Cahn–Hilliard image at $t\in\{0,T/4,T/2,3T/4,T\}$, and assert
  $\mathrm{Var}(x_t)\!\approx\!1$ numerically — the variance-preserving check the
  deck does on the board. Currently no DDPM forward path exists anywhere.
- [ ] **[P1] ε-prediction DDPM training loop + the "predict ε ⇔ know x₀" equivalence.**
  [CORE], the crux slide of the diffusion half ("predict $\epsilon$ ⇔ know
  $x_0$ ⇔ can take the reverse step"). The notebook trains a flow-matching
  velocity net but never a noise-prediction net, so students never run the
  5-step DDPM algorithm the exam explicitly tests. Reuse the existing `TinyUNet`
  with the same backbone, swap the FM target for the ε-MSE loss, train ~10
  epochs, and contrast its sample quality/step-count against the flow-matching
  model already in Block 4. This makes the deck's central "many easy steps,
  trained by MSE" claim a measured result.
- [ ] **[P1] Classifier-free guidance on the velocity/noise field.** [CORE] and
  the deck's single most important *practical* slide ("the difference between
  diffusion that ignores your prompt and Stable Diffusion"). Block 2's
  conditioning is CVAE-only; Block 4's flow matcher is unconditional. Add a
  conditional flow-matching variant trained with ~10% condition dropout and a
  sampling-time guidance scale $w$; sweep $w\in\{0,1,3,6\}$ and plot the
  diversity↔on-target trade-off (the deck's $w$ trade-off, and the ML-PC §43
  "conditional flow matching" callout simultaneously). This is the single
  highest-ROI addition: it closes an MFML [CORE] gap *and* the ML-PC
  inverse-design braid in one block.
- [ ] **[P2] Reparameterisation trick as a gradient-flow demonstration.** [CORE],
  the most-asked VAE exam question per the deck notes ("why reparameterisation?
  — differentiable sampling, not noise injection"). The homework *uses*
  `mu + exp(.5*log_var)*eps` but never demonstrates *why*. Add a 6-line cell to
  the homework helpers: show `torch.randn` blocks the gradient
  (`.grad is None`) vs the reparameterised path carries it — make the
  exam-critical point experimentally visible.
- [ ] **[P2] Posterior collapse — measured, not asserted.** [CORE] ("the #1 VAE
  failure in practice"; the deck wants the KL term *logged separately* and
  driven to ~0 at high β). Block 6 lists posterior collapse as exercise prose
  only; the homework β-sweep stops at β=4 and reports recon/KL but never names
  the collapse. Add an explicit high-β (β=8) run that plots KL→0 *with* degraded
  recon and labels it posterior collapse, plus the KL-warm-up mitigation
  (anneal β 0→1) as a one-cell remedy.
- [ ] **[P3] Latent diffusion = VAE + diffusion in one model.** [CORE] synthesis
  slide ("this is Stable Diffusion"; "both halves of today's lecture in one
  model"). The notebook trains a VAE and a flow matcher on the *same* CH data
  but never composes them. Stretch block: run the flow matcher in the trained
  VAE's 8-D latent instead of pixel space; note the cost argument (64× fewer
  dims) the deck makes.
- [ ] **[P3] Normalizing flows / exact-likelihood niche (Boltzmann generators).**
  [SUPPORT] but a deck [CORE]-table column ("the only model with exact
  likelihood *and* exact sampling"; Boltzmann generators for crystal-phase free
  energies — directly materials/thesis-relevant). Mentioned nowhere. At minimum
  a reading/stretch task framing the VAE↔diffusion↔flow trade-off table.
- [ ] **[P3] DDIM / deterministic fewer-step sampling & the quality-vs-NFE curve.**
  [SUPPORT]/[CORE]-adjacent (the "training and sampling are decoupled" point).
  The Heun sampler is fixed at 10 steps; add a steps∈{2,5,10,25} sweep on the
  trained FM model showing the quality-saturates-then-flatlines curve the deck
  plots, and state the DDIM "same weights, different sampler" decoupling.

## ML-PC — Inverse problems (braided Unit 8; see calendar caveat)

- [ ] **[P1] Non-uniqueness & the averaging trap — the reason a plain NN regressor fails.**
  [CORE] of Unit 8 (slides 04–05: many-to-one forward ⇒ one-to-many inverse;
  MSE-trained inverse net predicts the *mean of valid solutions*, which is
  itself invalid). The notebook's whole premise (CVAE / latent-GD / FM as the
  *fix*) is unmotivated without first showing the failure. Add a short block:
  train the existing `EnergyRegressor` *backwards* (energy → image via a small
  deconv net, MSE) for a target energy with multiple valid microstructures;
  show the blurred, averaged, off-manifold output — then point at Block 2/3 as
  the cure. This is the conceptual hinge of the braided lecture and is missing.
- [ ] **[P2] Conditional flow matching for inverse design (ML-PC §43, explicit).**
  [CORE] — the ML-PC slide the notebook *names* in Block 4's prose ("see ML-PC
  §Flow-matching microstructure inverse design"). Block 4's flow matcher is
  unconditional, so the braid is asserted, not delivered. Covered by the MFML
  classifier-free-guidance P1 above (shared block); cross-listed here because it
  is the ML-PC §43 [CORE] payload, not just an MFML nicety.
- [ ] **[P2] Regularisation as prior / Bayesian posterior framing of inverse design.**
  [CORE] of Unit 8 (slides 09–12: $\arg\min \|f(x)-y\|^2+\lambda R(x)$; Gaussian
  prior ⇔ Tikhonov; MAP ⇔ regularised solution). Block 3's latent-GD is
  *exactly* a regularised inverse solve (the VAE prior is the implicit
  regulariser) but the notebook never names this. Add 3–4 lines of prose +ONE
  experiment: add an explicit latent-norm penalty $\lambda\|z\|^2$ to
  `latent_gd`, sweep $\lambda$, and show the data-fidelity↔prior trade-off —
  making the deck's regularisation slide a measured curve and naming the
  Bayesian-MAP interpretation.
- [ ] **[P3] Honest OOD / non-identifiability framing (ML-PC §46).** [CORE]
  ("true non-uniqueness cannot be resolved by better algorithms — it is a
  property of the physics"). Block 6's OOD demo is good but is framed as a
  *generative* failure; add one sentence + the existing regressor check
  reframed as the Unit-8 "fundamentally unidentifiable from available
  measurements" point, and have Exercise 2 report it as such.
- [ ] **[P3] Calendar-W11 reality check (`unit10_automation`).** Not a notebook
  block — a lecturer note. The students sitting the 23.06 ML-PC lecture get
  **Automation in microscopy** (control theory, RL, autofocus/beam alignment,
  self-driving labs, conformal classification), *not* inverse problems. Nothing
  in this exercise braids that lecture. If the triad is meant to track the
  calendar, either (a) re-anchor the notebook's ML-PC strand to
  `unit10_automation` (a *different* exercise — closed-loop tuning, conformal
  prediction sets), or (b) accept that the ML-PC braid is topic-aligned to W9
  and document the slip in the notebook header (currently it claims "Unit 11").
  Flagging only; no code change proposed without a scope decision.

## MG — Representation learning + latent spaces (true W11)

- [ ] **[P1] Embedding diagnostics: linear probe vs random-init baseline.** [CORE]
  and the conceptual centre of the true W11 deck (§F, slides 41–42: "probe
  before projecting"; the random-init comparison is "the most-omitted
  comparison in published work"). The notebook never probes the TinyCGNN
  embedding at all. Add a block: freeze the trained CGNN encoder, fit a linear
  head on its embeddings to predict formation energy on a *held-out prototype*,
  and compare against (i) a random-init CGNN encoder and (ii) the raw
  composition/Magpie features already available — the four-row table the deck
  says the exercise *should* produce. This is the single largest true-W11 gap.
- [ ] **[P1] Nearest-neighbour retrieval as the honest diagnostic.** [CORE] (slides
  30, 43: "retrieval is honest, t-SNE is decorative"; generalises directly to
  the U13 discovery loop). Block 5 only PCA-scatters embeddings. Add: pick ~10
  query crystals, retrieve k-NN in full embedding dimension, report
  same-prototype / same-property-band hit-rate (precision@k). Cheap, and it is
  the diagnostic the deck trusts most.
- [ ] **[P1] The "pretty t-SNE / dead downstream" failure mode, demonstrated.**
  [CORE] (slides 45–46, the deck's signature anti-pattern: a clean projection
  that clusters on cell-size/atom-count metadata while the probe is at chance).
  Construct it: show the PCA/embedding scatter *looks* structured, then show a
  linear probe of a metadata artefact (e.g. number of atoms / prototype id)
  scores high while the property probe is weak — the exact lesson the deck
  builds §F around. Pairs naturally with the P1 probe block.
- [ ] **[P2] Why a crystal embedding ≠ an image embedding (priors).** [CORE]
  (slides 08–12: chemistry typing, periodicity/PBC supercell invariance,
  equivariance). TinyCGNN uses raw scalar distance and a generic element
  embedding with no invariance statement. Add an empirical PBC/supercell
  invariance check (duplicate a crystal graph 2×2×2, confirm the mean-pooled
  embedding is unchanged up to scaling) and a rotation/permutation invariance
  check — the deck's slide-11/12 [CORE] claims, currently unexercised.
- [ ] **[P2] Contrastive positive-pair physics (what is / is NOT a positive pair).**
  [CORE] and the deck's "deepest pedagogical hinge" (slide 25: element
  substitution is *not* a positive pair; the silent labelling bug with no
  pretraining-metric symptom). No contrastive setup exists. Add a minimal
  InfoNCE on CGNN embeddings using *valid* augmentations (supercell, rotation,
  thermal jitter) as positives, then a deliberately *broken* run that uses
  element substitution and show the downstream probe collapses (NaCl/KCl-style)
  — the canonical demonstration the deck asks for.
- [ ] **[P2] SOAP/Magpie baseline vs learned embedding decision (slide 47).** [CORE]
  ("'always use the foundation model' is wrong; the answer is task-driven").
  The notebook never compares the learned embedding to an engineered baseline.
  Fold into the P1 probe table: add the Magpie/composition-feature row and a
  small-N vs large-N comparison so the deck's four-on-four decision rule is a
  measured result, not a slide.
- [ ] **[P3] Latent-space *interpretation* (axis ↔ descriptor correlation).**
  [CORE]-adjacent (slide 48 — the *actual* W11 latent-space content, distinct
  from U12 generation). Block 5 does a *property-targeting walk* (U12 framing);
  add the W11 framing instead/also: correlate each embedding PC against known
  descriptors (mean atomic mass, electronegativity, formation energy) and
  report which axis aligns with what — the deck's interpretation example.
- [ ] **[P3] Frame Block 5 honestly as MG-U12 content, not W11.** Lecturer/header
  note. Block 5's stability-axis walk and the notebook's "MG W11 = latent-space
  arithmetic for property targeting" framing are MG **Unit 12** (Generative
  Models & Inverse Design, the W12 lecture), not W11. Either relabel Block 5 as
  "MG U12 preview" or re-scope it to the true W11 diagnostics content (P1 items
  above). At minimum fix the notebook header's MG bullet so it stops asserting
  a braid the W11 deck explicitly defers to U12.

---

## Cross-cutting note

The exercise is genuinely strong on the **MFML generative-machinery ↔ ML-PC
inverse-design** axis *for the lecture it chose to braid* (ML-PC Unit 8 §43–47):
flow matching, consistency distillation, conditional sampling and the OOD
honesty discipline are all exercised in code. Two structural problems blunt it
as a *triad-week-11* exercise. (1) **The ML-PC strand is calendar-misaligned**:
it braids the W9 inverse-problems lecture while ML-PC students' actual 23.06
lecture is automation in microscopy — a different exercise entirely. (2) **The
MG strand braids the wrong half of the right deck**: it does U12-style
latent-space *arithmetic for property targeting* while the true W11 MG lecture
is representation learning + the **embedding-diagnostics discipline** (linear
probe vs random-init, retrieval, the "pretty t-SNE / dead downstream" failure)
— almost none of which is exercised. Closing the MG [P1] items (probe +
random-init baseline + retrieval + the t-SNE failure-mode demo) and the MFML
[P1] diffusion-mechanics + classifier-free-guidance items is what would turn
this from a strong two-lecture (MFML×ML-PC-W9) exercise into a true
three-lecture Week-11 exercise. The two calendar/scope misalignments need a
lecturer decision, not just a code patch — they are flagged, not silently
audited around.
