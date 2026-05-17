# Week 9 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 9** · braided exercise =
`week9_latent_geometry.py` (main) + `week9_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-9 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/09_latent_spaces_advanced/01_intro.qmd` | Latent spaces & advanced representation learning (t-SNE, UMAP, contrastive, foundation embeddings) |
| ML-PC | `ml_for_characterization_and_processing/unit09_characterization_signals/10_characterization_signals.qmd` (delivered W10; folder index = delivered week post-2026-05-13) + companion `unit09b_transformers_for_materials/transformers_for_materials.qmd` | ML for characterization signals (PCA/AE/DAE/VAE/MAE/NMF on spectra) + Transformers for materials (ViT/Flash/Mamba) |
| MG | `materials_genomics/08_regression_and_generalization_in_materials_data/01_intro.qmd` (slug authoritative per REALIGNMENT_2026-05-13; folder 08 → Week 9, 09.06.2026) | Regression & generalization in materials data (split design, leakage, residual diagnostics, trustworthy reporting) |

**Caveats / corrections to the notebook's own framing:**

- **The notebook header mis-attributes the MG lecture.** `week9_latent_geometry.py` (lines 12–15) claims the MG triad partner is *"MG Unit 8 (delivered as W9): NN architectures for materials (SchNet/CGCNN/MEGNet/M3GNet)"*. Per `materials_genomics/REALIGNMENT_2026-05-13.md` the **folder index now equals the delivered week**: folder `08_regression_and_generalization_in_materials_data` → **Week 9** (09.06.2026); the SchNet/CGCNN/MEGNet/M3GNet architecture content is **MG Unit 9** (`09_neural_networks_for_materials_properties`), delivered **Week 10**. The actual MG W9 lecture is **regression & generalization** (split design, chemistry-family leakage, polymorph aliasing, baseline ladder, per-region residuals, the 7-point trustworthy-reporting checklist). The notebook's MG content (train a TinyCGNN, freeze, embed, probe) is *architecturally* aligned with MG **U9/W10**, not MG **U8/W9** — this is the dominant structural gap of the week and the reason the MG verdict below is WEAK.
- ML-PC W9 deck filenames are non-canonical: main deck is `10_characterization_signals.qmd` (not `01_intro.qmd`); companion is `transformers_for_materials.qmd`. AGENT_INSTRUCTIONS confirms `unit09_characterization_signals` is *delivered* W10 but is the **Week-9 braided partner** by curriculum number; the companion `unit09b` (ViT/Flash/Mamba) is explicitly *"theory in MFML W10"* and is a thin braided fit this week.
- MFML deck cleanly confirmed (title "Unit 9: Latent Spaces & Advanced Representation Learning").
- `ai4mat.datasets.CrystalGraphsDataset` is a **synthetic toy** generator (no lattice/PBC; toy formation energy from a known closed-form with prototype bias + cation/anion split). This bounds how far the MG regression-generalization content can be exercised, but it is *more* than enough for split-design / leakage / residual diagnostics — none of which the notebooks currently do.

## Verdict

- **MFML (latent spaces):** STRONG. PCA (hand-rolled, reconstruction-vs-k, anomaly), t-SNE (distance trap, perplexity), UMAP (n_neighbors sweep, runtime, global-vs-local), contrastive InfoNCE from scratch, MAE, linear probing, k-NN probe, DINOv2 foundation embedding + UMAP. Nearly the entire deck is exercised. Minor gaps: the four-desiderata rubric is never applied as an explicit measurement; CLIP/multimodal and "when foundation embeddings fail" are prose-only.
- **ML-PC (characterization signals):** STRONG on the spectra-latent half. Synthetic XRD → PCA → reconstruction-error anomaly (Block 2) and conv-AE bottleneck (Block 3) and MAE-on-spectra-analogue all land the deck's core PCA/AE/MAE-on-spectra story. **Gaps are real but second-tier:** no DAE (noisy→clean pair), no NMF (the deck's "natural choice for spectra"), no normalization-strategy step, no scree/elbow on spectra, and the **entire `unit09b` transformers companion is unexercised** (no ViT/patchify/SDPA — though MFML W10 owns that theory, so [P3]).
- **MG (regression & generalization):** WEAK — and mis-targeted. The notebook trains/freezes/probes a TinyCGNN as an *embedding* model (MG U9/W10 framing). **Almost none of the actual MG W9 lecture is exercised:** split-design taxonomy, chemistry-family leakage, polymorph aliasing, the mandatory baseline ladder, per-element / per-prototype residual tables, learning curves, leakage audits, the 7-point trustworthy-reporting checklist. The held-out-prototype probe (Block 5) and Exercise 3 (embedding-distance vs energy-distance) are the *only* faint contact points with "generalization", and neither is framed as a split-design / leakage / reporting exercise.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before the Week-9 session · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Latent spaces & advanced representation learning

- [ ] **[P2] Apply the four-desiderata rubric as a measurement, not prose.** The deck's spine is the rubric *compactness / separation / smooth interpolation / transferability* (slide "What is a good latent space?"), and "critique a latent space against the four desiderata" is learning-outcome #1. The notebook produces five embeddings (PCA, AE, contrastive, MAE, DINOv2) but never scores any of them on the rubric. Add a short Block-6 coda: a small table scoring AE-bottleneck vs contrastive vs MAE on (i) within-class compactness, (ii) silhouette/separation, (iii) linear-probe transfer — turning the existing embeddings into a rubric measurement.
- [ ] **[P3] Latent-space arithmetic / interpolation demo.** [SUPPORT] Two deck slides ("Three ways to shape a latent", "Latent space arithmetic — promise and limits") are prose-only. Optional: interpolate $z_A \to z_B$ in the conv-AE latent and decode the path, showing smooth-vs-"dead-region" behaviour — directly delivers the deck's "smooth interpolation" desideratum and the Unit-11 VAE bridge.
- [ ] **[P3] CLIP / multimodal & "when foundation embeddings fail".** [SUPPORT] Both are deck slides; only the success path (DINOv2 stretch ex.) is exercised. The deck's explicit *failure* cases (OOD modalities, specialized invariances) are never shown. Optional: embed Ising with DINOv2 and contrast with NEU-DET to make the "natural-image encoder is OOD for atomic-resolution / synthetic textures" point concretely.
- [ ] **[P3] PPCA / SVD-vs-eigendecomp framing.** [SUPPORT] Homework Part A computes PCA via covariance eigendecomp; the deck repeatedly motivates PCA via the PPCA generative view and the SVD path. One markdown cell + a `torch.linalg.svd` cross-check would close the deck's "From SVD to PCA" anchor referenced in the homework prose.

## ML-PC — ML for characterization signals (+ transformers companion)

- [ ] **[P1] Denoising autoencoder (DAE): noisy-in / clean-target on spectra.** [CORE] — a full deck section (slides 21–22, "Learning the Clean Manifold", Fe-L EELS case). The notebook trains a *standard* conv-AE on images only; the DAE noisy→clean pair, the central ML-PC W9 denoising story, is **absent**. Add a block to the synthetic-XRD pipeline in Block 2: corrupt spectra with Poisson/Gaussian noise, train an AE with clean targets, compare denoised reconstruction vs truncated-PCA denoising (the deck's explicit DAE-vs-PCA-denoising contrast).
- [ ] **[P1] NMF as the physically-natural spectral factorization.** [CORE] — the deck calls NMF "the natural choice for spectra" (slides 13, 34, 39: non-negative end-members + abundance maps, NMF-beats-PCA-for-interpretability). Nothing in either notebook uses NMF. Add a short block on the synthetic-XRD set: `sklearn.decomposition.NMF`, show non-negative end-member spectra vs PCA eigenspectra (which go negative), and recover per-prototype abundances.
- [ ] **[P2] Scree plot / intrinsic-dimensionality / elbow on spectra.** [CORE] (deck slides 9–10). Homework Part A does reconstruction-MSE-vs-k on *Ising images*; the deck's spectra-specific scree/eigenvalue-elbow + "intrinsic dim ≈ number of phases" argument is never made on the XRD spectra of Block 2 (where there are exactly 5 prototypes — a clean elbow demo). Add the eigenvalue scree plot of `eigvals_spec` and annotate the expected ~5-component elbow.
- [ ] **[P2] Normalization strategy as a pre-DR decision.** [CORE] (deck slide 31: total-count / peak-height / SNV / per-channel; "without normalization PCA learns intensity, not chemistry"). Block 2 feeds raw `synth_xrd` intensities straight into PCA. Add a contrast: PCA on raw vs total-count-normalized spectra, showing the chemistry (mean-Z) axis only emerges after normalization.
- [ ] **[P3] `unit09b` transformers companion — at least one patchify/SDPA touchpoint on a spectrum/image.** The companion deck (ViT, Flash/SDPA, anti-patterns) is wholly unexercised. The TinyMAE block (6b) already builds a 2-layer transformer encoder with patchify — frame it explicitly as the ViT/`F.scaled_dot_product_attention` exemplar (deck slides 4–5, anti-pattern (c) "no positional embeddings"), or add the deck's positional-embedding-ablation sanity check on TinyMAE. [P3] because the companion explicitly defers transformer *theory* to MFML W10.
- [ ] **[P3] Sim-to-real / synthetic-to-real transfer for spectra.** [SUPPORT] Deck slide 36 (pre-train on simulated spectra, fine-tune on real). The notebook only ever uses one synthetic XRD source. Optional: a "clean synthetic → noisy 'experimental' XRD" fine-tune step reusing the Block-2 generator with a heavier noise model.

## MG — Regression & generalization in materials data

- [ ] **[P1] Re-frame the MG block to the actual W9 lecture (regression & generalization), not U9/W10 architectures.** The whole MG arc in Blocks 4–5 + Ex 1–4 is an *embedding/representation* exercise (MG U9/W10). The MG W9 deck is about **trustworthy regression evaluation**. The single highest-priority fix: add an explicit MG-W9 regression block on `CrystalGraphsDataset`'s toy formation energy — train Magpie-style composition baseline + the TinyCGNN regressor, then run the lecture's diagnostics below. Re-title the Block 4 markdown so students are not told MG W9 = SchNet/CGCNN.
- [ ] **[P1] Split-design taxonomy: random vs group-aware (prototype-held-out / cation-family).** [CORE] — the deck's central message ("the test set's relationship to the training set *is* the scientific claim"; §B taxonomy, §C splits, slide 7 "random-fold CV estimates the wrong quantity"). The notebook uses a single random 80/20 split for CGNN training (Block 4) and the probe. Add the canonical contrast: random-split MAE vs prototype-held-out (and vs cation-element-held-out) MAE on the toy formation energy, reporting the gap as $\Delta_{\text{shift}}$ — the literal §A5 "fourth term" demo, exactly what the afternoon's MG exercise is specified to produce.
- [ ] **[P1] The mandatory baseline ladder.** [CORE] (§D29: constant → composition-linear → composition-GBT → structure-aware → GNN; "anything that doesn't beat tier 0 is broken"). The notebook reports CGNN MSE/MAE with no baseline at all. Add tiers 0–2 (training-mean constant; ridge on a composition feature vector; GBT) alongside the TinyCGNN, on the same split — this is also the deck's composition-vs-structure parity story (§31).
- [ ] **[P1] Polymorph aliasing / structure-awareness ablation.** [CORE] (slide 11; §F1/44 mandatory ablation: randomize positions, if MAE barely moves the model is composition-only in disguise). `CrystalGraphsDataset` has the same prototype appearing with different geometries — ideal for this. Add the ablation: shuffle/zero `edge_distance` (and/or collapse to composition-only features) and show the CGNN MAE collapse-or-not, demonstrating whether structure is actually used.
- [ ] **[P2] Per-region residual diagnostics (per-prototype / per-element table).** [CORE] (§E37–39: per-element, per-prototype, per-space-group MAE + signed bias; "global MAE hides localized failure"). Block 6 reports only aggregate metrics. Add a per-prototype MAE/bias table for the toy-formation-energy regressor — directly the deck's §E recipe and a checklist item.
- [ ] **[P2] OOD-vs-interpolation distance diagnostic.** [CORE] (slide 13: k-NN min-distance to training set, residuals vs distance quartile). The Block-4 embeddings + toy targets make this a 10-line add: bin test crystals by min-distance-to-train in embedding (or feature) space, report MAE per quartile, show the extrapolation tail.
- [ ] **[P2] Confidence interval on the headline metric.** [CORE] checklist item #6 (§B19 small-test-set problem; "point MAE without CI" is named anti-pattern #3-adjacent). Every reported number in the notebooks is a single point estimate. Add a bootstrap CI on the regressor's test MAE.
- [ ] **[P3] Leakage audit demo (preprocessing-before-split / dedup).** [CORE] §D32/§F: standard-scaler-fit-on-full-data, polymorph leakage. Optional but high-pedagogy: a deliberate "fit normalization on full data then split" vs "fit on train only" contrast showing the inflated test metric — the deck's most concrete leakage anti-pattern.
- [ ] **[P3] The 7-point trustworthy-reporting checklist as the block's closing rubric.** [SUPPORT→CORE-for-exercise] §F47. Add a short closing markdown that scores the (newly added) MG block against the deck's 7-point checklist — mirrors the afternoon MG exercise's stated 5/7 rubric and makes the braid explicit.

---

## Cross-cutting note

The exercise is excellently braided on the **MFML × ML-PC** axis: the *same*
PCA / t-SNE / UMAP / reconstruction-error / linear-probe / MAE machinery is
applied to images (MFML), to spectra (ML-PC characterization signals), and to a
trained network's penultimate layer — this is a genuine three-lens latent-space
exercise and the MFML and ML-PC core stories are largely covered.

The **MG leg is both thin and mis-targeted**. The notebook braids the MG
*architecture* lecture (U9/W10: CGNN as embedding model) instead of the MG
*Week-9* lecture (U8: regression & generalization). The toy
`CrystalGraphsDataset` is, ironically, almost perfectly suited to the real MG
W9 content — it has prototypes (→ prototype-held-out splits), a cation/anion
split (→ chemistry-family leakage), repeated prototypes at varied geometry
(→ polymorph aliasing & the structure-awareness ablation), and a closed-form
toy target (→ baseline ladder, residual diagnostics). Closing the four MG
[P1] items — re-framing the block, adding the split-design contrast, the
baseline ladder, and the polymorph/structure ablation — is what would make
Week 9 a true three-lecture braided exercise rather than a two-and-a-half.
