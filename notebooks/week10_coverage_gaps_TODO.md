# Week 10 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 10** · braided exercise =
`week10_attention_and_transfer.py` (main) + `week10_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-10 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/10_attention_transformers/01_intro.qmd` | Attention & Transformers (self-attention → ViT → foundation models) |
| ML-PC | `ml_for_characterization_and_processing/unit09b_transformers_for_materials/transformers_for_materials.qmd` (companion deck the notebook explicitly braids) | Transformers for materials (ViT, Flash Attention, Mamba) on a 1080 Ti budget |
| MG | `materials_genomics/09_neural_networks_for_materials_properties/01_intro.qmd` (slug authoritative per `REALIGNMENT_2026-05-13.md`; folder index = delivered week) | Neural networks for materials properties (SchNet/CGCNN/MEGNet/ALIGNN/M3GNet/NequIP/MACE; symmetries; equivariance) |

**Caveats / resolution decisions:**

- **ML-PC braid is correct as stated.** Per `AGENT_INSTRUCTIONS.md`, `unit09b_transformers_for_materials` is a *companion deck within ML-PC Week 9*, and `unit10_automation` ("Automation in microscopy and characterization") is the calendar-W10 ML-PC lecture but covers unrelated material (self-driving microscopes, RL/active learning). The notebook header explicitly chooses to braid 9b and to stay off Unit 10 Automation — this is the right call and Automation is **not** counted as a gap here.
- **MG braid is mislabeled in the notebook.** Both notebook headers claim the MG braid is *"MG Week 10 — Representation learning and feature discovery."* Per `REALIGNMENT_2026-05-13.md` (slugs authoritative, folder index = delivered week), the **delivered MG Week 10 (16.06.2026)** lecture is `09_neural_networks_for_materials_properties` ("Neural Networks for Materials Properties"). "Representation learning and feature discovery" is MG **Week 11** (`10_representation_learning_and_feature_discovery`, 23.06.2026). The notebook braids against the *wrong, next-week* MG lecture. This audit scores against the **actually delivered** MG W10 deck and treats the mislabel as the top cross-cutting issue.

## Verdict

- **MFML (attention & transformers):** STRONG. The conceptual spine — scaled dot-product attention, multi-head, permutation-equivariance, sinusoidal PE, the residual+LN transformer block, ViT patchify/CLS, ViT-vs-CNN data efficiency, attention-as-saliency, SDPA/Flash — is all genuinely exercised across homework + notebook. Residual gaps are pedagogical-mechanic (the √d_k *variance* argument, the hand-trace exam item, the "attention communicates / MLP computes" isolation).
- **ML-PC (transformers for materials):** STRONG on the transformer-mechanics half (SDPA dispatch, the explicit Pascal/1080 Ti caveat, naive-MHA-vs-SDPA wall-clock+VRAM bench, the three anti-patterns, ViT-vs-baseline). **The materials-application half is absent**: no 4D-STEM diffraction ViT, no LPBF cross-attention over a layer stack — the deck's two signature materials applications. Ising/Cahn-Hilliard/tensile stand in for diffraction/spectra.
- **MG (neural networks for materials properties):** WEAK / MISALIGNED. The notebook braids the wrong MG lecture. The delivered MG W10 deck is entirely about *graph/equivariant networks on atomic systems* (the four symmetries, invariance vs equivariance, SchNet/CGCNN/ALIGNN/MACE, autograd forces, extensive/intensive pooling). **None** of that machinery is in the notebook. Genuine contact exists only with §G/§43/§48 (transformer-on-crystals framing, pretrain→fine-tune recipe, "Unit 9 produces the embedding"), which the Block 5/6 transfer experiment touches *by analogy* on image data.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before delivery · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Attention & Transformers

- [ ] **[P2] The √d_k scaling: demonstrate the variance argument, don't just divide.** [CORE], flagged by the deck as "the highest concept-density slide" and an exam item. Both notebooks hard-code `/ np.sqrt(d_k)` but never *show* why. Add a 6-line block: sample random Q,K with iid unit-variance entries at d_k ∈ {4, 64, 512}, plot Var(qᵀk) ≈ d_k, then show the softmax collapsing to near-one-hot (and its Jacobian → 0) without scaling vs staying responsive with it. This is the single most exam-relevant MFML gap.
- [ ] **[P2] Hand-trace one attention computation, verify against autograd.** [CORE exam mechanic] — the deck names this the most-failed exam item and explicitly requests a "notebook 2: hand-trace vs autograd". Homework Part A jumps straight to the vectorised function. Add a tiny n=3, d_k=2 worked numeric trace (the deck's A/B/C/D toy values) with an `assert torch.allclose` against `scaled_dot_product_attention`.
- [ ] **[P2] "Attention communicates, the MLP computes" — isolate it.** [CORE] mnemonic and classic exam discriminator; the `TransformerBlock` implements both sublayers but the notebook never demonstrates that *only* attention moves information between tokens. Add a probe: zero/ablate the MLP sublayer vs ablate the attention sublayer on a permuted-token input and show only the attention path is position-mixing. Cheap (reuse `TransformerBlock`).
- [ ] **[P3] Head specialization is shown only untrained.** Homework Part C plots four heads with *random* projections; the deck's point is that specialization is *emergent after training*. Add a one-figure follow-up in notebook Block 4: re-plot the four trained `vit` heads' CLS-attention and comment on whether any head specializes (texture vs domain-edge) — closes the "emergent, not assigned" thread.
- [ ] **[P3] Learned vs sinusoidal positional encoding.** [SUPPORT] trade-off the deck makes (learned = best in-distribution, cannot extrapolate; the ViT resolution-change pitfall). Block 2 only does sinusoidal; RoPE is stretch Ex 4. Add a learned `nn.Embedding(T+1, d_model)` PE variant to `TinyViT` and a one-line accuracy contrast — also sets up the deck's "ViT trained at 224 breaks at 384" warning.
- [ ] **[P3] Depth / stacking effect.** [CORE] "depth = abstraction; keep L small for materials." `TinyViT` fixes `n_blocks=2`; nothing sweeps it. Add a small n_blocks ∈ {1,2,4} sweep on Ising-light and tie to the deck's "L=4–12 plenty for materials, deeper overfits small data" guidance.
- [ ] **[P3] Pre-norm vs post-norm.** [SUPPORT]. `TransformerBlock` is pre-norm (correctly the modern default) but the contrast with post-norm (needs LR warmup, less stable) is never shown. Optional: a post-norm variant + a training-stability curve. Low priority for the time budget.

## ML-PC — Transformers for Materials

- [ ] **[P1] No materials-application payoff — substitute a 4D-STEM / diffraction-style ViT.** [CORE]: ViT-on-4D-STEM (slide 06) is the slide the deck says "pays off the whole unit", and the deck's own exercise preview (slide 11) is *"Tiny ViT on 4D-STEM patches, bench it"*. The notebook runs ViT only on Ising microstructure. Add a block (or reframe Block 3) that builds a small synthetic CBED / diffraction-pattern dataset (paired Bragg discs across a central beam) and shows the *non-local* relationship a CNN cannot capture in one layer — this is the deck's central materials argument and is currently entirely on Ising.
- [ ] **[P2] Pretrain/MAE vs from-scratch ViT on a *small* materials set — actually run the anti-pattern.** [CORE] anti-pattern (b): "ViT from scratch on ~10³ labels underperforms ResNet18; pretrain with MAE then fine-tune." Notebook Block 3 only *discusses* the data-efficiency crossover in prose and trains both at one data size. Add a small-N sweep (e.g. N ∈ {200, 1000, 5000}) showing CNN ≥ ViT at small N and the gap closing — make the anti-pattern experimental, not narrated.
- [ ] **[P2] LPBF-style cross-attention over a layer/section stack.** [CORE] materials application 2 (slide 07): Q from the current layer, K/V from the previous k layers; the deck explicitly warns to use cross-attention (not self-attention on a concatenated stack) to keep cost linear in k. Nothing in the notebook does cross-attention at all (only self-attention everywhere). Add a minimal cross-attention block on a stacked-sequence analogue (e.g. successive Cahn-Hilliard time frames as the "layer stack") — this is the only braid that exercises *cross*-attention.
- [ ] **[P3] Make the SDPA bench the main exercise deliverable, with VRAM reported as the deck asks.** Block 3b already benches naive-MHA vs `nn.MultiheadAttention` vs SDPA and nails the Pascal/1080 Ti caveat (excellent, fully covers slides 05/09/10a). The deck's exercise (slide 11) wants val-accuracy + wall-clock/epoch + `torch.cuda.max_memory_allocated` reported *for the trained model*, not just a synthetic-tensor microbench. Promote the bench into the trained ViT path so peak-VRAM is reported on the real model. Low priority — the mechanics are already covered.

## MG — Neural Networks for Materials Properties

> The notebook braids MG "representation learning" (which is MG **Week 11**), not the delivered MG **Week 10** "Neural Networks for Materials Properties". The items below are the genuine W10 gaps. Several are deep architectural topics (graph/equivariant nets) that are out of scope for an attention-themed notebook — those are flagged [P3] honestly rather than demanded. The realistic fix is the cross-cutting note plus the two [P1]/[P2] braid-repair items.

> **UPDATE (2026-05-18): MG braid rebuilt.** The notebook now contains a from-scratch crystal-graph GNN lesson (main notebook Block 5/5a/5b/6 on `CrystalGraphsDataset`) and, as of this task, a **real-benchmark composition-ceiling block (Block 6c, `matbench_perovskites`) plus a QM9 molecular contrast (Block 6d)**; the homework Part D states the "a crystal is a graph, not a vector" thesis and now carries a real `matbench_perovskites` composition-only baseline (Part D.3). The MG items below are closed accordingly.

- [x] **[P1] Braid the delivered MG W10 content, not next week's.** Closed. Main notebook Block 5/5a/5b builds SchNet- and CGCNN-style message passing from scratch; Block 6 covers extensive/intensive pooling, the Magpie composition baseline, and the §43 pretrain→freeze→fine-tune recipe (frozen GNN trunk + Ridge head). **Block 6c** now confirms the slide-07 composition ceiling on REAL `matbench_perovskites` DFT data (composition-only Ridge/MagpieMLP MAE ≈ 0.43/0.55 eV/atom vs predict-the-mean 0.74), and **Block 6d** runs the same recipe on QM9 `gap` for the molecules-vs-crystals inductive-bias contrast. Genuine delivered-W10 content, on real data.
- [x] **[P2] Sum (extensive) vs mean (intensive) readout.** Closed by main notebook Block 6 (1): trained CGCNN with `readout="mean"` vs `readout="sum"`, probed on a 2x supercell — mean invariant, sum roughly doubles — with the extensive/intensive physics reasoning in prose. Homework Part D.2 reproduces the pooling-rule table directly.
- [x] **[P2] Permutation-equivariance as the *materials* symmetry, made explicit.** Closed by main notebook Block 5a (numeric permutation-invariance check on the trained crystal GNN, contrasted explicitly with the Block 2 image-patch case where PE is required) and homework Part D.2's permutation check.
- [x] **[P3] Invariance vs equivariance (scalars vs forces/tensors).** Closed (as a forward pointer): the GNN-lesson prose states the invariant-readout-for-scalars rule and the SchNet RBF autograd-forces pathway; full equivariant force nets remain explicitly out of scope, as flagged.
- [x] **[P3] The named atomic-system architectures (SchNet/CGCNN/MEGNet/ALIGNN/M3GNet/NequIP/MACE) and the four-symmetry / continuous-filter machinery.** Closed: SchNet continuous-filter and CGCNN gated message passing are implemented from scratch (Block 5), with the remaining named architectures cited as reading pointers in prose.

---

## Cross-cutting note

The exercise is genuinely **strong on the MFML×ML-PC attention axis**: scaled dot-product attention,
multi-head, permutation-equivariance + PE, the residual/LN block, ViT, the SDPA/Flash bench with the
correct Pascal/1080 Ti caveat, and attention-as-saliency are all really exercised — that half is a
true two-lecture braid and only needs the pedagogical-mechanic [P2]s (variance argument, hand-trace,
attention-vs-MLP isolation).

The **MG leg is broken at the source**: both notebook headers braid *"MG Week 10 — Representation
learning and feature discovery,"* but the authoritative `REALIGNMENT_2026-05-13.md` puts that topic
in MG **Week 11**; the delivered MG **Week 10** is *Neural Networks for Materials Properties* (graph
and equivariant nets on atomic systems). The notebook therefore exercises *next week's* MG lecture by
analogy and *this week's* not at all. Closing the single **[P1] MG braid-repair** item (explicitly
naming the §G/§43 transformer-on-crystals + foundation-model recipe that Block 5 already mirrors) plus
the two in-budget **[P2]s** (extensive/intensive pooling; permutation-equivariance as the materials
symmetry) is what would make this a true three-lecture exercise without turning it into a GNN notebook.
The ML-PC materials-application gap (4D-STEM ViT / LPBF cross-attention) is the secondary structural
hole — currently every "materials" input is Ising/Cahn-Hilliard/tensile, none of the deck's signature
diffraction/process applications.
