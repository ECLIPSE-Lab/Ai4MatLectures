# Week 6 exercise — triad coverage audit & gap to-do

**Audited:** 2026-05-17 · **Week 6 (lecture 19.05.2026)** · braided exercise =
`week6_optimization_and_finetuning.py` (main) + `week6_homework.py`.

**Triad lectures audited (post-restructure):**

| Course | Deck | Week-6 topic |
|---|---|---|
| MFML | `mathematical_foundations_of_ai_and_ml/06_loss_landscapes_optimization/01_intro.qmd` | Loss landscapes & optimization |
| ML-PC | `ml_for_characterization_and_processing/unit06_transfer_learning/01_intro.qmd` | Data scarcity & transfer learning |
| MG | `materials_genomics/07_graph_based_rep/01_intro.qmd` (slug authoritative; "Unit 7" label is stale) | Graph-based crystal representations + classical descriptor recap |

## Verdict

- **MFML (optimization):** STRONG. Core optimizer mechanics, schedules, edge-of-stability all exercised. A few genealogy/diagnostic gaps.
- **ML-PC (transfer learning):** STRONG on the fine-tuning loss-geometry half (catastrophic forgetting, discriminative LRs, warm-up/cosine, LP-FT). **The entire data-centric half of the lecture is absent** (augmentation, leakage-safe CV, synthetic/sim-to-real).
- **MG (graph reps):** WEAK. Only a toy hand-rolled GNN on fixed toy graphs. The lecture's CORE crystal-graph machinery (PBC, RBF/cutoff, invariance/equivariance, failure-mode diagnostics, ranking metrics) is not exercised.

Priority key: **[P1]** core lecture topic, currently absent/cosmetic — should add before 19.05 · **[P2]** core but partially covered — extend · **[P3]** support/nice-to-have.

---

## MFML — Loss landscapes & optimization

- [ ] **[P2] Adaptive-LR genealogy: AdaGrad → RMSProp.** Notebooks jump straight from SGD/momentum to Adam. The lecture builds AdaGrad (monotonic accumulator stall) → RMSProp (EMA fix) → Adam as one story. Add a 10-line AdaGrad + RMSProp to the Part A 2-D playground so the "why each fix exists" arc is exercised, not just Adam.
- [ ] **[P1] Normalization smooths the loss landscape.** A [CORE] MFML slide; not touched anywhere. Add a short block: same TinyCNN with vs without BatchNorm/LayerNorm, show max stable LR increases and loss curve smooths. Currently no normalization experiment exists.
- [ ] **[P2] Batch size ↔ gradient noise / linear scaling rule.** Part C varies full-batch vs batch=8 but never measures the gradient-noise / sharp-vs-flat / linear-LR-scaling relationship the lecture makes [CORE]. Add an explicit batch-size sweep (e.g. {8,32,full}) with the η∝B rule tested.
- [ ] **[P2] Flat vs sharp minima — measured, not just asserted.** [CORE] in both MFML and ML-PC; only referenced in prose (Part D, ML-PC narrative). Add a minimal sharpness probe (loss under small random/worst-case parameter perturbation, ε-ball) comparing two minima — also delivers the ML-PC "pretraining lands in flat basins" claim.
- [ ] **[P3] Weight initialization (Xavier/Glorot, He/Kaiming).** [SUPPORT] but pedagogically central to the landscape story; no init experiment. Optional: depth-vs-activation-variance demo, or at least a fixed-bad-init vs He-init convergence contrast.
- [ ] **[P3] Nesterov accelerated gradient.** Only heavy-ball momentum is hand-rolled; NAG look-ahead (and its O(1/t²) claim) is in the deck but not exercised. One extra variant in Part A.
- [ ] **[P3] SAM (sharpness-aware minimization).** [SUPPORT]; ties to the flat-minima probe above. Optional stretch variant.
- [ ] **[P3] Vanishing gradients / saturating activations & Newton–quasi-Newton (L-BFGS).** Exploding-gradient+clipping is covered (Block 6/Ex 2); the vanishing/plateau side and the second-order family (deck mentions L-BFGS, Sophia) are only prose. Low priority for a 90-min slot.

## ML-PC — Data scarcity & transfer learning

- [ ] **[P1] Data augmentation as encoding invariances.** [CORE], a large fraction of the ML-PC deck — **completely absent** from both notebooks. Add a block: train target task with vs without label-preserving augmentation (rotations/flips/noise) on the Cahn–Hilliard/Ising images; show the small-N gain.
- [ ] **[P1] Physically-illegal augmentations.** [CORE]. The materials-specific punchline (don't rotate rolled-texture micrographs / flip chiral structures). Fold into the augmentation block as a "which transforms are label-preserving here?" decision step.
- [ ] **[P1] Augmentation leakage — split before augmenting.** [CORE] leakage rule; nothing in the notebooks demonstrates it. Add the canonical contrast: augment-then-split (leaks, inflated val acc) vs split-then-augment.
- [ ] **[P1] Leakage-safe validation: K-fold & group-K-fold + mean±std.** [CORE]. Notebooks use a single seeded train/test split only. Add K-fold CV and a group-wise split (by specimen/microscope/day analogue, e.g. by Ising temperature bin or crystal prototype) reporting mean ± std.
- [ ] **[P2] Synthetic data & sim-to-real gap + ImageNet→synthetic→real cascade.** [CORE]. Exercise does single source→target only. Extend Block 1/3 into a 3-stage cascade, or add a "too-clean synthetic → noisy real" fine-tune step with the sim-to-real domain-gap discussion.
- [ ] **[P2] Held-out gold test discipline.** [CORE] "touch once" rule — currently implicit. Make the protocol explicit in the CV block (train/val for tuning, sealed test reported once).
- [ ] **[P3] Early stopping / patience; active learning; self-supervised backbones (DINOv2/MAE).** [SUPPORT], mentioned-not-implemented. Early stopping is a cheap add to existing training loops; active learning / SSL are stretch.
- [ ] **[P3] Gradual (progressive) unfreezing.** LP-FT (2-phase) is implemented; the deck's progressive layer-by-layer unfreezing is not. Optional Exercise-1 extension.

## MG — Graph-based crystal representations

- [ ] **[P1] Periodic boundary conditions / lattice-aware neighbor construction.** [CORE] and explicitly *mentioned-but-not-implemented*. `CrystalGraphsDataset` is fixed toy graphs with no PBC. Add real (or realistic) crystal-graph construction: minimum-image neighbor search within a cutoff with periodic images — this is the single biggest MG gap.
- [ ] **[P1] Gaussian RBF distance expansion + smooth cutoff envelope (and the hard-cutoff artifact).** [CORE]. TinyCGNN feeds raw scalar distance with a hard cutoff. Add RBF edge featurization and a smooth envelope; demonstrate the discontinuous-energy artifact a hard cutoff causes (deck [CORE] reproducibility point).
- [ ] **[P1] Shortcut-learning failure mode + feature-randomization diagnostic.** [CORE] MG diagnostic (GNN exploiting spurious volume–energy correlation). Add the diagnostic: shuffle/randomize a feature and show the metric collapse — directly braids with MFML "what did the optimizer actually fit."
- [ ] **[P2] Invariance vs equivariance, explicit (scalars vs forces).** [CORE] conceptual spine. Even without full equivariant nets, add an empirical invariance check: rotate/translate/permute a structure, confirm predicted (scalar) energy is unchanged; state where equivariance would be needed (forces).
- [ ] **[P2] Ranking / discovery metrics.** [CORE] for screening. Block 6 reports only MSE/MAE. Add Spearman ρ / Kendall τ and top-k recall — the metrics that matter for the discovery framing braided with MG.
- [ ] **[P2] Readout: sum (extensive) vs mean (intensive).** [CORE]. Only mean-pool is used. Add the sum-vs-mean contrast and the extensive/intensive reasoning (total vs per-atom energy).
- [ ] **[P2] Over-smoothing in deep GNNs (skip / jumping-knowledge).** [CORE]. TinyCGNN is shallow; add a depth sweep showing node-feature collapse and a skip-connection remedy — also a clean optimization-landscape tie-in.
- [ ] **[P3] Named architectures CGCNN / SchNet / MEGNet.** TinyCGNN is generic; deck treats these as [CORE]/[SUPPORT]. At minimum frame TinyCGNN explicitly as "CGCNN/SchNet minus RBF+PBC" and add one as a reading/stretch task.
- [ ] **[P3] Classical structural descriptor recap (SOAP / RDF / ACE), variable-size disjoint batching, O(L·k̄·N) scaling.** Magpie *composition* features are exercised (Part C/Block 5) but the lecture's *structural* descriptor recap and batching/scaling mechanics are not.

---

## Cross-cutting note

The exercise is well-braided on the **optimization ↔ fine-tuning** axis (MFML×ML-PC).
The **MG graph-representation lecture is only nominally braided** — the toy GNN exercises
*message passing as an optimization target* but almost none of the *crystal-representation*
content (PBC, RBF/cutoff, invariance, ranking, failure diagnostics) the MG students see
on 19.05. Closing the [P1] MG items is what would make this a true three-lecture exercise.
