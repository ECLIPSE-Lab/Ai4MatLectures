# MG Dataset → Exercise Integration Implementation Plan (rev 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put real Materials-Genomics data (the four new `ai4mat` datasets) into the braided weekly exercises **without breaking the existing lessons** — swap the data source only where a block is data-agnostic; otherwise *add* a complementary real-data block alongside the preserved lesson.

**Architecture:** Each MG block is classified **DATA-AGNOSTIC** (consumes flat features/targets → safe in-place data swap) or **ARCHITECTURE-COUPLED** (consumes graph tensors / fixed-dim CVAE / a bespoke descriptor → swapping the data source would gut the lesson → instead *add* a real-data block and keep the original as the explicitly-labelled toy/teaching core). This is why a blanket rewrite was rejected (rev 1 review): `MatBenchDataset.X`/`CDVAEMaterialsDataset.X` are flat 118-dim element-fraction vectors with **no graph tensors**, and W10's lesson literally argues "a crystal is a graph, not a vector." Homework follows its main. Notebooks stay headless-runnable; datasets cached, subsampled by slicing.

**Tech Stack:** Python, PyTorch, Jupytext percent notebooks, `ai4mat.datasets`, `jupytext` 1.19, `pytest`.

---

## Dataset API reference (verified against the implemented code)

```python
from ai4mat.datasets import MatBenchDataset, RMD17Dataset, CDVAEMaterialsDataset, QM9Dataset

MatBenchDataset(task="matbench_perovskites", root="data/matbench", download=True)
#   .X (N,118 elem-fraction f32)  .y (N,)  .folds = 5×(train_idx,test_idx) SURROGATE
#   (seeded KFold; official folds need the `matbench` pkg — .official_folds == False)
#   .feature_kind ("composition"|"structure")  .structures (list[dict]|None, RAW dicts:
#   sites[].species[].{element,occu}; NO edge_index/edge_distance)  .formulas|None
#   supported_tasks(); unknown task -> ValueError listing tasks

RMD17Dataset(molecule="aspirin"(default), n_samples=1000, split="train", seed=0,
             root="data/rmd17", download=True)
#   item (x=(4*n_atoms,) f32 [Z..||coords..], y=energy f32 kcal/mol)
#   .Z .coords .energies .forces (kcal/mol, Å); >1000 -> UserWarning. ~65–175 MB/molecule
#   one-time cache (benzene among the smaller ones). MOLECULES lists the 10.

CDVAEMaterialsDataset(subset="perov_5", split="train",
                      target="formation_energy_per_atom", root="data/cdvae")
#   .X (N,118 elem-frac)  .y  .df  .cif (list[str])  .ids  .feature_names
#   TARGET RULE (cdvae_materials.py:326-334): an explicit `target` that is NOT a
#   numeric column of that subset RAISES ValueError; the per-subset fallback applies
#   ONLY when target=="formation_energy_per_atom" (the global default). So you MUST
#   discover real columns first (instantiate with default, read the ValueError /
#   inspect _detect_numeric_targets) — DO NOT hard-code "e_above_hull".
#   perov_5 numeric: dir_gap, heat_all, heat_ref(default), ind_gap (NO hull, NO structure graph)
#   mp_20  numeric: band_gap, e_above_hull, formation_energy_per_atom(default), spacegroup.number
#   carbon_24: pure C — element-fraction X degenerate; DO NOT use for property/gen.
#   NO subsample arg -> slice ds.X[:k]/ds.y[:k] yourself.

QM9Dataset(target="gap", root="data/qm9", download=True)
#   item (x=(7,) [nC,nH,nN,nO,nF,n_heavy,n_tot], y=property f32)  .smiles
#   .property_names (19). NOTE: __init__ featurizes ALL 133,885 SMILES (~30 MB CSV,
#   ~tens-of-seconds featurize) every construction; slice AFTER for runtime.
```

Class-runtime rule: small `task`/`subset`/`molecule`; slice to a few k rows; first run downloads+caches under `data/<name>/` (network needed once — each task pre-warms it).

---

## Block classification & strategy (the heart of this plan)

| Wk / block | What it consumes today | Class | Action |
|---|---|---|---|
| **W9** tiers 0–2 + split-design (MG-U8) | a *bespoke* χ/radius/prototype descriptor + `cg.y` | DATA-AGNOSTIC | **swap** → `MatBenchDataset.X/.y/.folds`; rewrite the bespoke-descriptor cells |
| **W9** tier-4 `TinyCGNN` + structure-scramble ablation | `cg[i]["species"/"edge_index"/"edge_distance"]` | ARCH-COUPLED | **keep** on `CrystalGraphsDataset`, relabel as the explicit *synthetic structural appendix* |
| **W12** Block 6.5 generative CVAE (`x_dim=2`, `c_dim=1`, E/H) | toy 2-D `NanoindentationDataset` | ARCH-COUPLED | **keep** CVAE as toy generative-mechanics core; **add** a real-crystal *discovery-funnel + S.U.N. + uncertainty-triage* stage on `CDVAEMaterialsDataset.X` (tabular, fits) |
| **W13** Block 5b discovery loop | synthetic 1-D −E_hull | DATA-AGNOSTIC (tabular acquisition) | **swap** objective → real `CDVAEMaterialsDataset(subset="mp_20")` hull pool |
| **W8** (no MLIP/force host; MG anchor is graph-rep PBC/RBF) | — | n/a | **add** an rMD17 energy+force self-study block (additive only) |
| **W10** Block 5/5a/5b crystal-**graph** GNN (SchNet/CGCNN, perm-invariance, pooling) | `CrystalGraphsDataset` graph tensors | ARCH-COUPLED | **keep** the GNN lesson intact; **add** a "real benchmark" block that *proves the composition ceiling* (MatBench `.X` baseline + QM9 molecular contrast) — turns the contradiction into the intended teaching point |

Priority: **P1** = W9 (Task 1), W12 (Task 2), W13 (Task 3). **P2** = W8 (Task 4), W10 (Task 5). **P3** = bookkeeping (Task 6).

## Conventions every task follows

- Read the **entire** target `.py` first; preserve downstream variable names or update every consumer; verify by full headless run, not just `py_compile`.
- `.X`/`.y` only (no pymatgen/ase/rdkit); guard any `.cif`/`.structures`/`.smiles` use.
- Slice to subsample (no subsample arg on CDVAE/QM9). Pre-warm the dataset cache in the verify step.
- After each `.py` edit: `python3 -m py_compile`; `jupytext --to notebook <file>.py`; headless run `MPLBACKEND=Agg PYTHONPATH=. python3 <file>.py`.
- One commit per task, curated `git add` (never `-A`), on `main`, **no `Co-Authored-By`** line. `.ipynb` is regenerated, not hand-edited. Regenerate a paired `.ipynb` **only if one already exists** (e.g. `week13_homework` has **no** `.ipynb`).

---

### Task 1 (P1): W9 — swap the data-agnostic MG-U8 ladder/splits to MatBench; keep the GNN tier as a labelled synthetic appendix

**Files:** Modify `notebooks/week9_latent_geometry.py` (MG-U8 block: `cg = CrystalGraphsDataset()` at ~502; bespoke descriptor cells ~496–540; tier-4 `TinyCGNN` ~458/587–655/711–715; caveat prose ~730; checklist ~885) and `notebooks/week9_homework.py` (Part E, `cg = CrystalGraphsDataset()` ~519). Regenerate both `.ipynb`.

- [ ] **Step 1:** Read all of `week9_latent_geometry.py`; list every consumer of `cg`, the bespoke descriptor matrix, `embeds`, `proto_all`, `y_*`.
- [ ] **Step 2:** Tiers 0/1/2 + split-design: replace the bespoke χ/radius/prototype descriptor construction **and** `cg.y` with `MatBenchDataset` (default `task="matbench_perovskites"`; `.X` 118-dim, `.y` real target). Replace the synthetic prototype/cation split with **`ds.folds`** (label it "reproducible surrogate folds — official folds need the `matbench` pkg; fine for split-design teaching") plus one composition-family hold-out derived from `.X`. Delete/rewrite the now-orphaned descriptor cells.
- [ ] **Step 3:** Tier-4 `TinyCGNN` + structure-scramble ablation: **keep on `CrystalGraphsDataset`**, but rewrite the surrounding prose to frame it explicitly as a *synthetic structural appendix* ("real composition benchmark above; synthetic graph below to show what structure adds"). Rewrite the ~730 "toy energy is a real lesson" caveat and the ~885 checklist "test-set construction N/A" to the real-benchmark situation for the MatBench part.
- [ ] **Step 4:** Mirror at homework depth in `week9_homework.py` Part E: MatBench ladder + one `ds.folds` vs random contrast (no GNN tier needed).
- [ ] **Step 5: Verify:** `python3 -m py_compile` both; pre-warm `PYTHONPATH=. python3 -c "from ai4mat.datasets import MatBenchDataset; MatBenchDataset(task='matbench_perovskites')"`; `jupytext --to notebook` both; headless-run both end-to-end. Expected: ladder MAE on real targets; fold-vs-random gap visible; synthetic appendix still runs.
- [ ] **Step 6:** Tick closed MG items in `notebooks/week9_coverage_gaps_TODO.md`.
- [ ] **Step 7: Commit** the 5 files: `"week9: MG-U8 ladder+splits on real MatBench; GNN tier kept as synthetic appendix"`.

**Done when:** both W9 notebooks run headless; U8 split-design lesson uses real MatBench data+folds; the GNN/ablation still works and is labelled synthetic.

---

### Task 2 (P1): W12 — keep the 2-D CVAE toy core; ADD a real-crystal discovery stage on CDVAE

**Files:** Modify `notebooks/week12_uncertainty_and_discovery.py` (Block 6.5 starts ~1081; CVAE `x_dim=2/c_dim=1` ~1142–1165) and `notebooks/week12_homework.py` (Part E). Regenerate both `.ipynb`.

- [ ] **Step 1:** Read Block 6.5 + homework Part E; note CVAE/CFG/funnel/S.U.N./uncertainty-triage variable names.
- [ ] **Step 2:** **Do not re-architect the 2-D CVAE.** Keep it as the generative-mechanics toy (relabel its markdown "toy 2-D substrate for CVAE/CFG mechanics"). **Append** a new sub-block: load `CDVAEMaterialsDataset(subset="perov_5", split="train")` (discover its real numeric targets by instantiating with the default first and reading `_detect_numeric_targets`/the ValueError — do **not** hard-code), use `ds.X` as real crystal feature vectors, and run the *discovery-funnel + S.U.N. + uncertainty-triage* logic on real crystals (this part is tabular and dimension-flexible). Slice `ds.X[:6000]` for runtime.
- [ ] **Step 3:** Mirror at homework depth in `week12_homework.py` Part E: a tiny funnel on `CDVAEMaterialsDataset(subset="perov_5").X` (no CVAE rebuild).
- [ ] **Step 4: Verify:** py_compile both; pre-warm `CDVAEMaterialsDataset(subset='perov_5', split='val')`; jupytext regen; headless-run both. Expected: toy CVAE unchanged & runs; new real-crystal funnel produces a non-degenerate waterfall.
- [ ] **Step 5:** Tick MG generative items in `notebooks/week12_coverage_gaps_TODO.md`.
- [ ] **Step 6: Commit** 5 files: `"week12: add real-crystal CDVAE discovery-funnel stage; keep 2-D CVAE toy core"`.

**Done when:** W12 main+homework run headless; real CDVAE crystals drive the discovery funnel; CVAE mechanics lesson intact.

---

### Task 3 (P1): W13 — discovery loop acquires on the real mp_20 hull pool

**Files:** Modify `notebooks/week13_physics_and_gp.py` (Block 5b, synthetic −E_hull at ~701). `notebooks/week13_homework.py` has **no `.ipynb`** and likely no discovery stub — touch only if a stub exists. Regenerate `week13_physics_and_gp.ipynb`.

- [ ] **Step 1:** Read Block 5b (E-hull objective + EI/UCB/Thompson + regret/recall plots) and downstream cells.
- [ ] **Step 2:** Replace the synthetic objective with a real pool: `CDVAEMaterialsDataset(subset="mp_20", split="train")`; **discover the hull column name at runtime** (instantiate default, inspect numeric targets) rather than assuming `"e_above_hull"`; then re-instantiate with that target. Surrogate (GP/ridge) predicts hull from `ds.X[:5000]`; "discovery" = recall of truly-stable (hull ≈ 0) over rounds; keep EI/UCB/Thompson/random + regret/recall plots; rewrite the synthetic-objective prose.
- [ ] **Step 3: Verify:** py_compile; pre-warm `CDVAEMaterialsDataset(subset='mp_20', split='val')`; jupytext regen `week13_physics_and_gp.py`; headless-run. Expected: acquisition beats random on real stable-material recall.
- [ ] **Step 4:** Tick MG discovery items in `notebooks/week13_coverage_gaps_TODO.md`.
- [ ] **Step 5: Commit:** `"week13: discovery loop acquires on real mp_20 hull pool"`.

**Done when:** W13 discovery loop runs headless on the real mp_20 pool; EI/UCB/Thompson beat random on stable recall.

---

### Task 4 (P2): W8 — ADD an rMD17 MLIP energy+force self-study block (additive only)

**Files:** Modify `notebooks/week8_uncertainty_and_robustness.py` (additive cells **after the existing Block 8/8b MG graph-rep anchor** — W8's MG content is graph-rep PBC/RBF; SOAP/MLIPs were moved to MG W6 per the week8 TODO, so this is purely an additive robustness/MLIP self-study extension, not a host-swap). Homework file is `week8_uncertainty_and_robustness_homework.py` (not touched). Regenerate `week8_uncertainty_and_robustness.ipynb`.

- [ ] **Step 1:** Read W8; confirm the Block 8/8b graph-rep anchor; pick a clean additive insertion point.
- [ ] **Step 2: Add** "Block — MLIP energy regression on rMD17 (why ≤1000 samples)": `RMD17Dataset(molecule="benzene", n_samples=1000, split="train")` + held-out `split="test"`; tiny MLP on `.X`→energy; show train/test error; then raise `n_samples` past 1000 and show the **correlated-sample trap** (test error becomes misleadingly optimistic) — a built-in generalization/robustness lesson braiding W8's theme. One `.forces`-magnitude diagnostic cell (no full force-matching). Small/fast.
- [ ] **Step 3: Verify:** py_compile; pre-warm `RMD17Dataset(molecule='benzene', n_samples=200)` (one-time cached download, ~65–175 MB depending on molecule; benzene among the smaller — acceptable); jupytext regen; headless-run end-to-end. Expected: new block runs; oversampling lesson visible; existing blocks untouched.
- [ ] **Step 4:** Tick the MLIP/robustness item in `notebooks/week8_coverage_gaps_TODO.md`.
- [ ] **Step 5: Commit:** `"week8: add rMD17 MLIP energy+force self-study block (correlated-sample lesson)"`.

**Done when:** W8 runs headless with the additive rMD17 block; the >1000-sample lesson is demonstrated; no existing block changed.

---

### Task 5 (P2): W10 — keep the GNN lesson; ADD a real-benchmark block proving the composition ceiling + a QM9 molecular contrast

**Files:** Modify `notebooks/week10_attention_and_transfer.py` (ADD after Block 5/6 GNN lesson; `cg = CrystalGraphsDataset(n_total=200, seed=0)` at 636 stays) and `notebooks/week10_homework.py` (Part D — add a light teaser, do not replace the graph thesis). Regenerate both `.ipynb`.

- [ ] **Step 1:** Read W10 Block 5/5a/5b (GNN) and homework Part D (thesis: "a crystal is a graph, not a vector; composition can't tell polymorphs apart").
- [ ] **Step 2: Add** "Block — the composition ceiling on a real benchmark": `MatBenchDataset(task="matbench_perovskites").X/.y` sliced; train the same simple regressor on composition-only features and show it *underperforms* the graph GNN from the preceding block → empirically confirms the lesson's thesis on real data (not a contradiction — the intended payoff). Then **add** a short `QM9Dataset(target="gap")` cell (slice after the ~tens-of-seconds full featurize): same simple model on molecules — "same recipe, different inductive bias," as the MG-U9 deck names QM9.
- [ ] **Step 3:** Homework Part D: add one light cell (MatBench composition baseline number to compare against the Part-D graph point); keep the graph thesis intact.
- [ ] **Step 4: Verify:** py_compile both; pre-warm MatBench perovskites + QM9; jupytext regen; headless-run both. Expected: composition baseline visibly worse than the GNN; QM9 cell runs.
- [ ] **Step 5:** Tick MG items in `notebooks/week10_coverage_gaps_TODO.md`.
- [ ] **Step 6: Commit:** `"week10: add real MatBench composition-ceiling + QM9 contrast (GNN lesson preserved)"`.

**Done when:** W10 main+homework run headless; composition-ceiling block confirms the graph thesis on real data; QM9 contrast present; GNN lesson unchanged.

---

### Task 6 (P3): Bookkeeping + full sanity

**Files:** `notebooks/week{8,9,10,12,13}_coverage_gaps_TODO.md`, `notebooks/coverage_gaps_INDEX.md`.

- [ ] **Step 1:** Check off the MG-realism gaps now closed per week; add an INDEX row per week ("MG now on real data: <dataset>; <swap|added>").
- [ ] **Step 2: Verify suite:** `PYTHONPATH=. python3 -m pytest -q -m "not slow" tests/` (expect green); confirm `git status` shows only intended files.
- [ ] **Step 3: Commit:** `"docs: record MG real-dataset integration across W8–W13 exercises"`.

**Done when:** fast suite green; INDEX/TODO updated; tree clean.

---

## Risks & mitigations

- **Lesson integrity (the rev-1 flaw):** never swap data into an architecture-coupled block; W9-GNN/W10-GNN/W12-CVAE are preserved, real data is *added* alongside. This is the central design correction.
- **CDVAE target API:** explicit non-default target raises if the column name is wrong → every CDVAE task discovers real numeric columns at runtime before selecting (Tasks 2, 3).
- **No graph tensors in MatBench/CDVAE `.X`:** they are composition vectors; only used where composition is pedagogically correct (U8 ladder, composition-ceiling demo, discovery features) — never to "replace" a GNN.
- **First-run download / class network:** cached under `data/<name>/`, pre-warmed per task. Sizes: QM9 ~30 MB (+~tens-of-sec featurize-all-134k on every construct → slice after), CDVAE perov_5 small, MatBench task-dependent, rMD17 ~65–175 MB/molecule (benzene among the smaller). Pick smallest viable.
- **Runtime:** slice to a few k rows (no subsample arg on CDVAE/QM9); keep models tiny.
- **MatBench `.folds` are seeded surrogates** (`.official_folds == False`) — prose says "reproducible surrogate folds," no leaderboard-parity claim.
- **carbon_24 excluded** (degenerate pure-C features). **rMD17 default molecule is `aspirin`** (largest) — always pass `molecule="benzene"` explicitly.
- **week13_homework has no `.ipynb`** and no discovery stub — Task 3 touches the W13 main only.
- **Scope:** P1 (1–3) before P2 (4–5); homework follows its main.

## Rollback

One commit per task on `main`; revert with `git revert <sha>`. `.ipynb` regenerable from `.py` via `jupytext --to notebook`. No lecture decks touched.
