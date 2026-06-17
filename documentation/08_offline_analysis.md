# Topic 8 — Offline Analysis: `run_analysis.py` Orchestration, K-Sweep, and Result Summarisation

All claims verified against code on 2026-06-13. Line numbers refer to the current working tree.
Primary sources read in full: `run_analysis.py` (1889 lines, repo root), `sweep_clusters.py` (141 lines, repo root), `summarize_results.py` (849 lines, repo root). Cross-checked against `ATOMs_Analysis/atoms_config.py`, `ATOMs_Analysis/utils/visualization_carla.py:1005-1027` (`make_output_dirs`), `CLAUDE.md` (the pipeline table at "Analysis Pipeline (`run_analysis.py`)"), `documentation/06_perturbations.md`, `documentation/07_distances_and_detectors.md`.

---

## 1. Purpose & scope

This document covers the **orchestration layer** of the offline experiment — the three repo-root scripts that drive the entire clean→profile→detect→report flow:

1. **`run_analysis.py`** — the single end-to-end entry point. It loads the agent + LRP, computes (or loads) baseline ATOMs profiles, fits every baseline-side model (Mahalanobis, GMM, MDX-v1, MDX-v2), builds the labeled test set, scores every detector, evaluates ROC/AUC, breaks results down per perturbation, and writes all figures and JSON. One invocation produces one `(agent, mode, K)` result snapshot.
2. **`sweep_clusters.py`** — a thin driver that calls `run_analysis.py --gmm-k K` once per K in a list, copying each run's output into a `<K> clusters/` snapshot folder.
3. **`summarize_results.py`** — a post-hoc aggregator that reads every snapshot's `summary.json` / `results_per_perturbation.json` / `results_knn_*.json` and writes `SUMMARY.md` + heatmaps, including the **val-set K-selection recommendation** (its Section 3).

This topic is the layer that *calls* everything documented in Topics 3–7. It does not re-derive detector math (Topic 7), perturbation semantics (Topic 6), the ATOMs profile (Topic 4), or LRP (Topic 3) — it documents the control flow, the I/O contract of each stage, the caching/recompute gates, and the agent-specific branching. The most load-bearing single finding of this review is that **the in-code `STEP N` banner numbering does not match `CLAUDE.md`'s pipeline-table step numbering** (§3.1, finding 8.1), and that the true control-flow order interleaves a *disabled* trajectory block between the val load and detector scoring (§3.2, §3.9, finding 8.2).

Scope note: `run_analysis.py`'s module docstring (`:14-30`) lists "Steps 1–12" as a *logical* outline; the in-code banners run 1→14; and `CLAUDE.md`'s table uses yet a third numbering (1, 2, 2.5, 2.5-v2, 3, 4, 5, 6, 7, 8, 8.5, 9, 9.5, 10, 11, 12). All three disagree. We treat the **code banners** as ground truth and map the other two to them.

---

## 2. Key design decisions

### 2.1 Recompute-flag caching: expensive stages are gated, everything downstream is cheap

The pipeline is built around the assumption that the two expensive stages — LRP-driven ATOMs profile computation on baseline (Step 2) and test (Step 9) frames, and MDX feature extraction (Step 3) — run *once*, are persisted to disk, and are *reloaded* on every subsequent invocation. Five boolean config flags gate this (all default `False`, `atoms_config.py:37-42`):

| Flag | Default | Gates | Artifact |
|---|---|---|---|
| `RECOMPUTE_BASELINE` | False | Step 2 baseline ATOMs | `baseline_<mode>.npz` |
| `RECOMPUTE_MDX_BASELINE` | False | Step 3 MDX-v1 fit | `mdx_parameters.pkl` |
| `RECOMPUTE_MDX_V2_BASELINE` | False | Step 2.5-v2 MDX-v2 fit | `mdx_v2_parameters.pkl` |
| `REAPPLY_PERTURBATIONS` | False | Step 8 labeled-set build | `test_labeled.npz` |
| `RECOMPUTE_TEST_ATOMS` | False | Step 9 test ATOMs | `test_profiles_<mode>.npy` |
| `RECOMPUTE_MDX_TEST_SCORES` | False | Step 11 §9d MDX test scoring | `mdx_scores_<mode>.npy` |

**Rationale.** With all flags `False`, a `run_analysis.py` invocation skips every model forward pass and only re-fits the Gaussian/GMM, re-scores the (cached) profiles, and re-renders figures. This is what makes `sweep_clusters.py` viable: re-running 3–10 K values is fast because only the GMM and downstream scoring change with K (`sweep_clusters.py:18-23` documents exactly this prerequisite). The flags are an OR with file-existence: each stage recomputes if its flag is True **or** the cached artifact is missing (`run_analysis.py:228`, `:680`, `:732` gate on the flag; the MDX paths additionally branch on file existence, `:268`, `:322`, `:356-375`).

### 2.2 `MODE_ANALYSIS`-suffixed artifacts: two attribution modes coexist on disk

Every profile/logit/score artifact is suffixed with the active `MODE_ANALYSIS` (`_1` or `_2`): `baseline_{_mode}.npz` (`:225`), `test_profiles_{_mode}.npy` (`:806`), `test_logits_{_mode}.npy` / `test_speed_logits_{_mode}.npy` (`:816-818`), `mdx_scores_{_mode}.npy` (`:1317`), and the output directory itself `atoms_analysis_mode_{conf.MODE_ANALYSIS}` (`:110`). The MDX `*_parameters.pkl` files are the exception — they are **not** mode-suffixed (`:308`, `:354`, `:415`), because MDX consumes backbone/F_c features, not the mode-dependent ATOMs profile, so a single MDX fit is shared across both modes.

**Rationale.** Mode 1 (node-level, paper default) and Mode 2 (layer-level) produce *different* ATOMs profiles from the same frames (Topic 4). Mode-suffixing lets both sets of profiles coexist so the analysis can be re-run for either mode without recomputation — `summarize_results.py` exploits this to build a mode-1-vs-mode-2 comparison (its Section 8). The live config runs `MODE_ANALYSIS = 2` (`atoms_config.py:23`).

### 2.3 Val-set K selection is offline (via `sweep_clusters` + `summarize_results`), not at runtime

A single `run_analysis.py` invocation **fixes K** (resolved §3.5) — it does *not* sweep K internally. The BIC/AIC sweep is computed and plotted (Step 5) but its result is almost always discarded because `NUM_GMM_CLUSTERS = 10` overrides it (finding 7.1). To choose K on a principled, leakage-free criterion the project pushes K selection *out* of the runtime loop:

- `run_analysis.py` reports a single scalar per run, `__val_auc_gmm_avg__` — the **mean val-set AUC across the GMM detector variants at this run's K** (`:1531-1544`), embedded in `summary.json` (`:1586-1587`).
- `sweep_clusters.py` runs `run_analysis.py --gmm-k K` for each K and snapshots each `summary.json` into a `<K> clusters/` folder.
- `summarize_results.py` reads `__val_auc_gmm_avg__` from each snapshot and produces a ranked **val-set K recommendation** (its Section 3, §3.11 here).

**Rationale.** K is one of only two tuned hyperparameters in the detection layer (the other is k-NN k, selected on val *at runtime*, Topic 7 §2.7). Selecting K by maximising *test* AUC would be hyperparameter leakage. The val set (disjoint Town05 routes under the original split, or a disjoint random split under the alternative variant; Topic 5 §2.5) exists precisely to break this. Doing the selection *offline* over the snapshot folders keeps a single runtime invocation deterministic in K (necessary for the snapshot semantics) while still surfacing the val signal. `summarize_results.py` explicitly warns against the test-set best K ("Do *not* use the test-set best K — that constitutes hyperparameter leakage", `:466-467`).

### 2.4 The trajectory step is dead code; on-disk `trajectory_analysis/` figures are stale

The in-code STEP 10 (`run_analysis.py:905-1106`) is the "perturbation trajectory analysis in ATOMs attention space". **Almost the entire block is commented out** — every executable line from the `TRAJ_OUT_DIR` definition (`:936`) through the four figure-saving calls (`:1075-1106`) is prefixed with `#`. Only the section banner, the prose docstring (`:906-933`), and two bare `print` statements (`:1072` "Generating figures..." and an empty body) survive. No `trajectory_*.png`, `displacement_*.png`, or `displacement_stats.txt` is written by the current code. This **confirms `CLAUDE.md`'s claim** that step 8.5 is disabled and any `trajectory_analysis/` figures on disk are stale (finding 8.3 records the additional staleness that the orphaned `print("[Step 10f] Generating figures...")` still fires, printing a misleading progress line for a step that produces nothing). The dependencies it documents in its docstring (`baseline_profiles`, `clean_test_profiles.npy`) are likewise never produced.

### 2.5 HPC-vs-local division of profile computation (forward ref to Topic 10)

The two expensive stages (baseline + test ATOMs profiles, and MDX backbone features) are designed to run **either locally or on HPC**, with the local path acting as a fallback. For MDX-v1, `run_analysis.py` first looks for a pre-computed `mdx_features.npz` (gathered from `compute_baseline_chunk.py` on Viper) and only extracts features frame-by-frame locally if that file is absent, printing a "slow" warning and a tip to use HPC (`:266-277`, `:321-332`). For the test profiles, the deferred-PGD scheme (Topic 6 §2.3) means the *local* recompute path produces non-adversarial PGD profiles — the HPC-crafted profiles must be merged instead, and two guards warn about this (`:735-748` at compute time, `:840-865` an alignment-key check at load time). The HPC mechanics (chunking, array jobs, gather scripts, `mdx_features.npz` production) are Topic 10; here we document only the local consumption and the fallback boundary.

### 2.6 `EXPERIMENT_VARIANT` silently re-roots every path

All four data/result roots resolve through `EXPERIMENT_VARIANT` (`atoms_config.py:65-74`): `"alternative"` (the live setting) points `BASELINE_DATA_DIR`/`TEST_DATA_DIR`/`VAL_DATA_DIR`/`RESULTS_DIR` at the `*_data_alt` / `results_alt` counterparts. `run_analysis.py`, `sweep_clusters.py`, and `summarize_results.py` all read these through `conf` (or, for `summarize_results.py`, via the `--alt` flag that switches `results_subdir` to `results_alt` and the output to `results_summary_alt/`, `:815-817`). No code change is needed to switch splits — but a reader following `CLAUDE.md`'s canonical-path examples will inspect the wrong directory under the live config (finding 5.7, Topic 5).

---

## 3. Implementation details

### 3.1 The in-code STEP ↔ CLAUDE.md-step ↔ actual-content mapping (the central artefact)

The code's `STEP N` banner comments do **not** line up with `CLAUDE.md`'s pipeline-table numbering. The table below is the authoritative reconciliation; every mismatch is flagged as finding 8.1.

| In-code banner (file:line) | What actually happens | `CLAUDE.md` table step | Match? |
|---|---|---|---|
| STEP 1 (`:126`) | Load agent model + LRP wrapper + `ATOMsCarla`; set `action_logits_available` / `speed_logits_available` | 1 | ✓ |
| STEP 2 (`:214`) | Baseline ATOMs profiles → `baseline_<mode>.npz`; load mean/cov/series | 2 | ✓ |
| STEP 3 (`:247`) | Fit **MDX-v1** (backbone-512 / WoR penultimate) → `mdx_parameters.pkl` | **2.5** | ✗ (banner 3 ≠ table 2.5) |
| (nested) Step 2.5-v2 (`:387`) | Fit **MDX-v2** (F_c-256 *intended*, backbone-512 *as configured*) → `mdx_v2_parameters.pkl` | 2.5-v2 | ✓ banner label, but nested *inside* the STEP 3 block (`:377-419`) |
| STEP 4 (`:423`) | Fit single-Gaussian `MahalanobisDetector`; 99th-pct threshold → `mahal_detector.npz` | **3** | ✗ (banner 4 ≠ table 3) |
| STEP 5 (`:448`) | BIC/AIC sweep K=1..MAX_K; resolve K; plot `gmm_model_selection.png` | **4** | ✗ |
| STEP 6 (`:485`) | Fit `GMMClustering` at K; assign clusters → `gmm.npz` | **5** | ✗ |
| STEP 7 (`:513`) | Visualise baseline (attention bars, per-cluster, per-cmd, representative imgs, PCA, t-SNE) | **6** | ✗ |
| STEP 8 (`:670`) | Build labeled test set via `PerturbationApplier.apply` → `test_labeled.npz` | **7** | ✗ |
| STEP 9 (`:727`) | Test ATOMs profiles + logits → `test_profiles_<mode>.npy` (+ keys, logits) | **8** | ✗ |
| STEP 9.5 (`:872`) | Load val profiles + `val_labeled.npz`; set `_has_val` | 9.5 | ✓ label, but precedes STEP 10/11 |
| STEP 10 (`:905`) | Trajectory analysis — **DISABLED / commented out** | **8.5** | ✗ (banner 10 ≠ table 8.5; and it is dead) |
| STEP 11 (`:1109`) | Score all detectors (sub-banners 9a–9e: Mahal/Euclid/k-NN/JSD/GMM-variants/Entropy/MDX/PEOC) | **9** | ✗ |
| STEP 12 (`:1408`) | Evaluate ROC/AUC/Youden; val k selection; `__val_auc_gmm_avg__`; write `summary.json` + per-detector JSON | **9.5 + 10 + 12 (partial)** | ✗ (one banner spans three table steps) |
| STEP 13 (`:1593`) | Per-perturbation breakdown → `results_per_perturbation.json` | **11** | ✗ |
| STEP 14 (`:1701`) | Detection figures (ROC views, score dists, k-NN sensitivity, PCA/t-SNE OOD) | **12** | ✗ |

Additional naming drift *inside* STEP 11/12/14: the sub-banner comments still use the old `9a`/`9b`/`9c`/`9d`/`9e` and `12a`–`12g` labels (`:1114`, `:1164`, `:1308`, `:1314`, `:1373`, `:1706`, `:1746`, …) even though they sit under STEP 11/14 banners. So a single function spans three numbering schemes simultaneously. The module docstring's "Steps 1–12" (`:14-27`) is yet a fourth, idealised numbering that matches neither the banners nor `CLAUDE.md`.

### 3.2 True control-flow order (banners are misleading)

Because of the dead STEP 10 block sitting between the val load and the scoring, the *executed* order is:

1–7 baseline side (load → baseline profiles → MDX-v1 → MDX-v2 → Mahalanobis → BIC/AIC + K → GMM fit → baseline viz)
→ **8** build labeled test set
→ **9** test profiles + logits
→ **9.5** load val (sets `_has_val`)
→ **10** *(no-op: trajectory block commented out; only prints "[Step 10f] Generating figures...")*
→ **11** detector scoring (this is where every score array is built)
→ **12** evaluation, val-AUC k selection, `__val_auc_gmm_avg__`, `summary.json`
→ **13** per-perturbation
→ **14** figures.

So **detector scoring (STEP 11) runs *after* the (dead) trajectory block (STEP 10)**, not before it, and *after* the val load (STEP 9.5). This is the opposite of what `CLAUDE.md`'s table implies (table step 9 "scoring" comes *before* table step 9.5 "val"). In reality val is loaded (STEP 9.5) before any scoring, the dead trajectory block is between them, and scoring (STEP 11) consumes `_has_val` to also build the val score arrays inline (finding 8.2).

### 3.3 STEP 1 — model + LRP + ATOMs load (`:126-209`)

Branches on `conf.AGENT`. **TFV6** (`:130-166`): loads `TrainingConfig` from `pcla_agents/transfuserv6_pretrained/visiononly_resnet34/config.json`, instantiates `TFv6`, loads the **first** `model*.pth` checkpoint (`sorted(...)[0]`, dropping shape-mismatched keys, `:141-152`), wraps it in `LRPTFv6Model(backbone_eval=..., planning_decoder=...)`, sets `action_logits_available=False`, `speed_logits_available=True`. **WoR** (`:168-187`): loads `CameraModel` from `config_leaderboard.yaml` + `main_model_10.th`, wraps in `LRPCameraModel`, sets `action_logits_available=True`, `speed_logits_available=False`. The class map is `TFV6_CLASSES` (10-class grouped) for TFV6 else `CARLA_CLASSES` (`:192`). `ATOMsCarla` is built with `p_relevance=conf.FC_RELEVANCE_FILTER` (0.9), `default_cmd=conf.DEFAULT_CMD` (2), `mode_analysis=conf.MODE_ANALYSIS`, `use_reduced=False` (`:198-205`).

**Hardcoded model paths** (finding 1.2): the checkpoint directories are string literals at `:134` (TFV6) and `:172` (WoR); the module docstring's reference to `conf.MODEL_PATH` (`:11`) is to a config attribute that does not exist. The `action_logits_available`/`speed_logits_available` pair is the agent switch that drives every later PEOC/Action-entropy/MDX branch.

### 3.4 STEP 2/3/2.5-v2 — baseline profiles and the three baseline-side fits

**STEP 2** (`:214-244`): if `RECOMPUTE_BASELINE` or the npz is missing, `BaselineComputer(lrp, atoms).compute_and_save()` writes `baseline_<mode>.npz`; then the npz is loaded into `baseline_series [N,C]`, `baseline_mean [C]`, `baseline_cov [C,C]` (float64).

**STEP 3 = MDX-v1** (`:247-375`): banner says "Compute MDX baseline" (table calls it 2.5). For WoR (`action_logits_available`) it uses `model.get_features()` (penultimate) + the joint-action proxy via `ImageAgent`; for TFV6 it uses `lrp.get_backbone_features` (512-d) + the speed-derived proxy `[0, min(spd/25,1), 1 if spd<0.5 else 0]` (`:346`). The fast path reads HPC-precomputed `mdx_features.npz`; the slow path extracts locally. Fitted MDX saved to `mdx_parameters.pkl`. When `RECOMPUTE_MDX_BASELINE=False`, it loads the pkl (or fits from `mdx_features.npz` if the pkl is missing, `:356-375`).

**STEP 2.5-v2 = MDX-v2** (`:377-419`, *nested inside the TFV6 branch of STEP 3*): if `RECOMPUTE_MDX_V2_BASELINE`, it extracts F_c-256 features via `lrp.get_planning_action_and_features` **only when `MDX2_USE_FC_FEATURES=True`** — otherwise it falls back to the 512-d backbone + the same speed proxy (`:397-404`). The live config has `MDX2_USE_FC_FEATURES=False` (finding 7.6), so as configured MDX-v2 differs from MDX-v1 only in `bin_strategy` (quantile, `MDX2_USE_QUANTILE_BINNING=True`). Runs locally only (no HPC fast path). Saved to `mdx_v2_parameters.pkl`. **MDX-v2 produces no AUC** because its scoring/eval blocks are commented out downstream (finding 7.7; §3.9, §3.10).

### 3.5 STEP 4/5/6 — Mahalanobis, K resolution, GMM (`:423-509`)

**STEP 4** (`:423-444`): `MahalanobisDetector(ridge=conf.MAHAL_RIDGE)`.fit on `baseline_series`; `fit_threshold(percentile=99.0)`; save to `OUT_DIR/mahal_detector.npz`. This is the only detector with a saved, label-free runtime threshold (Topic 7 §2.6).

**STEP 5** (`:448-481`): `GMMClustering.select_n_components(..., criterion="bic")` and `"aic"` over `MAX_K=conf.GMM_MAX_K` (10); plots `clustering/gmm_model_selection.png`. **K resolution** in assignment order (`:475-479`): `N_COMPONENTS = best_k_bic`, overwritten by `conf.NUM_GMM_CLUSTERS` if not None, overwritten by `_cli.gmm_k` if not None — effective precedence **CLI `--gmm-k` > config `NUM_GMM_CLUSTERS` > BIC** (matches CLAUDE.md). With `NUM_GMM_CLUSTERS=10` the BIC result is discarded unless `--gmm-k` is passed (which is exactly what `sweep_clusters.py` does).

**STEP 6** (`:485-509`): `GMMClustering(n_components=N_COMPONENTS, covariance_type=conf.GMM_COV_TYPE, random_state=conf.RANDOM_SEED, ridge=conf.MAHAL_RIDGE).fit`; save `OUT_DIR/gmm.npz`; `predict_batch` → `baseline_cluster_labels`; log cluster sizes. Note `random_state=conf.RANDOM_SEED=17` here, whereas `GMMClustering.select_n_components` and its internal fit use a hardcoded `random_state=42` (Topic 7 §4) — the swept GMM and the model-selection GMM are seeded differently (finding 8.5).

### 3.6 STEP 7 — baseline visualisation (`:513-666`)

Produces, all via `save_figure` into the `make_output_dirs` tree: `attention/baseline_attention_bar.png`, `attention/attention_by_cluster.png`, per-cluster `attention/cluster_*_attention_bar.png`, `attention/attention_by_command.png` (only if >1 cmd present), `attention/cluster_<k>_representative.png` (the frame nearest each cluster mean), `pca/pca_baseline_by_run.png`, `pca/pca_baseline_clusters.png`, `tsne/tsne_baseline_by_run.png`, `tsne/tsne_baseline_clusters.png`. It also fits and stores `pca_obj` (`:640`) for reuse in the OOD overlays so test points project into the *same* baseline PCA space. Detailed figure semantics → Topic 12.

### 3.7 STEP 8 — labeled test set (`:670-723`)

Gated by `REAPPLY_PERTURBATIONS` OR missing `test_labeled.npz`. Defines the 5-way 20% spec (TFV6 at `:696-702`, WoR at `:704-710`), instantiates `PerturbationApplier(pm, model)`, calls `.apply(spec, seed=42, output_name="test_labeled")`. The TFV6 spec uses `intensity=conf.PGD_EPSILON` (14.0) with `fgsm_target=conf.PGD_TARGET` ("brake"); the WoR spec uses `intensity=conf.EPSILON` (8.0) with `fgsm_target="max_steer"`. The deferred-PGD scheme and the seed-42 local↔HPC parity are Topic 6 §2.3/§3.5. Always reloads via `LabeledTestLoader.load()` and prints a summary (`:720-722`).

### 3.8 STEP 9 — test profiles + logits (`:727-868`)

If `RECOMPUTE_TEST_ATOMS`: first the **deferred-PGD guard** warns for TFV6 if `pgd` frames are present (`:739-748`); then `atoms.reset()` and a per-frame loop calls `atoms.process_frame(wide, narrow, seg_wide, seg_narr, cmd=cmd)` — **without `spd`** for the test loop (finding 4.4, Topic 4), forcing the speed token to 0.0 for every test frame even though baseline profiles used the true speed. Per frame it also collects WoR `get_action_logits` (28-dim, if `action_logits_available`) or TFV6 `get_speed_logits` (8-bin, if `speed_logits_available`). Saves `test_profiles_<mode>.npy`, a `test_profiles_<mode>.keys.npy` alignment key of `(run_id, frame_idx)` (`:810-814`), and the logits arrays. The **else** branch (`:821-868`) reloads cached profiles, raises on length mismatch (`:833-838`), and enforces the `(run_id, frame_idx)` alignment key when present (raising on mismatch, `:850-856`) or warns if the key file is absent (HPC data predating the guard, `:858-865`). This alignment guard is what makes merging HPC-crafted PGD profiles safe.

### 3.9 STEP 9.5 — val load (`:872-902`) and STEP 10 — dead trajectory block (`:905-1106`)

**STEP 9.5**: if `val_profiles_<mode>.npy` AND `val_labeled.npz` both exist, loads `val_data` (via `LabeledTestLoader.load_val()`), `val_profiles`, `val_labels`, sets `_has_val=True`. Otherwise `_has_val=False` and emits a `warnings.warn` plus a print that "k selected on test (leakage)" (`:894-902`) — the fallback hazard of finding 7.8. **STEP 10**: dead (§2.4). The orphaned `print("[Step 10f] Generating figures...")` at `:1072` still executes (finding 8.3).

### 3.10 STEP 11 — detector scoring (`:1109-1405`)

This is where every score array is built (sub-banners 9a–9e). All distance scorers go through the stateless `DistanceComputer` (Topic 7 §2.1), not the stateful detector classes:

- **9a single-Gaussian** (`:1114-1162`): `scores_mahal_single` (`compute_mahalanobis`, `regularization=conf.MAHAL_RIDGE`), `scores_euclid_single`, `scores_knn_by_k` for `KNN_K_VALUES=[1,5,10,25,50,100,250]` (`normalize=True`), `scores_jsd_single` (vs `baseline_mean`). Wasserstein single is commented out (`:1156-1162`).
- **9b GMM** (`:1164-1243`): `scores_mahal_gmm` (`compute_gmm_distance` mode `"nearest"`, also yields `nearest_clusters`), `scores_euclid_gmm`, `scores_jsd_gmm`. Wasserstein-GMM commented out. When `_has_val`, the same three GMM scorers run on `val_profiles` → `scores_*_gmm_val` (`:1210-1243`).
- **9b.3 GMM k-NN** (`:1245-1306`): for each k, find the nearest GMM centroid, run k-NN within that cluster's baseline subset (fallback to full baseline if the cluster is smaller than k). When `_has_val`, the val variants `scores_knn_val_by_k` and `scores_knn_gmm_val_by_k` are also built.
- **9c Action entropy** (`:1308-1312`): WoR only — `ActionEntropyDetector(from_logits=True).score_batch(test_logits_all)`.
- **9d MDX** (`:1314-1348`): cached read of `mdx_scores_<mode>.npy` when valid (cache invalidated by `RECOMPUTE_MDX_TEST_SCORES`, `RECOMPUTE_MDX_BASELINE`, or `RECOMPUTE_TEST_ATOMS`, `:1318-1323`); otherwise per-frame feature extraction (WoR `get_features` / TFV6 `get_backbone_features`) + `mdx.score`, cached to disk. **MDX-v2 scoring (9d-v2) is entirely commented out** (`:1350-1371`).
- **9e PEOC** (`:1373-1379`): TFV6 only — `ActionEntropyDetector(from_logits=True).score_batch(test_speed_logits)` over the 8-bin speed distribution.

A sanity table (`:1381-1405`) prints clean/perturbed mean separation per detector; MDX-v2 is omitted from this roster (`:1387`).

### 3.11 STEP 12 — evaluation, k selection, `__val_auc_gmm_avg__`, save (`:1408-1590`)

`DetectorEvaluator().evaluate(scores, labels, name)` computes AUC + Youden-J per detector (Topic 7 §3.11). The k-NN k and GMM-k-NN k are each selected by **val AUC** when `_has_val` (`:1486-1489`, `:1517-1520`), else by test-AUC argmax with a leakage warning (`:1491-1493`, `:1522-1525`). The headline `__val_auc_gmm_avg__` is the mean over four GMM detector val AUCs — Mahalanobis-GMM, Euclidean-GMM, JSD-GMM, k-NN-GMM (Wasserstein-GMM commented out, `:1534-1543`). **Note the CLAUDE.md description says "five GMM detector variants" but only four are live** because Wasserstein-GMM is disabled (finding 8.4). Results: per-detector `results_<name>.json`, per-k `results_knn_k*.json` / `results_knn_gmm_k*.json` (written in STEP 14, `:1825-1828`), and the combined `summary.json` (`:1585-1589`) carrying `{detector: {auc, youden_j}}` plus `__val_auc_gmm_avg__`. MDX-v2 results are commented out (`:1552-1556`), so `all_results` (`:1564-1576`) excludes it.

### 3.12 STEP 13 — per-perturbation breakdown (`:1593-1698`)

`LabeledTestLoader.split_by_perturbation`; for each non-clean perturbation, builds an `eval_mask = clean | this-perturbation` and evaluates each detector on that subset against `eval_labels` (clean vs *this* perturbation only). Writes `results_per_perturbation.json` as `{pert: [{detector_name: auc}, …]}` (`:1693-1697`). MDX-v2 and Wasserstein rows are commented out.

### 3.13 STEP 14 — detection figures (`:1701-1888`)

ROC views (`roc/roc_all_detectors.png`, `roc_static_detectors.png`, `roc_gmm_vs_static.png`, `roc_top5.png`), per-detector score distributions (`scores/score_dist_*_single|gmm.png`), per-perturbation ROC (`roc/roc_<pert>_*.png`), k-NN sensitivity (`roc/knn_k_sensitivity.png`, `knn_gmm_k_sensitivity.png` — showing **val** AUC when available, else test, `:1804-1812`), and PCA/t-SNE OOD overlays (`pca/pca_ood_*.png`, `tsne/tsne_ood_*.png`). Figure semantics → Topic 12.

### 3.14 Results-directory layout (`make_output_dirs`, `visualization_carla.py:1005-1027`)

`OUT_DIR = RESULTS_DIR / "atoms_analysis_mode_<mode>"` (`:110`). `make_output_dirs` creates six subdirectories under it: `pca/`, `tsne/`, `roc/`, `attention/`, `scores/`, `clustering/`. JSON results (`summary.json`, `results_*.json`, `results_per_perturbation.json`, `results_knn_*.json`) and the saved models (`mahal_detector.npz`, `gmm.npz`) are written at the `OUT_DIR` root, not in a subdirectory. Note: the directory name is `atoms_analysis_mode_<mode>`, whereas `CLAUDE.md`'s Data-Layout section calls it `atoms_analysis/` (and references a `trajectory_analysis/` subdir that the dead STEP 10 no longer creates) — finding 8.6.

### 3.15 `sweep_clusters.py` mechanics (`sweep_clusters.py:1-141`)

`K_VALUES_DEFAULT = [10, 8, 12]` (`:33`). The `--k-values` help string says "default: 2 4 6 8 … 20" (`:42`) — **stale**, contradicting both the actual default and the docstring example `--k-values 2 4 6 8` (cross-ref finding 1.5; the literal is now `[10,8,12]`, not the `[18,20,22]` recorded in 1.5, so the help text is wrong against a different literal — finding 8.7). The driver: reads `conf` for `RESULTS_DIR`/`MODE_ANALYSIS`, warns if any expensive recompute flag is True (`:73-91`, prompting unless `--yes`), then for each K runs `subprocess.run([python, "run_analysis.py", "--gmm-k", str(K)])` (`:108-111`), and on success copies `RESULTS_DIR/atoms_analysis_mode_<mode>/` (the scratch output) into `RESULTS_DIR/<K> clusters/atoms_analysis_mode_<mode>/` (`:96`, `:123-127`). It never edits `atoms_config.py` — K is passed purely via `--gmm-k`. The snapshot folders are what `summarize_results.py` treats as authoritative; the bare scratch folder reflects only the last K run.

### 3.16 `summarize_results.py` and SUMMARY.md (`summarize_results.py:1-849`)

`discover_runs` (`:182-233`) globs every `*/<results_subdir>/**/summary.json`, infers `agent` (top path part), `mode` (`mode_(\d)` regex), and K (`_gmm_k_from_keys`, parsed from a `K=<n>` token in a detector name, `:118-123`), then de-duplicates by `(agent, mode, K)` preferring the named `<K> clusters/` snapshot over bare scratch folders (`:228-233`). Bare folders with no mode suffix get their mode inferred by matching the **K-invariant non-GMM AUC fingerprint** (`_nongmm_fingerprint`, `:175-179`) against a labelled mode's fingerprint — exploiting the fact that non-GMM detectors are constant across K (a key assumption stated at `:14-18`). `load_summary` reads `__val_auc_gmm_avg__` into `Run.val_auc_gmm_avg` (`:129`, `:194-203`).

`build_markdown` (`:332-666`) emits **SUMMARY.md** with these sections:
1. **Headline** — best detector + AUC + best-K per (agent, mode), plus the MDX feature-space AUC.
2. **AUC matrix** — detector × K, marking K-dependence and best K (only `-GMM` detectors vary with K).
3. **Which cluster count (K) works best?** — three sub-tables: (a) best K per GMM detector vs its non-GMM twin; (b) aggregate test-set best mean-K; and (c) the **val-set K selection** (only emitted when any run has `val_auc_gmm_avg`, `:435-469`): a ranked top-5 table by val avg GMM AUC, with the test avg at that K and Δ(test−val), plus the recommendation prose "Use the #1 K … Do *not* use the test-set best K — that constitutes hyperparameter leakage" (§2.3).
4. **Distance robustness** — per-agent mean AUC and worst-perturbation AUC ranking.
5. **Per-perturbation breakdown** — best AUC over K per (perturbation, detector), with the winning detector per perturbation.
6. **k-NN k sweep** — AUC vs k from `results_knn_k*.json`, marking the chosen k.
7. **Live-perturbation inventory** — qualitative-only (no AUC), from PNG stems in `*live*` folders (`inventory_live`, `:236-249`).
8. **Mode 1 vs Mode 2 comparison** — per-detector and per-perturbation Δ, emitted only when both modes are present.

It also writes PNG heatmaps (`heatmap_K_*`, `heatmap_pert_*`, `curve_meanGMM_vs_K_*`, `mode_comparison_scatter.png`) to `results_summary/` (or `results_summary_alt/` with `--alt`). **The K-recommendation** (Section 3c): for each `(agent, mode)`, rank the snapshot Ks by their stored `val_auc_gmm_avg` descending; recommend the top-ranked K, reporting the test avg GMM AUC at that K as the *quoted-but-not-selected* number — implementing the offline, leakage-free K selection of §2.3.

---

## 4. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| `RECOMPUTE_BASELINE` | False | `atoms_config.py:37`; gate `run_analysis.py:223,228` | config | recompute Step 2 baseline ATOMs |
| `RECOMPUTE_TEST_ATOMS` | False | `atoms_config.py:38`; gate `:732` | config | recompute Step 9 test ATOMs |
| `REAPPLY_PERTURBATIONS` | False | `atoms_config.py:39`; gate `:680` | config | rebuild Step 8 `test_labeled.npz` |
| `RECOMPUTE_MDX_BASELINE` | False | `atoms_config.py:40`; gate `:263,317` | config | refit MDX-v1 (Step 3) |
| `RECOMPUTE_MDX_V2_BASELINE` | False | `atoms_config.py:41`; gate `:384` | config | refit MDX-v2 (Step 2.5-v2) |
| `RECOMPUTE_MDX_TEST_SCORES` | False | `atoms_config.py:42`; gate `:1320` | config | re-score MDX on test set |
| `MODE_ANALYSIS` | 2 | `atoms_config.py:23`; used `:110,116,225,806,…` | config | attribution mode; suffixes all profile/score artifacts and `OUT_DIR` |
| `NUM_GMM_CLUSTERS` | 10 | `atoms_config.py:21`; `:476-477` | config | forced K; overrides BIC; `None` → use BIC |
| `GMM_MAX_K` | 10 | `atoms_config.py:89`; `:456` | config | upper K in BIC/AIC sweep |
| `GMM_COV_TYPE` | "full" | `atoms_config.py:90`; `:462,492` | config | GMM covariance type |
| `--gmm-k` (CLI) | None | `run_analysis.py:53-54`; applied `:478-479` | CLI | overrides config K; set by `sweep_clusters.py` |
| `RANDOM_SEED` | 17 | `atoms_config.py:91`; `:493` | config | live GMM fit seed (≠ select_n_components' hardcoded 42) |
| GMM model-selection seed | 42 | `clustering.py` (Topic 7 §4) | code | BIC/AIC sweep GMM seed |
| `MAHAL_RIDGE` | 0.01 | `atoms_config.py:88`; `:431,1121,1176,1218,494` | config | covariance ridge (single + GMM Mahalanobis) |
| Mahalanobis threshold pct | 99.0 | `run_analysis.py:440` | code | label-free 99th-pct runtime decision boundary |
| `KNN_K_VALUES` | `[1,5,10,25,50,100,250]` | `run_analysis.py:1134` | code | k sweep for k-NN and GMM-k-NN |
| k-NN `normalize` | True | `:1142,1269,1283,1303` | code | L2-normalise before k-NN distance |
| labeling seed | 42 | `run_analysis.py:715` | code | frame-to-entry shuffle (must match HPC) |
| `FC_RELEVANCE_FILTER` | 0.9 | `atoms_config.py:24`; `:200` | config | ATOMs p_relevance mass filter |
| `DEFAULT_CMD` | 2 | `atoms_config.py:87`; `:201` | config | ATOMs default command (comment says 3=FOLLOW_LANE; finding 1.7) |
| `EXPERIMENT_VARIANT` | "alternative" | `atoms_config.py:63` | config | re-roots all 4 data/result dirs to `*_alt` |
| `MDX2_USE_FC_FEATURES` | False | `atoms_config.py:45`; `:397-404` | config | False → MDX-v2 on 512-d backbone (finding 7.6) |
| `MDX2_USE_QUANTILE_BINNING` | True | `atoms_config.py:46`; `:411` | config | MDX-v2 binning strategy |
| `MDXDetector.n_pca_components` | 50 | `:305,351,367,412` | code | MDX PCA dim |
| `K_VALUES_DEFAULT` (sweep) | `[10, 8, 12]` | `sweep_clusters.py:33` | code | default K sweep (help text stale; finding 8.7) |
| `_VAL_TOP_N` (summary) | 5 | `summarize_results.py:434` | code | val K-recommendation top-N shown |
| heatmap AUC range | `[0.40, 0.75]` | `summarize_results.py:672` | code | summary heatmap colour scale |
| `--data-root` / `--out` | `data` / `results_summary` | `summarize_results.py:808-817` | CLI | summary I/O roots (`--alt` → `*_alt`) |
| model paths | string literals | `run_analysis.py:134,172` | hardcoded | TFV6/WoR checkpoint dirs (finding 1.2) |

---

## 5. Known limitations & open issues

- **In-code STEP banners do not match `CLAUDE.md`'s pipeline-table numbering** (finding 8.1) — the single largest documentation hazard here. The banner→table offset is +1 from STEP 4 onward (banner 4 = table 3, …, banner 9 = table 8), with banner 3 = table 2.5, banner 10 = table 8.5 (dead), banner 11 = table 9, banner 12 spanning table 9.5+10+12, banner 13 = table 11, banner 14 = table 12. The module docstring's "Steps 1–12" is a fourth, idealised numbering. §3.1 is the reconciliation table.
- **Detector scoring runs after the (dead) trajectory block and after the val load** (finding 8.2) — `CLAUDE.md`'s table implies scoring (step 9) precedes val (step 9.5); the code loads val (STEP 9.5) *before* scoring (STEP 11), with the dead STEP 10 trajectory block between them. Scoring consumes `_has_val` to build val score arrays inline.
- **Trajectory step is dead; on-disk `trajectory_analysis/` figures are stale** (finding 8.3, confirms `CLAUDE.md`) — `run_analysis.py:936-1106` is commented out; only the orphaned `print("[Step 10f] Generating figures...")` (`:1072`) still fires, printing a misleading progress line for a step that writes nothing. No `trajectory_*.png` / `displacement_*` are produced.
- **`__val_auc_gmm_avg__` averages four GMM detectors, not five** (finding 8.4) — `CLAUDE.md` (step 9.5) and `summarize_results.py` prose say "five GMM detector variants"; the live code (`run_analysis.py:1534-1543`) includes only Mahalanobis-GMM, Euclidean-GMM, JSD-GMM, k-NN-GMM because Wasserstein-GMM is commented out (Topic 7 §5). The reported mean is over four.
- **`select_n_components` and the live GMM fit use different seeds** (finding 8.5) — the BIC/AIC sweep GMM is seeded `random_state=42` (hardcoded in `clustering.py`) while the fitted GMM at the chosen K uses `random_state=conf.RANDOM_SEED=17` (`run_analysis.py:493`). The model-selection GMM and the deployed GMM are therefore not the same fit even at identical K, a subtle reproducibility wrinkle.
- **`OUT_DIR` name and `trajectory_analysis/` subdir disagree with `CLAUDE.md`'s Data Layout** (finding 8.6) — the real output dir is `atoms_analysis_mode_<mode>/` (`run_analysis.py:110`), not `atoms_analysis/`; and `make_output_dirs` creates only `{pca,tsne,roc,attention,scores,clustering}/`, never the `trajectory_analysis/` subdir the docs list (it was created by the now-dead STEP 10).
- **`sweep_clusters.py` help text stale** (finding 8.7, extends 1.5) — `--k-values` help says "default: 2 4 6 8 … 20" while the actual default is `K_VALUES_DEFAULT = [10, 8, 12]` (`:33`). Finding 1.5 recorded the literal as `[18,20,22]`; the literal has since changed to `[10,8,12]` but the help string was never synced — the inconsistency persists against a different literal.
- **Hardcoded model paths / nonexistent `conf.MODEL_PATH`** (finding 1.2) — TFV6/WoR checkpoint dirs are string literals at `run_analysis.py:134,172`; the docstring references `conf.MODEL_PATH` which does not exist.
- **MDX-v2 fitted but never scored → no AUC** (finding 7.7) — the entire MDX-v2 test-scoring (`:1350-1371`), evaluation (`:1552-1556`), sanity-roster entry (`:1387`), `all_results` entry (`:1573`), and per-perturbation block (`:1662-1665`) are commented out. MDX-v2 is fitted/saved (Step 2.5-v2) but contributes no number; the thesis should report MDX-v1 only or re-enable v2.
- **MDX-v2 feature ablation disabled** (finding 7.6) — `MDX2_USE_FC_FEATURES=False` collapses MDX-v2 to MDX-v1-with-quantile-bins on 512-d backbone features.
- **k-on-test leakage when val is absent** (finding 7.8) — both k-NN selections fall back to test-AUC argmax with a non-enforced `WARNING` print (`:1491-1493,1522-1525`); STEP 9.5 also warns at the val-load level (`:894-902`).
- **Wasserstein fully disabled** (Topic 7 §5) — every Wasserstein single/GMM/per-pert call and the `WassersteinDetector` are commented out across `run_analysis.py` (`:1156-1162,1200-1206,1236-1242,1455-1465,1642-1649,1754-1755`), because TFV6 profiles are signed (finding 4.5).
- **Test ATOMs computed without `spd`** (finding 4.4) — `run_analysis.py:768` omits `spd=`, forcing the test-frame speed token to 0.0 while baseline used true speed (a baseline/test conditioning asymmetry independent of perturbations).

---

## 6. Cross-references

- **01_architecture_overview.md** — `atoms_config.py` as single source of truth for every recompute flag, `MODE_ANALYSIS`, `NUM_GMM_CLUSTERS`/`GMM_MAX_K`/`GMM_COV_TYPE`, `MAHAL_RIDGE`, `RANDOM_SEED`, `EXPERIMENT_VARIANT` path re-rooting; the `--gmm-k` CLI override; findings 1.2 (hardcoded model paths / `conf.MODEL_PATH`), 1.5 (sweep help text), 1.7 (DEFAULT_CMD), 1.10 (K vs MAX_K), 5.7 (alternative-variant paths).
- **02_agents.md** — the feature/logit extractors this orchestration calls per agent: TFV6 `get_backbone_features`/`get_planning_action_and_features`/`get_speed_logits`, WoR `model.get_features`/`get_action_logits`; the single-member vs ensemble caveat (finding 2.5) — `run_analysis.py:141` loads `sorted(...)[0]` only.
- **04_atoms.md** — `ATOMsCarla.process_frame` is the per-frame call in Steps 2 and 9; MODE_ANALYSIS 1 vs 2; finding 4.4 (test profiles computed without `spd`), 4.5 (signed TFV6 profiles → Wasserstein disabled).
- **05_dataset_creation.md** — the baseline/test/val frame sets consumed here (`BaselineDataLoader`, `LabeledTestLoader.load`/`load_val`); the val set whose disjointness underpins leakage-free k/K selection; the alternative-split paths.
- **06_perturbations.md** — the 5-way 20% spec at `run_analysis.py:696-710`; `PerturbationApplier.apply` (Step 8); the deferred-PGD guard (Step 9, `:735-748`) and the alignment-key check (`:840-865`); the per-perturbation breakdown labels.
- **07_distances_and_detectors.md** — every detector scored in STEP 11 and evaluated in STEP 12; the detector roster table; the `DistanceComputer`-vs-detector-class split (stateless scoring); k selection via val AUC (§2.7 there); findings 7.1 (forced K), 7.6/7.7 (MDX-v2), 7.8 (k leakage), 7.11 (clamping).
- **09_online_analysis.md** — `run_online_analysis.py` reuses the same detectors but only the 99th-pct label-free threshold is deployable; live runs feed `summarize_results.py`'s Section 7 inventory (qualitative, no AUC).
- **10_hpc_pipeline.md** — `mdx_features.npz` precomputation that the Step 3 fast path reads; chunked baseline/test/val profile computation; the HPC-crafted PGD profiles merged before STEP 9's load path.
- **11_validation_and_testing.md** — the LRP/ATOMs correctness suites that validate the profiles this pipeline scores.
- **12_visualization.md** — `make_output_dirs` tree and every figure produced in Steps 7 and 14; `summarize_results.py` heatmaps; `viz_config.py` styling.
- **99_bugs_and_findings.md** — Topic 8 findings 8.1–8.7; cross-references 1.2, 1.5, 1.7, 1.10, 2.5, 4.4, 4.5, 5.7, 7.1, 7.6, 7.7, 7.8.
