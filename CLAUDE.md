# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **master's thesis research project** on explainability-based out-of-distribution (OOD) detection for autonomous driving agents. The codebase has two layers:

1. **PCLA** (Pretrained CARLA Leaderboard Agents) — infrastructure for deploying pretrained agents in the CARLA simulator. Used to run agents and collect driving data.
2. **ATOMs_Analysis** — the research core. Implements the full pipeline for computing attention profiles via LRP + ATOMs, fitting baseline distributions, and detecting OOD inputs by measuring divergence from those baselines.

The scientific goal is to show that attention profiles derived from LRP are a meaningful signal for OOD detection: when the agent encounters perturbed or adversarial inputs, its attention distribution shifts measurably away from the clean-driving baseline.

**Primary agent: TransFuser v6 (TFV6).** The World on Rails (WoR) agent is also supported but TFV6 is the current experimental focus. Set `conf.AGENT = "TFV6"` or `"WOR"` in `atoms_config.py` to switch.

Target platform: **Linux Ubuntu 22**, **CARLA 0.9.16**, **Python 3.10**.

---

## Key Concepts

### LRP (Layer-wise Relevance Propagation)
Backpropagates relevance from the network output to the input pixels. Two implementations:

- **WoR** (`ATOMs_Analysis/saliency/lrp_analysis.py`, `LRPCameraModel`): z⁺-rule throughout via `zennit`. Single-pass joint LRP over the dual-camera WoR `CameraModel`.
- **TFV6** (`ATOMs_Analysis/saliency/lrp_transfuser.py`, `LRPTFv6Model`): AttnLRP (Achtibat et al. 2024) — custom autograd Functions for softmax (Prop 3.1) and bilinear matmul (Prop 3.3); ε-rule for attention Linear layers; AlphaBeta for Conv/FFN. Wraps backbone + PlanningDecoder in `TFv6FullModelForLRP` so a single zennit composite context covers the full attribution graph. Every residual/skip-connection `+` (ResNet34 `BasicBlock`s in the image *and* lidar encoders, GPT fusion blocks, backbone cross-modal fuse, PlanningDecoder self/cross-attn + FFN residuals) is routed through `LRPResidualAdd`, a custom autograd Function implementing Ratio-Based Relevance Splitting (Otsuki et al. 2024, arXiv:2407.09115) — `R_a = R·|a|/(|a|+|b|+ε)` — so relevance is conserved rather than duplicated at every junction. See `docs/design_decisions.md` ("TFV6 LRP: residual/skip-connection conservation") for why absolute value (not zennit's signed `Norm` rule) is used, and `docs/lrp_todo.md` Issue 8 for status (fixed and HPC-verified 2026-07-01: 12/13 diagnostics pass — see D13 in `tfv6_lrp_diagnostics.py`). Baseline/test/val ATOMs profiles on disk predate this fix and should be recomputed before drawing OOD-AUC conclusions.

**TFV6 two-pass scheme:**
- **LRP1** (`beg="output", end="fc"`): softmax-distribution seed over 8 speed bins → backprop through `target_speed_decoder` (Linear→ReLU→Linear) to `speed_query` level. Returns 256-dim node relevance vector.
- **LRP2** (`beg="fc", end="input"` with `node_id`): one-hot at F_c node k → backprop through full model to input pixels. Returns [1,3,H,W] pixel map.
- **Output→input** (`beg="output", end="input"`): same softmax seed, backprop all the way to pixels in one pass.

**LRP1 seed rationale:** `target_speed_decoder` is trained with a two-hot target over 8 speed bins [0, 4, 8, 10, 13.9, 16, 17.8, 20] m/s. Using argmax causes discontinuities at bin boundaries. The softmax distribution seed is used instead (`grad_outputs = softmax(speed_logits.detach())`), giving a smooth attribution that reflects the full predicted speed distribution.

**F_c layer:** 256-dim `speed_query` token output by `PlanningDecoder.transformer_decoder`, just before `target_speed_decoder`. This is the closest equivalent to the ATOMs paper's F_c ("the final world model on which the agent chooses its action"). See `docs/lrp_todo.md` Decision A/B for rationale.

### ATOMs (Attention-Oriented Metrics)
Introduced by Beylier et al. (NeurIPS 2024 workshop). Converts raw LRP heatmaps into structured, object-level attention vectors by intersecting relevance maps with semantic segmentation masks. Two levels:
- **Hierarchical-attention** `h(o)`: the sum (or mean, see `NORMALIZE_BY_PIXEL_COUNT`) of relevance over object `o`'s pixels that carry nonzero relevance, then re-normalized across objects. The paper (Beylier et al.) uses mean (R̄); the current default (`NORMALIZE_BY_PIXEL_COUNT = False`) uses raw sum on advisor recommendation.
- **Combinatorial-attention** `c(T)`: fraction of frames where a neuron attends jointly to a subset of objects `T` (using threshold α = 0.25).

For TFV6, CARLA's grouped semantic segmentation (10 classes, `save_grouped_semantic=True`) is used via `TFV6_CLASSES` in `atoms_carla.py`. Implemented in `ATOMs_Analysis/saliency/atoms_carla.py`.

**ATOMs saliency attributes** (set after each `process_frame`):
- `saliency_data_wide_default` — map from the default (softmax) seed; this is what feeds `_hierarchical`.
- `saliency_data_wide_brake` / `saliency_data_wide_drive` — forced brake/drive maps, populated only when `PLOT_COMPARATIVE_REL=True`; do **not** feed `_hierarchical`.
- When `PLOT_COMPARATIVE_REL=False`, brake/drive slots mirror the default map.

### OOD Detection Strategy
1. Collect clean baseline driving frames; compute one ATOMs profile per frame. Dimensionality is agent-dependent: 29 for WOR (full `CARLA_CLASSES`, tags 0–28) and 10 for TFV6 (grouped `TFV6_CLASSES`). (Earlier docs said "23-dim"; that was a stale CARLA tag count.) Optional extra profile blocks exist behind `conf.ADD_BRAKE_SEEDS` (brake-counterfactual seed, output→input) and `conf.ADD_WAYPOINT_SEEDS` (waypoint-query activation seed, `wp_fc`→input) — each adds `num_classes` dims and one backward pass, blocks normalized to sum 1/n_blocks. **Both default False:** measured 2026-07-02, every decoder-level seed produces a pixel map with cosine ≥ 0.994 to the default speed-seeded map — TFV6 LRP maps are effectively seed-invariant at the planning-decoder level, so extra blocks are redundant copies (see "Seed-invariance of TFV6 LRP maps" in `docs/design_decisions.md`). Use `atoms.profile_dim` / `atoms.profile_names`, not `atoms.num_classes` / `class_names`, for profile array shapes and labels; `run_analysis.py` fails loudly on a dim mismatch with cached profiles.
2. Fit a statistical model (Gaussian or GMM) on the baseline profile cloud.
3. At test time, score each frame's profile by its distance from the baseline distribution.
4. High distance → OOD flag. Evaluated against ground-truth perturbation labels via ROC / AUC.

---

## Analysis Pipeline (`run_analysis.py`)

The main entry point. Runs end-to-end and produces all figures and JSON results for the thesis.

| Step | What happens |
|------|-------------|
| 1 | Load `CameraModel` weights; initialize `LRPCameraModel` and `ATOMsCarla` |
| 2 | Compute ATOMs profiles on baseline frames → `baseline.npz` (mean, covariance, per-frame series) |
| 2.5 | Fit `MDXDetector` (MDX-v1) on baseline — 512-d backbone features + speed-derived action proxy. For TFV6: reads pre-computed `mdx_features.npz` if available; otherwise extracts locally. Controlled by `RECOMPUTE_MDX_BASELINE`. |
| 2.5-v2 | Fit `MDXDetector` (MDX-v2) on baseline — 256-d F_c (`speed_query`) features + waypoint steer proxy + quantile binning. Runs locally only (full planning-decoder forward per frame). Controlled by `RECOMPUTE_MDX_V2_BASELINE`. Saves `mdx_v2_parameters.pkl`. |
| 3 | Fit single-Gaussian `MahalanobisDetector` on baseline profiles; set threshold at 99th percentile |
| 4 | BIC/AIC sweep over K=1..MAX_K to select GMM component count. K is then resolved in priority order: (1) `--gmm-k` CLI arg (used by `sweep_clusters.py`), (2) `conf.NUM_GMM_CLUSTERS` if not None, (3) BIC-selected K. |
| 5 | Fit `GMMClustering` with selected K; assign baseline frames to clusters |
| 6 | Visualize baseline: attention bar chart, per-cluster attention comparison, PCA coloured by run and cluster |
| 7 | Apply perturbation mix to test set → `test_labeled.npz` with ground-truth labels |
| 8 | Compute ATOMs profiles + action logits on labeled test set → `test_profiles.npy` |
| 8.5 | Trajectory analysis: match clean↔perturbed pairs by (run_id, frame_idx); compute displacement stats and PCA trajectories per perturbation type. **DISABLED — this block is currently commented out in `run_analysis.py`; any `trajectory_analysis/` figures on disk are stale.** |
| 9 | Score test profiles with all detectors: Mahalanobis-single, Mahalanobis-GMM, Euclidean, k-NN, JSD, MDX-v1, MDX-v2, Action Entropy. Also computes **PGD success rate** (TFV6 only): counts frames where the brake-target attack forced softmax(speed\_logits)[0] ≥ 99.9%; printed to stdout and saved to `summary.json` as `__pgd_success__`. |
| 9.5 | Load val profiles + `val_labeled.npz`. When present: score val profiles with all GMM detectors (Mahalanobis-GMM, Euclidean-GMM, JSD-GMM, Wasserstein-GMM, k-NN-GMM); compute per-detector val AUC and their mean → `__val_auc_gmm_avg__`; use val AUC to select k-NN/GMM-kNN k. Falls back to test-set k selection with a warning when val files are absent. |
| 10 | Evaluate each detector: ROC curve, AUC, Youden-J optimal threshold |
| 11 | Per-perturbation breakdown: evaluate each detector separately on each perturbation type |
| 12 | Save all figures (PNG) and results (JSON) to `conf.RESULTS_DIR/atoms_analysis/`. `summary.json` includes `__val_auc_gmm_avg__` when val set is present; `sweep_clusters.py` copies this per-K. `summarize_results.py` reads it to produce a **val-set K selection recommendation** (Section 3 of SUMMARY.md). |

Key flags in `atoms_config.py` that control re-computation: `RECOMPUTE_BASELINE`, `RECOMPUTE_MDX_BASELINE`, `RECOMPUTE_MDX_V2_BASELINE`, `REAPPLY_PERTURBATIONS`, `RECOMPUTE_TEST_ATOMS`. Profile-enrichment flags (2026-07-02): `ADD_BRAKE_SEEDS` / `ADD_WAYPOINT_SEEDS` (extra profile blocks, +1 backward/frame each — **both default False after the seed-invariance measurement**, see `docs/design_decisions.md`) and `USE_REAL_TARGET_POINTS` (real TP conditioning from npz, default True). All require recomputing profiles when toggled. GMM cluster count K is overridden at runtime by the `--gmm-k` CLI arg (passed automatically by `sweep_clusters.py`). PGD settings: `PGD_TARGET="brake"`, `PGD_EPSILON=4.0`, `PGD_N_STEPS=5` — must stay in sync with `hpc/prep_test.py` (ε) and `hpc/array_test_task.sh` (target, steps).

---

## `ATOMs_Analysis/` Structure

```
ATOMs_Analysis/
├── atoms_config.py            # Central ExperimentConfig — all paths, hyperparameters
├── perturbation_manager.py    # Registry of image perturbations (@_register_wide decorator)
├── saliency/
│   ├── lrp_analysis.py        # LRPCameraModel — zennit LRP for WoR (z+-rule, dual-camera)
│   ├── lrp_transfuser.py      # LRPTFv6Model — AttnLRP for TFV6 (see Key Concepts above)
│   └── atoms_carla.py         # ATOMsCarla — per-frame hierarchical + combinatorial attention
├── detection/
│   ├── baseline_dataset.py    # BaselineDataCollector, BaselineComputer, BaselineDataLoader
│   ├── dataset.py             # LabeledTestLoader, PerturbationApplier, PerturbationSpec
│   ├── detectors.py           # MahalanobisDetector, ActionEntropyDetector,
│   │                          #   MDXDetector (bin_strategy="equal-width"|"quantile"),
│   │                          #   EuclideanDetector, KNNDetector, JensenShannonDetector,
│   │                          #   DetectorEvaluator (ROC / AUC / Youden)
│   └── clustering.py          # GMMClustering — BIC/AIC sweep + GMM fit + per-cluster scoring
└── utils/
    ├── visualization_carla.py # All plotting functions (PCA, ROC, bar charts, trajectory plots)
    ├── viz_config.py          # Shared style config (colors, DPI, thesis-wide formatting)
    ├── distance_computer.py   # DistanceComputer — Mahalanobis, Euclidean, k-NN, JSD (stateless)
    ├── lrp_test_suite.py      # Diagnostic suite for validating LRP + ATOMs correctness (WoR)
    └── tfv6_test_suite.py     # TFV6TestSuite — property-based tests for LRPTFv6Model + ATOMs
```

### Key design notes
- `atoms_config.py` is the single source of truth for all paths and hyperparameters. Edit it rather than hardcoding values in scripts.
- `ATOMsCarla.process_frame(wide, narr, seg_wide, seg_narr, cmd, spd, data, target_points)` is the per-frame API; call `atoms.reset()` between datasets. For TFV6, `process_frame` is called without `data=`; `_make_minimal_data` rebuilds the conditioning dict from `cmd`/`spd`/`target_points`. With `conf.USE_REAL_TARGET_POINTS = True` (default since 2026-07-02) the three TP tokens (current/previous/next, ego frame) come from the npz via `extract_target_points(data, i)` — matching the deployed model's route conditioning. Older npz without TP keys fall back to zeros with a one-time warning (`acceleration` stays zero; the leaderboard config doesn't use it).
- `DistanceComputer` is stateless (static methods); detectors (`MahalanobisDetector`, etc.) are stateful (fit/save/load).
- Visualization functions return `matplotlib.Figure` objects; use `save_figure(fig, path)` to write them.
- `viz_config.py` defines `apply_default_style()` — call it at the top of any new plotting script to keep figures consistent across the thesis.
- For `LRPTFv6Model`, see `docs/lrp_todo.md` for the full history of design decisions and remaining open issues (BatchNorm canonization, contrastive seeding).

### Perturbation types (via `PerturbationManager`)
| Name | Description |
|------|-------------|
| `gaussian_noise` | Additive Gaussian noise on wide/narrow camera images |
| `brightness_scale` | Multiplicative brightness change |
| `camera_loss` | Zeros out one camera stream (simulates sensor dropout) |
| `pgd` | PGD adversarial attack — target=brake, ε=4 pixel units, 5 steps (TFV6). Attack is deferred to the HPC array job; `test_labeled.npz` stores clean pixels with `label=1` until then. |

---

## Raw Data Creation Pipeline (TFV6)

All TFV6 driving footage originates from the **lead360 dataset** (`ln2697/lead360` on Hugging Face, 6-camera, 354 GB total), downloaded to `D:\Carla_tfv6_data_lead360\`. Set `conf.N_CAMERAS = 6` when using this dataset.

> **Legacy dataset (archived):** The earlier 3-camera dataset was stored at `D:\Carla_tfv6_data\data\carla_leaderboard2\`. It produced incorrect attributions because TFV6 was trained on 6-camera inputs. Set `conf.N_CAMERAS = 3` if re-running archival comparisons.

The pipeline from raw footage to analysis-ready `.npz` files has three stages.

### Stage 1 — Select and unzip routes (`unzip_routes.ps1`)

Raw footage is stored as per-route `.zip` files in:
```
D:\Carla_tfv6_data_lead360\noScenarios\
```

Available routes by town (1431 zips total):

| Town | Zips | Used for |
|------|------|---------|
| Town03 | 34 | baseline |
| Town04 | 217 | baseline |
| Town05 | 114 | **test / val (held out)** |
| Town06 | 320 | baseline |
| Town07 | 165 | baseline |
| Town10 | 27 | baseline |
| Town15 | 554 | baseline |

**Town05 is the designated test/validation town and must be excluded from baseline extraction.**

`unzip_routes.ps1` groups zips by town, picks the N largest (by file size) routes per town, and extracts them to:
```
D:\Carla_tfv6_data_lead360\data\noScenarios\<route_name>\
```

Each extracted route has three subdirectories:
- `rgb/NNNN.jpg` — concatenated 6-camera JPEG (384×2304 px)
- `semantics/NNNN.png` — grouped semantic segmentation (class IDs in channel 0)
- `metas/NNNN.pkl` — XZ-compressed pickle with: `next_commands` (list of RoadOption ints), `speed` (float64 m/s), `brake` (bool), `town` (str)

**Common invocations:**
```powershell
# Preview: 10 routes per non-Town05 town
.\unzip_routes.ps1 -ExcludeTowns "Town05" -DryRun

# Extract baseline routes (10 per town, Town05 withheld)
.\unzip_routes.ps1 -ExcludeTowns "Town05"

# Extract Town05 test/val routes (N per run)
.\unzip_routes.ps1 -ExcludeTowns "Town03,Town04,Town06,Town07,Town10,Town15" -RoutesPerTown 13

# Extract more Town05 routes for the val set (e.g. 26 total, top-13 already used for test)
.\unzip_routes.ps1 -ExcludeTowns "Town03,Town04,Town06,Town07,Town10,Town15" -RoutesPerTown 26
```

### Stage 2 — Migrate to npz format (`migrate_lead_to_baseline.py`)

Converts the extracted LEAD routes into the npz format consumed by the analysis pipeline. Reads from the `noScenarios/` directory, applies even-spaced frame sampling across all routes of each active town, and writes one `.npz` per route.

**Output schema** (each npz): `wide_rgb [N,3,H,W] uint8`, `seg_red_wide [N,H,W] uint8`, `cmd [N] int32`, `speed [N] float32`, `is_brake [N] int8`, `frame_idx [N] int32`, plus (since 2026-07-02) `target_point / target_point_previous / target_point_next [N,2] float32` — ego-frame target points reproducing the **training dataloader recipe** exactly (`pos_global` + `theta`, `next/previous_target_points_3.25` meta keys with duplicate-merge; `TP_POP_DISTANCE = 3.25` must equal `TrainingConfig.tp_pop_distance`). No narrow camera (TFV6 is wide-only).

**Command mapping:** CARLA `RoadOption` integers (1-based: LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, ...) → 0-based indices (0–5). When TP extraction succeeds, `cmd` is taken from the filtered `next_commands_3.25` list (what the model saw in training); the unsuffixed `next_commands` reflects a different route-planner pop state and can genuinely differ (e.g. LANEFOLLOW vs RIGHT at junctions).

```bash
# Baseline: sample from all non-Town05 towns → data/TFV6/baseline_data/frames/
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data_lead360\data\noScenarios" \
    --mode baseline \
    --n_frames 3000 \
    --exclude_towns Town05

# Test set: sample from Town05 only → data/TFV6/test_data/frames/
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data_lead360\data\noScenarios" \
    --mode testset \
    --testset_towns Town05 \
    --testset_n_frames 500

# Validation set: sample from Town05 routes NOT yet in test_data/frames/
# → data/TFV6/val_data/frames/  (auto-excludes test routes by reading test_data/frames/ npz stems)
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data_lead360\data\noScenarios" \
    --mode valset \
    --testset_towns Town05 \
    --testset_n_frames 500
```

The `valset` mode auto-excludes routes already present in `test_data/frames/` by reading the existing npz stems — no manual exclusion list needed.

**Alternative same-distribution split** (`alt_split` mode): pools all towns, shuffles routes with a fixed seed (`conf.RANDOM_SEED`), and writes disjoint baseline / test / val sets to `*_data_alt/frames/`. OOD signal comes exclusively from perturbations, not domain shift. Requires `EXPERIMENT_VARIANT = "alternative"` in `atoms_config.py` so paths resolve to the `_alt` directories.

```bash
# All three sets at once, ~5000 baseline + 1000 test + 1000 val frames
# Set EXPERIMENT_VARIANT = "alternative" in atoms_config.py first
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data_lead360\data\noScenarios" \
    --mode alt_split \
    --baseline_n 5000 --test_n 1000 --val_n 1000
```

### Stage 3 — HPC profile computation

Once npz frame files exist in `baseline_data/frames/` or `test_data/frames/`, the HPC pipeline takes over (see `docs/cluster_explanations.md`). This stage is identical for baseline, test, and val data — only the source frames directory and output paths differ.

---

## Data Layout

```
data/
  baseline_data/
    frames/run_*.npz         # Raw baseline driving frames (migrated from LEAD, non-Town05 towns)
    baseline_1.npz           # Computed profiles for MODE_ANALYSIS=1
    baseline_2.npz           # Computed profiles for MODE_ANALYSIS=2
    mdx_parameters/          # Saved MDXDetector (v1) parameters — 512-d backbone + quantile bins
    mdx_v2_parameters/       # Saved MDXDetector (v2) parameters — 256-d F_c + quantile bins
  val_data/                  # Validation set — Town05 routes NOT used in test_data (planned)
    frames/run_*.npz         # Raw clean validation frames
    val_labeled.npz          # Perturbed+labeled val set (same 5-way 20% mix as test)
    attention/
      val_profiles_1.npy
      val_profiles_2.npy
      val_speed_logits_1.npy
      val_speed_logits_2.npy
  test_data/
    frames/run_*.npz         # Raw clean test frames (migrated from LEAD, Town05 only)
    test_labeled.npz         # Perturbed+labeled test set
    attention/
      test_profiles_1.npy    # Per-frame ATOMs profiles, MODE_ANALYSIS=1
      test_profiles_2.npy    # Per-frame ATOMs profiles, MODE_ANALYSIS=2
      test_logits_1.npy      # Per-frame action logits (WOR, mode 1)
      test_logits_2.npy      # Per-frame action logits (WOR, mode 2)
      live_pert/<pert>/
        live_pert_profiles_<variant>_<mode>.npy          # perturbed-run ATOMs profiles
        live_pert_profiles_<variant>_clean_<mode>.npy    # clean-RGB counterpart (overlay line)
        live_pert_speed_logits_<variant>_<mode>.npy      # TFV6 8-bin speed logits → PEOC
        live_pert_speed_logits_<variant>_clean_<mode>.npy
        live_pert_mdx_scores_<variant>.npy               # cached MDX scores (cache_live_mdx_scores.py;
        live_pert_mdx_scores_<variant>_clean.npy         #   no mode suffix — MDX is mode-independent)
        # WOR uses live_pert_action_logits_* instead of speed logits
  results/
    atoms_analysis/          # All output figures and JSON results
      trajectory_analysis/   # Displacement/trajectory figures
```

Profile filenames are always suffixed with the `MODE_ANALYSIS` value (`_1` or `_2`). `run_analysis.py` and `run_online_analysis.py` load the file matching `conf.MODE_ANALYSIS`; `BaselineComputer` saves to the same mode-specific path. To compare modes, set `MODE_ANALYSIS = 1` or `2` in `atoms_config.py` and re-run the analysis without recomputing.

**Alternative split** (`EXPERIMENT_VARIANT = "alternative"` in `atoms_config.py`): all four path variables (`BASELINE_DATA_DIR`, `TEST_DATA_DIR`, `VAL_DATA_DIR`, `RESULTS_DIR`) resolve to `*_data_alt` / `results_alt` counterparts. No other code changes needed — flip the flag to switch splits entirely.

Frame `.npz` files contain: `wide_rgb`, `narr_rgb`, `seg_red_wide`, `seg_red_narr`, `cmd`, `speed`, `run_id`, `frame_idx`. Labeled test `.npz` also contains `label` (0=clean, 1=perturbed) and `perturbation` (string name).

---

## TransFuser v6 (TFV6) Agent — primary

Entry point: `pcla_agents/transfuserv6/lead/inference/sensor_agent_data_collection.py`. The neural network is `TFv6` in `tfv6.py`:
- **Input**: 6-camera RGB concatenated → [B, 3, 384, 2304]; LiDAR as 2-ch positional grid (LTF mode)
- **Architecture**: timm ResNet34 image encoder + ResNet34 LiDAR encoder → 4× GPT cross-modal fusion → BEV features → PlanningDecoder (6-layer TransformerDecoder) → speed/waypoint predictions
- **F_c node space**: 256-dim `speed_query` token, output of `PlanningDecoder.transformer_decoder` at index `_speed_query_idx`
- **Speed output**: `target_speed_decoder` (Linear 256→256 → ReLU → Linear 256→8) → 8-bin two-hot speed distribution → decoded scalar in m/s
- **Speed bins**: [0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0] m/s
- **LRP model**: `LRPTFv6Model` in `lrp_transfuser.py`. Construct with `LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder)`
- **Feature extraction methods** (on `LRPTFv6Model`):
  - `get_backbone_features(wide_rgb)` → `np.ndarray[512]` — ResNet34 GAP output; used by MDX-v1.
  - `get_fc_features(wide_rgb, cmd, spd)` → `np.ndarray[256]` — F_c (`speed_query`); used by MDX-v2 test scoring.
  - `get_planning_action_and_features(wide_rgb, cmd, spd)` → `(np.ndarray[256], steer, throttle, brake)` — single forward pass returning F_c feature + action proxy; used by MDX-v2 baseline fit.
  - `get_speed_logits(wide_rgb, cmd, spd)` → `np.ndarray[8]` — raw speed bin logits; used by PEOC.
- **`TFv6FullModelForLRP.forward`** now accepts `_return_wps=True` to also return predicted future waypoints `[B, N_wp, 2]` alongside `speed_query`; all existing callers use the default `False`.

## World on Rails (WoR) Agent — secondary

Entry point: `pcla_agents/wor/image_agent.py` (`ImageAgent`). The neural network is `pcla_agents/wor/rails/models/main_model.py` (`CameraModel`):
- **Input**: wide RGB (160×704) + narrow RGB (88×352), both preprocessed to float tensors
- **Architecture**: two ResNet streams → shared FC layers → action heads (steer/throttle/brake logits per speed bin)
- **LRP target layer**: the second hidden FC layer (256-dim), referred to as F_c in the ATOMs paper
- **LRP model**: `LRPCameraModel` in `lrp_analysis.py`
- **Weights**: `pcla_agents/wor_pretrained/leaderboard_weights/main_model_10.th`
- **Config**: `pcla_agents/wor_pretrained/leaderboard_weights/config_leaderboard.yaml`

Requires CARLA launched with the `-vulkan` flag for online data collection.

---

## PCLA Infrastructure (used for data collection only)

### Core Flow
1. Create a CARLA client and vehicle actor
2. Instantiate `PCLA(agent_name, vehicle, route_xml, client)` with `agent_name="wor_lb"`
3. `PCLA.__init__` → `setup_agent()` → `setup_route()` → `setup_sensors()`
4. Each frame: `pcla.get_action()` returns a `carla.VehicleControl`; also capture sensor data for storage
5. `pcla.cleanup()` destroys sensors and vehicle

### Key Commands
```bash
# Download pretrained weights
python pcla_functions/download_weights.py

# View spawn points in the current CARLA map
python pcla_functions/spawn_points.py

# Run the sample script (wor_lb agent in Town02)
python sample.py
```

### Leaderboard Compatibility Layer (`leaderboard_codes/`)
- `autonomous_agent_local.py` — base class `AutonomousAgent`; defines `setup()`, `sensors()`, `run_step()`, `destroy()`
- `carla_data_provider.py` — singleton providing map/actor/traffic-light data
- `sensor_interface.py` — manages sensor callbacks; pseudo-sensors `SpeedometerReader`, `OpenDriveMapReader`
- `route_manipulation.py` — interpolates GPS trajectories from route XML files
- `watchdog.py` — kills hung agent initialization after 260 s

### Route XML Files
Pre-built routes for Town01–07 are in the repo root. Generate custom routes:
```python
from PCLA import location_to_waypoint, route_maker
waypoints = location_to_waypoint(client, startLoc, endLoc)
route_maker(waypoints, "my_route.xml")
```

---

## Thesis Figures

Final thesis-quality figures are produced by `make_thesis_figures.py` (repo root)
and written to `thesis_figures/` as `.pdf` + `.png`. The visual style is shared
with the Atari/ATOMs chapter and defined by `thesis_style.py` (rcParams, CVD-safe
metric palette, save helper) with the full written spec in
`documents/14_thesis_figure_style.md` — no titles, one legend/colorbar per figure,
serif fonts sized for placement at exactly `\textwidth` (6.3 in). Follow that spec
for any new thesis figure; exploratory pipeline plots (`visualization_carla.py` /
`viz_config.py`) are unaffected.

Current figures (alternative split, TFV6 mode 2, val-selected K=8): GMM AUROC vs
K sweep, baseline PCA (by run vs by GMM cluster), score distributions of the two
best GMM detectors (Mahalanobis-GMM vs kNN-GMM), per-perturbation AUROC bars, a
GMM-vs-single parity scatter, three cluster-attention figures (per-cluster
bar grid, the same grid with representative frames, and the all-clusters grouped
bars), three live-perturbation score-trace figures (Mahalanobis-GMM | MDX |
PEOC per panel; one variant each of brightness_scale / gaussian_noise / pgd —
MDX scores must be pre-cached once via `cache_live_mdx_scores.py`, which needs
the `PCLA` env with torch/timm), a K-selection view (val mean GMM AUROC solid
vs test dotted), and a kNN k-selection figure (val AUROC vs k, single vs GMM).
The per-perturbation bar chart also carries plain (non-GMM) kNN in a lighter
green tint (`KNN_SINGLE_COLOR`) — kNN is the one detector where
cluster-restricted pools are conceptually questionable. Cluster figures (PCA +
attention bars) share `CLUSTER_COLORS`, eight hues deliberately outside the
metric families (orange/cyan/magenta/brown/gray) so cluster and detector
figures can't be confused. Score arrays are
recomputed from saved detector parameters; the script
asserts the recomputed AUCs match the stored results JSONs. Since the
**uniform-shrinkage fix (2026-07-16**, see `docs/design_decisions.md`), the
pipeline uses the shrinkage-regularised covariances everywhere — GMM scoring,
GMM cluster prediction, and single-Gaussian Mahalanobis scoring — so the
figure script simply reuses the stored `gmm.npz`/`mahal_detector.npz`
parameters (results computed before that date mixed shrunk and raw
covariances and do not reproduce from the stored parameters alone). Note the
summary.json test AUROCs (and hence the AUROC-vs-K figure) include
gaussian-noise frames as OOD; only the val-side `__val_auc_*__` K-selection
metrics exclude them. Runs in either conda env (`PCLA`, numpy 1.x, or
`atoms3`, numpy 2.x): a shim in the script aliases `numpy._core` →
`numpy.core` so the numpy-2-pickled object arrays in the alt-split npz files
load under numpy 1.x.

---

## Documentation Policy

Whenever you make a meaningful change to the codebase, **update the relevant `.md` files** to reflect that change. Do not let documentation go stale. The living documentation files in this project are:

| File | What it tracks |
|------|----------------|
| `CLAUDE.md` | Architecture, pipeline, key concepts, module responsibilities |
| `docs/design_decisions.md` | Design choices for the ATOMs/LRP pipeline — why things are done the way they are |
| `docs/docs/lrp_todo.md` | Open questions, remaining work, and decision history for the LRP implementation |
| `docs/cluster_explanations.md` | HPC/Viper file transfer and job submission how-to |

**Rules:**
- If you add or change a module, class, or key function → update `CLAUDE.md` (module structure, key concepts, or pipeline sections as appropriate).
- If you make a design choice that isn't obvious from the code (algorithm selection, rule variant, architectural tradeoff) → record it in `docs/design_decisions.md`.
- If you resolve an open issue or make a decision on something tracked in `docs/lrp_todo.md` → mark it resolved and document the outcome there.
- If a change doesn't fit any existing file, you may create a new `.md` file — but **ask the user first**.
- If you are unsure which file to update, or whether a change warrants documentation, **ask the user** before proceeding.
