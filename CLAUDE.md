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
- **TFV6** (`ATOMs_Analysis/saliency/lrp_transfuser.py`, `LRPTFv6Model`): AttnLRP (Achtibat et al. 2024) — custom autograd Functions for softmax (Prop 3.1) and bilinear matmul (Prop 3.3); ε-rule for attention Linear layers; AlphaBeta for Conv/FFN. Wraps backbone + PlanningDecoder in `TFv6FullModelForLRP` so a single zennit composite context covers the full attribution graph.

**TFV6 two-pass scheme:**
- **LRP1** (`beg="output", end="fc"`): softmax-distribution seed over 8 speed bins → backprop through `target_speed_decoder` (Linear→ReLU→Linear) to `speed_query` level. Returns 256-dim node relevance vector.
- **LRP2** (`beg="fc", end="input"` with `node_id`): one-hot at F_c node k → backprop through full model to input pixels. Returns [1,3,H,W] pixel map.
- **Output→input** (`beg="output", end="input"`): same softmax seed, backprop all the way to pixels in one pass.

**LRP1 seed rationale:** `target_speed_decoder` is trained with a two-hot target over 8 speed bins [0, 4, 8, 10, 13.9, 16, 17.8, 20] m/s. Using argmax causes discontinuities at bin boundaries. The softmax distribution seed is used instead (`grad_outputs = softmax(speed_logits.detach())`), giving a smooth attribution that reflects the full predicted speed distribution.

**F_c layer:** 256-dim `speed_query` token output by `PlanningDecoder.transformer_decoder`, just before `target_speed_decoder`. This is the closest equivalent to the ATOMs paper's F_c ("the final world model on which the agent chooses its action"). See `docs/lrp_todo.md` Decision A/B for rationale.

### ATOMs (Attention-Oriented Metrics)
Introduced by Beylier et al. (NeurIPS 2024 workshop). Converts raw LRP heatmaps into structured, object-level attention vectors by intersecting relevance maps with semantic segmentation masks. Two levels:
- **Hierarchical-attention** `h(o)`: the mean relevance over object `o`'s pixels that carry nonzero relevance (R̄ in Beylier et al.), then re-normalized across objects — not a raw fraction of total relevance.
- **Combinatorial-attention** `c(T)`: fraction of frames where a neuron attends jointly to a subset of objects `T` (using threshold α = 0.25).

For TFV6, CARLA's grouped semantic segmentation (10 classes, `save_grouped_semantic=True`) is used via `TFV6_CLASSES` in `atoms_carla.py`. Implemented in `ATOMs_Analysis/saliency/atoms_carla.py`.

**ATOMs saliency attributes** (set after each `process_frame`):
- `saliency_data_wide_default` — map from the default (softmax) seed; this is what feeds `_hierarchical`.
- `saliency_data_wide_brake` / `saliency_data_wide_drive` — forced brake/drive maps, populated only when `PLOT_COMPARATIVE_REL=True`; do **not** feed `_hierarchical`.
- When `PLOT_COMPARATIVE_REL=False`, brake/drive slots mirror the default map.

### OOD Detection Strategy
1. Collect clean baseline driving frames; compute one ATOMs profile per frame. Dimensionality is agent-dependent: 29 for WOR (full `CARLA_CLASSES`, tags 0–28) and 10 for TFV6 (grouped `TFV6_CLASSES`). (Earlier docs said "23-dim"; that was a stale CARLA tag count.)
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
| 4 | BIC/AIC sweep over K=1..MAX_K to select GMM component count |
| 5 | Fit `GMMClustering` with selected K; assign baseline frames to clusters |
| 6 | Visualize baseline: attention bar chart, per-cluster attention comparison, PCA coloured by run and cluster |
| 7 | Apply perturbation mix to test set → `test_labeled.npz` with ground-truth labels |
| 8 | Compute ATOMs profiles + action logits on labeled test set → `test_profiles.npy` |
| 8.5 | Trajectory analysis: match clean↔perturbed pairs by (run_id, frame_idx); compute displacement stats and PCA trajectories per perturbation type. **DISABLED — this block is currently commented out in `run_analysis.py`; any `trajectory_analysis/` figures on disk are stale.** |
| 9 | Score test profiles with all detectors: Mahalanobis-single, Mahalanobis-GMM, Euclidean, k-NN, JSD, MDX-v1, MDX-v2, Action Entropy |
| 10 | Evaluate each detector: ROC curve, AUC, Youden-J optimal threshold |
| 11 | Per-perturbation breakdown: evaluate each detector separately on each perturbation type |
| 12 | Save all figures (PNG) and results (JSON) to `conf.RESULTS_DIR/atoms_analysis/` |

Key flags in `atoms_config.py` that control re-computation: `RECOMPUTE_BASELINE`, `RECOMPUTE_MDX_BASELINE`, `RECOMPUTE_MDX_V2_BASELINE`, `REAPPLY_PERTURBATIONS`, `RECOMPUTE_TEST_ATOMS`.

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
- `ATOMsCarla.process_frame(wide, narr, seg_wide, seg_narr, cmd, spd, data)` is the per-frame API; call `atoms.reset()` between datasets. For TFV6, `process_frame` is called without `data=`; `_make_minimal_data` rebuilds the conditioning dict from `cmd`/`spd`. The command one-hot is built from `cmd` (round-trips exactly with the agent's stored `cmd = argmax(command)`), but `target_point` and `acceleration` are zero — route conditioning is therefore absent, which can shift attributions relative to the live agent.
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
| `pgd` | PGD adversarial attack (targeted max-steer) |

---

## Raw Data Creation Pipeline (TFV6)

All TFV6 driving footage originates from the **LEAD CARLA leaderboard dataset**, pre-downloaded to `D:\Carla_tfv6_data\`. The pipeline from raw footage to analysis-ready `.npz` files has three stages.

### Stage 1 — Select and unzip routes (`unzip_routes.ps1`)

Raw footage is stored as per-route `.zip` files in:
```
D:\Carla_tfv6_data\data\carla_leaderboard2\zip\noScenarios\
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
D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios\<route_name>\
```

Each extracted route has three subdirectories:
- `rgb/NNNN.jpg` — concatenated 6-camera JPEG (384×1152 px)
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

**Output schema** (each npz): `wide_rgb [N,3,H,W] uint8`, `seg_red_wide [N,H,W] uint8`, `cmd [N] int32`, `speed [N] float32`, `is_brake [N] int8`, `frame_idx [N] int32`. No narrow camera (TFV6 is wide-only).

**Command mapping:** CARLA `RoadOption` integers (1-based: LEFT=1, RIGHT=2, STRAIGHT=3, LANEFOLLOW=4, ...) → 0-based indices (0–5).

```bash
# Baseline: sample from all non-Town05 towns → data/TFV6/baseline_data/frames/
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios" \
    --mode baseline \
    --n_frames 3000 \
    --exclude_towns Town05

# Test set: sample from Town05 only → data/TFV6/test_data/frames/
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios" \
    --mode testset \
    --testset_towns Town05 \
    --testset_n_frames 500

# Validation set: sample from Town05 routes NOT yet in test_data/frames/
# → data/TFV6/val_data/frames/  (auto-excludes test routes by reading test_data/frames/ npz stems)
python migrate_lead_to_baseline.py \
    --lead_dir "D:\Carla_tfv6_data\data\carla_leaderboard2\data\noScenarios" \
    --mode valset \
    --testset_towns Town05 \
    --testset_n_frames 500
```

The `valset` mode auto-excludes routes already present in `test_data/frames/` by reading the existing npz stems — no manual exclusion list needed.

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
    mdx_parameters/          # Saved MDXDetector (v1) parameters — 512-d backbone + equal-width bins
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
        live_pert_profiles_1.npy
        live_pert_profiles_2.npy
  results/
    atoms_analysis/          # All output figures and JSON results
      trajectory_analysis/   # Displacement/trajectory figures
```

Profile filenames are always suffixed with the `MODE_ANALYSIS` value (`_1` or `_2`). `run_analysis.py` and `run_online_analysis.py` load the file matching `conf.MODE_ANALYSIS`; `BaselineComputer` saves to the same mode-specific path. To compare modes, set `MODE_ANALYSIS = 1` or `2` in `atoms_config.py` and re-run the analysis without recomputing.

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
