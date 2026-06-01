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
- **Hierarchical-attention** `h(o)`: fraction of total relevance falling on object `o`.
- **Combinatorial-attention** `c(T)`: fraction of frames where a neuron attends jointly to a subset of objects `T` (using threshold α = 0.25).

For TFV6, CARLA's grouped semantic segmentation (10 classes, `save_grouped_semantic=True`) is used via `TFV6_CLASSES` in `atoms_carla.py`. Implemented in `ATOMs_Analysis/saliency/atoms_carla.py`.

**ATOMs saliency attributes** (set after each `process_frame`):
- `saliency_data_wide_default` — map from the default (softmax) seed; this is what feeds `_hierarchical`.
- `saliency_data_wide_brake` / `saliency_data_wide_drive` — forced brake/drive maps, populated only when `PLOT_COMPARATIVE_REL=True`; do **not** feed `_hierarchical`.
- When `PLOT_COMPARATIVE_REL=False`, brake/drive slots mirror the default map.

### OOD Detection Strategy
1. Collect clean baseline driving frames; compute one ATOMs profile (23-dim attention vector) per frame.
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
| 2.5 | Fit `MDXDetector` on baseline (feature-space Mahalanobis baseline from Zhang et al. 2024) |
| 3 | Fit single-Gaussian `MahalanobisDetector` on baseline profiles; set threshold at 99th percentile |
| 4 | BIC/AIC sweep over K=1..MAX_K to select GMM component count |
| 5 | Fit `GMMClustering` with selected K; assign baseline frames to clusters |
| 6 | Visualize baseline: attention bar chart, per-cluster attention comparison, PCA coloured by run and cluster |
| 7 | Apply perturbation mix to test set → `test_labeled.npz` with ground-truth labels |
| 8 | Compute ATOMs profiles + action logits on labeled test set → `test_profiles.npy` |
| 8.5 | Trajectory analysis: match clean↔perturbed pairs by (run_id, frame_idx); compute displacement stats and PCA trajectories per perturbation type |
| 9 | Score test profiles with all detectors: Mahalanobis-single, Mahalanobis-GMM, Euclidean, k-NN, JSD, MDX, Action Entropy |
| 10 | Evaluate each detector: ROC curve, AUC, Youden-J optimal threshold |
| 11 | Per-perturbation breakdown: evaluate each detector separately on each perturbation type |
| 12 | Save all figures (PNG) and results (JSON) to `conf.RESULTS_DIR/atoms_analysis/` |

Key flags in `atoms_config.py` that control re-computation: `RECOMPUTE_BASELINE`, `RECOMPUTE_MDX_BASELINE`, `REAPPLY_PERTURBATIONS`, `RECOMPUTE_TEST_ATOMS`.

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
│   ├── detectors.py           # MahalanobisDetector, ActionEntropyDetector, MDXDetector,
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
- `ATOMsCarla.process_frame(wide, narr, seg_wide, seg_narr, cmd, spd, data)` is the per-frame API; call `atoms.reset()` between datasets. Pass the real `data` dict (from the frame npz) for TFV6 — the fallback `_make_minimal_data` uses a zero command vector which distorts LRP attributions.
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

## Data Layout

```
data/
  baseline_data/
    frames/run_*.npz       # Raw baseline driving frames
    baseline.npz           # Computed profiles (mean, cov, series)
    mdx_parameters/        # Saved MDXDetector parameters
  test_data/
    frames/run_*.npz       # Raw clean test frames
    test_labeled.npz       # Perturbed+labeled test set
    attention/
      test_profiles.npy    # Per-frame ATOMs profiles (test set)
      test_logits.npy      # Per-frame action logits (for entropy detector)
  results/
    atoms_analysis/        # All output figures and JSON results
      trajectory_analysis/ # Displacement/trajectory figures
```

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
