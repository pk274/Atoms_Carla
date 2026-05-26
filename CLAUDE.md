# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **master's thesis research project** on explainability-based out-of-distribution (OOD) detection for autonomous driving agents. The codebase has two layers:

1. **PCLA** (Pretrained CARLA Leaderboard Agents) — infrastructure for deploying pretrained agents in the CARLA simulator. Used here solely to run the **World on Rails (WoR)** agent and collect driving data.
2. **ATOMs_Analysis** — the research core. Implements the full pipeline for computing attention profiles via LRP + ATOMs, fitting baseline distributions, and detecting OOD inputs by measuring divergence from those baselines.

The scientific goal is to show that attention profiles derived from LRP are a meaningful signal for OOD detection: when the agent encounters perturbed or adversarial inputs, its attention distribution shifts measurably away from the clean-driving baseline.

Target platform: **Linux Ubuntu 22**, **CARLA 0.9.16**, **Python 3.10**.

---

## Key Concepts

### LRP (Layer-wise Relevance Propagation)
Backpropagates relevance from the network output to the input pixels, producing a saliency/heatmap that indicates which pixels drove the network's decision. Uses the z⁺-rule throughout (`zennit` library). Implemented in `ATOMs_Analysis/saliency/lrp_analysis.py` as `LRPCameraModel`, which wraps the WoR `CameraModel` for a single-pass joint LRP through both camera streams.

### ATOMs (Attention-Oriented Metrics)
Introduced by Beylier et al. (NeurIPS 2024 workshop). Converts raw LRP heatmaps into structured, object-level attention vectors by intersecting relevance maps with semantic segmentation masks. Two levels:
- **Hierarchical-attention** `h(o)`: fraction of total relevance falling on object `o`.
- **Combinatorial-attention** `c(T)`: fraction of frames where a neuron attends jointly to a subset of objects `T` (using threshold α = 0.25).

In this project ATOMs is applied to the WoR `CameraModel`'s penultimate FC layer (256-dim). CARLA's red-channel semantic segmentation (23 classes) provides the object labels. Implemented in `ATOMs_Analysis/saliency/atoms_carla.py`.

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
│   ├── lrp_analysis.py        # LRPCameraModel — zennit-based LRP over WoR dual-camera model
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
    └── lrp_test_suite.py      # Diagnostic suite for validating LRP + ATOMs correctness
```

### Key design notes
- `atoms_config.py` is the single source of truth for all paths and hyperparameters. Edit it rather than hardcoding values in scripts.
- `ATOMsCarla.process_frame(wide, narr, seg_wide, seg_narr, cmd)` is the per-frame API; call `atoms.reset()` between datasets.
- `DistanceComputer` is stateless (static methods); detectors (`MahalanobisDetector`, etc.) are stateful (fit/save/load).
- Visualization functions return `matplotlib.Figure` objects; use `save_figure(fig, path)` to write them.
- `viz_config.py` defines `apply_default_style()` — call it at the top of any new plotting script to keep figures consistent across the thesis.

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

## World on Rails (WoR) Agent

The only agent analyzed in this project. Entry point: `pcla_agents/wor/image_agent.py` (`ImageAgent`). The neural network is `pcla_agents/wor/rails/models/main_model.py` (`CameraModel`):
- **Input**: wide RGB (160×704) + narrow RGB (88×352), both preprocessed to float tensors
- **Architecture**: two ResNet streams → shared FC layers → action heads (steer/throttle/brake logits per speed bin)
- **LRP target layer**: the second hidden FC layer (256-dim), referred to as F_c in the ATOMs paper
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
