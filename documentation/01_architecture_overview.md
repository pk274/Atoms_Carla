# 01 — Architecture Overview & Configuration System

> Documentation sweep topic 1. All claims verified against code state of 2026-06-12.
> File references are relative to the repo root `PCLA/` unless stated otherwise.

---

## 1. Purpose & scope

This document describes (a) the two-layer structure of the repository, (b) the end-to-end
data flow from raw driving footage to OOD-detection results, (c) what each top-level entry
script does, and (d) the central configuration system `ATOMs_Analysis/atoms_config.py` —
every attribute, its current value, its consumers, and all magic constants hardcoded
outside the config. Internals of individual pipeline stages (LRP rules, detector math,
HPC job mechanics, …) are covered by the other topic docs (Section 7).

---

## 2. Repository layers

### 2.1 Layer A — PCLA infrastructure (data collection)

Everything needed to deploy pretrained agents in a live CARLA simulator and record
sensor data. Not part of the research contribution; used only to produce driving frames
(and, for TFV6, only for the *live-perturbation* recordings — the bulk of TFV6 data comes
from the pre-recorded LEAD dataset, see 2.3).

| Path | Role |
|---|---|
| `PCLA.py` | Facade class `PCLA(agent_name, vehicle, route_xml, client)`; route helpers `location_to_waypoint`, `route_maker` |
| `agents.json` | Agent-name → module registry |
| `leaderboard_codes/` | CARLA leaderboard compatibility layer (`autonomous_agent_local.py`, `carla_data_provider.py`, `sensor_interface.py`, `route_manipulation.py`, `watchdog.py`) |
| `pcla_agents/wor/`, `pcla_agents/wor_pretrained/` | World on Rails agent + weights (`leaderboard_weights/main_model_10.th`) |
| `pcla_agents/transfuserv6/`, `pcla_agents/transfuserv6_pretrained/` | TransFuser v6 agent (LEAD codebase) + checkpoint (`visiononly_resnet34/`) |
| `map_manupulation/` | `generate_traffic.py` (`TrafficOrganizer`), `dynamic_weather.py` (`WeatherOrganizer`) — used by `sample.py` |
| `pcla_functions/` | Utilities: `download_weights.py`, `spawn_points.py` |
| `sample.py` | Live CARLA session driver (see §3) |
| `route_Town0*.xml`, `route_creation.py` | Pre-built / generated route files |

### 2.2 Layer B — ATOMs_Analysis research core

The thesis contribution: LRP → ATOMs attention profiles → baseline distribution →
OOD scoring.

```
ATOMs_Analysis/
├── atoms_config.py            # ExperimentConfig — single source of truth (§4)
├── perturbation_manager.py    # PerturbationManager — image perturbation registry (doc 06)
├── saliency/
│   ├── lrp_analysis.py        # LRPCameraModel — WoR z+-rule LRP (doc 03)
│   ├── lrp_transfuser.py      # LRPTFv6Model — TFV6 AttnLRP (doc 03)
│   ├── lrp_lbc.py             # LBC LRP — dead end, out of scope (see 00_PLAN.md)
│   └── atoms_carla.py         # ATOMsCarla + CARLA_CLASSES / TFV6_CLASSES (doc 04)
├── detection/
│   ├── baseline_dataset.py    # BaselineDataCollector / BaselineComputer / BaselineDataLoader (doc 05)
│   ├── dataset.py             # LabeledTestLoader, PerturbationApplier, PerturbationSpec (docs 05/06)
│   ├── detectors.py           # Mahalanobis, Euclidean, KNN, JSD, Wasserstein, MDX, ActionEntropy, DetectorEvaluator (doc 07)
│   └── clustering.py          # GMMClustering — BIC/AIC sweep + per-cluster scoring (doc 07)
└── utils/
    ├── visualization_carla.py # All plotting functions (doc 12)
    ├── viz_config.py          # Thesis-wide plot style, shared with the Atari repo (doc 12)
    ├── distance_computer.py   # DistanceComputer — stateless metric functions (doc 07)
    ├── lrp_test_suite.py, tfv6_test_suite.py, atoms_test_suite.py,
    ├── wor_lrp_diagnostics.py, tfv6_lrp_diagnostics.py   # validation (doc 11)
```

Supporting top-level pieces: `hpc/` (Slurm chunked profile computation, doc 10),
`docs/` (legacy living docs: `design_decisions.md`, `lrp_todo.md`, `cluster_explanations.md`,
`code_review.md`, `interpretation_hypothesis.md`), `data/` (per-agent data trees, see 2.4),
`results_summary/` + `results_summary_alt/` (output of `summarize_results.py`),
`documentation/` (this sweep).

### 2.3 End-to-end data flow (TFV6 primary path)

```
D:\Carla_tfv6_data\...\zip\noScenarios\*.zip      (1431 LEAD route zips, 7 towns)
        │  unzip_routes.ps1  — NOT in the repo (referenced only in CLAUDE.md; see BUG-8)
        ▼
D:\...\data\noScenarios\<route>\{rgb/, semantics/, metas/}
        │  migrate_lead_to_baseline.py  (--mode baseline|testset|valset|alt_split)
        ▼
data/TFV6/{baseline,test,val}_data[/­_alt]/frames/run_*.npz       (wide_rgb, seg_red_wide, cmd, speed, is_brake, frame_idx)
        │  hpc/prep_test.py → test_labeled.npz / val_labeled.npz   (5-way 20% perturbation mix; PGD deferred)
        │  hpc/submit_baseline.sh / submit_test.sh / submit_val.sh — Slurm array jobs
        │  hpc/compute_*_chunk.py  → per-chunk profiles;  hpc/gather_*.py → merged .npy/.npz
        ▼
baseline_{1|2}.npz, test_profiles_{1|2}.npy, val_profiles_{1|2}.npy, *_speed_logits_{1|2}.npy,
mdx_features.npz   (downloaded back to data/TFV6/... via hpc/collect_results.sh)
        │  run_analysis.py  (Steps 1–12; fits Mahalanobis/GMM/MDX, scores test, ROC/AUC)
        │  sweep_clusters.py  (re-runs per K, snapshots results)
        ▼
data/TFV6/results[_alt]/atoms_analysis_mode_{1|2}/   figures (PNG) + summary.json + results_*.json
data/TFV6/results[_alt]/<K> clusters/atoms_analysis_mode_{1|2}/   per-K snapshots
        │  summarize_results.py
        ▼
results_summary[_alt]/SUMMARY.md + heatmaps   (cross-K report, val-based K recommendation)
```

Parallel online branch: `sample.py` (live CARLA, `sensor_agent_live_perturbation.py`
injecting a perturbation at `conf.INJECTION_TIME`) → `data/TFV6/test_data/live_pert_frames/`
→ `hpc/submit_live_pert.sh` → `run_online_analysis.py` → `results/atoms_analysis_live_mode_{1|2}/`.

WoR variant: same flow, but frames are recorded live from CARLA (`image_agent.py` with
`BaselineDataCollector`/`TestDataCollector`), profiles are 29-dim, and `*_wor` HPC scripts
are used.

### 2.4 Data layout (verified on disk)

`data/<AGENT>/` exists for `TFV6` and `WOR`; under `TFV6` all eight directories exist:
`baseline_data`, `baseline_data_alt`, `test_data`, `test_data_alt`, `val_data`,
`val_data_alt`, `results`, `results_alt`. The non-`_alt` vs `_alt` pair is selected
exclusively by `EXPERIMENT_VARIANT` (§4.4).

---

## 3. Entry points (top-level scripts)

**`run_analysis.py`** (~1850 lines) — the main offline pipeline. Loads the agent model
(TFV6 checkpoint dir hardcoded at line 134, WoR at line 172), builds `LRPTFv6Model` /
`LRPCameraModel` + `ATOMsCarla`, then executes Steps 1–12 (baseline profiles, MDX-v1/v2
fit, Mahalanobis fit, BIC/AIC GMM-K sweep, GMM fit, baseline visualization, perturbation
application, test profiles, detector scoring, val-based k/K selection, ROC/AUC evaluation,
per-perturbation breakdown, JSON+figure export). Consumes the config pervasively (≈60
`conf.*` references); accepts exactly one CLI arg `--gmm-k` (lines 51–54, applied at
lines 478–479). Output root: `conf.RESULTS_DIR / f"atoms_analysis_mode_{conf.MODE_ANALYSIS}"`
(line 110). Details in doc 08.

**`run_online_analysis.py`** (~900 lines) — variant of the same pipeline for
live-perturbation recordings. Re-uses Steps 1–5 (baseline + detectors), then loads frames
from `conf.TEST_DATA_DIR / "live_pert_frames"` (line 374) instead of the labeled test set,
keyed by `LIVE_PERT_NAME = conf.PERTURBATION` (line 93). Output root:
`conf.RESULTS_DIR / f"atoms_analysis_live_mode_{conf.MODE_ANALYSIS}"` (line 89). Its module
docstring is a verbatim copy of `run_analysis.py`'s (BUG-11). Details in doc 09.

**`migrate_lead_to_baseline.py`** — offline converter from LEAD route directories to the
`run_*.npz` frame format; five modes (`baseline`, `testset`, `valset`, `both`, `alt_split`;
argparse at lines 574–646). Consumes the config only for output paths
(`conf.BASELINE_DATA_DIR`, `conf.TEST_DATA_DIR`, `conf.VAL_DATA_DIR`, `conf._DATA_ROOT`)
and the `alt_split` shuffle seed (`seed: int = conf.RANDOM_SEED`, line 500). Details in doc 05.

**`sweep_clusters.py`** — runs `run_analysis.py --gmm-k K` as a subprocess for each K in
`--k-values` (default `K_VALUES_DEFAULT = [10, 8, 12]`, line 33 — the literal has since changed from the `[18, 20, 22]` recorded in finding 1.5; see finding 8.7) and snapshots the result
dir to `conf.RESULTS_DIR / f"{K} clusters" / f"atoms_analysis_mode_{mode}"`. Reads the
config only for paths/mode and to warn when expensive `RECOMPUTE_*` flags are left on
(lines 72–78). Never modifies the config — K travels via CLI.

**`summarize_results.py`** — config-independent (CLI: `--data-root` default `data`,
`--out` default `results_summary`, `--alt` switches to `*_data_alt`/`results_summary_alt`;
lines 808–817). Crawls `data/<AGENT>/results/<K> clusters/atoms_analysis_mode_<m>/summary.json`
across both agents, builds detector×K AUC matrices, and writes `SUMMARY.md` + heatmaps,
including the val-set K recommendation from `__val_auc_gmm_avg__`. Details in doc 08.

**`sample.py`** — live CARLA driver for online data collection. Connects to
`localhost:2000`, loads `conf.TOWN`, sets weather via `WeatherOrganizer(conf.WEATHER)`,
spawns the ego vehicle at `conf.SPAWN_INDEX`, places the spectator at
`conf.SPEC_POS`/`conf.SPEC_ROT`, instantiates `PCLA`, and steps the agent. The agent-side
recording behavior is governed by the three `*_RECORDING_MODE` flags (§4.2).

**`route_creation.py`** — 20-line helper to generate a route XML from spawn-point indices
(hardcoded indices 398/287/312/368); ad-hoc tooling, not part of the pipeline.

**HPC scripts** (`hpc/*.py`, `hpc/*.sh`) are deliberately config-independent: all paths,
chunk sizes, seeds, and `MODE_ANALYSIS` arrive via CLI args or environment variables
(e.g. `hpc/array_task.sh:48` defaults `--mode-analysis "${MODE_ANALYSIS:-1}"`). The single
exception is `hpc/gather_live_pert.py`, which imports `atoms_config`. Consequence: the
local config's `MODE_ANALYSIS = 2` is *not* propagated to HPC automatically — the operator
must pass the matching mode (doc 10).

---

## 4. Configuration system — `ATOMs_Analysis/atoms_config.py`

### 4.1 Mechanics

`ExperimentConfig` is a plain class (138 lines) that is **never instantiated**; every
attribute is a class attribute and every consumer imports it as
`from ATOMs_Analysis.atoms_config import ExperimentConfig as conf`. There is no
environment/CLI override layer — changing an experiment means editing the file. Two
non-obvious mechanics:

- **Import-time path resolution.** `_DATA_ROOT`, the four `*_DATA_DIR`s, `RESULTS_DIR`,
  and the town-dependent `SPAWN_INDEX`/`SPEC_POS`/`SPEC_ROT` are computed in the class
  body at import time from `AGENT`, `EXPERIMENT_VARIANT`, and `TOWN` (lines 58–74,
  100–134). They are *not* re-derived if a script mutates `conf.AGENT` later.
- **Mutable global state.** `image_counter` (line 138) is a class attribute used as a
  global counter, incremented in `ATOMs_Analysis/detection/baseline_dataset.py:552` to
  enumerate saved segmentation/relevance example images
  (`baseline_dataset.py:522–525`).

`_DATA_ROOT` is a hardcoded absolute Windows path
(`C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/data`, line 58) — the config is
machine-specific (see §6).

### 4.2 Top-level switches

| Attribute (line) | Value | Meaning | Consumers |
|---|---|---|---|
| `AGENT` (10) | `"TFV6"` | Active agent; selects data subtree (`data/<AGENT>/`), LRP wrapper class, segmentation class map, perturbation spec branch. Docstring claims `"WOR" \| "LBC" \| "TFV6"`, but `run_analysis.py:130` branches only `TFV6` vs `else→WoR` — `"LBC"` would silently run the WoR model path (BUG-13). | `run_analysis.py:64,130,192,313,689`; `run_online_analysis.py:59,111,138,144,201,209,416`; `sample.py:58`; `sweep_clusters.py` |
| `MODE_ANALYSIS` (23) | `2` | ATOMs analysis mode (1 = paper default node-level, 2 = alternative; semantics in doc 04). Suffixes *every* profile artifact filename (`baseline_{m}.npz`, `test_profiles_{m}.npy`, …) and the results directory `atoms_analysis_mode_{m}`. | `run_analysis.py:110,116,202`; `run_online_analysis.py:89,97,149`; `sweep_clusters.py:63`; `BaselineComputer` save path. HPC defaults to mode **1** unless `MODE_ANALYSIS` env var is set (`hpc/array_task.sh:48`). |
| `EXPERIMENT_VARIANT` (63) | `"original"` | `"original"` = Town05-held-out split; `"alternative"` = same-distribution random route split. Resolves all four path variables to `*_data` vs `*_data_alt` / `results` vs `results_alt` (lines 65–74). No other code reads the flag — the switch is purely path-based. | path block lines 65–74 only |
| `TOWN` (12) | `"Town05"` | Live-collection town; selects `SPAWN_INDEX`/`SPEC_POS`/`SPEC_ROT` block (lines 100–134). | `sample.py:16,64` (loads world, picks `route_{TOWN}.xml`) |
| `WEATHER` (13) | `"sunny"` | Live-collection weather preset (`sunny/cloudy/night/rainy/foggy`). | `map_manupulation/dynamic_weather.py:10–18`; `generate_traffic.py:41` |
| `SPEED_MODE` (14) | `False` | WoR-only throttle modification during live runs. | `pcla_agents/wor/image_agent.py:310` |
| `HIGH_SPEED_MODE` (15) | `False` | WoR-only high-speed variant. | `image_agent.py:299,312` |
| `BASELINE_RECORDING_MODE` (17) | `False` | Live session records clean baseline frames. | `image_agent.py:251`, `lbc_agent.py:222`, `sensor_agent_data_collection.py:83` |
| `TESTSET_RECORDING_MODE` (18) | `False` | Live session records clean test frames. | same files |
| `LIVE_PERTURBATION_RECORDING_MODE` (19) | `True` | Live session injects `PERTURBATION` at `INJECTION_TIME` and records. Also flips the Town05 spawn-block branch (line 127 — currently dead, see BUG-6) and the agent class in `sample.py:58`. | `sensor_agent_live_perturbation.py:110`; `image_agent.py:255`; `lbc_agent.py:226` |

### 4.3 Analysis hyperparameters

| Attribute (line) | Value | Meaning / consumers |
|---|---|---|
| `NUM_GMM_CLUSTERS` (21) | `12` | GMM component count override. Resolution priority in `run_analysis.py:475–479`: `--gmm-k` CLI > `NUM_GMM_CLUSTERS` (if not `None`) > BIC-selected K. Note: 12 > `GMM_MAX_K`=10, so the forced K lies outside the BIC sweep range — BIC/AIC figures don't cover the chosen K (BUG-10). |
| `FC_RELEVANCE_FILTER` (24) | `0.9` | Cumulative-relevance mass filter p for selecting F_c nodes in LRP1 (`ATOMsCarla(p_relevance=…)`, `run_analysis.py:200`, `run_online_analysis.py:147`; applied in `atoms_carla.py:420` via `_relevance_filter`). |
| `DEFAULT_CMD` (87) | `2` | Default command index passed to `ATOMsCarla` (`run_analysis.py:201`). The inline comment there claims "3 = FOLLOW_LANE" while the value is 2 (= STRAIGHT in the 0-based mapping of `migrate_lead_to_baseline.py:66`) — value/comment mismatch, BUG-7. |
| `MAHAL_RIDGE` (88) | `0.01` | Covariance ridge for all Mahalanobis fits and `DistanceComputer` calls (11 usages, e.g. `run_analysis.py:431,494,1121,1176,1218`). `MahalanobisDetector.__init__` default is `1e-6` (`detectors.py:163`); the `run_analysis.py:431` comment still describes the 1e-6 default (BUG-14). |
| `GMM_MAX_K` (89) | `10` | Upper bound of the BIC/AIC sweep (`run_analysis.py:456`). |
| `GMM_COV_TYPE` (90) | `"full"` | GMM covariance type (`run_analysis.py:462,468,492`). |
| `RANDOM_SEED` (91) | `17` | GMM `random_state` (`run_analysis.py:493`) and `alt_split` route shuffle (`migrate_lead_to_baseline.py:500`). **Not** used for perturbation assignment — that seed is hardcoded 42 (§4.7). |
| `WIDE_ONLY_PROFILE` (97) | `True` | Build profiles from the wide-camera relevance map only; narrow contribution skipped in `atoms_carla.py:656`. Required for TFV6 (no narrow camera). |
| `MODE` filename suffix | — | see `MODE_ANALYSIS` above. |

### 4.4 Path variables

| Attribute (lines) | Value (original variant) | Consumers |
|---|---|---|
| `_DATA_ROOT` (58) | `C:/Users/paulk/.../PCLA/data/<AGENT>` | base for the four below; `migrate_lead_to_baseline.py:559` (`alt_split`) |
| `BASELINE_DATA_DIR` (71) | `_DATA_ROOT/baseline_data` | 38 usages — frame loading, `baseline_{m}.npz`, `mdx_parameters/`, `mdx_v2_parameters/`, `mdx_features.npz`, example-image dumps |
| `TEST_DATA_DIR` (72) | `_DATA_ROOT/test_data` | 22 usages — frames, `test_labeled.npz`, `attention/`, `live_pert_frames/`, `relevance_live_pert/` |
| `VAL_DATA_DIR` (73) | `_DATA_ROOT/val_data` | `run_analysis.py:77` (Step 9.5), `migrate_lead_to_baseline.py:485` |
| `RESULTS_DIR` (74) | `_DATA_ROOT/results` | output roots in both analysis scripts and `sweep_clusters.py` |

With `EXPERIMENT_VARIANT = "alternative"` all four resolve to the `_alt` counterparts
(lines 66–69). `summarize_results.py` does **not** use these — it takes `--data-root`/`--alt`.

### 4.5 Recompute flags (cache control)

All default `False`; each guards an expensive stage in `run_analysis.py` /
`run_online_analysis.py`, falling back to cached artifacts:

| Flag (line) | Guards | Cached artifact |
|---|---|---|
| `RECOMPUTE_BASELINE` (37) | Step 2 baseline ATOMs profiles | `BASELINE_DATA_DIR/baseline_{m}.npz` |
| `RECOMPUTE_TEST_ATOMS` (38) | Step 8 test ATOMs profiles | `TEST_DATA_DIR/attention/test_profiles_{m}.npy` |
| `REAPPLY_PERTURBATIONS` (39) | Step 7 labeled-set generation (also auto-runs if `test_labeled.npz` missing, `run_analysis.py:680`) | `TEST_DATA_DIR/test_labeled.npz` |
| `RECOMPUTE_MDX_BASELINE` (40) | Step 2.5 MDX-v1 fit | `BASELINE_DATA_DIR/mdx_parameters/` (+ `mdx_features.npz` from HPC) |
| `RECOMPUTE_MDX_V2_BASELINE` (41) | Step 2.5-v2 MDX-v2 fit (local-only) | `BASELINE_DATA_DIR/mdx_v2_parameters.pkl` |
| `RECOMPUTE_MDX_TEST_SCORES` (42) | Step 9 MDX test-feature extraction | cached MDX test scores (`run_analysis.py:1320–1322`) |

`sweep_clusters.py:72–78` warns when any of the first four (excl. `REAPPLY`, `MDX_V2`) is
left `True` before a sweep.

### 4.6 Perturbation / attack parameters

| Attribute (line) | Value | Meaning / consumers |
|---|---|---|
| `NOISE_INTENSITY` (27) | `21` | Gaussian-noise σ for the labeled-set mix (`run_analysis.py:698,706`). Comment: "25 for day, 21 by night". |
| `BRIGHTNESS_INTENSITY` (28) | `3` | Brightness scale factor for the mix (`run_analysis.py:699,707`). |
| `PERTURBATION` (30) | `"brightness_scale"` | Live-injection perturbation name; also keys the online analysis dirs (`run_online_analysis.py:93`; `sensor_agent_live_perturbation.py:113`; `image_agent.py:171–218`). |
| `INTENSITY` (31) | `4` | Live-injection intensity (`sensor_agent_live_perturbation.py:218`; `image_agent.py:173`). |
| `INJECTION_TIME` (32) | `10` | Sim-seconds after which the live perturbation activates (`sensor_agent_live_perturbation.py:176`; `image_agent.py:167–169`). |
| `AFFECT_BOTH_CAMS` (33) | `True` | **Unused anywhere in the codebase** (BUG-3). |
| `CAM_INDEX` (34) | `None` | Which of the 6 (TFV6) / 2 (WoR) cameras to perturb; `None` = all (`sensor_agent_live_perturbation.py:219`; `image_agent.py:173`). |
| `MANUAL_SPAWNS` (35) | `True` | Traffic spawning strategy (`generate_traffic.py:48`). |
| `ADD_AUTOPILOT_VEHICLES` (76) | `True` | **Unused anywhere in the codebase** (BUG-3). |
| `FRAMES_TO_SKIP` (78) | `0` | Live PGD attack recomputed every `FRAMES_TO_SKIP+1` frames (`perturbation_manager.py:78`; stale comment "every 3rd frame", BUG-9). |
| `EPSILON` (79) | `8.0` | ℓ∞ budget for the **WoR** live FGSM/PGD and the WoR labeled-set PGD entry (`image_agent.py:212,218`; `run_analysis.py:709`; also TFV6 *live* PGD, `sensor_agent_live_perturbation.py:140`). Comment "TF: 12" is informational only. |
| `PGD_TARGET` (83) | `"brake"` | TFV6 PGD objective (`brake`/`max_speed`/`steer_left`/`steer_right`); used in the TFV6 labeled-set spec (`run_analysis.py:701`) and live injection (`sensor_agent_live_perturbation.py:139`). |
| `PGD_EPSILON` (84) | `14.0` | TFV6 labeled-set PGD ε (`run_analysis.py:701`). Comment says it "must match `hpc/prep_test.py` PGD_EPSILON default" — but that default is **12.0** (`hpc/prep_test.py:42,55`) → mismatch, BUG-1. |
| `PGD_N_STEPS` (85) | `8` | PGD iteration count for TFV6 live injection (`sensor_agent_live_perturbation.py:141`). |

### 4.7 Sampling / buffer sizes (live collection + plotting)

| Attribute (line) | Value | Meaning / consumers |
|---|---|---|
| `IMAGE_SAMPLE_INTERVAL` (53) | `25` | Keep every 25th frame during live baseline recording (`BaselineDataCollector`; `baseline_dataset.py:143`; `image_agent.py:90`; `lbc_agent.py:112`). |
| `TEST_SAMPLE_INTERVAL` (54) | `5` | Same for test / live-pert recording (`image_agent.py:91`; `sensor_agent_live_perturbation.py:112`). |
| `MAX_BASELINE_SIZE` (55) | `100` | Flush threshold (frames per saved npz) for live baseline buffer (`baseline_dataset.py:211`). |
| `MAX_TEST_SIZE` (56) | `200` | Flush threshold for live test buffer (`dataset.py:163`). |
| `MAX_LIVE_PERT_SIZE` (57) | `100` | Flush threshold for live-pert buffer (`dataset.py:163`). |
| `PLOT_SEG_AND_REL` (48) | `True` | Dump segmentation/relevance example images during baseline computation (`baseline_dataset.py:521`). |
| `PLOT_COMPARATIVE_REL` (49) | `True` | Compute forced-brake/forced-drive comparative relevance maps (`run_online_analysis.py:450`; `atoms_carla.py` comparative slots). |
| `PLOT_INTERVAL` (50) | `20` | Every Nth frame gets example images (`baseline_dataset.py:521`). |

### 4.8 Live-session spawn constants

Per-town `SPAWN_INDEX`, `SPEC_POS` (spectator location), `SPEC_ROT` (lines 100–134),
selected by `TOWN`. Consumed by `sample.py:43,51–52` and
`generate_traffic.py:123` (ego spawn point removed from the traffic pool). The Town05
block contains a deliberately disabled branch `if LIVE_PERTURBATION_RECORDING_MODE and
False:` (line 127, BUG-6) — the alternative spawn 235 is dead code; spawn 152 is always
used.

### 4.9 Magic constants hardcoded outside the config

All values below are **hardcoded** (not configurable via `atoms_config.py`):

| Location | Value | Meaning |
|---|---|---|
| `run_analysis.py:440` | `percentile = 99.0` | In-distribution threshold percentile for the single-Gaussian Mahalanobis detector (marked `<<< ADJUST`). |
| `run_analysis.py:697–710` | `fraction=0.20` ×5 | 5-way labeled-set mix: clean / gaussian_noise / brightness_scale / camera_loss / pgd, 20% each (both agent branches). Must structurally match `hpc/prep_test.py:_SPEC`. |
| `run_analysis.py:715` | `seed = 42` | Perturbation frame-assignment seed; must match `hpc/prep_test_task.sh:36` (`--seed 42`) and `hpc/prep_test.py` default (line 52). Independent of `conf.RANDOM_SEED`=17. |
| `run_analysis.py:1134` | `KNN_K_VALUES = [1, 5, 10, 25, 50, 100, 250]` | k-NN k sweep grid (test + val + GMM variants). |
| `run_analysis.py:134` | `"pcla_agents/transfuserv6_pretrained/visiononly_resnet34"` | TFV6 checkpoint dir (config has no `MODEL_PATH` despite docstring claims, BUG-2). |
| `run_analysis.py:172,281` | `"pcla_agents/wor_pretrained/leaderboard_weights"` | WoR weights dir (+ `main_model_10.th`, `config_leaderboard.yaml`). |
| `hpc/prep_test.py:35–42` | `_SPEC` intensities: noise 21.0, brightness 3.0, pgd 4.0; fractions 0.20 | HPC-side mirror of the perturbation mix (CLI-overridable: `--noise-intensity`, `--brightness-intensity`, `--pgd-epsilon`). |
| `hpc/array_task.sh:48` etc. | `MODE_ANALYSIS:-1` | HPC default analysis mode = 1 (local config currently 2 — must be passed explicitly). |
| `detectors.py:163` | `ridge=1e-6` | `MahalanobisDetector` default (overridden everywhere by `conf.MAHAL_RIDGE`=0.01). |
| `detectors.py:216` | `percentile=99.0` | `fit_threshold` default. |
| `detectors.py:542` | `k=50` | `KNNDetector` default k. |
| `detectors.py:615` | `n_projections=200, random_state=42` | Sliced-Wasserstein defaults. |
| `atoms_carla.py:240` | `p_relevance=0.9` | `ATOMsCarla` default (overridden by `conf.FC_RELEVANCE_FILTER`). |
| `atoms_carla.py:57–109` | `CARLA_CLASSES` (29 entries, tags 0–28), `TFV6_CLASSES` (10 entries), `REDUCED_CLASS_IDS` (8 WoR ids) | Profile dimensionality source: 29-dim WoR vs 10-dim TFV6. |
| `migrate_lead_to_baseline.py:66` | `_ROAD_OPTION_TO_IDX = {1:0,…,6:5}` | RoadOption → 0-based cmd mapping. |
| `migrate_lead_to_baseline.py:600–628` | defaults `n_frames=3000`, `testset_n_frames=500`, `baseline_n=5000`, `test_n=1000`, `val_n=1000`, `exclude_towns=["Town05"]` | Sampling targets (CLI-overridable). |
| `sweep_clusters.py:33` | `K_VALUES_DEFAULT = [10, 8, 12]` | Default K sweep (literal changed from `[18,20,22]`; help text still disagrees, BUG-5 → finding 8.7). |
| `sample.py:13–14` | `"localhost"`, port `2000`, timeout `10.0` | CARLA connection. |
| `leaderboard_codes/watchdog.py` (per CLAUDE.md) | 260 s | Agent-init watchdog timeout. |

---

## 5. Key design decisions & rationale

**Single static config class, no CLI layer.** Every script and both agents import the same
`ExperimentConfig` class; an experiment is fully specified by one file diff. This was
chosen over per-script argparse for reproducibility (the config file can be committed
alongside results) and because the agents run inside the CARLA leaderboard harness where
CLI plumbing is awkward. Cost: import-time path resolution makes runtime mutation unsafe,
and the file accumulates dead attributes (§6). The one exception — `--gmm-k` on
`run_analysis.py` — was added 2026-06-09 specifically because `sweep_clusters.py` runs
the analysis as a subprocess and previously its `--gmm-k` flag was silently ignored
(`docs/design_decisions.md`, "Val/Test split…" section; verified: argparse at
`run_analysis.py:51–54`).

**Mode-suffixed artifact filenames** (`baseline_{1|2}.npz`, `test_profiles_{1|2}.npy`,
`atoms_analysis_mode_{1|2}/`). The two ATOMs analysis modes produce incompatible profile
semantics; suffixing lets both coexist on disk so switching `MODE_ANALYSIS` re-runs the
cheap analysis against cached expensive profiles without recomputation (CLAUDE.md "Data
Layout"; verified at `run_analysis.py:110,116,225` and `hpc/submit_val.sh:44–45`).

**`EXPERIMENT_VARIANT` as a pure path switch.** The alternative (same-distribution) split
exists to isolate perturbation-induced OOD signal from town-domain shift. Implementing it
as four path substitutions (lines 65–74) means zero downstream code changes — every
loader/saver resolves through the config — at the cost of having to keep two full data
trees (verified on disk, §2.4).

**Agent-dependent profile dimensionality (29 vs 10).** WoR data carries raw CARLA semantic
tags 0–28 → 29-dim profiles (`CARLA_CLASSES`, `atoms_carla.py:57`, `NUM_CARLA_CLASSES=29`
line 109). TFV6/LEAD data is stored with `save_grouped_semantic=True`, i.e. the LEAD
exporter already collapsed 32 raw tags into 10 driving-relevant groups → 10-dim profiles
(`TFV6_CLASSES`, `atoms_carla.py:92`). The class map is selected once per run
(`run_analysis.py:192`). CLAUDE.md explicitly retracts the older "23-dim" claim; a stale
"23-class" comment survives at `run_analysis.py:190–191` (BUG-12).

**HPC scripts decoupled from the config.** Profile computation must run on Linux/Slurm
where the Windows `_DATA_ROOT` is meaningless; all HPC parameters travel via CLI/env
(§3). This is why the perturbation seed (42) and the mix spec are duplicated rather than
imported — duplication is the consistency mechanism, enforced only by comments
(`run_analysis.py:692–695,715`), which is exactly where BUG-1 (ε 14 vs 12) crept in.

**Val-set-driven hyperparameter selection.** k (k-NN) and K (GMM) were originally picked
on the test set — data leakage, documented and fixed per `docs/design_decisions.md`
("Val/Test split for hyperparameter selection", 2026-06-08/09). Verified in code:
`run_analysis.py:1486–1492` (val-AUC k selection with test fallback + warning),
`__val_auc_gmm_avg__` written to `summary.json`, consumed by `summarize_results.py`
Section 3.

**Cache-flag pattern instead of a build system.** Each expensive stage is guarded by a
`RECOMPUTE_*` boolean plus an existence check on its artifact. Simple and transparent, but
staleness is the user's responsibility (no dependency tracking — e.g. changing
`FC_RELEVANCE_FILTER` does not invalidate cached profiles).

---

## 6. Known limitations / open issues

1. **Machine-specific config**: `_DATA_ROOT` hardcoded Windows path (line 58); the repo
   cannot run elsewhere without editing the config. `hpc/gather_live_pert.py` imports the
   config on Linux and inherits this fragility.
2. **Cross-environment constant duplication** (perturbation mix, seed 42, PGD ε, mode
   default 1 vs 2) between `run_analysis.py`, `atoms_config.py`, and `hpc/` — kept in sync
   only by comments; already drifted once (BUG-1).
3. **Dead/unused config attributes**: `AFFECT_BOTH_CAMS`, `ADD_AUTOPILOT_VEHICLES` (BUG-3);
   dead Town05 spawn branch (BUG-6); `"LBC"` agent value not actually routed (BUG-13).
4. **No validation of config consistency**: nothing checks `NUM_GMM_CLUSTERS ≤ GMM_MAX_K`
   (currently violated, BUG-10), nor that the active `MODE_ANALYSIS` matches the
   downloaded HPC artifacts.
5. **Import-time side effects**: path and spawn resolution at class-body execution means
   scripts must not mutate `AGENT`/`TOWN`/`EXPERIMENT_VARIANT` at runtime.
6. **`image_counter` as mutable global state** breaks if two computations run in one
   process.
7. **Stage-1 tooling outside the repo**: `unzip_routes.ps1` is documented in CLAUDE.md but
   absent from version control (BUG-8) — the dataset-selection step is not reproducible
   from the repo alone.

---

## 7. Cross-references

| Doc | Covers |
|---|---|
| `02_agents.md` | TFV6 / WoR network architectures, F_c choice, checkpoint contents |
| `03_lrp.md` | `lrp_transfuser.py`, `lrp_analysis.py`, AttnLRP rules, two-pass scheme |
| `04_atoms.md` | `ATOMsCarla`, MODE_ANALYSIS 1 vs 2 semantics, class maps, `_make_minimal_data`; **note**: the combinatorial-attention c(T) claimed in CLAUDE.md is not implemented (BUG-4) |
| `05_dataset_creation.md` | `migrate_lead_to_baseline.py` internals, loaders, Town05 holdout, alt split |
| `06_perturbations.md` | `PerturbationManager`, `PerturbationApplier`/`PerturbationSpec`, PGD details |
| `07_distances_and_detectors.md` | all detectors, `GMMClustering`, threshold policy (99th pct, Youden-J) |
| `08_offline_analysis.md` | `run_analysis.py` step-by-step, `sweep_clusters.py`, `summarize_results.py` |
| `09_online_analysis.md` | `run_online_analysis.py`, live injection agents |
| `10_hpc_pipeline.md` | `hpc/` scripts, env-var contract, mode-1 default pitfall |
| `11_validation_and_testing.md` | test suites and diagnostics under `ATOMs_Analysis/utils/` |
| `12_visualization.md` | `visualization_carla.py`, `viz_config.py` shared styling |
| `99_bugs_and_findings.md` | running bug log (BUG-1 … BUG-14 from this doc) |
