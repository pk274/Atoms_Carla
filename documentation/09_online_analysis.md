# Topic 9 — Online Analysis: `run_online_analysis.py`, Mid-Drive Injection, and Distance-over-Time

All claims verified against code on 2026-06-14. Line numbers refer to the current working tree.
Primary sources read in full: `run_online_analysis.py` (689 lines, repo root), `pcla_agents/transfuserv6/lead/inference/sensor_agent_live_perturbation.py` (306 lines), `hpc/prep_live_pert.py` (91 lines), `hpc/prep_live_pert_wor.py` (93 lines), `hpc/compute_live_pert_chunk.py` (204 lines), `hpc/gather_live_pert.py` (107 lines). Cross-checked against `ATOMs_Analysis/atoms_config.py`, `ATOMs_Analysis/utils/visualization_carla.py:1359-1414` (`plot_distance_over_time`), `ATOMs_Analysis/detection/dataset.py:120-180,461-470` (`TestDataCollector`), `CLAUDE.md`, `documentation/06_perturbations.md`, `documentation/07_distances_and_detectors.md`, `documentation/08_offline_analysis.md`, and the on-disk `data/TFV6/test_data/live_pert_frames/` + `data/TFV6/test_data/attention/live_pert/` trees.

---

## 1. Purpose & scope

This document covers the **online (closed-loop) experiment** — the half of the project that injects a perturbation *mid-drive*, while the agent is actually controlling the car in CARLA, and then traces how each detector's anomaly score evolves *around the injection point*. It is the online counterpart to Topic 8's offline `run_analysis.py`. Two scripts span the experiment:

1. **`sensor_agent_live_perturbation.py`** (the *recording* side, Topic 6 §3.6 introduced it) — a `DataCollectionSensorAgent` subclass that drives the TFV6 ensemble in CARLA, flips an injection flag at `conf.INJECTION_TIME` seconds, applies the chosen perturbation to the live camera input from that point on, and records a continuous time-series of frames (clean before injection, perturbed after) plus a parallel pre-perturbation "clean RGB" copy.
2. **`run_online_analysis.py`** (the *analysis* side, repo root) — reuses the entire baseline-fitting half of `run_analysis.py` (load model → baseline ATOMs → MDX → Mahalanobis → GMM), then, **per recorded live-pert file**, computes-or-loads ATOMs profiles, scores them with a reduced detector roster, and produces one **distance-over-time figure per detector** with the injection frame marked and the clean series overlaid.

The HPC chain (`prep_live_pert*.py`, `compute_live_pert_chunk.py`, `gather_live_pert.py`) is the live-pert analogue of the offline HPC chain and produces the ATOMs profiles that `run_online_analysis.py` consumes when `RECOMPUTE_TEST_ATOMS=False`.

This topic documents control flow, I/O contracts, the local-vs-HPC split, and the magic constants. It does **not** re-derive detector math (Topic 7), perturbation pixel transforms or the live PGD ensemble attack (Topic 6 §2.3/§3.4/§3.6), or the ATOMs profile (Topic 4). The two load-bearing findings of this review are: (a) **the online experiment produces no ROC/AUC** — it is qualitative-only, a visual distance-over-time trace, never a quantitative detector ranking (§2.2, §3.7); and (b) **scores are plotted completely raw — there is no windowing, rolling mean, EMA, or any score smoothing whatsoever** (§2.5, §3.7, definitively confirmed against both `run_online_analysis.py` and `plot_distance_over_time`).

Scope note: `run_online_analysis.py`'s module docstring (`:1-30`) is a **verbatim copy of `run_analysis.py`'s** — it describes the offline ROC/AUC pipeline that this script does not run (finding 1.11, Topic 1; restated as 9.1 with the new detail that the copied "Steps 1–12" outline bears no relation to the actual control flow). We treat the **code** as ground truth.

---

## 2. Key design decisions

### 2.1 Closed-loop live injection vs offline static perturbation

The offline experiment (Topic 8) perturbs *pre-recorded* clean Town05 frames and scores them in isolation: each frame is an independent IID test point, the agent never reacts, and the labels come from a fixed 5-way 20 % mix. The online experiment is fundamentally different: the perturbation is injected into the **live sensor stream of a driving agent**, so

- the agent *reacts* to the corrupted input — a PGD "brake" attack can actually stop the car, a phantom obstacle can make it swerve — and every subsequent frame is influenced by that reaction (feedback loop absent offline);
- frames form a **temporally ordered trajectory through one drive**, not a shuffled IID set; consecutive frames are highly correlated;
- the "label" is a single **change-point** in time (`is_perturbed` flips 0→1 once, at `INJECTION_TIME`), not a per-frame independent draw.

The online experiment therefore answers a different question than the offline one: *does the detector's anomaly score visibly rise when the perturbation actually starts affecting the running agent?* This is closer to the deployment scenario the thesis motivates (a runtime OOD monitor on a driving car) than the offline IID-classification framing.

### 2.2 Distance-over-time traces, not ROC/AUC — qualitative only

Because the online signal is a single change-point in a correlated time-series rather than a labeled IID set, the script does **not** compute ROC curves, AUC, or Youden thresholds. Instead, for each detector it calls `plot_distance_over_time(...)` (`run_online_analysis.py:677-685`), which renders the per-frame score against frame index, draws a red dotted vertical line at the injection frame, and overlays the clean-series score as a dashed line. The reader judges *by eye* whether the perturbed score departs from the clean baseline after the injection line.

**Rationale.** A single drive has one injection point; you cannot build a meaningful ROC from one positive change-point and a handful of correlated frames. The intended evidence is visual: a detector "works" online if its score jumps at the injection line and stays elevated, and stays flat on the clean overlay. This is exactly the **qualitative-only** status that `summarize_results.py` records in its Section 7 "Live-perturbation inventory" (Topic 8 §3.16), which lists the live figures by PNG stem with **no AUC** (cross-ref Topic 8 §6). The consequence is that the online experiment provides *illustrative* rather than *quantitative* detector evidence (finding 9.6) — the thesis cannot rank detectors from it.

### 2.3 The baseline-fitting half is duplicated from `run_analysis.py`, not imported

`run_online_analysis.py` STEPS 1–5 (`:106-343`) are a near-verbatim copy of `run_analysis.py`'s baseline-side STEPS 1, 2, 2.5(MDX-v1), 4, 5, 6 — load agent + LRP + ATOMs, compute/load baseline ATOMs (`baseline_<mode>.npz`), fit/load MDX-v1, fit single-Gaussian Mahalanobis, BIC/AIC K-sweep, fit GMM. It does **not** call into `run_analysis.py`; the two share no orchestration code, only the same underlying library classes (`BaselineComputer`, `MahalanobisDetector`, `GMMClustering`, `MDXDetector`, `DistanceComputer`). The same recompute gates apply: with all flags `False` (the live config) the baseline side **loads cached artifacts** rather than recomputing — `baseline_<mode>.npz` is loaded (`:173-186`), `mdx_parameters.pkl` is loaded (`:238-242`), Mahalanobis and GMM are re-fit cheaply from the cached baseline series (`:268-331`), so a live-analysis run is fast on the baseline side and only the per-file live-pert loop is heavy. This duplication is a maintenance hazard: any baseline-side fix in `run_analysis.py` (e.g. the alignment-key guard, the MDX-v2 fit, the val handling) is **absent** here (finding 9.2).

Differences from the offline baseline side: (i) the online script has **no MDX-v2 (Step 2.5-v2) block** at all; (ii) it has no Step 7 baseline visualisation (PCA/t-SNE/attention bars); (iii) the GMM K-resolution lacks the `--gmm-k` CLI override — it resolves K as `best_k_bic` overridden by `conf.NUM_GMM_CLUSTERS` only (`:313-315`), so there is no sweep driver for the online path.

### 2.4 HPC computes the profiles, the local script plots them

Mirroring the offline split (Topic 8 §2.5), the expensive per-frame ATOMs computation is designed to run on HPC, with a local fallback gated by `RECOMPUTE_TEST_ATOMS`:

- **`RECOMPUTE_TEST_ATOMS=True`** (local fallback): the per-file loop runs LRP+ATOMs on the GPU locally, frame by frame, saving `live_pert_profiles_{variant}_{mode}.npy` (and a clean companion) plus relevance PNGs (`:421-524`).
- **`RECOMPUTE_TEST_ATOMS=False`** (the live config, default): the loop **loads** the cached profile `.npy`, and **hard-errors with a pointer to the HPC chain** if it is missing — `"Run HPC (submit_live_pert.sh + collect_results.sh) first"` (`:527-532`).

Unlike the offline PGD deferral, **no perturbation is crafted on HPC for the live path** — the live perturbation (including PGD) was already baked into the *recorded pixels* during driving (§3.8). So the HPC live-pert chain is pure profile computation over already-perturbed frames; `prep_live_pert*.py` are concatenation-only (Topic 6 §3.6) and `compute_live_pert_chunk.py` does no attack crafting (§3.9). The deferred-PGD / clean-pixel-but-perturbed-label complication of the offline path does **not** apply online.

### 2.5 Scores are plotted raw per frame — no smoothing or windowing anywhere

This is a question the plan asks to answer definitively. **Answer: there is no smoothing, rolling mean, moving average, EMA, Savitzky-Golay filter, or windowing of any kind, in either `run_online_analysis.py` or `plot_distance_over_time`.** The score arrays are built one entry per frame (`:559-657`), passed unmodified to `plot_distance_over_time`, and plotted with a marker-line `"o-"` directly against `np.arange(len(dist))` (`visualization_carla.py:1388,1392`). A grep for `rolling|moving.average|smooth|window|convolve|ema|savgol|uniform_filter` over both files returns nothing. Each plotted point is one frame's raw distance.

**Consequence.** Because the underlying ATOMs profiles are noisy per frame (LRP attribution varies frame-to-frame, the speed token is set per frame, segmentation masks shift), the raw distance-over-time traces are inherently jittery. A detector whose mean score rises after injection but with high per-frame variance is hard to read off an unsmoothed plot; a moving-average or short-window mean would make the change-point far more legible. The absence of any smoothing is a deliberate-or-overlooked design gap that weakens the qualitative read (finding 9.5).

### 2.6 `EXPERIMENT_VARIANT` re-roots the inputs but not the live results, and the live data lives under the original split

Like every path in the project, `TEST_DATA_DIR` resolves through `EXPERIMENT_VARIANT` (`atoms_config.py:65-74`): under the live `"alternative"` setting it points at `test_data_alt`. **But the live-pert frames and profiles physically exist only under the original `test_data/`** (`data/TFV6/test_data/live_pert_frames/` is populated; `data/TFV6/test_data_alt/live_pert_frames/` is *empty*, verified on disk). So running `run_online_analysis.py` as configured (`EXPERIMENT_VARIANT="alternative"`) globs `test_data_alt/live_pert_frames/` and **raises `FileNotFoundError`** (`:379-383`) — the live experiment was conducted under the original split and is not reachable with the live config without flipping `EXPERIMENT_VARIANT` back to `"original"`. Separately, the **results** path is also inconsistent internally: `OUT_DIR` is `RESULTS_DIR/atoms_analysis_live_mode_<mode>` (so `results_alt/...` under the live config), yet the per-detector figures are written to `RESULTS_DIR/live_perturbation/<pert>/` *for one branch* and to `OUT_DIR` *for another* (§3.7). On disk the live figures sit under the non-alt `data/TFV6/results/live_perturbation/`, again confirming the experiment ran under the original split. This is finding 9.3 (a live-path variant of finding 5.7).

### 2.7 One global `PERTURBATION` name keys both the recording and the analysis

The whole online pipeline is keyed off the single global `LIVE_PERT_NAME = conf.PERTURBATION` (`run_online_analysis.py:93`). It (i) builds the glob `run_{LIVE_PERT_NAME}_live_pert_*.npz` that selects which recorded files to analyse (`:376`), (ii) names the output attention dir `attention/live_pert/{LIVE_PERT_NAME}/` (`:95`), the relevance dir, and the results dir (`:660`). There is no per-file perturbation field; the perturbation is inferred purely from the filename prefix. Consequently a recording made with `PERTURBATION="pgd"` and one made with `PERTURBATION="brightness_scale"` write profiles into *different* folders, but if the config's `PERTURBATION` is later changed and the script re-run, profiles for the new perturbation's variants land in the new folder while stale profiles from earlier runs remain — on disk the `attention/live_pert/brightness_scale/` folder contains profiles named after **PGD variants** (`brake_205328_000`, `nocrash_155706_000`, …) that the brightness glob never matches, i.e. stale leftovers from runs under a different `PERTURBATION` setting (finding 9.4). The single-global-name design makes the folders accumulate mis-keyed artifacts.

---

## 3. Implementation details

### 3.1 Top-level structure and the copied docstring (`:1-102`)

The module docstring (`:1-30`) is byte-for-byte `run_analysis.py`'s, down to "`run_analysis.py`" as its title and the offline "Steps 1–12" outline that ends in "Evaluate each detector (ROC, AUC, Youden threshold)" — none of which this script does (finding 9.1, extends 1.11). After imports, two output anchors are set: `OUT_DIR = RESULTS_DIR/atoms_analysis_live_mode_<mode>` (`:89`) and `ATT_DIR = TEST_DATA_DIR/attention/live_pert/<LIVE_PERT_NAME>` (`:95`), with `_mode = conf.MODE_ANALYSIS` (`:97`). The banner it prints still says "ATOMs Analysis Pipeline" (`:100`).

### 3.2 STEP 1 — model + LRP + ATOMs (`:106-156`)

Identical agent branch to offline STEP 1 (Topic 8 §3.3): TFV6 loads `TrainingConfig` from `visiononly_resnet34/config.json`, instantiates `TFv6`, loads the **first** `model*.pth` (`ckpt_files[0]`, dropping shape-mismatched keys, `:117-124`) — i.e. the single-member analysis of finding 2.5 — and wraps it in `LRPTFv6Model`. WoR loads `CameraModel` + `main_model_10.th` + `LRPCameraModel`. `action_logits_available = (conf.AGENT == "WOR")` (`:138`). `ATOMsCarla` is built with `p_relevance=conf.FC_RELEVANCE_FILTER` (0.9), `default_cmd=conf.DEFAULT_CMD` (2), `mode_analysis=conf.MODE_ANALYSIS`, `use_reduced=False` (`:145-152`). The inline comments here are stale in the same ways flagged for offline: "7 driving-relevant classes instead of all 23" (`:142`, finding 4.10) and "3 = FOLLOW_LANE" for `DEFAULT_CMD=2` (`:148`, finding 1.7/2.12).

### 3.3 STEP 2 / 2.5 — baseline ATOMs and MDX-v1 (`:159-257`)

**STEP 2** (`:159-190`): gated by `RECOMPUTE_BASELINE` OR missing `baseline_<mode>.npz`; otherwise loads the cached npz into `baseline_series [N,C]`, `baseline_mean [C]`, `baseline_cov [C,C]` (all float64). Identical to offline STEP 2. **STEP 2.5 = MDX-v1** (`:192-257`): if `RECOMPUTE_MDX_BASELINE`, extracts features (TFV6 `lrp.get_backbone_features` 512-d + speed proxy `[0, min(spd/25,1), 1 if spd<0.5 else 0]`, `:209-213`; WoR via `model.get_features` + `ImageAgent` action proxy, `:214-229`) and fits `MDXDetector(n_pca_components=50)`. Otherwise it loads `mdx_parameters.pkl`, or fits from `mdx_features.npz` (the HPC-precomputed file) if the pkl is missing, or **raises** if neither exists (`:253-257`). There is **no MDX-v2** block (§2.3). The WoR config path is a **hardcoded absolute Windows path** at `:202` (finding 9.7).

### 3.4 STEP 3 / 4 / 5 — Mahalanobis, K, GMM (`:260-343`)

**STEP 3** (`:260-282`): `MahalanobisDetector(ridge=conf.MAHAL_RIDGE).fit(baseline_series)`; `fit_threshold(percentile=99.0)`; save `OUT_DIR/mahal_detector.npz`. This 99th-percentile, label-free threshold is the **only deployable runtime decision boundary** in the whole project (Topic 7 §2.6) — but note it is computed and saved here yet **never actually applied** to flag frames in the online script; the distance-over-time plots show raw scores, not thresholded alarms (finding 9.6 facet). **STEP 4** (`:285-317`): BIC/AIC sweep over `MAX_K=GMM_MAX_K` (10); plot `gmm_model_selection.png`; resolve `N_COMPONENTS = best_k_bic`, overridden by `conf.NUM_GMM_CLUSTERS` (10) when not None (`:313-315`). No `--gmm-k` CLI path. **STEP 5** (`:320-343`): `GMMClustering(n_components=N_COMPONENTS, covariance_type=conf.GMM_COV_TYPE, random_state=conf.RANDOM_SEED, ridge=conf.MAHAL_RIDGE).fit`; save `OUT_DIR/gmm.npz`; log cluster sizes. The same seed mismatch as offline applies (live fit seed 17 vs model-selection seed 42; finding 8.5).

### 3.5 The per-file / per-variant loop (`:347-687`)

This is the heart of the online script and has **no offline analogue**. After the baseline side, the script (banner "STEPS 7–9", `:347-348`) globs the recorded frame files and processes each one **independently** as a self-contained drive:

- **Glob** (`:374-383`): `frames_dir = TEST_DATA_DIR/live_pert_frames`; `frame_files = sorted(run_{LIVE_PERT_NAME}_live_pert_*.npz)` *excluding* any stem ending `_clean_rgb` (`:377`). Raises if none match (`:379-383`).
- **Variant parsing** (`:390-392`): `_prefix = "run_{LIVE_PERT_NAME}_live_pert_"`; `variant = frame_file.stem[len(_prefix):]`, e.g. `"brake_205328_000"`. A **variant** is therefore the per-recording suffix (a human-chosen scenario tag like `brake`/`nocrash`/`heavycrash`/`smallstreetbrake`/`noreaction` + a timestamp + a run counter) that uniquely names one driving episode for that perturbation. Each variant is loaded, profiled, scored, and plotted in isolation — there is no pooling across variants.
- **Frame load** (`:357-371`, `_load_single_live_pert_file`): reads `wide_rgb`, `seg_red_wide`, `cmd`, `speed`, `frame_idx`, optional `narr_rgb`/`seg_red_narr`/`is_brake`, and **`is_perturbed`** (defaulting to all-zeros if absent). For TFV6 there is no narrow camera, so `narr_rgb`/`seg_red_narr` are `None`.
- **Clean-RGB pairing** (`:402-410`): if a sibling `<stem>_clean_rgb.npz` exists, `clean_data` is built as a *copy of the perturbed dict with `wide_rgb` swapped for the clean pixels* — i.e. the segmentation masks, cmd, speed, frame_idx are **borrowed from the perturbed file**, only the RGB differs. This is the source of the dashed "Clean" overlay (§2.2). The clean overlay therefore measures the attention-profile distance the *same* drive would have had with un-perturbed pixels but identical metadata.

### 3.6 Profile compute-or-load (Step 8, `:420-554`)

**Compute branch** (`RECOMPUTE_TEST_ATOMS=True`, `:421-524`): `atoms.reset()`, then per frame `atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)` — note `spd` **is** passed here (unlike the offline test loop, finding 4.4; the online and HPC paths condition on true speed). Relevance PNGs are written to `relevance_live_pert/<pert>/<variant>/` via `visualize_relevance`, with the comparative brake/drive maps when `PLOT_COMPARATIVE_REL=True` (`:441-478`). WoR action logits are collected per frame (`:480-481`). Profiles saved to `ATT_DIR/live_pert_profiles_{variant}_{mode}.npy`; logits to `live_pert_{action|speed}_logits_{variant}_{mode}.npy` (`:412-418,488-491`). The **clean** frames, if present, are profiled in a second loop with the same metadata but clean RGB → `live_pert_profiles_{variant}_clean_{mode}.npy` (`:494-524`).

**Load branch** (the live config, `:526-554`): loads `live_pert_profiles_{variant}_{mode}.npy`; **hard-errors** with the HPC pointer if missing (`:527-532`); raises on a row-count vs frame-count mismatch (`:535-541`); loads the logits and the clean profiles when present (skipping clean scoring with a warning if the clean profile file is absent, `:548-554`). There is **no `(run_id, frame_idx)` alignment-key guard** here (the offline STEP 9 guard of Topic 8 §3.8 is absent), so a profile file silently mis-aligned with its frame file would only be caught by the length check (finding 9.2 facet).

### 3.7 Scoring (Step 9) and the online detector roster (`:556-685`)

For every frame in the variant the script computes (`:559-657`):

| Detector | Call | Cross-ref |
|---|---|---|
| Mahalanobis (single-Gaussian) | `DistanceComputer.compute_mahalanobis(baseline_mean, baseline_cov, …, MAHAL_RIDGE)` | Topic 7 §2.2 |
| Euclidean (single) | `DistanceComputer.compute_euclidean(baseline_mean, …)` | Topic 7 §2.3 |
| k-NN (single) | `DistanceComputer.compute_knn_distance(baseline_series, …, k=25, normalize=True)` | Topic 7 §2.4 |
| JSD (single) | `DistanceComputer.compute_jsd(p=baseline_mean, q=…)` | Topic 7 §2.5 |
| Mahalanobis-GMM (nearest) | `DistanceComputer.compute_gmm_distance(gmm.means_/covariances_/weights_, mode="nearest", MAHAL_RIDGE)` | Topic 7 §2.6/§4 |
| Action entropy / PEOC | `ActionEntropyDetector(from_logits=True).score_batch(test_logits_all)` (only when logits available) | Topic 7 §2.8/§2.9 |
| MDX-v1 | per-frame `lrp.get_backbone_features` (TFV6) / `model.get_features` (WoR) → `mdx.score` | Topic 7 §2.7 |

The same five distance scorers are run on the **clean** profiles when present (`:613-642`), yielding the dashed-overlay arrays `_cs_mahal/_cs_euclid/_cs_knn/_cs_jsd/_cs_mahal_gmm`. The MDX block (`:644-657`) extracts backbone features *fresh from the perturbed RGB* each frame (no caching, no clean MDX overlay).

**Online vs offline roster (key difference).** The online roster is a **reduced subset** of the offline one. Online scores: Mahalanobis-single, Euclidean-single, **k-NN-single at a fixed k=25**, JSD-single, Mahalanobis-GMM-nearest, Action-entropy/PEOC, MDX-v1 — **seven** traces. It does **not** score: Euclidean-GMM, JSD-GMM, k-NN-GMM, Wasserstein (single or GMM), the k-NN *k-sweep* (offline sweeps `[1,5,10,25,50,100,250]`; online hardcodes k=25, `:581,629`), MDX-v2 (never even fitted online), and there is no per-perturbation breakdown. The online k=25 is a magic constant with no val-based selection (offline selects k on val AUC, Topic 7 §2.7) — finding 9.8.

**Injection-frame determination** (`:663-673`): the injection frame is `int(np.argmax(test_data["is_perturbed"]))` — the saved-frame index of the **first** post-injection frame — *whenever any `is_perturbed` entry is 1* (`:664-666`). The fallback, used only when the `is_perturbed` key is absent or all-zero, is `int(np.searchsorted(test_data["frame_idx"], conf.INJECTION_TIME * _CARLA_HZ))` with **`_CARLA_HZ = 20`** hardcoded inline (`:668-671`). Because `frame_idx` is the raw CARLA tick index sampled every `TEST_SAMPLE_INTERVAL` ticks (5 in the live config), `INJECTION_TIME=10 s × 20 Hz = tick 200`, which `searchsorted` maps to saved-index ≈ 40 — consistent with the observed `argmax(is_perturbed)=40` on a real file (76 frames, 36 perturbed). The `_CARLA_HZ=20` constant is hardcoded and **not** sourced from any config (finding 9.9); the argmax path is the one actually taken on all recorded files (they all carry `is_perturbed`).

**Plotting** (`:675-685`): `_plot_label = f"{LIVE_PERT_NAME}_{variant}"`; then one `plot_distance_over_time(scores, _plot_label, distance_type, OUT_DIR, _injection_frame, dist_clean=…)` per detector. Note the figures go to **`OUT_DIR`** (= `RESULTS_DIR/atoms_analysis_live_mode_<mode>`), even though `live_pert_dir = RESULTS_DIR/live_perturbation/<pert>` is created at `:660-661` and then **never used as the save target** — a dead directory creation (finding 9.10). `plot_distance_over_time` (`visualization_carla.py:1359-1414`) plots `dist` with `"o-"` and the optional dashed `dist_clean`, draws the red dotted injection line, and saves `{distance_type}_{perturbation}.png`. No AUC, no threshold line, no smoothing.

### 3.8 The recording side — `sensor_agent_live_perturbation.py` (concise; see Topic 6 §3.6)

`LivePerturbationSensorAgent` subclasses `DataCollectionSensorAgent`. Activation requires `conf.LIVE_PERTURBATION_RECORDING_MODE=True` (`:110`, the live config). The injection flag flips in `run_step` once `timestamp >= conf.INJECTION_TIME` (`:175-185`), printing a one-time "PERTURBATION ACTIVATED" banner. Two perturbation application paths:

- **Non-PGD** (`tick`, `:214-222`): applied to the uint8 image via `perturb_tfv6_image(..., n_cameras=_N_FORWARD_CAMS=3)`. Crucially, the perturbation is written into `input_data["rgb"]` (the *model's* input), but the comment warns that the **save crop** is only the first 1152 px (`save_rgb = full_rgb[..., :fwd_width]`, `:204-206`) — the model actually sees the full strip while only the 3-forward-camera crop is recorded (Topic 6 §3.6, finding context).
- **PGD** (`_perturb_tensor_hook`, `:131-169`): applied to the float tensor *after* tensor prep via `pgd_attack_tfv6(nets=self.closed_loop_inference.nets, …, epsilon=conf.EPSILON=8.0, n_steps=conf.PGD_N_STEPS=8, target=conf.PGD_TARGET)` — i.e. against the **full 3-model ensemble** (Topic 6 §3.4/§3.6; ensemble-vs-single-member finding 6.8/2.5; live ε=8 vs offline ε=14 finding 6.4). The PGD frame is recorded *after* the attack so the saved pixels match what the model saw; `tick` defers seg/cmd/speed/clean as `_pending_*` for the hook to consume (`:245-251`).

Per frame the collector records `wide_rgb` (3-cam crop, perturbed-or-clean), `seg_red_wide` (forward-cam grouped semantics, `:231-241`), `cmd = argmax(command)`, `speed`, `is_brake`, `frame_idx` (raw tick), and **`is_perturbed`** (= `self._injection_active`, `:261`). `destroy()` flushes the buffer and, separately, stacks the parallel `self._clean_frames` list into a `<stem>_clean_rgb.npz` companion holding only `wide_rgb` (`:276-303`). The collector auto-saves when the buffer hits `MAX_LIVE_PERT_SIZE=100` (`dataset.py:163`), naming files `run_{pert}_live_pert_{ts}_{run_count:03d}.npz` (`dataset.py:174-178`).

### 3.9 The live-pert HPC chain I/O

The chain is the live analogue of the offline HPC chain (Topic 10), but **does no perturbation crafting** — the perturbation is already in the recorded pixels (§2.4).

1. **`prep_live_pert.py`** (TFV6) / **`prep_live_pert_wor.py`** (WoR): pure NumPy concatenation. Globs `run_{perturbation}_live_pert_*.npz`, concatenates `wide_rgb`/`seg_red_wide`/`cmd`/`speed`/`is_brake`/`frame_idx` and a synthesised `run_id` into `live_pert_concat.npz`, and writes `live_pert_meta.txt` (total frame count) to size the SLURM array (`prep_live_pert.py:60-86`). The WoR variant additionally preserves `narr_rgb` + `seg_red_narr` and *requires* `narr_rgb` (`prep_live_pert_wor.py:58-69`). **Neither carries `is_perturbed`** — the injection index is reconstructed locally at plot time from the *frame* file, not the concat (so this loss is harmless for the current plotting path, but means the concat alone cannot locate the injection point).
2. **`compute_live_pert_chunk.py`** (GPU worker): loads `live_pert_concat.npz`, processes a `[chunk_start:chunk_end)` slice through LRP + ATOMs, and writes a partial `.npz` with `profiles`, `chunk_start/end`, `class_ids/names`, and a logit array keyed `speed_logits` (TFV6) or `action_logits` (WoR) for PEOC (`:157-196`). It builds `ATOMsCarla` with **`p_relevance=0.25, default_cmd=2`** hardcoded (`:139-146`) — the same 0.25-vs-0.9 HPC discrepancy as the baseline/test chunk workers (finding 4.3) — and `--mode-analysis` defaults to **1** (`:45-46`), the mode where `p_relevance` matters. `process_frame` is called with `spd` (`:165`). No attack is crafted (`:9-11` "already perturbed; no perturbation application needed").
3. **`gather_live_pert.py`**: sorts partials by `chunk_start`, concatenates into `live_pert_profiles.npy` + `live_pert_{speed|action}_logits.npy`, and prints copy-into-repo instructions targeting `data/{agent}/test_data/attention/live_pert/$PERT/` (`:73-101`).

**HPC ↔ local naming divergence (load contract mismatch).** `gather_live_pert.py` writes **`live_pert_profiles.npy`** — *no variant, no mode suffix* — whereas the local loader expects **`live_pert_profiles_{variant}_{mode}.npy`** (`run_online_analysis.py:412`). On disk both schemes coexist in `attention/live_pert/pgd/` (`live_pert_profiles_1.npy` AND `live_pert_profiles_brake_205328_000_1.npy`), so the gathered HPC file would **not** be picked up by the per-variant load branch as written — it must be manually renamed per variant before the local script can consume it. The HPC chain concatenates *all* variant files into one profile array (the concat loses the per-variant boundaries), while the local analysis is strictly per-variant — the two granularities are incompatible without a manual split/rename (finding 9.11).

### 3.10 On-disk live-pert data layout (verified)

Under the **original** split (where the data actually lives):

```
data/TFV6/test_data/
  live_pert_frames/
    run_<pert>_live_pert_<variant>.npz            # perturbed time-series (one drive)
    run_<pert>_live_pert_<variant>_clean_rgb.npz  # parallel clean RGB only
  live_pert_frames_backup/                         # stale: *.npz + *.npz.bak + *.npz.bak2
  alternative live_perts/                          # stale, NAME HAS SPACES: run_pgd_live_pert_e16n5_*, e24n5_* (ε/step ablations)
  relevance_live_pert/<pert>/<variant>/            # per-frame relevance PNGs (local compute branch)
  attention/live_pert/<pert>/
    live_pert_profiles_<variant>_<mode>.npy        # local naming (per variant, mode-suffixed)
    live_pert_profiles_<variant>_clean_<mode>.npy  # clean companion
    live_pert_speed_logits_<variant>_<mode>.npy    # TFV6 (action_logits for WoR)
    live_pert_profiles_<mode>.npy                  # HPC naming (gathered, no variant) — coexists, mismatched
data/TFV6/results/live_perturbation/<pert>/        # created but figures actually go to results/atoms_analysis_live_mode_<mode>/
data/WOR/test_data/live_pert_frames/               # WoR equivalents (both cameras); + live_pert_videos/
```

**One perturbed live-pert npz schema** (verified, `run_pgd_live_pert_nocrash_155706_000.npz`, 76 frames):
`wide_rgb [N,3,384,1152] uint8`, `seg_red_wide [N,384,1152] uint8`, `cmd [N] int32`, `speed [N] float32`, `is_brake [N] int8`, `frame_idx [N] int32` (raw tick, step 5: `[0,5,10,…]`), `is_perturbed [N] int8` (0 before injection, 1 after; 36 of 76 set, first-1 at index 40). **No `narr_rgb`/`seg_red_narr`** for TFV6 (wide-only). The `_clean_rgb.npz` companion holds only `wide_rgb [N,3,384,1152] uint8`.

The `test_data/` live-pert tree contains files for **multiple** perturbations (`brightness_scale`, `pgd`, `phantom_obstacle`) plus the cross-keyed stale profiles noted in §2.7. The `alternative live_perts/` directory (note the **space** in the name) holds ε/step-ablation PGD recordings (`e16n5`, `e16p5`, `e24n5` = ε16/24, n5 steps) that are not addressable by the standard glob and document the live ε-sweep informally (finding 9.4 context).

---

## 4. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| `LIVE_PERTURBATION_RECORDING_MODE` | True | `atoms_config.py:19` | config | enables the recording agent (§3.8) |
| `PERTURBATION` (= `LIVE_PERT_NAME`) | "brightness_scale" | `atoms_config.py:30`; `run_online_analysis.py:93` | config | keys glob, att dir, results dir (§2.7) |
| `INTENSITY` (live) | 4 | `atoms_config.py:31` | config | non-PGD perturbation strength |
| `CAM_INDEX` (live) | None (all cams) | `atoms_config.py:34` | config | per-camera restriction |
| `INJECTION_TIME` | 10 (s) | `atoms_config.py:32`; `run_online_analysis.py:670,672` | config | live injection time; fallback injection-frame seed |
| `_CARLA_HZ` | 20 | `run_online_analysis.py:668` | **hardcoded inline** | fallback injection-frame conversion (finding 9.9) |
| `TEST_SAMPLE_INTERVAL` | 5 | `atoms_config.py:54` | config | tick-sampling stride for `frame_idx` |
| `MAX_LIVE_PERT_SIZE` | 100 | `atoms_config.py:57`; `dataset.py:163` | config | live-pert buffer auto-save size |
| `EPSILON` (live PGD ε) | 8.0 | `atoms_config.py:79`; hook `:140` | config | live PGD ℓ∞ budget (≠ offline 14.0, finding 6.4) |
| `PGD_N_STEPS` (live) | 8 | `atoms_config.py:85`; hook `:141` | config | live PGD iterations |
| `PGD_TARGET` (live) | "brake" | `atoms_config.py:83`; hook `:139` | config | live PGD objective |
| `MODE_ANALYSIS` | 2 | `atoms_config.py:23`; suffixes all live profiles & `OUT_DIR` | config | attribution mode |
| online k-NN k | 25 | `run_online_analysis.py:581,629` | **hardcoded** | fixed; no val selection (finding 9.8) |
| Mahalanobis threshold pct | 99.0 | `run_online_analysis.py:278` | code | saved but never applied online (finding 9.6) |
| `MAHAL_RIDGE` | 0.01 | `atoms_config.py:88`; `:269,564,602,…` | config | covariance ridge |
| `NUM_GMM_CLUSTERS` | 10 | `atoms_config.py:21`; `:314-315` | config | forced K (no `--gmm-k` online) |
| `GMM_MAX_K` / `GMM_COV_TYPE` | 10 / "full" | `atoms_config.py:89-90`; `:294,327` | config | BIC sweep cap / GMM cov |
| `RANDOM_SEED` | 17 | `atoms_config.py:91`; `:328` | config | live GMM fit seed (≠ select 42, finding 8.5) |
| `FC_RELEVANCE_FILTER` | 0.9 (local) / 0.25 (HPC chunk) | `atoms_config.py:24`; `compute_live_pert_chunk.py:141` | config / hardcoded | ATOMs mass filter (HPC discrepancy, finding 4.3) |
| `RECOMPUTE_TEST_ATOMS` | False | `atoms_config.py:38`; gate `:421` | config | compute (local) vs load (HPC) profiles |
| `RECOMPUTE_MDX_BASELINE` | False | `atoms_config.py:40`; gate `:196` | config | refit/load MDX-v1 |
| `_N_FORWARD_CAMS` / `_CAM_PX` | 3 / 384 | `sensor_agent_live_perturbation.py:67-68` | code | save-crop / model-input mismatch (Topic 6) |
| HPC chunk `--mode-analysis` default | 1 | `compute_live_pert_chunk.py:45` | CLI default | mode where p_relevance matters (finding 4.3) |
| WoR MDX config path | absolute Windows literal | `run_online_analysis.py:202` | hardcoded | non-portable (finding 9.7) |
| `EXPERIMENT_VARIANT` | "alternative" | `atoms_config.py:63` | config | re-roots inputs to `*_alt` (live data is under non-alt; finding 9.3) |

---

## 5. Known limitations & open issues

- **Online experiment is qualitative-only — no ROC/AUC, no detector ranking** (finding 9.6) — `run_online_analysis.py` produces distance-over-time figures, never AUC; the saved 99th-pct Mahalanobis threshold (`:278`) is computed but never applied to flag frames. `summarize_results.py` Section 7 inventories the live figures with no number (Topic 8 §3.16). The thesis can use these only as illustration, not as quantitative evidence.
- **Scores plotted raw — no smoothing/windowing of any kind** (finding 9.5) — verified across `run_online_analysis.py` and `plot_distance_over_time`; per-frame distances are plotted with `"o-"` markers directly. The traces are inherently jittery; a moving-average overlay would make the injection change-point legible. State this in the thesis if the online plots are shown.
- **Module docstring is a verbatim copy of `run_analysis.py`** (finding 9.1, extends 1.11) — describes an offline ROC/AUC "Steps 1–12" pipeline this script does not run; even the title says `run_analysis.py`.
- **Baseline-side code is duplicated, not imported; lacks offline fixes** (finding 9.2) — STEPS 1–5 copy `run_analysis.py`'s baseline half but omit MDX-v2, baseline visualisation, the `--gmm-k` sweep, and the `(run_id, frame_idx)` alignment-key guard. Divergence over time is inevitable.
- **Live data lives under the original split but the config points at `*_alt`** (finding 9.3, live variant of 5.7) — with `EXPERIMENT_VARIANT="alternative"` the script globs the empty `test_data_alt/live_pert_frames/` and raises; the actual recordings/profiles/figures are all under non-alt `test_data/` and `results/live_perturbation/`. The online experiment is only runnable after flipping the flag to `"original"`.
- **Results path is internally inconsistent** (finding 9.10) — `live_pert_dir = RESULTS_DIR/live_perturbation/<pert>` is created (`:660-661`) but figures are saved to `OUT_DIR` instead; `live_pert_dir` is dead.
- **Single global `PERTURBATION` keys folders → stale cross-keyed profiles** (finding 9.4) — `attention/live_pert/brightness_scale/` holds profiles named after PGD variants the brightness glob never matches; messy dirs (`live_pert_frames_backup/`, `alternative live_perts/` *with a space*) accumulate. No per-file perturbation field.
- **HPC ↔ local profile-naming mismatch** (finding 9.11) — `gather_live_pert.py` writes `live_pert_profiles.npy` (no variant/mode suffix, all variants concatenated) but the local loader expects `live_pert_profiles_{variant}_{mode}.npy` per variant; the gathered file is not picked up without manual per-variant rename/split. Both schemes coexist on disk in `attention/live_pert/pgd/`.
- **Online detector roster differs from offline; k=25 fixed, no val selection** (finding 9.8) — online scores only single + Mahalanobis-GMM-nearest + PEOC + MDX-v1 with a **hardcoded k=25** k-NN; it omits the GMM variants (Euclidean/JSD/k-NN-GMM), Wasserstein, the k-sweep, and MDX-v2. The k is not selected on val (offline uses val AUC, Topic 7 §2.7).
- **`_CARLA_HZ=20` hardcoded inline** (finding 9.9) — the fallback injection-frame estimate hardcodes 20 Hz with no config source; harmless because the `argmax(is_perturbed)` path is always taken on recorded files, but a silent magic constant.
- **Hardcoded absolute Windows path for WoR MDX config** (finding 9.7) — `run_online_analysis.py:202` is a non-portable literal (`C:/Users/paulk/...`), unlike STEP 1's relative paths.
- **HPC chunk p_relevance=0.25 ≠ config 0.9; mode default 1** (finding 4.3, Topic 4) — `compute_live_pert_chunk.py:141` hardcodes the 0.25 mass filter (with `default_cmd=2`) and defaults to mode 1, so HPC-computed live-pert mode-1 profiles are not reproducible with local defaults.
- **Ensemble-vs-single-member asymmetry** (findings 2.5/6.8) — the live recordings are driven and PGD-attacked against the **3-model ensemble**, but `run_online_analysis.py` (and the HPC chunk worker) explain/profile only the **single** `sorted(...)[0]` member; the analyzed policy differs from the one that drove the recordings.

---

## 6. Cross-references

- **01_architecture_overview.md** — `atoms_config.py` as single source of truth for `PERTURBATION`/`INTENSITY`/`CAM_INDEX`, `INJECTION_TIME`, `EPSILON`/`PGD_N_STEPS`/`PGD_TARGET`, `MODE_ANALYSIS`, `NUM_GMM_CLUSTERS`/`GMM_MAX_K`, `MAHAL_RIDGE`, `RANDOM_SEED`, `TEST_SAMPLE_INTERVAL`/`MAX_LIVE_PERT_SIZE`, `LIVE_PERTURBATION_RECORDING_MODE`, `EXPERIMENT_VARIANT` path re-rooting; findings 1.7 (DEFAULT_CMD), 1.11 (copied docstring → 9.1).
- **02_agents.md** — single-member `sorted(...)[0]` load vs 3-model live ensemble (finding 2.5); the `get_speed_logits` (TFV6) / `get_action_logits` (WoR) extractors used for PEOC; `get_backbone_features` for MDX-v1.
- **04_atoms.md** — `ATOMsCarla.process_frame` (with `spd`) is the per-frame call in the live loop and the HPC chunk worker; mode 1 vs 2; the HPC `p_relevance=0.25` discrepancy (finding 4.3).
- **05_dataset_creation.md** — `TestDataCollector` save naming, `live_pert_frames/` layout, the test/val split context; the live data is a separate recording, not a migrated set.
- **06_perturbations.md** — the live perturbation path (§3.6 there): `sensor_agent_live_perturbation.py`, non-PGD uint8 vs PGD float-tensor application, the live PGD ensemble attack, `prep_live_pert*.py` concatenation; live ε=8 vs offline ε=14 (finding 6.4); ensemble-vs-single PGD (finding 6.8); the registered-but-offline-unused perturbations *are* exercised here.
- **07_distances_and_detectors.md** — every detector scored online routes through `DistanceComputer` (stateless); the online roster is a subset of the detector table; k-NN k=25 fixed vs val-AUC selection; `ActionEntropyDetector` for PEOC; MDX-v1; finding 7.11 (clamping) applies to the online scores too.
- **08_offline_analysis.md** — the offline counterpart whose baseline half (STEPS 1–6) this script duplicates; the offline ROC/AUC roster that the online experiment deliberately does not compute; the val-set/`__val_auc_gmm_avg__` machinery that has no online analogue; the `summarize_results.py` Section 7 live-figure inventory (qualitative, no AUC).
- **10_hpc_pipeline.md** — the live-pert array-job mechanics: `prep_live_pert*.py` sizing via `live_pert_meta.txt`, `array_live_pert_task.sh`, `compute_live_pert_chunk.py`, `gather_live_pert.py`, and the copy-into-repo step; the HPC↔local naming mismatch (finding 9.11) to be reconciled there.
- **11_validation_and_testing.md** — the LRP/ATOMs correctness suites that validate the profiles this online script scores.
- **12_visualization.md** — `plot_distance_over_time` semantics (the only live-experiment figure type), `FIGSIZE_DISTANCE_OVER_TIME`, `DISTANCE_TYPE_COLORS`/`YLABELS`, and `visualize_relevance`/`visualize_comparative_relevance` per-frame relevance PNGs from the local compute branch.
- **99_bugs_and_findings.md** — Topic 9 findings 9.1–9.11; cross-references 1.7, 1.11, 2.5, 4.3, 5.7, 6.4, 6.8, 7.11, 8.5.
