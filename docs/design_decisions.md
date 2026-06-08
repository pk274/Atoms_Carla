# Design Decisions

This document records key architectural and methodological choices made during
implementation of the ATOMs + LRP OOD-detection pipeline.  Update this file
whenever a significant design choice is revisited.

---

## Val/Test split for hyperparameter selection (planned, 2026-06)

**Problem:** `run_analysis.py` currently selects k for k-NN (and GMM k-NN) by maximising AUC
on the test set (lines ~1336, ~1351). This is a form of data leakage — the test labels are
used to choose hyperparameters, then the same labels are used to report performance.

**Decision:** Introduce a **validation set** drawn from the same held-out town (Town05) using
different routes than the test set. Hyperparameter selection (k in k-NN, k in GMM-kNN) is
done exclusively on the val set. Final AUC/ROC numbers reported in the thesis use only the
test set. GMM cluster count K stays selected by BIC on the baseline, which is already clean.

**Data split strategy:**
- Both sets use Town05 routes (the only held-out town in the LEAD dataset).
- The 13 routes currently in `test_data/frames/` are the test set and are NOT re-used for val.
- 13 additional Town05 routes (not previously extracted) become the val set.
- Both sets use the same 5-way 20% perturbation mix (clean / gaussian_noise / brightness_scale
  / camera_loss / pgd) and the same HPC pipeline.

**Concrete creation steps (see `CLAUDE.md` — Raw Data Creation Pipeline for tool invocations):**
1. Extract 13 more Town05 routes via `unzip_routes.ps1 -RoutesPerTown 26` (top-26 by size; the
   top-13 are already extracted and used for test; routes 14–26 are new).
2. Run `migrate_lead_to_baseline.py --mode valset` with `--exclude_routes` pointing at the
   existing test frames, writing to `val_data/frames/`.  *(The `valset` mode and `--exclude_routes`
   flag need to be added to `migrate_lead_to_baseline.py`.)*
3. Apply perturbations to val frames (locally via `PerturbationApplier` or via HPC `prep_test.py`),
   producing `val_labeled.npz`.
4. Run HPC ATOMs profile computation on the val set (`submit_val.sh`), producing
   `val_profiles_{MODE}.npy` and `val_speed_logits_{MODE}.npy`.
5. In `run_analysis.py`: load val profiles after Step 9; compute k-NN scores on val; pick
   `best_k` by val AUC; use that k to index into already-computed test scores for reporting.

**Code changes (all implemented 2026-06-08):**
- `migrate_lead_to_baseline.py` — added `valset` mode; `exclude_routes` param to `build_sampling_plan`; `migrate_valset()` auto-excludes test routes by reading `test_data/frames/` npz stems.
- `atoms_config.py` — added `VAL_DATA_DIR = _DATA_ROOT / "val_data"`.
- `dataset.py` — `PerturbationApplier.__init__` accepts optional `data_dir`; `LabeledTestLoader.load_val()` added.
- `hpc/gather_test_task.sh` — `SPEED_LOGITS_OUT` and `LABELED_FILE` now overridable via env vars (backward-compatible).
- `hpc/submit_val.sh` — new script; reuses existing task scripts with val-specific paths.
- `hpc/collect_results.sh` — added `val` pipeline case → `data/<AG>/val_data/attention/`.
- `run_analysis.py` — Step 9.5 loads val profiles; k-NN/GMM-kNN k selected on val AUC; sensitivity plot shows val AUC; falls back to test AUC with a warning when val is absent.

---

## Agent support

| Agent key | Model | Status |
|-----------|-------|--------|
| `WOR`     | World on Rails (CameraModel) | ✅ implemented |
| `LBC`     | Learning by Cheating (RGBPointModel) | ✅ implemented (no weights available as of 2025-05) |
| `TFV6`    | TransFuser v6 (`visiononly_resnet34`, LTF mode) | ✅ implemented |

The `AGENT` key in `atoms_config.py` selects which data subfolder and which
LRP wrapper class is used.  Downstream analysis code (detectors, visualization,
`run_analysis.py`) is agent-agnostic.

---

## ATOMs class: single shared implementation

`ATOMs_Analysis/saliency/atoms_carla.py` (`ATOMsCarla`) is reused for all
agents.  It was made wide-only compatible when LBC was added
(`WIDE_ONLY_PROFILE = True` in `atoms_config.py`).  For TFV6 the same flag
applies: the full 6-camera concatenated image is treated as the "wide" image,
and `narr_rgb` / `narr_seg` are passed as `None`.

---

## TFV6: FC-layer equivalent (Option B — PlanningDecoder speed_query 256-dim)

**Choice (updated 2026-05-27):** F_c = the **256-dim speed-query token** from
the `PlanningDecoder`'s `TransformerDecoder`, extracted after the final norm
layer.  This is the representation from which `target_speed_decoder` directly
predicts the target speed distribution — the closest equivalent of "the layer
just before the output" described in the ATOMs paper.

**Rationale:**
- Option A (512-dim backbone) described the image encoder's representation,
  not the driving decision layer.  ATOMs profiles under Option A captured
  "what the visual encoder attends to," not "what the decision-maker attends to."
- The TransformerDecoder's cross-attention attends over BEV + status tokens,
  making the speed_query a truly decision-conditioned representation.
- All AttnLRP rules needed for Option B are now implemented:
  `LRPSoftmax`, `LRPMatMul`, `MultiheadAttentionExplicit`,
  `TransformerDecoderLayerExplicit`.

**LRP1 seed:** One-hot at the argmax of `target_speed_decoder(speed_query)`.
**Node space:** 256 dimensions (speed_query token, F_c per ATOMs paper).

**Alternative worth trying — Option A (512-dim backbone output):**
The globally averaged backbone output (`avgpool_final` → flatten → `[B, 512]`)
is a simpler F_c candidate.  Switching back to it requires:
1. Change `TFv6FullModelForLRP.forward` to return `avgpool_final(image_features).flatten(1)` instead of `speed_query`.
2. Remove `target_speed_decoder` from the LRP1 path — seed directly from the
   backbone output (e.g. positive activations, or backprop from the planning
   decoder logits all the way through).
3. Update `node_dim = 512`.
The `SelfAttentionExplicit`, `LRPSoftmax`, `LRPMatMul` GPT-block improvements
carry over unchanged regardless of which option is used.
Option A profiles describe "what the image encoder attends to" rather than
"what the decision-maker attends to", which may be useful for comparison.

---

## TFV6: LTF mode — no real LiDAR sensor

`config.LTF = True` in the `visiononly_resnet34` checkpoint.  The backbone
generates a 2-channel deterministic x/y coordinate grid instead of real LiDAR.
This means:
- No LiDAR sensor needs to be attached during data collection.
- The LiDAR grid is created fresh inside `_forward()` without `requires_grad`,
  so autograd attribution flows only through the RGB path.

---

## TFV6 LRP: AttnLRP for attention blocks (updated 2026-05-27)

**Problem:** Both the GPT backbone fusion blocks (using
`F.scaled_dot_product_attention`) and the PlanningDecoder TransformerDecoder
(using `nn.MultiheadAttention` internally via `nn.TransformerDecoderLayer`)
use fused CUDA kernels opaque to zennit.

**Solution:**

### GPT backbone blocks
`SelfAttention` → `SelfAttentionExplicit`:
- K/Q/V/proj wrapped as `AttentionLinear` (subclass of `nn.Linear`)
- Q·K^T and A·V computed via `LRPMatMul.apply` (AttnLRP Prop 3.3)
- Softmax via `LRPSoftmax.apply` (AttnLRP Prop 3.1)

### PlanningDecoder TransformerDecoder
`nn.TransformerDecoderLayer` → `TransformerDecoderLayerExplicit`:
- Self-attn and cross-attn use `MultiheadAttentionExplicit`
  - Extracts Q/K/V from `in_proj_weight` as separate `AttentionLinear` layers
  - Uses `LRPSoftmax` and `LRPMatMul` for AttnLRP-compliant backward

### Composite rule split (Bug 3 fix)
- `AttentionLinear` → `Epsilon(ε=1e-6)`  (K/Q/V/proj in all attention blocks)
- `Convolution` → `AlphaBeta(α=1, β=0)`
- `nn.Linear` (FFN) → `AlphaBeta(α=1, β=0)`
- `BatchNorm`, `LayerNorm`, activations → `Pass`

---

## TFV6 LRP: zennit composite — no canonizer

timm's ResNet34 `BasicBlock` type differs from torchvision's, so
`zennit.torchvision.ResNetCanonizer` cannot be used.

**Solution:** Use a plain `SpecialFirstLayerMapComposite` without canonizers:
- First `Convolution` → `WSquare`
- `AttentionLinear` → `Epsilon(ε=1e-6)` (K/Q/V/proj; matched before AnyLinear)
- `Convolution` → `AlphaBeta(α=1, β=0)`
- `nn.Linear` (FFN/classification) → `AlphaBeta(α=1, β=0)`  — no `zero_params`
- `BatchNorm`, `LayerNorm`, activations → `Pass`

Without canonization, BatchNorm is not merged into the preceding Conv.
`Pass` on BatchNorm means its scaling factor is ignored in LRP, which can
introduce small relevance-conservation errors.  Acceptable for the thesis;
proper canonization would eliminate this.

Residual additions in ResNet and GPT are handled automatically by autograd.

---

## Bug fix: BaselineDataCollector path

`BaselineDataCollector.__init__` had:
```python
self._data_dir = Path(getattr(conf.BASELINE_DATA_DIR, "baseline_data_dir", "baseline_data"))
```
`getattr` on a `Path` object for attribute `"baseline_data_dir"` always returns the default
`"baseline_data"`, so data was saved to `./baseline_data/frames/` (relative CWD) instead of
`conf.BASELINE_DATA_DIR / "frames"`.  Fixed to `Path(conf.BASELINE_DATA_DIR)`.

---

## TFV6 data collection: semantic cameras

The existing `SensorAgent` calls `av_sensor_setup(sensor_agent=True, ...)`,
which skips semantic segmentation cameras (they are only added in training mode
with `sensor_agent=False`).

**Solution:** `DataCollectionSensorAgent` (new subclass in
`sensor_agent_data_collection.py`) overrides `sensors()` to append one
`sensor.camera.semantic_segmentation` sensor per RGB camera, using the same
pose and intrinsics from `config.camera_calibration`.  Semantic data is
captured in the overridden `tick()` and concatenated horizontally (matching
how RGB cameras are concatenated), then saved to disk via a simple frame
collector.

The red channel (`[:, :, 2]` in BGRA output) of each semantic camera contains
the CARLA semantic class ID (0–22 in CARLA 0.9.16).

---

## Comparative relevance maps in mode 1: LRP1-reweighted node maps

`PLOT_COMPARATIVE_REL=True` renders `saliency_data_wide_drive - saliency_data_wide_brake`
to show what the model attends to differently when braking vs driving.

**Bug:** `_lrp2_pixels` (mode 1, node-level) was computing forced maps via `beg="fc",
end="input"` with `forced_brake=True` / `forced_drive=True`.  In the `fc→input` path,
`forced_brake`/`forced_drive` only update `is_brake` as a side effect — the actual backward
seed is always the one-hot at `node_id`.  Both forced calls returned identical maps →
`drive - brake = 0` (uniform).

**Fix:** `_compute_node_level` now caches the per-node LRP2 pixel maps during the main
node loop, then calls `_set_comparative_maps_node_level` after the loop:

1. Run LRP1 (`output→fc`) with `forced_brake` → per-node weight vector `r_brake`
2. Run LRP1 (`output→fc`) with `forced_drive` → per-node weight vector `r_drive`
3. Re-weight the cached LRP2 maps:
   `saliency_wide_brake = Σ_k |r_brake[k]| * lrp2_map[k]`
   `saliency_wide_drive = Σ_k |r_drive[k]| * lrp2_map[k]`

Only two extra LRP1 passes (FC-only, no ResNet backward — cheap).  The LRP2 maps are
reused from the main loop.  The comparative map is non-trivial when the brake and drive
LRP1 weight distributions differ across nodes, which they do for TFV6.  For WoR, the GAP
collapse makes all LRP2 maps identical, so the comparative map remains flat regardless.

---

## WoR and LBC: no changes

`lrp_analysis.py`, `lrp_lbc.py`, and `atoms_carla.py` are not modified.
The `atoms_config.py` change is strictly additive (`TFV6` branch added).

---

## WoR: per-FC-node pixel maps are identical (GAP collapse — architectural limit)

`forward_relevance(beg='fc', end='input', node_id=k)` produces the **same pixel
map for every k** in WoR. This is not a code bug.

**Mechanism:**
1. ResNet backbone outputs `[B, 512, H', W']` (spatial feature map).
2. `AdaptiveAvgPool2d((1,1))` collapses it to `[B, 512, 1, 1]` — all spatial
   information averaged away.
3. AvgPool has **no LRP rule** registered (intentionally excluded from the
   composite; see `_create_composite` comment). Standard autograd backward
   uniformly redistributes each channel's scalar relevance back to all H'×W'
   positions.
4. The ResNet z+ backward then uses the same fixed activation patterns
   (`R_i = a_i^+ * w^+/z^+`) regardless of which FC node was seeded.

Result: all 256 FC nodes produce cosine ≈ 1.0 pixel maps — determined by
backbone activations, not by the node identity.

**Implication:** The `fc→input` attribution path is uninformative for WoR.
Only `output→input` (full-path) and `output→fc` (node relevance vector) are
meaningful.  This is the primary motivation for adopting TFV6, whose
`speed_query` token is produced by attention (no GAP) so per-node LRP gives
genuinely distinct spatial maps.

The `W07_fc_node_cosine_matrix` diagnostic test now reports WARN (not FAIL)
when all pairs are cosine ≈ 1.0, with a note explaining the mechanism.

---

## TFV6 baseline data: LEAD dataset migration

Instead of collecting baseline frames live in CARLA (0.5 fps on CPU due to
the model being trained on 4× L40S GPUs), the official LEAD dataset is used:

    git clone https://huggingface.co/datasets/ln2697/lead_carla data/carla_leaderboard2/zip

The dataset stores per-route data as:
- `rgb/{frame:04d}.jpg`      — all 6 cameras concatenated horizontally (expected 2304×384)
- `semantics/{frame:04d}.png` — channel 0 = CARLA semantic class IDs (same layout)
- `metas/{frame:04d}.pkl`    — pickle dict with at least `speed`, `command`, `brake`

`migrate_lead_to_baseline.py` (project root) converts these into the standard
`conf.BASELINE_DATA_DIR/frames/run_<town>_<route>.npz` format consumed by
`BaselineDataLoader`.  It groups routes by CARLA town and samples
`~n_frames / n_towns` frames per town; Town05 is reserved for the test set.

**Three TODOs remain until a sample file is inspected:**
- `TODO_SHAPE`: confirm image dimensions are (384, 2304, 3)
- `TODO_CMD`:   verify meta dict command key name and integer encoding
- `TODO_TOWN`:  verify meta dict town key name

`noScenarios` routes are used for the clean driving baseline; accident/obstacle
scenarios are reserved for the test set perturbation mix.

---

## TFV6 minimal data dict: command one-hot fix

`_make_minimal_data` (fallback used when no full data dict is available) was
building `command = torch.zeros(1, 6)` — an all-zero vector that is never a
valid one-hot.  `PlanningContextEncoder` passes this through
`command_encoder` (Linear 6→256); the resulting command token was entirely
bias-driven with no directional information, distorting the
TransformerDecoder cross-attention and causing ~80% of baseline frames to
predict speed bin 0 (stop).

**Fix:**
- `_make_minimal_data(spd, device, cmd=3)` now accepts a `cmd` integer
  (0–5, leaderboard one-hot index) and sets `cmd_vec[0, cmd] = 1.0`.
  Default is 3 (FOLLOW_LANE).
- `LRPTFv6Model.update_context` gains `cmd: Optional[int] = None` and
  passes it to `_make_minimal_data`.
- `ATOMsCarla.process_frame` detects TFV6 via `hasattr(lrp, '_data_cache')`
  and passes `cmd` to `update_context` when no full data dict is supplied.

`target_point` and `acceleration` remain zero (not stored in npz files).
These are secondary conditioning inputs; their effect on LRP is smaller
than the command, which governs the primary cross-attention token.

---

## Data dict backport: why `BaselineComputer` deliberately omits `data=`

An HPC agent suggested passing the full frame data dict to `process_frame`
(instead of just `cmd`/`spd` scalars) to improve TFV6 LRP conditioning.
This was assessed and rejected for the following reasons:

**The `.npz` files do not contain the missing fields.**
The only keys stored are `wide_rgb`, `narr_rgb`, `seg_red_wide`, `seg_red_narr`,
`cmd`, `speed`, `is_brake`, `frame_idx`. The fields that `_make_minimal_data`
zeroes out (`target_point` ×3, `acceleration`) are not in the files, so
constructing a data dict from the `.npz` would still zero those fields — no
improvement over `_make_minimal_data`.

**Passing a raw `.npz` dict would break inference.**
`planning_decoder.py` uses direct `data["key"]` indexing with no graceful
fallback. The file stores `"cmd"` (int scalar) but the model expects `"command"`
(one-hot float32 tensor `[1,6]`). Passing the raw dict causes a `KeyError`
immediately.

**No LiDAR cheating risk in LTF mode.**
TFV6 is run in LTF mode (`config.LTF = True`). In this mode
`transfuser_backbone.py` generates LiDAR as a deterministic 2-channel
positional grid — it never reads `data["rasterized_lidar"]`. Even if the
`.npz` contained recorded LiDAR it would be ignored. `LRPTFv6Model.__init__`
now asserts `backbone_eval.config.LTF` to make this invariant explicit and
prevent silent breakage if the config is changed.

**Current approach is already correct.**
`_make_minimal_data` (with the correct `cmd` one-hot fix) provides exactly the
same information that a properly constructed data dict from the `.npz` would
provide. `BaselineComputer.compute_and_save` passes `cmd=cmd` and `spd=spd`
scalars, giving `_make_minimal_data` the only frame-specific information
available. A comment at the call site documents this intent.

---

## BaselineDataLoader: narr_rgb now optional

`BaselineDataLoader.load_run()` and `load_all_runs()` used to assume `narr_rgb`
and `seg_red_narr` keys always exist in npz files.  TFV6 (wide-only) npz files
do not contain these keys (matching `DataCollectionSensorAgent` which passes
`narr_rgb=None`).

**Fix:** both methods now return `None` for missing narr keys.
`BaselineComputer.compute_and_save()` gates narr access on `has_narr` / `has_seg_narr`
booleans derived from the loaded data.  `reference_narr` is only saved when present.

---

## run_analysis.py: agent-conditional loading

`run_analysis.py` is the single entry-point for the full pipeline and must
support both WoR and TFV6.  The adaptation strategy is:

- **Step 1**: conditional on `conf.AGENT`.  WoR loads `CameraModel` + `LRPCameraModel`;
  TFV6 loads `TFv6` + `LRPTFv6Model` (backbone_eval = `net.backbone`, planning_decoder = `net.planning_decoder`).
- **`action_logits_available` flag**: set `True` for WoR, `False` for TFV6.
  Gates WoR-style MDX fit (Step 3), steer/throt/brake logit collection in Step 9,
  and `ActionEntropyDetector` scoring.
- **`speed_logits_available` flag**: set `True` for TFV6, `False` for WoR.
  Gates TFV6 MDX fit (Step 3), speed logit collection in Step 9 (saved as
  `test_speed_logits.npy`), PEOC scoring (Step 11e), and PEOC evaluation.
- **narr_rgb guards**: all `data["narr_rgb"]` accesses are conditioned on
  `data["narr_rgb"] is not None` (returns `None` from the patched loader).
- **ATT_DIR**: fixed to `conf.TEST_DATA_DIR / "attention"` (was hardcoded WoR path).

---

## WoR PEOC detector — corrected implementation (2026-06-02)

**Bug (fixed):** The old implementation concatenated all 4 speed bins into a 52-element
vector `[steer_flat(36), throt_flat(12), brake_flat(4)]` and computed entropy of
`softmax([52])`.  This is not H(π(a|s)) because the 4 speed bins represent the same
decision at different speeds — only the two bins bracketing the actual vehicle speed
are relevant, and steer/throt/brake are not 52 mutually exclusive outcomes.

**Correct implementation:** WoR's true action space is 28-dimensional:
27 joint (steer × throt) actions + 1 brake, built by `action_logits()` in `main_model.py`
as `steer_j + throt_i` (a factored joint distribution).  At the actual vehicle speed
the model linearly interpolates between the two adjacent speed bins (x0, x1).

**Fix:** `LRPCameraModel.get_action_logits(wide, narr, cmd, spd)` was added.  It calls
`model.forward()` (which runs `action_logits()` internally), selects the active command,
and lerp-interpolates to the actual speed — exactly mirroring `_build_drive_brake_selector`.
The returned [28] numpy array is H(softmax([28])) under `ActionEntropyDetector`.

Files changed: `lrp_analysis.py` (new method), `run_analysis.py`, `run_online_analysis.py`.

---

## TFV6 PEOC detector (Sedlmeier et al., 2020)

**PEOC = Policy Entropy Out-of-distribution Classifier.**  H(π) of the 8-bin
speed distribution from `target_speed_decoder` is used as the OOD score:
high entropy → the agent is uncertain → likely OOD.  This is exactly the
existing `ActionEntropyDetector(from_logits=True, cmd=None)` applied to the
speed logits — no new class is needed.

Speed logits are extracted via `LRPTFv6Model.get_speed_logits(wide_rgb, cmd, spd)`,
which runs a no-grad forward through `full_model` + `target_speed_decoder`.

---

## TFV6 MDX feature extraction fix

`lrp.backbone_model` was referenced in Steps 3 and 11 of `run_analysis.py` but
never existed as an attribute on `LRPTFv6Model`.  Fixed by adding
`LRPTFv6Model.get_backbone_features(wide_rgb)` which calls
`full_model._run_backbone()`, applies global average pooling, and clamps with
ReLU to produce a 512-dim feature vector matching the MDX paper's penultimate-layer
feature extraction.

Also fixed: `_make_minimal_data` created all tensors on CPU regardless of the
`device` argument.  All tensor constructors now pass `device=device`.

---

## HPC gather: lexicographic sort bug (found 2026-06-07)

**Problem**: `gather_baseline.py` used `sorted(partials_dir.glob("partial_*.npz"))` to
reassemble partial results from SLURM array tasks.  Python's default sort is
lexicographic on the full path string, so for 39 tasks (indices 0–38) the assembly
order was `0, 1, 10, 11, …, 19, 2, 20, …, 9` — not the intended numeric order
`0, 1, 2, …, 38`.  SLURM task K processed run file K from the sorted list, so
`partial_10.npz` contained profiles from run 10, but was placed in position 2 of the
gathered series.  This caused a large-scale frame-to-profile mismatch: 37 of 39 run
files had mismatched series entries.  The symptom was "spurious Biker attention in
Town07 frames that have zero biker pixels" — those series positions actually held Town10
profiles (where bikers are present).

**Impact**:
- OOD detection AUC results remain valid: GMM/Mahalanobis/kNN fitting is
  order-independent, and all profiles are real attention vectors.
- Representative frame images and run-level PCA coloring are broken in the scrambled
  baseline (the wrong RGB frame is shown for each cluster representative).

**Fix**: `gather_baseline.py` line 51 now uses a numeric sort key:
```python
partial_files = sorted(args.partials_dir.glob("partial_*.npz"),
                       key=lambda f: int(f.stem.split("_")[-1]))
```
The existing `baseline_1.npz` must be regenerated on the HPC with the fixed gather
script before the representative-frame visualization can be trusted.

---

## Code-review fixes — 2026-06-08

Applied after the thorough code review documented in `docs/code_review.md`. Only
the "easy / unambiguous" fixes were applied; the validation-set redesign (§2.1),
MDX binning (§2.4) and the WoR steer objective rework remain open for discussion.

### PGD attack sign correction (review §2.2)
`ATOMs_Analysis/perturbation_manager.py`. Both `pgd_attack` (WoR) and
`pgd_attack_tfv6` (TFV6) take gradient-**ascent** steps, but the per-target losses
were written as quantities to *minimise* toward the target, so the attack drove the
agent *away* from its stated objective. Each objective is now a **reward maximised
under ascent**:
- TFV6: `brake`/`max_speed` → `reward = -CE(speed_logits, target_bin)`;
  `steer_left` → `-mean(wp_x)`; `steer_right` → `+mean(wp_x)`.
- WoR: `brake` → `+brake_logits`; `max_steer` → `+|steer_logits|`.
- WoR `steer_left/right` still use the raw steer-logit sum, which is shift-invariant
  under softmax and therefore a weak proxy; flagged in-code for a later rework to the
  decoded steering value `steers·softmax(steer_logits)`.

**Consequence:** any previously generated PGD test/profile data was produced with the
inverted attack and must be regenerated (TFV6 PGD profiles are recomputed on the HPC).

Verified by replicating the corrected PGD loop on toy linear models: every target now
moves its metric the right way (e.g. TFV6 `brake` raises P(bin 0) 0→1; `steer_right`
increases mean waypoint-x while `steer_left` decreases it).

### Mahalanobis double-sqrt (review §3.1)
`ATOMs_Analysis/detection/detectors.py`, `MahalanobisDetector.score`. It applied a
second `sqrt` to a value that `DistanceComputer.compute_mahalanobis` already returns
as a distance, yielding `sqrt(distance)` and a scale inconsistent with the GMM path
(which returns the distance). Now returns the distance directly. (The main ROC/AUC
path already used `DistanceComputer` directly and was unaffected; this only corrected
the class and the threshold saved in `mahal_detector.npz`.) Verified numerically
against `compute_mahalanobis` and a hand-computed distance.

### Deferred-PGD guard (review §4.2)
For TFV6, PGD frames are stored with **clean pixels** but `label=1` (the adversarial
image is crafted on the HPC). Added a `warnings.warn` in
`PerturbationApplier.apply` (`detection/dataset.py`) and in `run_analysis.py` Step 9
so that recomputing ATOMs locally for these frames no longer silently produces
non-adversarial "PGD" profiles.

### Profile↔label alignment guard (review §3.3)
`run_analysis.py` Step 9 now persists a companion `test_profiles_{mode}.keys.npy`
holding each profile row's `(run_id, frame_idx)`, and verifies it against
`test_labeled.npz` on load (replacing the length-only check). A reordered or
different-but-same-length test set now raises instead of silently pairing profiles
with the wrong labels. Falls back to a warning when no key file is present
(e.g. HPC-produced data predating this guard).

### Documentation sync
`CLAUDE.md`: corrected the attention-profile dimensionality (29 for WOR / 10 for
TFV6, not "23-dim"); reworded the hierarchical-attention definition to the
nonzero-pixel mean (R̄); marked the Step-8.5 trajectory analysis as disabled; and
replaced the stale "zero command vector" note with the current `_make_minimal_data`
behaviour. `lrp_transfuser.py` docstring: corrected the AttentionLinear ε from 1e-6
to 1e-2 to match the composite.

*Not applied (need discussion): validation split for k/K selection (§2.1), WoR
steer-objective rework, and dead-code removal.*

---

## MDX-v2: F_c features + waypoint steer proxy + quantile binning (2026-06-08)

**Motivation.** The original TFV6 MDX detector (MDX-v1) had two degeneracies, flagged
in review §2.4:

1. **Degenerate steer proxy.** `run_analysis.py` hardcoded `steer=0.0` for all TFV6
   baseline frames. With equal-width binning, `np.linspace(0,0,4)=[0,0,0,0]`, so all
   frames land in steer-bin 0. Only `throttle × brake = 2×2 = 4` of the intended 12
   action classes were ever populated, making the class-conditional Gaussian structure
   largely vacuous.

2. **Suboptimal feature layer.** MDX-v1 used the 512-d globally-pooled ResNet backbone
   output (`get_backbone_features`), which precedes the TransformerDecoder and lacks
   cross-modal fusion and planning-level representations. The ATOMs paper defines F_c
   as "the final world model on which the agent chooses its action" — for TFV6 that is
   the 256-d `speed_query` token output by `PlanningDecoder.transformer_decoder` just
   before `target_speed_decoder`.

**MDX-v2 is additive — MDX-v1 is left untouched** (same code path, same saved parameters).

### Feature: 256-d speed_query (F_c)

MDX-v2 builds its class-conditional Gaussians over 256-d `speed_query` vectors
extracted by `LRPTFv6Model.get_fc_features`. This is the same node used as the LRP
attribution seed and is the natural TFV6 equivalent of F_c in the ATOMs paper.
PCA (50 components) is applied before fitting, matching MDX-v1's compression approach.

`TFv6FullModelForLRP.forward` is extended with `_return_wps: bool = False`. When
`True`, the method also returns the predicted future waypoints so that baseline fitting
can retrieve both the speed_query and the planned trajectory in a single forward pass.
All existing callers pass no argument and are unaffected.

### Steer proxy: mean lateral waypoint offset

Instead of the constant `0.0`, MDX-v2 uses the mean lateral (x) offset of the model's
predicted future waypoints (`pred.pred_future_waypoints[..., 0].mean()`). This is
non-degenerate even on straight roads and captures the geometry of the planned path.
Waypoints are decoded by `wp_decoder` inside `TFv6FullModelForLRP` when `_return_wps=True`.

The combined baseline-fit helper `get_planning_action_and_features(wide_rgb, cmd, spd)`
returns `(feature[256], steer, throttle, brake)` in one forward pass; the test-scoring
helper `get_fc_features(wide_rgb, cmd, spd)` returns only the 256-d feature.

### Binning: quantile edges

MDX-v1 used equal-width bin edges, which collapse when a dimension is near-constant.
MDX-v2 uses `bin_strategy="quantile"` in `MDXDetector`: edges are placed at the
empirical quantiles of each action dimension over the baseline set so every bin has
roughly equal population. `_build_bin_edges` handles the constant-dimension case via
`np.unique` collapse with a ±1e-6 fallback interval so binning never fails.

### `bin_strategy` parameter in `MDXDetector`

`MDXDetector.__init__` accepts `bin_strategy: str = "equal-width"` (default). The
default preserves exact backward compatibility with MDX-v1. Passing `"quantile"`
activates the new scheme. `discretise_action` is unchanged — `np.digitize + np.clip`
work with any edge layout.

### Configuration

`atoms_config.py` exposes `RECOMPUTE_MDX_V2_BASELINE` (default `True`; set `False`
after first run). Fit result saved to `baseline_data/mdx_v2_parameters/` alongside
the existing `mdx_parameters/`. Controlled by `RECOMPUTE_MDX_V2_BASELINE` flag only;
all other `RECOMPUTE_*` flags are independent.

Files changed: `lrp_transfuser.py` (`_return_wps` flag, two new extraction methods),
`detectors.py` (`bin_strategy` param + quantile `_build_bin_edges`), `atoms_config.py`
(new flag), `run_analysis.py` (fit + score + evaluate blocks for TFV6 only).
