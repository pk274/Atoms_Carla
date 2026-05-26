# Design Decisions

This document records key architectural and methodological choices made during
implementation of the ATOMs + LRP OOD-detection pipeline.  Update this file
whenever a significant design choice is revisited.

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

## TFV6: FC-layer equivalent (Option A — backbone 512-dim)

**Choice:** The FC-layer node space for TFV6 is the **512-dim globally averaged
image backbone output** produced after 4 rounds of ResNet stage + GPT fusion.

**Rationale:**
- The `visiononly_resnet34` planning decoder (`PlanningDecoder`) contains a
  6-layer `nn.TransformerDecoderLayer` that also uses fused flash-attention
  internally.  Propagating LRP through it would require implementing AttnLRP
  for those layers too, roughly tripling implementation complexity.
- The 512-dim pooled backbone is the direct equivalent of LBC's 512-dim node
  space and is analogous to WoR's 256-dim FC node space.
- The backbone output already captures 4 rounds of cross-modal image-LiDAR
  fusion, making it a meaningful high-level visual representation.

**Future work:** Option B (256-dim target-speed query after TransformerDecoder)
would tie attribution more tightly to the driving decision.  It requires
wrapping `nn.TransformerDecoderLayer` for AttnLRP.

---

## TFV6: LTF mode — no real LiDAR sensor

`config.LTF = True` in the `visiononly_resnet34` checkpoint.  The backbone
generates a 2-channel deterministic x/y coordinate grid instead of real LiDAR.
This means:
- No LiDAR sensor needs to be attached during data collection.
- The LiDAR grid is created fresh inside `_forward()` without `requires_grad`,
  so autograd attribution flows only through the RGB path.

---

## TFV6 LRP: SelfAttention replacement

The GPT fusion blocks in `TransfuserBackbone` use
`torch.nn.functional.scaled_dot_product_attention` — a fused CUDA kernel that
appears as a single opaque node in the autograd graph and cannot be intercepted
by zennit.

**Solution:** At LRP-model construction time, every `SelfAttention` module in
the deep-copied GPT blocks is replaced with `SelfAttentionExplicit`, which
computes the same result using:
```
scale   = (head_dim)**-0.5
scores  = Q @ K^T * scale     # explicit matmul
weights = softmax(scores)     # explicit softmax
out     = weights @ V          # explicit matmul
```
The constituent `nn.Linear` layers (key, query, value, proj) are shared with
the copied transformer, so zennit's AlphaBeta rule applies to them normally.
Gradient flow through `softmax` and `matmul` uses standard autograd
(not the full AttnLRP DTD rule), which is a conservative but common
simplification.

---

## TFV6 LRP: zennit composite — no canonizer

timm's ResNet34 `BasicBlock` type differs from torchvision's, so
`zennit.torchvision.ResNetCanonizer` cannot be used.

**Solution:** Use a plain `LayerMapComposite` without canonizers:
- `Convolution` → `AlphaBeta(α=1, β=0)`
- `Linear`      → `AlphaBeta(α=1, β=0)`
- `BatchNorm`, `LayerNorm`, activations → `Pass`
- First `Convolution` (via `SpecialFirstLayerMapComposite`) → `WSquare`

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

## WoR and LBC: no changes

`lrp_analysis.py`, `lrp_lbc.py`, and `atoms_carla.py` are not modified.
The `atoms_config.py` change is strictly additive (`TFV6` branch added).
