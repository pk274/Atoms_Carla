# 02 — Driving Agents: Architectures and the F_c Layer

> Documentation sweep topic 2. All claims verified against code state of 2026-06-12.
> File references are relative to the repo root `PCLA/` unless stated otherwise.
> Line numbers refer to the current working tree.

---

## 1. Purpose & scope

This document specifies (a) the network architecture of the two driving agents whose
attention is analyzed in the thesis — **TransFuser v6 (TFV6, primary)** and
**World on Rails (WoR, secondary)** — including exact tensor shapes, preprocessing,
output heads and the inference-time control derivation, and (b) the choice of the
**F_c layer** for each agent: the representation at which LRP relevance is seeded,
corresponding to the ATOMs paper's notion of "the final world model on which the agent
chooses its action". LRP rule internals are doc 03; the conversion of relevance maps
into ATOMs profiles is doc 04. LBC is out of scope (dead end, see `00_PLAN.md`).

---

## 2. TFV6 architecture

### 2.1 Provenance, checkpoint, config resolution

TFV6 is the **LEAD** codebase (paper: `papers/LEAD tranmsfuser v6 agent.pdf`), vendored
under `pcla_agents/transfuserv6/lead/`. The network class is `TFv6`
(`pcla_agents/transfuserv6/lead/tfv6/tfv6.py:23`), composed of
`TransfuserBackbone` (`tfv6.py:35`) plus optional perception decoders and the
`PlanningDecoder` (`tfv6.py:85-88`).

**Checkpoint used throughout the thesis:**
`pcla_agents/transfuserv6_pretrained/visiononly_resnet34/` containing `config.json` and
three ensemble members `model_0030_{0,1,2}.pth` (wandb id
`744_radarless_LTF_010_postrain32_0_251102_123841`, trained 31 epochs on the 6-camera
`carla_leaderboad2_v10` dataset). It is registered in `agents.json` under the `tfv6`
variants `visiononly`, `datacollect` (data-collection subclass) and `livepert`
(live-perturbation subclass).

- **Live agent** (`lead/inference/open_loop_inference.py:54-81`): loads **all three**
  checkpoints and drives with an **ensemble** — speed logits, waypoints and route are
  averaged across the nets (`open_loop_inference.py:86-140`).
- **Analysis pipeline** (`run_analysis.py:141-152`, `hpc/compute_baseline_chunk.py:89-101`,
  `hpc/compute_test_chunk.py`, `run_online_analysis.py`): loads **only the first**
  checkpoint, `sorted(glob("model*.pth"))[0]` = `model_0030_0.pth`. The analyzed model is
  therefore a single ensemble member, not the deployed ensemble (see §7 / bug log 2.5).

**Config mechanics.** `TrainingConfig(json.load(config.json))`
(`lead/training/config_training.py:24-36`) applies the stored JSON via `setattr`; keys
that correspond to `@property` attributes raise and are **silently skipped**
(`lead/common/config_base.py:393-398, 422-427`), so all derived values are recomputed
from class logic. Crucially, `target_dataset` is inferred from the `carla_root` string
(`config_training.py:38-52`): `"carla_leaderboad2_v10"` →
`CARLA_LEADERBOARD2_6CAMERAS` → `num_cameras = 6` (`config_base.py:108-115`).
Key resolved flags for `visiononly_resnet34`:

| Flag | Value | Source |
|---|---|---|
| `LTF` (latent TransFuser, no real LiDAR) | `True` | config.json (default `False`, `config_training.py:738`) |
| `use_planning_decoder` | `True` | config.json (default `False`, `config_training.py:631`) |
| `use_radar_detection` / `radar_detection` | `False` | config.json |
| `use_semantic`, `use_depth`, `use_bev_semantic`, `detect_boxes` | all `False` | config.json (vision-only posttrain) |
| `image_architecture` / `lidar_architecture` | `resnet34` / `resnet34` | `config_training.py:734,736` |
| `carla_leaderboard_mode` | `True` (property, `config_training.py:836-845`) | drives most other properties |

Because the perception decoders are disabled, the effective network is
**backbone + PlanningDecoder** only. (`TFv6.forward` still computes
`backbone.top_down(bev_features)` unconditionally at `tfv6.py:124` — dead compute in
this configuration.)

### 2.2 Inputs & preprocessing

**Cameras (live agent).** 6 RGB cameras, each 384×384 px, FOV 60°, yaws
−57.5/0/+57.5/+122.5/180/−122.5°, mounted at z = 2.25 m
(`lead/common/config_base.py:120-170`). Images are horizontally concatenated →
`rgb [H=384, W=6·384=2304, 3]`. At inference the agent additionally:

1. simulates JPEG compression at quality 90 to avoid a train-test mismatch
   (`lead/inference/sensor_agent.py:536-541`; `config_closed_loop.py:17`),
2. optionally slices out unused cameras (`sensor_agent.py:543-554`; all 6 used here),
3. transposes to CHW and batches → `rgb [1, 3, 384, 2304]` float32 in raw [0, 255]
   (`sensor_agent.py:640`).

ImageNet normalization happens *inside* the backbone:
`fn.normalize_imagenet` (`lead/tfv6/fn.py:17-29`): `x/255`, mean
`(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. (The LRP wrapper replaces this
functional with an `nn.Module` clone, `NormalizeImageNet`,
`ATOMs_Analysis/saliency/lrp_transfuser.py:349-357`, because the in-place channel
writes in `fn.normalize_imagenet` break zennit hook pairing.)

**Offline analysis frames are narrower.** The LEAD dataset migrated by
`migrate_lead_to_baseline.py` is the **3-camera** `carla_leaderboard2` collection:
RGB shape `(384, 1152, 3)` — 3 cameras × 384 px (`migrate_lead_to_baseline.py:38,150`;
verified on disk: `data/TFV6/baseline_data/frames/*.npz` has
`wide_rgb [N, 3, 384, 1152]`). The 3-camera rig differs geometrically from the front
half of the 6-camera rig (yaw ±54.5° at x=0.1/0.35 vs ±57.5° at x=0.0/0.25;
`config_base.py:171-197` vs `:120-170`). The model tolerates the width change only
because every fusion stage adaptively pools the image features to a fixed anchor grid
(§2.4); this is a train/analysis input mismatch, flagged in §7. The docstrings in
`lrp_transfuser.py:7` and `CLAUDE.md` state `[B, 3, 384, 2304]`, which is correct only
for the live agent.

**LiDAR (LTF mode).** With `LTF=True` no LiDAR sensor is used. The backbone fabricates
a deterministic 2-channel positional grid: channel 0 = top-down y-coordinates, channel
1 = left-right x-coordinates, both `linspace(0,1)`
(`lead/tfv6/transfuser_backbone.py:134-146`). Grid size = BEV raster size:

- planning area (leaderboard mode): x ∈ [−32, 64] m, y ∈ [−40, 40] m
  (`config_training.py:81-108`), `pixels_per_meter = 4.0` (`config_training.py:78`)
- → `lidar_width_pixel = 96·4 = 384`, `lidar_height_pixel = 80·4 = 320`
  (`config_training.py:110-118`)
- → LTF grid `[B, 2, 320, 384]`. Created without `requires_grad`, so LRP attribution
  flows exclusively through the RGB path (`docs/design_decisions.md:132-140`;
  mirrored in `lrp_transfuser.py:439-448`).

**Status inputs** (consumed by the PlanningDecoder context encoder, §2.5): ego speed
(m/s, scalar), discrete command as 6-dim one-hot
(`carla_dataset_utils.command_to_one_hot`, `lead/data_loader/carla_dataset_utils.py:1304-1321`:
1-based CARLA `RoadOption` − 1; negative → 4 before decrement; out-of-range → index 3),
and three target points (previous/current/next) in ego coordinates produced by the
route planner (`sensor_agent.py:468-524`; pop distance 5.0 m, adaptive fallback 4.0 m,
`config_closed_loop.py:35`, `sensor_agent.py:558-569`).

### 2.3 Encoders — two timm ResNet34 branches

`TransfuserBackbone` (`lead/tfv6/transfuser_backbone.py:15-97`):

- **Image branch**: `timm.create_model("resnet34", pretrained=True, features_only=True)`
  (`:35`). Stage channels 64/128/256/512; `num_image_features = 512` (`:40`).
- **LiDAR branch**: `timm.create_model("resnet34", pretrained=False, in_chans=2)` for
  LTF (`:43-44`); `num_lidar_features = 512` (`:49`).
- Per-stage 1×1 channel adapters in both directions
  (`lidar_channel_to_img` / `img_channel_to_lidar`, `:50-69`).
- Stage-wise execution: the timm layer iterator is advanced one return-layer block at a
  time (`forward_layer_block`, `:191-208`); after each of the 4 stages a fusion block
  runs (`_forward`, `:149-189`).

Shape walk-through (image side, live input 384×2304; offline 1152-wide values in
parentheses):

| Stage | Stride | Shape |
|---|---|---|
| input (normalized) | 1 | `[B, 3, 384, 2304]` (`[B,3,384,1152]`) |
| stage 1 | /4 | `[B, 64, 96, 576]` (`…, 96, 288]`) |
| stage 2 | /8 | `[B, 128, 48, 288]` |
| stage 3 | /16 | `[B, 256, 24, 144]` |
| stage 4 | /32 | `[B, 512, 12, 72]` (`…, 12, 36]`) |

LiDAR side (input `[B,2,320,384]`): stages `[B,64,80,96] → [B,128,40,48] →
[B,256,20,24] → [B,512,10,12]`.

### 2.4 GPT cross-modal fusion (4 stages)

After each ResNet stage, `fuse_features` (`transfuser_backbone.py:210-243`):

1. `avgpool_img` → fixed anchor grid `(img_vert_anchors, img_horz_anchors) = (12, 72)`
   (`:36`; anchors = `final_image_height//32 = 12`,
   `num_used_cameras·384//32 = 72`, `config_training.py:720-728`). `avgpool_lidar` →
   `(10, 12)` (`:70`; `lidar_height_pixel//32`, `lidar_width_pixel//32`,
   `config_training.py:469-477`). **Note:** for offline 1152-wide inputs the stage-4
   image map is 12×36, which `AdaptiveAvgPool2d` *upsamples* to 12×72 by duplicating
   columns — this is what makes the width mismatch silently legal.
2. LiDAR pooled features are channel-adapted to the image width, both token sets are
   concatenated (864 image + 120 LiDAR = 984 tokens) and passed through a `GPT` module
   (`transfuser_backbone.py:246-345`): learnable positional embedding `[1, 984, C]`
   (`:265-272`), dropout 0.1, then `n_layer = 2` pre-norm transformer `Block`s
   (`:348-392`) with `n_head = 4`, MLP expansion `block_exp = 4`, ReLU MLP
   (`:370-375` — "changed from GELU"), attention/residual dropout 0.1, final LayerNorm.
   `SelfAttention` (`:394-456`) uses `F.scaled_dot_product_attention` (`:449`) — the
   fused kernel that the LRP wrapper must replace (doc 03). GPT init: linear
   N(0, 0.02), LayerNorm weight 1.0 (`config_training.py:753-758`).
3. Fused tokens are split back, channel-adapted, bilinearly interpolated to the stage
   resolution and **added residually** to both branches (`:232-242`).

`TransfuserBackbone.forward` returns
`(lidar_features [B,512,10,12], image_features [B,512,12,72])` (`:189`); the BEV
(lidar) map feeds planning, the image map feeds the (disabled) perception decoders and
MDX-v1 feature extraction.

### 2.5 PlanningDecoder

`lead/tfv6/planning_decoder.py:20-155`.

**PlanningContextEncoder** (`planning_decoder.py:310-489`) builds the cross-attention
memory:

- `dimension_adapter`: 1×1 conv 512→256 on the BEV map (`:369`) → `[B, 256, 10, 12]`;
  added 2-D sine positional embedding (`PositionEmbeddingSine`, `:492-524`,
  temperature 10000, normalized to 2π); flattened → **120 BEV tokens**.
- Status tokens (one each, all projected to 256):
  speed: `Linear(1→256)` on `speed / max_speed` with `max_speed = 25.0` m/s
  (`:319-324, 406-411`; `config_training.py:531-544`);
  command: `Linear(6→256)` on the one-hot (`:333-336, 422-426`;
  `discrete_command_dim = 6`, `config_training.py:509-522`);
  target point previous/current/next: a *shared* `tp_encoder Linear(2→256)` on points
  normalized by `[200.0, 50.0]` (`:338-349, 429-451`;
  `target_points_normalization_constants`, `config_training.py:582-585`).
  Acceleration and radar tokens are disabled in leaderboard mode
  (`config_training.py:546-549`; `use_radar_detection=False`).
- Learnable `status_pos_embedding [1, 5, 256]` added to the status tokens (`:367, 484`).
- Context = concat → **125 tokens × 256**.

**Query set.** A single learnable parameter `query [1, 19, 256]`
(`planning_decoder.py:30-45`, uniform init `:81-82`): 10 route queries
(`num_route_points_prediction = 10`, `config_training.py:656`) + 8 waypoint queries
(`num_way_points_prediction = 8` at 4 Hz over 2 s, `config_training.py:660-671`) +
**1 target-speed query**.

**Transformer decoder.** `nn.TransformerDecoder` with 6 ×
`nn.TransformerDecoderLayer(d_model=256, nhead=8, activation=GELU, batch_first=True)`
and a final LayerNorm (`planning_decoder.py:47-56`;
`transfuser_num_bev_cross_attention_layers = 6`, `…_heads = 8`,
`transfuser_token_dim = 256`, `config_training.py:610-615`). `dim_feedforward` is not
passed → PyTorch default **2048**; dropout default 0.1; post-norm layout. Each layer:
self-attention over the 19 queries, cross-attention onto the 125 context tokens, FFN.

**Output split** (`planning_decoder.py:121-147`):

| Queries | Head | Output |
|---|---|---|
| `[:, 0:10]` | `route_decoder = Linear(256→2)` + `cumsum` (`:60, 129-132`) | route checkpoints `[B, 10, 2]` (m, ego frame) |
| `[:, 10:18]` | `wp_decoder = Linear(256→2)` + `cumsum` (`:62, 134-139`) | future waypoints `[B, 8, 2]` |
| `[:, 18]` | `target_speed_decoder = Sequential(Linear(256→256), ReLU(inplace), Linear(256→8))` (`:65-73`) | speed logits `[B, 8]` |

The cumulative sums make each head predict per-step *offsets*; the heading decoder
(`:63-64`) exists only for NavSim training and is absent here.

### 2.6 Speed output: 8-bin two-hot representation

`target_speed_classes = [0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0]`
m/s (`config_training.py:633-645`; also serialized in the checkpoint's `config.json`
as `target_speeds`). The bins 13.8̅ and 17.7̅ are 50 and 64 km/h.

- **Training target**: `encode_two_hot` (`planning_decoder.py:265-307`) places linear
  interpolation weights on the two neighbouring bins (one-hot at bin 0 when the expert
  brakes; one-hot at the last bin when ≥ 20 m/s); loss = cross-entropy between
  predicted logits and the two-hot distribution (`:177-186`).
- **Decoding**: `decode_two_hot` (`:246-262`) = expectation
  `Σ softmax(logits)·bins` → scalar target speed in m/s (`:141-147`, computed in fp32
  under disabled autocast `:145`).
- This two-hot training scheme is the reason the LRP1 seed uses the full softmax
  distribution rather than the argmax bin (Decision D, `docs/lrp_todo.md:284-294`):
  argmax seeding is discontinuous when probability mass straddles a bin boundary.

### 2.7 From predictions to vehicle control (live agent)

`ClosedLoopInference.ensemble` (`lead/inference/closed_loop_inference.py:173-220` +
`open_loop_inference.py:86-140`):

1. Average speed logits / waypoints / route over the 3 ensemble nets.
2. Brake override: if `softmax(speed_logits)[0] > brake_threshold = 0.9`, target speed
   is forced to 0 (`open_loop_inference.py:115-119`; `config_open_loop.py:17`).
3. **Steer** comes from the *route* head via `LateralPIDController`
   (`steer_modality = "route"`, `config_closed_loop.py:21`;
   `execute_route_and_target_speed`, `closed_loop_inference.py:78-114`).
4. **Throttle/brake** come from the *target speed* head
   (`throttle_modality = brake_modality = "target_speed"`, `config_closed_loop.py:23,25`):
   brake if `target < 0.01` m/s or `speed/target > brake_ratio`; throttle via
   `get_throttle` PID.
5. A waypoint-based alternative controller exists (`execute_waypoints`, `:116-172`)
   but is not selected.
6. `ForceMovePostProcessor` (`sensor_agent.py:821-874`) overrides throttle when the
   car is stuck (speed < 0.1 m/s for `sensor_agent_stuck_threshold` ticks), guarded by
   a LiDAR safety box; the first `inital_frames_delay = 1` step is forced full brake
   (`config_base.py:320`, `sensor_agent.py:689-691`).

Relevant to the thesis: the longitudinal decision (brake/throttle) is a direct function
of the speed-bin distribution, while the lateral decision is a function of the route
queries — this asymmetry matters for the F_c choice (§3.4).

### 2.8 Data-collection subclass

`DataCollectionSensorAgent` (`lead/inference/sensor_agent_data_collection.py:66-181`)
adds one semantic-segmentation camera per RGB camera (same calibration, `:102-128`),
converts raw CARLA class IDs (0–28) to the grouped 10-class TFV6 scheme via
`SEMANTIC_SEGMENTATION_CONVERTER` (`:51-57`), and feeds `BaselineDataCollector`
(`:160-167`) producing the npz schema consumed by the analysis (wide-only; `narr_rgb =
None`). Its `destroy()` flush is commented out (`:175-181`), so only full buffers
(`conf.MAX_BASELINE_SIZE = 100` frames) are written
(`ATOMs_Analysis/detection/baseline_dataset.py:211-214`); a final partial buffer is
discarded.

---

## 3. The F_c choice for TFV6

### 3.1 What F_c means

The ATOMs paper defines F_c as the layer just before the output layer — "the final
world model on which the agent chooses its action". ATOMs computes (i) LRP1: relevance
of each F_c node for the output, and (ii) LRP2: per-node pixel maps from F_c back to
the input; the node-weighted combination yields the object-level attention profile
(doc 04). The choice of F_c therefore determines *whose* attention the profiles
describe.

### 3.2 What was chosen: the `speed_query` token (Option B)

**F_c = the 256-dim target-speed query token output by
`PlanningDecoder.transformer_decoder`** (after the decoder's final LayerNorm), i.e.
`queries[:, 18]` — the exact tensor from which `target_speed_decoder` predicts the
speed distribution.

Code anchors:

- `TFv6FullModelForLRP` recomputes the query index exactly as
  `PlanningDecoder.forward` does:
  `idx = 10 (route) + 8 (waypoints) = 18` → `_speed_query_idx`
  (`ATOMs_Analysis/saliency/lrp_transfuser.py:424-430`; mirror of
  `planning_decoder.py:121-143`).
- `TFv6FullModelForLRP.forward` returns `queries[:, self._speed_query_idx]` → `[B, 256]`
  (`lrp_transfuser.py:534-546`).
- `LRPTFv6Model.node_dim = 256` (`lrp_transfuser.py:609`).
- LRP1 (`beg="output", end="fc"`) backprops from
  `target_speed_decoder(speed_query)` to `speed_query` with the softmax-distribution
  seed (`_attribute_to_fc`, `lrp_transfuser.py:780-801`; seed construction
  `_make_speed_seed`, `:743-772`). LRP2 (`beg="fc", end="input"`) one-hot-seeds a
  single F_c node and backprops to pixels (`_attribute_backbone`, `:803-819`).
- Construction interface: `LRPTFv6Model(backbone_eval=model.backbone,
  planning_decoder=model.planning_decoder, device=…)` (`run_analysis.py:155`,
  `hpc/compute_baseline_chunk.py:104-108`); `planning_decoder=None` raises
  (`lrp_transfuser.py:599-603`) and `LTF=True` is asserted (`:614-622`).

### 3.3 Alternatives considered and decision history

Documented in `docs/lrp_todo.md:240-265` (Decisions A/B, made 2026-05-27) and
`docs/design_decisions.md:96-128`; originally raised as Design Issue 6
(`docs/lrp_todo.md:129-150`).

| Option | Layer | Pro | Con |
|---|---|---|---|
| **A** (initial implementation) | 512-dim globally averaged backbone image features (`avgpool` of `image_features [B,512,12,72]`) | Simple; no AttnLRP through the PlanningDecoder needed | Not the decision layer — describes "what the image encoder attends to", not "what the decision-maker attends to" |
| **B** (chosen) | 256-dim `speed_query` token inside the PlanningDecoder | Paper-faithful: directly precedes the speed output; cross-attends over BEV + status tokens, hence decision-conditioned | Requires explicit AttnLRP-capable reimplementations of `nn.TransformerDecoderLayer` / `nn.MultiheadAttention` (`lrp_transfuser.py:196-342`) |

Decision B is coupled with Decision B′ ("what to seed LRP1 from"): with Option B the
seed is naturally the speed output. The seed itself evolved: argmax one-hot
(2026-05-27, still stated at `docs/design_decisions.md:114` — **stale**) → softmax
distribution over the 8 bins (Decision D, 2026-05-28, `docs/lrp_todo.md:284-294`),
implemented at `lrp_transfuser.py:771`. Two further coupled decisions: negative F_c
relevances from AttnLRP are absolute-valued before use in ATOMs (Decision C,
`lrp_todo.md:267-282`), and forced-brake / forced-drive counterfactual seeds exist for
comparative plots (`_make_speed_seed`, `lrp_transfuser.py:743-772`).

Option A was not deleted: the 512-dim pooled backbone feature survives as the **MDX-v1
detector feature space** (`get_backbone_features`, `lrp_transfuser.py:922-945`), while
MDX-v2 uses the F_c vector itself (`get_fc_features`, `:995-1022`;
`get_planning_action_and_features`, `:950-993`; rationale
`docs/design_decisions.md:549-578`). This gives the thesis an implicit
backbone-vs-decision-layer comparison through the two MDX variants.

### 3.4 Implications and caveats of F_c = `speed_query`

1. **Longitudinal-only decision coverage.** The speed query feeds only
   throttle/brake; steering is derived from the 10 route queries (§2.7). ATOMs
   profiles therefore describe the attention behind the *speed* decision. The lateral
   decision pathway is unattributed (a waypoint/route-query F_c was never implemented;
   `wp_decoder` is exposed in the LRP wrapper only as an action proxy for MDX-v2,
   `lrp_transfuser.py:421-422, 541-544`).
2. **Decision-conditioned by construction.** The token is produced by 6 rounds of
   self-attention (against the other 18 queries) and cross-attention (against 120 BEV +
   5 status tokens), so it aggregates fused image/BEV evidence *and* command/target-point
   conditioning — exactly the property the ATOMs paper requires of F_c.
3. **Post-LayerNorm node space.** `speed_query` is taken after the decoder's final
   `LayerNorm`, so node magnitudes are normalized per frame; per-node LRP relevances
   (not raw activations) carry the inter-frame signal. Unlike WoR's GAP-pooled F_c,
   no spatial pooling is involved — the token is attention-pooled.
4. **Offline conditioning gap.** When profiles are computed from npz frames,
   `_make_minimal_data` (`lrp_transfuser.py:1040-1060`) reconstructs the status dict
   with real `speed` and command one-hot but **zero** `target_point*` and
   `acceleration`. Route conditioning is thus absent offline, which can shift
   attributions relative to the live agent (also noted in `CLAUDE.md`). Default
   `cmd=3` (LANEFOLLOW in the 0-based scheme) at `:1040`; the analysis pipeline passes
   `default_cmd=2` in some places (`hpc/compute_baseline_chunk.py:133` — see bug log
   1.7).
5. **F_c doubles as detector feature.** The same 256-dim vector is the MDX-v2 feature
   space, making MDX-v2 an attribution-free control experiment on the identical
   representation.

---

## 4. WoR architecture and its F_c (secondary)

### 4.1 Network — `CameraModel`

`pcla_agents/wor/rails/models/main_model.py:7-50`, weights
`pcla_agents/wor_pretrained/leaderboard_weights/main_model_10.th`, config
`config_leaderboard.yaml` (same dir). Loaded on **CPU** by `ImageAgent`
(`pcla_agents/wor/image_agent.py:66-72`).

- **Wide stream**: custom `resnet34` (`pcla_agents/wor/common/resnet.py`) on the
  concatenated wide image; output `[B, 512, h, w]`, global average pooled → 512
  (`main_model.py:20, 57, 63-68`).
- **Narrow stream**: `resnet18` → GAP 512 → `bottleneck_narr = Linear(512→64) + ReLU`
  (`main_model.py:23-28, 61-66`).
- **Joint embedding**: concat → **576** (= 512 + 64).
- **Action head** (`main_model.py:42-48`):
  `Linear(576→256) → ReLU → Linear(256→256) → ReLU → Linear(256→312)`.
  `num_acts = num_cmds·num_speeds·(num_steers+num_throts+1) = 6·4·13 = 312`
  (`main_model.py:30-31`; `all_speeds: True`).
- `action_logits` (`main_model.py:137-148`) expands each 13-logit block into a
  9×3 = 27-way steer×throttle joint grid plus 1 brake logit → 28 logits per
  (command, speed-bin) cell.
- Segmentation heads (`main_model.py:21, 24, 41`) are training-time auxiliaries only.
- Normalization: `x/255` then ImageNet mean/std (`main_model.py:50, 57`).

### 4.2 Inputs (verified against code and on-disk data)

`ImageAgent.sensors()` (`image_agent.py:115-138`): 3 wide RGB cameras 160×240 px,
FOV 60°, yaws −55/0/+55° plus one narrow camera 384×240 px, FOV 50°, all at
(x=1.5, z=2.4) (yaml `camera_x/camera_z`; note the yaml's `camera_yaws: [0,-30,30]`
is *not* what the agent uses). Matching semantic cameras are added for ATOMs labeling
(`:129-136`).

Preprocessing (`image_agent.py:144-206`):

- wide: crop top 48 rows (`wide_crop_top`, yaml) → 192×160 each, BGR→RGB, concat
  3 cameras → **`[1, 3, 192, 480]`**;
- narrow: crop bottom 80 rows (`narr_crop_bottom`, yaml) → **`[1, 3, 160, 384]`**.

Verified on disk: `data/WOR/baseline_data/frames/*.npz` →
`wide_rgb (N,3,192,480)`, `narr_rgb (N,3,160,384)`. (`CLAUDE.md`'s "wide RGB
(160×704) + narrow RGB (88×352)" is **wrong** — bug log 2.1. The stored
`seg_red_narr` is `(N,192,384)`, 32 rows taller than `narr_rgb` — bug log 2.7.)

Command mapping (`image_agent.py:187-201`): `cmd.value − 1` (1-based `RoadOption` →
0-based), negatives → 3 (LANEFOLLOW); lane-change commands (4/5) are suppressed to 3
after 200 consecutive frames.

### 4.3 Action selection

`CameraModel.policy` (`main_model.py:86-116`) slices the `[B, 6, 4, 28]` grid at the
current command; `ImageAgent._lerp` (`image_agent.py:271-282`) linearly interpolates
the per-speed-bin logits at the current ego speed across the 4 bins spanning
`min_speeds = 0.0` to `max_speeds = 8.0` m/s (yaml). Controls
(`image_agent.py:241-248, 284-323`):

- steer = `steers · softmax(steer_logits)` with `steers = linspace(−1, 1, 9)`;
- throttle = `throts · softmax(throt_logits)` with `throts = linspace(0, 1, 3)`;
- brake probability = last element of the 28-way joint softmax; `> 0.5` → full stop;
- post-processing: throttle floor 0.4 (0.6 in `HIGH_SPEED_MODE`), speed caps
  10 km/h (turns) / 20 km/h (default; raised by `SPEED_MODE`/`HIGH_SPEED_MODE`).

### 4.4 WoR F_c

**F_c = the 256-dim ReLU output of the second hidden FC layer of `act_head`**, i.e.
the output of `act_head[0:4]` (`Linear(576→256), ReLU, Linear(256→256), ReLU`), the
direct input of the final `Linear(256→312)`:

- `JointCameraToFC` truncates the model at `act_head.children()[:4]`
  (`ATOMs_Analysis/saliency/lrp_analysis.py:54-79`).
- LRP1 attributes through the final linear `act_head[4]` only
  (`_attribute_to_fc`, `lrp_analysis.py:378-390`); LRP2 seeds one F_c node and
  backprops through `act_head[:4]` + both ResNet streams to both camera inputs.

This mirrors the original ATOMs setup (a plain MLP head after a conv encoder): F_c is
the penultimate fully-connected layer feeding the action logits — structurally exact,
unlike TFV6 where the "penultimate layer" had to be identified inside a transformer
decoder. Differences to TFV6's F_c: (i) WoR's F_c precedes the *joint*
steer/throttle/brake logits, so its attribution covers the full action, not only the
longitudinal decision; (ii) it is built from GAP-pooled CNN features rather than an
attention-pooled token; (iii) dual-camera relevance is attributed jointly in one
backward pass (the narrow share is discarded when
`conf.WIDE_ONLY_PROFILE = True`, `ATOMs_Analysis/atoms_config.py:97`).

---

## 5. Parameters & magic constants

"cfg" = configurable (read from a config file/object at runtime); "hard" = hardcoded
in source; "ckpt-cfg" = fixed by the shipped checkpoint's `config.json` (changing it
invalidates the weights).

### 5.1 TFV6

| Constant | Value | Where | Kind |
|---|---|---|---|
| Cameras (live) | 6 × 384×384 px, FOV 60°, yaws ±57.5/0/±122.5/180° | `config_base.py:120-170` | hard (selected by `target_dataset`) |
| Live input tensor | `[1, 3, 384, 2304]` | derived; `sensor_agent.py:640` | — |
| Offline analysis input | `[1, 3, 384, 1152]` (3-camera LEAD data) | `migrate_lead_to_baseline.py:38`; npz on disk | data property |
| JPEG simulation quality | 90 | `config_closed_loop.py:17` | cfg |
| ImageNet mean/std | (0.485,0.456,0.406)/(0.229,0.224,0.225) | `fn.py:26-28`; `lrp_transfuser.py:351-352` | hard (duplicated) |
| Planning area | x∈[−32,64] m, y∈[−40,40] m | `config_training.py:81-108` | hard (mode-dependent property) |
| `pixels_per_meter` | 4.0 | `config_training.py:78` | hard |
| LTF grid / BEV raster | 320×384 px | derived `config_training.py:110-118` | — |
| Encoders | timm `resnet34` ×2 (image pretrained, lidar not) | `config_training.py:734,736`; `transfuser_backbone.py:35,43` | ckpt-cfg |
| Image/LiDAR anchors | (12, 72) / (10, 12) | `config_training.py:469-477, 720-728` | derived |
| GPT depth/heads/MLP/dropout | `n_layer=2`, `n_head=4`, `block_exp=4`, all dropouts 0.1 | `config_training.py:741-752` | ckpt-cfg |
| GPT init | linear N(0, 0.02), LN weight 1.0 | `config_training.py:753-758` | hard |
| GPT token count | 984 = 12·72 + 10·12 | `transfuser_backbone.py:265-272` | derived |
| Token dim (`transfuser_token_dim`) | 256 | `config_training.py:615` | ckpt-cfg |
| Decoder layers / heads | 6 / 8 | `config_training.py:611,613` | ckpt-cfg |
| Decoder FFN dim | 2048 (PyTorch default, not set) | `planning_decoder.py:47-56` | hard (implicit) |
| Queries | 19 = 10 route + 8 waypoints + 1 speed | `planning_decoder.py:30-45`; `config_training.py:656,660-664` | ckpt-cfg |
| `_speed_query_idx` | 18 | `lrp_transfuser.py:424-430` | derived |
| Speed head | Linear 256→256 → ReLU → Linear 256→8 | `planning_decoder.py:65-73` | ckpt-cfg |
| Speed bins (m/s) | `[0, 4, 8, 10, 13.88888888, 16, 17.77777777, 20]` | `config_training.py:633-645` | hard |
| Rounded duplicate of bins | `[0, 4, 8, 10, 13.89, 16, 17.78, 20]` | `lrp_transfuser.py:948` (`_SPEED_BINS_PROXY`) | hard (duplicate, bug 2.4) |
| Status normalizers | `max_speed=25.0` m/s; tp `[200, 50]` | `config_training.py:531-544, 582-585` | hard |
| Command dim | 6 | `config_training.py:509-522` | hard (mode-dependent) |
| Brake threshold (inference) | P(bin 0) > 0.9 | `config_open_loop.py:17` | cfg |
| Control modalities | steer="route", throttle/brake="target_speed" | `config_closed_loop.py:21-25` | cfg |
| Route-planner pop distance | 5.0 m (adaptive fallback 4.0) | `config_closed_loop.py:35`; `sensor_agent.py:558-569` | cfg / hard |
| `inital_frames_delay` | 1 | `config_base.py:320` | hard |
| Sine pos-emb temperature | 10000 | `planning_decoder.py:493` | hard |
| Checkpoint glob → analysis model | `sorted("model*.pth")[0]` = `model_0030_0.pth` | `run_analysis.py:141-145` | hard |
| Ensemble size (live) | 3 (`model_0030_{0,1,2}.pth`) | `open_loop_inference.py:54-81` | data property |

### 5.2 WoR

| Constant | Value | Where | Kind |
|---|---|---|---|
| Wide cameras | 3 × 160×240 px, FOV 60°, yaws ±55/0° | `image_agent.py:119-124` | hard |
| Narrow camera | 384×240 px, FOV 50° | `image_agent.py:125-126` | hard |
| Crops | wide top 48; narrow bottom 80 | yaml `wide_crop_top`/`narr_crop_bottom`; `image_agent.py:147,182` | cfg |
| Model inputs | wide `[1,3,192,480]`, narrow `[1,3,160,384]` | derived; verified npz | — |
| Camera mount | x=1.5, z=2.4 | yaml `camera_x`,`camera_z` | cfg |
| Action grid | 6 cmds × 4 speed bins × (9 steers + 3 throts + 1 brake) = 312 logits | `main_model.py:30-31`; yaml | cfg |
| Speed bin range | 0.0–8.0 m/s, 4 bins, linear interp | yaml `min/max_speeds`,`num_speeds`; `image_agent.py:271-282` | cfg |
| Steer/throttle decode grids | linspace(−1,1,9) / linspace(0,1,3) | `image_agent.py:81-82` | derived from yaml |
| F_c | `act_head[:4]` output, 256-dim | `main_model.py:42-46`; `lrp_analysis.py:70-72` | hard |
| Joint embed dim | 576 = 512 wide + 64 narrow bottleneck | `main_model.py:43` | derived |
| Brake threshold | 0.5 | `image_agent.py:295` | hard |
| Throttle floor | 0.4 (0.6 high-speed) | `image_agent.py:299-302` | hard (mode flag cfg) |
| Speed caps | {turn: 10, else 20} km/h (modes raise) | `image_agent.py:309-313` | hard |
| Lane-change suppression | 200 frames | `image_agent.py:195` | hard |
| Weights | `main_model_10.th`, CPU | yaml `main_model_dir`; `image_agent.py:66-69` | cfg |

---

## 6. Key design decisions with rationale

1. **F_c = `speed_query` (Option B) instead of pooled backbone (Option A).**
   Paper-faithfulness: the ATOMs F_c must be the layer the action is chosen from;
   Option A only described the perception encoder. Cost: explicit AttnLRP wrappers
   for the whole decoder. (`docs/lrp_todo.md:240-265`; `docs/design_decisions.md:96-128`;
   §3.3.) Option A survives as the MDX-v1 feature space, giving a built-in comparison.
2. **`visiononly_resnet34` checkpoint (LTF, radarless).** LTF removes the LiDAR sensor
   dependency and guarantees attribution flows only through RGB
   (`docs/design_decisions.md:132-140`); enforced by an assert
   (`lrp_transfuser.py:614-622`). Radarless removes 20 radar tokens from the
   planning context, simplifying the attribution graph.
3. **Softmax-distribution LRP1 seed** (Decision D) — consequence of the two-hot
   training target (§2.6): smooth across bin boundaries, weights attribution by the
   model's full predicted speed distribution (`docs/lrp_todo.md:284-294`).
4. **Speed-query index computed, not hardcoded.** `TFv6FullModelForLRP` re-derives
   `_speed_query_idx` from the same config predicates as `PlanningDecoder.forward`
   (`lrp_transfuser.py:424-430`), so a config change cannot silently de-synchronize
   the F_c location.
5. **WoR F_c = second hidden FC** — direct transplant of the ATOMs paper setup; the
   truncation point `act_head[:4]` keeps the final action projection as the LRP1
   "output layer" (`lrp_analysis.py:54-79, 378-390`).
6. **Lenient checkpoint loading** (`run_analysis.py:146-152`,
   `open_loop_inference.py:66-78`): weights with mismatched shapes are dropped and
   loading is non-strict, tolerating config-derived size changes (e.g. positional
   embedding lengths) at the cost of silently ignoring genuine mismatches.
7. **TFV6 treated as wide-only in ATOMs** (`conf.WIDE_ONLY_PROFILE = True`): the full
   camera concatenation is "the wide image", `narr_rgb=None`
   (`docs/design_decisions.md:90-92`; `sensor_agent_data_collection.py:160-167`).

---

## 7. Known limitations / open issues

1. **Train/analysis camera mismatch (med-high).** The checkpoint was trained on
   6-camera 2304-px-wide images; all offline baseline/test/val frames are 3-camera
   1152-px-wide LEAD data (§2.2). `AdaptiveAvgPool2d` to the fixed (12, 72) anchor
   grid makes this dimensionally legal but horizontally stretches the token layout
   and changes the camera geometry the GPT positional embeddings were trained on.
   Live-perturbation recordings (6-camera, 2304) and LEAD-based baselines (1152) also
   differ in input geometry from each other.
2. **Ensemble vs single member.** The deployed agent is a 3-model ensemble; LRP/ATOMs
   analyze only `model_0030_0.pth`. Attribution explains a related but not identical
   policy to the one that drove (relevant for live-collected data).
3. **Longitudinal-only F_c.** Steering attribution is structurally out of scope of the
   chosen F_c (§3.4.1). An additional waypoint-query F_c would be needed for lateral
   decisions.
4. **Offline conditioning gap.** Zero `target_point*`/`acceleration` in
   `_make_minimal_data` (§3.4.4) removes route conditioning from offline attributions.
5. **Inference-time post-processing is invisible to attribution.** Brake-threshold
   override (0.9), PID controllers, `ForceMovePostProcessor` and WoR's `post_process`
   all transform model outputs after F_c; ATOMs explains the network decision, not the
   final actuated control.
6. **Stale docs.** `CLAUDE.md` WoR input sizes (160×704 / 88×352) are wrong;
   `design_decisions.md:114` still documents the superseded argmax seed; the
   `[B,3,384,2304]` docstrings do not hold for offline data. See bug log Topic 2.

---

## 8. Cross-references

- **01_architecture_overview.md** — repo layers, config system, entry points; how
  `conf.AGENT` selects the model load path (`run_analysis.py:64-71, 130-176`).
- **03_lrp.md** — the LRP rules applied on top of these architectures:
  AttnLRP custom autograd, explicit attention reimplementations
  (`SelfAttentionExplicit`, `MultiheadAttentionExplicit`,
  `TransformerDecoderLayerExplicit`), composite/canonizer setup, two-pass scheme;
  WoR z⁺-rule pipeline in `lrp_analysis.py`.
- **04_atoms.md** — how F_c node relevances and per-node pixel maps are
  combined with semantic masks into attention profiles; the 10-class TFV6 vs 29-class
  WoR schemes; `_make_minimal_data` consequences.
- **05_dataset_creation.md** — the LEAD 3-camera npz schema referenced in
  §2.2 and §4.2.
- `docs/lrp_todo.md:240-294` — primary decision record for F_c (Decisions A–D).
- `docs/design_decisions.md:96-140, 549-578` — Option A/B rationale, LTF mode, MDX-v2
  reuse of F_c.
