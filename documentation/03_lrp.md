# 03 — LRP Implementations

Technical reference for the two Layer-wise Relevance Propagation backends:

| Agent | File | Class | Approach |
|---|---|---|---|
| TFV6 (primary) | `ATOMs_Analysis/saliency/lrp_transfuser.py` | `LRPTFv6Model` | AttnLRP (Achtibat et al. 2024) on top of zennit |
| WoR (secondary) | `ATOMs_Analysis/saliency/lrp_analysis.py` | `LRPCameraModel` | z⁺-rule (AlphaBeta α=1, β=0) via zennit |

`lrp_lbc.py` is a dead end and out of scope (see `documentation/00_PLAN.md`). Both classes expose the same public interface (`update_context`, `forward_relevance(wide, narr, cmd, spd, node_id, raw, beg, end, forced_brake, forced_drive)` returning `(wide_rel, narr_rel, wide_frac, is_brake)`), so `ATOMsCarla` (Topic 04) is agent-agnostic (`atoms_carla.py:474-622`).

All claims below verified against code; `docs/lrp_todo.md` and `docs/design_decisions.md` were used as decision history only (several entries are stale — see §8 and `99_bugs_and_findings.md` §Topic 3). zennit version used for semantics verification: 0.5.1 (NOTE: zennit is **not pinned in `environment.yml`** — finding 3.10).

---

## 1. Purpose & scope

LRP backpropagates the model's output decision to earlier layers, producing (a) a relevance vector over the F_c node space (the 256-d decision layer, see `02_agents.md`) and (b) pixel-level relevance maps over the input camera image(s). These are the raw inputs of the ATOMs metric (Topic 04): pixel maps are intersected with semantic segmentation masks, weighted by F_c node relevances, and aggregated into per-frame attention profiles for OOD detection.

This document covers: the LRP rules and their assignment to layers, the model-wrapping needed to make the networks zennit-attributable, the custom AttnLRP autograd functions, the three attribution pass modes and their seeds, all numeric constants, and the verified decision history.

## 2. Background: rules used

Notation: layer computes z_j = Σ_i x_i w_ij (+ b_j); R_j is relevance at the layer output, R_i at its input.

- **ε-rule**: R_i = Σ_j x_i w_ij / (z_j + ε·sign(z_j)) · R_j. Small ε stabilizes near-zero denominators; large ε absorbs (dampens) relevance. zennit `Epsilon`.
- **AlphaBeta(α, β)** with α−β=1: separates positive and negative pre-activations, R_i = Σ_j (α·x_i w_ij⁺/z_j⁺ − β·x_i w_ij⁻/z_j⁻)·R_j. **α=1, β=0 is the z⁺-rule**: only positive contributions propagate, output is non-negative for non-negative seeds. zennit `AlphaBeta`.
- **WSquare**: R_i = Σ_j w_ij²/Σ_k w_kj² · R_j — input-independent; standard choice for the *first* layer whose input is real-valued/unbounded (raw pixels). zennit `WSquare`.
- **Pass**: identity (R_i = R_j elementwise). Used for activations and norm layers.
- **AttnLRP** (Achtibat et al. 2024, `papers/attention lrp.pdf`) extends LRP to the non-linear ops inside attention:
  - **Proposition 3.1 (Eq. 13), softmax**: Taylor decomposition at the input point x gives R_i^{l−1} = x_i (R_i^l − s_i Σ_j R_j^l), s = softmax(x).
  - **Proposition 3.3 (Eq. 15), bilinear matmul** O = A·B: sequential ε-rule + uniform rule over the two factors; conservation enforced by dividing by 2: R_{A,ji} = Σ_p A_ji B_ip / (2·O_jp + ε·sign) · R_jp (symmetrically for B).
  - **Proposition 3.4, LayerNorm/RMSNorm**: identity rule — so mapping `nn.LayerNorm → Pass` is paper-compliant (unlike BatchNorm→Pass, which is an approximation; see §3.4).
  - The paper recommends the ε-rule for the linear (Q/K/V/out) projections inside attention, and a noise-suppressing rule (γ in the paper; z⁺ chosen here) for Conv/FFN layers in vision transformers.
- **zennit mechanics**: rules are zennit `Hook`s attached to `nn.Module`s by a `Composite`; inside `composite.context(model)`, a call to `torch.autograd.grad` computes LRP because the hooks replace the modules' backward with the rule computation. Anything that is *not* an `nn.Module` (functional softmax/matmul, fused attention kernels, residual `+`) falls through to plain autograd — the core motivation for the explicit re-implementations in §3.1.

## 3. TFV6 AttnLRP implementation (`lrp_transfuser.py`)

### 3.1 Model wrapping — `TFv6FullModelForLRP` (lines 364–546)

A single `nn.Module` wrapping backbone + PlanningDecoder so **one** composite context covers the entire attribution graph `rgb → speed_query`:

- **Shared by reference** (read-only): `image_encoder`, `lidar_encoder`, `lidar_channel_to_img`, `img_channel_to_lidar`, `avgpool_img/lidar` (lines 390–395). **Deep-copied and modified**: the 4 GPT fusion transformers (line 402, each `block.attn` replaced by `SelfAttentionExplicit`, 403–405) and the entire PlanningDecoder (line 408, each `nn.TransformerDecoderLayer` replaced by `TransformerDecoderLayerExplicit`, 410–415). Caveat: the BatchNorm canonizer temporarily mutates the *shared* conv weights of the live backbone inside every composite context (restored on exit) — see finding 3.9.
- **`NormalizeImageNet`** (349–357) replaces the functional `normalize_imagenet`, because the functional version's `x.clone()` + in-place channel writes create `CopySlices` autograd nodes that break zennit hook pairing (comment at 396–399). It is not matched by any rule, so backward through it is plain autograd — a per-channel constant scaling of pixel relevance by 1/(255·σ_c). Constants: ImageNet mean/std (351–352, hardcoded).
- **LiDAR (LTF mode)**: `_make_lidar` (439–448) builds the deterministic 2-channel x/y grid *without* `requires_grad`, so attribution flows exclusively through the RGB path. `LRPTFv6Model.__init__` asserts `config.LTF` (614–622) to fail loudly if a non-LTF checkpoint were used (non-LTF reads `rasterized_lidar` from the data dict, which npz frames never contain). Verified by diagnostic D04 (Topic 11).
- **`_run_backbone`** (477–497) re-implements `TransfuserBackbone.forward` for the vision path: normalize → interleaved ResNet34 stages and 4 GPT fusions (`_fuse`, 457–475, with `F.interpolate` upsampling handled by autograd) → returns (lidar/BEV features, image features).
- **`forward`** (503–546): backbone → `planning_context_encoder` (BEV tokens + status tokens from the data dict) → `transformer_decoder` over the learned query set → returns `queries[:, _speed_query_idx]`, the 256-d `speed_query` token = F_c. `_speed_query_idx` mirrors `PlanningDecoder.forward` query layout (424–433). With `_return_wps=True` it additionally decodes future waypoints via `wp_decoder` + `cumsum` (541–544; used only by MDX-v2, §3.7).

### 3.2 Explicit attention modules and custom autograd functions

Reason for re-implementation: the GPT blocks use `F.scaled_dot_product_attention` (fused kernel) and the PlanningDecoder uses `nn.MultiheadAttention` (also fused) — both opaque to zennit hooks (`docs/lrp_todo.md` Bug 2, fixed).

- **`LRPSoftmax`** (64–80): `forward` returns softmax and saves `(x, s)`; `backward` returns `x * (R − s·ΣR)` — exactly Prop 3.1/Eq. 13. Unit-tested by diagnostic D01.
- **`LRPMatMul`** (83–109): `backward` computes `scaled_R = R / (2·O + EPS·sign(O))`, `R_A = (scaled_R @ Bᵀ)·A`, `R_B = (Aᵀ @ scaled_R)·B` — exactly Prop 3.3/Eq. 15 incl. the conservation factor 2. `EPS = 1e-6` (line 92, hardcoded class attribute). `sign(0)` mapped to +1 (103–104). Unit-tested by D02.
- **`AttentionLinear`** (116–122): empty `nn.Linear` subclass used as a *marker type* so the composite can give Q/K/V/out projections the ε-rule while ordinary Linears get AlphaBeta. `_make_attn_linear` (125–131) shares weight/bias tensors with the source Linear.
- **`SelfAttentionExplicit`** (138–189), for the 4×2 GPT fusion blocks: K/Q/V/proj as `AttentionLinear`; `Q·Kᵀ` and `A·V` via `LRPMatMul.apply`; softmax via `LRPSoftmax.apply`. The 1/√d_head scale is applied as a plain multiplication *outside* `LRPMatMul` (line 184) — see finding 3.7 (systematic per-layer attenuation of the q/k-branch relevance; cancels under ATOMs normalization).
- **`MultiheadAttentionExplicit`** (196–269): splits `in_proj_weight` into separate q/k/v `AttentionLinear`s (`from_module`, 218–239, clones weights), same LRP-aware score/softmax/value pipeline (253–268), returns `(out, None)` to match the `nn.MultiheadAttention` signature. Masks are supported syntactically (259–264) but never used by the PlanningDecoder (called without masks, 534–537); if they were, the `-inf` entries would NaN through `LRPSoftmax.backward` — finding 3.6.
- **`TransformerDecoderLayerExplicit`** (276–342): post-norm layout (self-attn → norm1 → cross-attn → norm2 → FFN → norm3), activation hardcoded `nn.GELU()` (295–298). Verified to match the real PlanningDecoder construction: `nn.TransformerDecoderLayer(..., activation=nn.GELU(), batch_first=True)` with default `norm_first=False` (`pcla_agents/transfuserv6/lead/tfv6/planning_decoder.py:47-56`). `from_module` copies attention + linear1/2 + norm1/2/3 weights (300–316); dropouts are dropped (identity in eval). Neither the activation nor `norm_first` is copied/checked from the source layer — latent mismatch risk if the training config ever changes (finding 3.8). The final `TransformerDecoder.norm` LayerNorm is retained from the deep copy and receives `Pass`.

### 3.3 Composite construction — `_create_composite` (650–667)

`SpecialFirstLayerMapComposite` with `canonizers=[SequentialMergeBatchNorm()]`:

| Match (in order) | Rule | Notes |
|---|---|---|
| `Activation` | `Pass()` | ReLU/GELU etc. |
| `nn.BatchNorm2d` | `Pass()` | no-op fallback; BN is folded into the preceding Conv by the canonizer first (§3.4) |
| `nn.LayerNorm` | `Pass()` | paper-compliant (AttnLRP Prop 3.4 identity rule) |
| `AttentionLinear` | `Epsilon(epsilon=1e-2)` | Q/K/V/out in all attention blocks; **must precede** `AnyLinear` (mapping is first-match) |
| `Convolution` | `AlphaBeta(α, β)` | α,β = (1,0) default; (2,1) if `uitb=True` (591–592) |
| `AnyLinear` (= `zennit.types.Linear`) | `AlphaBeta(α, β)` | FFN/decoder Linears. zennit's `Linear` type *includes* Convolution, so the ordering above is load-bearing |
| first `Convolution` encountered | `WSquare()` | `first_map` (663); zennit assigns this to the *first* matching module in traversal order only — here the image-encoder stem conv (image_encoder is declared before lidar_encoder, line 391 vs 392), which is correct since there is a single trainable input stream |

ε = **1e-2** (line 658) is deliberate and history-laden: ε=1e-6 caused ~10¹⁴× relevance amplification with sign oscillation, because LayerNorm-fed attention Linears have near-zero pre-activation sums (`docs/lrp_todo.md` Bug C; pos_frac 0.45→0.993, profile CoV 2.67→0.15 after the change). Note `docs/design_decisions.md` still documents ε=1e-6 — stale (finding 3.1).

`uitb` (AlphaBeta(2,1)) is plumbed through both LRP classes but never enabled by any caller (`run_analysis.py`, `run_online_analysis.py:134`, all `hpc/compute_*_chunk.py` use the default).

### 3.4 BatchNorm handling / canonization status

**Status: canonized** (since 2026-05-28). `SequentialMergeBatchNorm` walks the module tree depth-first in declaration order and folds every BatchNorm that directly follows a Conv/Linear leaf into that layer's weights, replacing the BN by identity for the duration of the composite context (verified against zennit 0.5.1 source). timm's ResNet34 declares `conv1, bn1, …, conv2, bn2` and `downsample = Sequential(conv, bn)` in forward order, so all ~36 BN pairs of both encoders are merged; the `(nn.BatchNorm2d, Pass())` entry only catches hypothetical leftovers (comment 653–655).

History (`docs/lrp_todo.md` Issue 7, measured by diagnostics D05/D06): with Pass-only BN, LRP1 conservation was fine (Σ node_rel ≈ 0.983) but backbone LRP2 amplified relevance by 10¹³–10¹⁷ with oscillating sign. Canonization reduced amplification by ~100×; residual amplification (~2×10⁷, from residual connections handled by plain autograd, plus the ε-rule and the 1/√d scaling) is systematic across frames and cancels in the ATOMs per-frame normalization. Note: `lrp_todo` originally claimed a *custom* canonizer would be required because "timm differs from torchvision"; the actual fix uses zennit's generic `SequentialMergeBatchNorm`, which is order-based, not type-based (finding 3.2). `docs/design_decisions.md` §"zennit composite — no canonizer" is stale (finding 3.1).

### 3.5 The three pass modes — `forward_relevance` (680–737)

`beg`/`end` select the segment of the model the relevance traverses. All passes run inside `self.composite.context(self.full_model)` with `torch.enable_grad()`; relevance is extracted with `torch.autograd.grad(outputs, inputs, grad_outputs=seed)`. The conditioning data dict comes from `update_context` (`_data_cache`, set per frame by `ATOMsCarla.process_frame`) or is rebuilt by `_make_minimal_data` (§3.6).

| Mode | beg→end | Method | Seed (grad_outputs) | Returns | ATOMs use |
|---|---|---|---|---|---|
| **LRP1** | `output→fc` | `_attribute_to_fc` (780–801) | `_make_speed_seed` over the 8 speed logits (§3.6) | `[256]` node relevance at `speed_query`, + `is_brake` | MODE_ANALYSIS=1: node weights + node filter (`atoms_carla.py:416-420`) |
| **LRP2** | `fc→input`, `node_id=k` | `_attribute_backbone` (803–819) | one-hot at `speed_query[:,k]` (`_one_hot_node`, 1028–1033) | `[1,3,H,W]` pixel map (signed) | MODE_ANALYSIS=1: per-node pixel maps |
| layer-level | `fc→input`, `node_id=None` | `_attribute_fc_to_input` (821–836) | `speed_query.clamp(min=0).detach()` — positive **activations**, not decision relevance (explicit in docstring) | `[1,3,H,W]` | MODE_ANALYSIS=2 |
| output-weighted | `output→input` | `_attribute_true_output_to_input` (838–886) | two-step: LRP1 seed → node_rel → used as LRP2 seed | `[1,3,H,W]`, + `is_brake` | MODE_ANALYSIS=3 |

Details:

- LRP1 runs the full forward but the backward stops at `speed_query` (`inputs=[speed_query]`), i.e. it propagates only through `target_speed_decoder` (Linear 256→256 → ReLU → Linear 256→8, AlphaBeta rule). Conservation here is excellent (D05: Σ ≈ 0.983).
- **The `output→input` mode is deliberately two-step** (LRP1 with `retain_graph=True`, then LRP2 seeded with `node_rel.detach()`, 866–886). The naive single backward `speed_logits → rgb` was numerically unstable: a *distributed* seed backpropagated through the 6 decoder layers hits `LRPMatMul` denominators `2·O + 1e-6` with `O ≈ 0` (near-uniform attention × cancelling values), cascading over 12 matmuls to ~10¹⁵ amplification (`docs/lrp_todo.md` Bug F). Because `autograd.grad` is linear in `grad_outputs`, the two-step result equals Σ_k R_k · pixel_map_k exactly (verified by diagnostic D07, rel_L∞ = 0).
- `fc→input` skips the output forward entirely and hardcodes `is_brake=False` in the return (727) to avoid an extra pass; `ATOMsCarla._compute_node_level` saves/restores the LRP1 `is_brake` around the LRP2 loop (`atoms_carla.py:417-419, 437`).
- For TFV6, `narr_rel` is always `None`, `wide_frac` always `1.0`, and the `raw` kwarg is accepted but ignored — no cross-camera normalization exists (single image stream); pixel maps are returned **raw and signed** (AttnLRP rules produce negative relevance; cf. WoR §4 where maps are |·|-normalized). All normalization happens downstream in ATOMs.

### 3.6 Seed construction and rationale

`_make_speed_seed(speed_logits, forced_brake, forced_drive)` (743–772):

| Condition | Seed over the 8 speed bins | Rationale |
|---|---|---|
| default | `softmax(speed_logits.detach())` | **Decision D** (`docs/lrp_todo.md`): `target_speed_decoder` is trained with a *two-hot* target (mass split across two adjacent bins of [0, 4, 8, 10, 13.89, 16, 17.78, 20] m/s). An argmax one-hot seed is discontinuous at bin boundaries — two near-identical frames whose mass straddles a boundary differently would get different LRP1 vectors. The softmax seed weights every bin by model confidence, is smooth in the logits, and represents the full predicted speed distribution. Total seed mass = 1. |
| `forced_brake=True` | one-hot at bin 0 (0 m/s) | **Decision E**: counterfactual "what would the model look at if it had to brake"; used only by `PLOT_COMPARATIVE_REL` visualization (Topic 04/12). |
| `forced_drive=True` | one-hot at `argmax(logits[1:]) + 1` | best counterfactual *drive* bin; equals the normal argmax when the model is already driving. |

`is_brake` (4th return of `forward_relevance`) always reflects the model's **actual** argmax (`== bin 0`), independent of forced flags (762; verified by D09). For TFV6 MODE_ANALYSIS=1, forced flags only change the LRP1 node weights — the per-node LRP2 maps are seed-independent — so comparative maps are built by re-weighting cached LRP2 maps in ATOMs (`atoms_carla.py:509-575`), not inside the LRP class.

**Conditioning dict** `_make_minimal_data(spd, device, cmd=3)` (1040–1060): speed scalar, command one-hot of length 6 (clamped to 0–5; an earlier all-zero command vector distorted cross-attention and made ~80% of frames predict bin 0 — `docs/design_decisions.md` §"command one-hot fix"), and **zeros** for `target_point` ×3 and `acceleration` (not stored in npz). Consequence: route conditioning is absent offline, which can shift attributions relative to the live agent (acknowledged in `CLAUDE.md`). Default `cmd=3` here vs. `default_cmd=2` in the ATOMs pipeline is inconsistent (already logged as findings 1.7/2.12).

### 3.7 Inference helpers (no LRP, `torch.no_grad`)

Interface only — consumers are documented in Topic 07:

| Method (lines) | Returns | Consumer |
|---|---|---|
| `get_speed_logits(wide_rgb, cmd=3, spd=0.0)` (892–920) | `np.ndarray[8]` raw speed logits | PEOC (policy entropy) |
| `get_backbone_features(wide_rgb)` (922–945) | `np.ndarray[512]`, GAP over image features, `clamp(min=0)` | MDX-v1 |
| `get_fc_features(wide_rgb, cmd=4, spd=0.0)` (995–1022) | `np.ndarray[256]` speed_query (F_c) | MDX-v2 test scoring |
| `get_planning_action_and_features(wide_rgb, cmd=4, spd=0.0)` (950–993) | `(feature[256], steer, throttle, brake)` — steer = mean lateral waypoint offset; throttle = `min(E[v]/20, 1)`; brake = `E[v] < 0.5` | MDX-v2 baseline fit (single forward via `_return_wps=True`) |

`_SPEED_BINS_PROXY` (948) is a hardcoded, rounded duplicate of `config.target_speed_classes` (already logged as finding 2.4). Note the differing `cmd` defaults (3 vs 4) across these helpers; harmless because all pipeline callers pass `cmd` explicitly.

### 3.8 Instantiation

`LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder, device=...)`; both submodels must be in `.eval()` (asserted, 597/604). Call sites: `run_analysis.py:155`, `run_online_analysis.py:126`, `hpc/compute_baseline_chunk.py:104-108`, `hpc/compute_test_chunk.py:102`, `hpc/compute_live_pert_chunk.py:94`, `hpc/compute_mdx_features.py:83`. All use the first checkpoint (`sorted(model*.pth)[0]`) of the 3-member ensemble (single-member analysis — finding 2.5).

## 4. WoR z⁺ implementation (`lrp_analysis.py`)

Differences from TFV6 and why:

- **Pure CNN → no AttnLRP needed.** The `CameraModel` is two ResNets + 3-layer FC head; the standard zennit composite covers everything. No custom autograd functions.
- **Dual camera, single joint backward.** `JointCameraForLRP` (32–51) re-implements the forward with both cameras feeding `act_head` so one `autograd.grad(output, [wide_x, narr_x], grad_outputs=mask)` attributes both images simultaneously (`_attribute_joint`, 361–376). The inline `.mean(dim=[2,3])` of the original forward is replaced by explicit `AdaptiveAvgPool2d((1,1))+Flatten` modules so the graph is module-complete (docstring 12–13). `JointCameraToFC` (54–79) is the same pipeline truncated at `act_head[:4]` (output = the 256-d second hidden FC = F_c).
- **Composite** (`_create_composite`, 176–191): `Activation→Pass`, `Convolution→AlphaBeta(1,0)`, `AnyLinear→AlphaBeta(1,0, zero_params='bias')` (bias excluded from the z⁺ denominator — classical LRP convention; the AttnLRP bias warning of `lrp_todo` Bug 3 applies to the TFV6 composite, where `zero_params` was removed), `first_map: Convolution→WSquare`, `canonizers=[ResNetCanonizer()]`. **AvgPool deliberately has no rule** (comment 179–182): `Pass` would return the pooled `[B,C,1,1]` gradient against a `[B,C,H,W]` input (shape mismatch); plain autograd backward spreads each channel's relevance uniformly over H'×W' — this is precisely what makes all per-node `fc→input` maps identical for WoR (GAP collapse, `docs/design_decisions.md` §"per-FC-node pixel maps are identical"), the architectural motivation for moving to TFV6.
- **Canonizer caveat (verified, finding 3.3):** WoR uses a *local copy* of torchvision's ResNet (`pcla_agents/wor/common/resnet.py`, own `BasicBlock` class, `main_model.py:20,23`). zennit's `ResNetBasicBlockCanonizer`/`ResNetBottleneckCanonizer` match `isinstance(module, torchvision.models.resnet.BasicBlock/Bottleneck)` and therefore silently **no-op**; only the `SequentialMergeBatchNorm` component of `ResNetCanonizer` is active. Net effect: BN *is* merged, but residual additions are handled by plain autograd (same accepted simplification as TFV6), contrary to the module docstring (lines 10–11).
- **First-layer rule asymmetry (verified, finding 3.4):** `SpecialFirstLayerMapComposite` assigns `WSquare` to only the *first* matching Convolution in traversal order — the **wide** backbone's stem conv. The narrow backbone's stem conv gets plain `AlphaBeta(1,0)`.
- **Seeds** (`_build_drive_brake_selector`, 439–521): the act_head output is flat `[num_cmds·num_speeds·13] = [6·4·13] = 312` (9 steer + 3 throttle + 1 brake logits per (cmd, speed-bin)). The seed mask is built in this flat space for the active `cmd` and the two speed bins `x0, x1` bracketing the actual speed (`_lerp_bins`, 168–174; weights `1−w, w`):
  - **brake mode** (`brake_prob > 0.5` or `forced_brake`): mass `(1−w, w)` on the two brake logits.
  - **drive mode**: mass distributed over the 9 steer logits proportionally to `softmax(steer_logits)` per bin (493–503); optionally split 0.5/0.5 with throttle softmax weights when `include_throttle=True` (default False, 505–514).
  - `brake_prob` is computed by `softmax(cat[steer.repeat(3), throt.repeat_interleave(9), brake])[-1]` (479–484) — a 55-element ad-hoc distribution, but it **exactly replicates the live agent's own brake decision** (`image_agent.py:284-300`, `action_prob` + `post_process` threshold 0.5), so the LRP seed matches the deployed policy's switching rule.
- **Pass modes**: `output→input` = full joint path; `output→fc` = `_attribute_to_fc` (378–390): forward to the FC activation with `no_grad`, then LRP through **only the final Linear** `act_head[4]` from a detached FC leaf (the analogue of TFV6's LRP1 through `target_speed_decoder`); `fc→input` = one-hot node (or `selector=None` → all-ones seed) through `JointCameraToFC`.
- **Post-processing unique to WoR**: (a) *ResNet-amplification undo* (`undo_resnet_amplification=True` default; blocks 263–276 and 310–323): the wide/narrow relevance *fractions* measured at the pixel level are rescaled to match the fractions measured at the 576-d concat point (`_attribute_to_concat`, 394–433), so the deeper wide ResNet34 (vs. narrow ResNet18 + 512→64 bottleneck) does not distort the inter-camera split; (b) *cross-normalization* (`_cross_normalize`, 334–345): each camera map is mapped to `abs()`, normalized to unit sum, then scaled by `wide_fraction` / `1−wide_fraction` → joint mass 1, non-negative maps (skipped when `raw=True`). TFV6 has neither step (single camera, signed maps).
- `get_action_logits` (534–586): no-LRP helper returning the speed-interpolated 28-d joint action logits (27 steer×throttle + brake, via `action_logits()` in `main_model.py:137-148`) for the WoR PEOC/ActionEntropy detector.
- `attribute_action` (592–631) is broken dead code (missing `narr_rgb`/`spd` arguments, computed `action_idx` unused, no callers) — finding 3.5.

## 5. Parameters & magic constants

“cfg” = settable via constructor/`atoms_config.py`; “hard” = hardcoded.

### TFV6 (`lrp_transfuser.py`)

| Constant | Value | Where | Status |
|---|---|---|---|
| ε (AttentionLinear Epsilon rule) | 1e-2 | 658 | hard; deliberate (Bug C history, §3.3) |
| `LRPMatMul.EPS` (matmul stabilizer) | 1e-6 | 92 | hard (class attribute) |
| AlphaBeta (Conv + FFN Linear) | α=1, β=0 (z⁺); (2,1) iff `uitb=True` | 591–592, 659–660 | cfg (`uitb` ctor flag; never enabled by callers) |
| First-conv rule | WSquare | 663 | hard |
| BN/LN/Activation | Pass | 654–656 | hard |
| Canonizer | `SequentialMergeBatchNorm` | 666 | hard |
| Attention scale | `d_head^-0.5` outside LRPMatMul | 184, 257 | hard (architecture) |
| ImageNet mean/std | (0.485,0.456,0.406)/(0.229,0.224,0.225); `/255` | 351–352, 357 | hard |
| `node_dim` | 256 | 609 | hard (F_c size) |
| LRP1 seed | softmax(speed_logits) | 771 | hard (Decision D) |
| forced_brake bin / forced_drive | bin 0 / argmax(bins 1–7) | 764–769 | hard (Decision E) |
| `_make_minimal_data` default cmd | 3 (clamped to [0,5]) | 1040, 1052 | hard (vs. pipeline `DEFAULT_CMD=2`, findings 1.7/2.12) |
| `_SPEED_BINS_PROXY` | [0,4,8,10,13.89,16,17.78,20] m/s | 948 | hard (duplicate of config, finding 2.4) |
| MDX-v2 throttle/brake proxies | `min(E[v]/20, 1)`; `E[v] < 0.5` | 993 | hard |
| backbone-feature clamp | `clamp(min=0)` | 944 | hard |
| helper cmd defaults | `get_speed_logits` cmd=3; `get_fc_features`/`get_planning_action_and_features` cmd=4 | 896, 953, 998 | hard (callers always pass cmd) |

### WoR (`lrp_analysis.py`)

| Constant | Value | Where | Status |
|---|---|---|---|
| AlphaBeta | (1,0); (2,1) iff `uitb` | 118–119, 183–184 | cfg (ctor) |
| `zero_params='bias'` on Linear | bias excluded | 184 | hard |
| First-conv rule | WSquare (wide stem only, §4) | 187 | hard |
| Canonizer | `ResNetCanonizer` (BN-merge part effective only) | 190 | hard |
| `min_speeds`/`max_speeds` (lerp range) | 0.0 / 8.0 m/s | 111–112 | cfg (ctor defaults) |
| `num_speeds` | 4 | 127 | hard (ignores `model_eval.num_speeds`; matches checkpoint) |
| `include_throttle` | False (steer-only drive seed; 0.5/0.5 split if True) | 113, 495 | cfg (ctor) |
| `undo_resnet_amplification` | True | 116, 133 | cfg (ctor) |
| brake threshold | `brake_prob > 0.5` | 484 | hard (mirrors `image_agent.py:296`) |
| numeric guards | 1e-12 (sum stabilizers), 1e-9 (fraction guards) | 271–276, 317–323, 337–344, 430 | hard |
| act-head layout | base=13, stride=52, 312 outputs | 459–460 | derived from model config |

Related but owned by other topics: `p_relevance`/`FC_RELEVANCE_FILTER` = 0.25 node filter and `PLOT_COMPARATIVE_REL` (ATOMs, Topic 04); `MODE_ANALYSIS` 1/2/3 dispatch (`atoms_carla.py:264-268`).

## 6. Key design decisions (verified against code)

1. **F_c = `speed_query` (Option B)** — Decision A/B in `docs/lrp_todo.md` (history in `02_agents.md`). Consequence for LRP: the whole PlanningDecoder had to become attributable → `MultiheadAttentionExplicit`/`TransformerDecoderLayerExplicit`. Implemented as documented.
2. **AttnLRP rather than gradient fallback for attention** — original code let softmax/matmul fall through to plain autograd (Bug 2); fixed with `LRPSoftmax`/`LRPMatMul`. Verified: all four GPT fusion blocks and all 6 decoder layers (self- + cross-attention) route through the custom functions.
3. **ε-rule for attention Linears, z⁺ for Conv/FFN** (Bug 3 fix) — follows the AttnLRP recommendation of ε inside attention; the paper's γ-rule suggestion for ViT Conv/FFN was implemented as AlphaBeta(1,0) (z⁺) instead, consistent with the WoR backend and the ATOMs paper's z⁺ assumption. `zero_params='bias'` removed for TFV6 per AttnLRP Remark A.2.2 (sign-flip warning); retained for WoR (classical z⁺ convention).
4. **ε=1e-2** (Bug C) — stability over fidelity: the larger ε absorbs relevance in near-singular attention Linears; quantified improvement documented in §3.3.
5. **Softmax-distribution LRP1 seed** (Decision D) — smoothness across two-hot bin boundaries; implemented at 771.
6. **Two-step `output→input`** (Bug F) — exact by linearity, avoids LRPMatMul denominator blow-up; implemented at 838–886 and regression-tested (D07).
7. **abs() of negative node weights** (Decision C) — lives in ATOMs (`atoms_carla.py:431-435`), but is a direct consequence of AttnLRP producing signed F_c relevance; preserves non-negative attention profiles for the detectors.
8. **BN canonization via SequentialMergeBatchNorm** (Issue 7) — see §3.4.
9. **Contrastive seeding** (Decision E) — `forced_brake`/`forced_drive` implemented for both agents; for TFV6 meaningful in modes 2/3 directly and in mode 1 via LRP1-reweighting of cached LRP2 maps (ATOMs side).
10. **WoR seeds mirror the deployed policy** — speed-bin lerp and the 0.5 brake threshold replicate `ImageAgent` inference exactly, so attribution explains the action actually taken.

## 7. Validation hooks (detail in `11_validation_and_testing.md`)

- `ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py` — mathematical diagnostics D01–D12: D01/D02 unit-test the `LRPSoftmax`/`LRPMatMul` backward formulas against the paper equations; D03 seed construction; D04 LiDAR gradient isolation; D05 LRP1 conservation; D06 backbone amplification budget; D07 two-step ≡ output→input consistency; D08/D09 forced-seed behaviour; D10 per-node map cosine; D11 bias fraction; D12 determinism.
- `ATOMs_Analysis/utils/tfv6_test_suite.py` — output-sanity tests L01–L07 (NaN/Inf, conservation stability CoV, spatial Gini/entropy, positive dominance with AttnLRP-adjusted thresholds, per-node distinctiveness) + ATOMs integration A01–A05.
- `ATOMs_Analysis/utils/lrp_test_suite.py` and `atoms_test_suite.py` — WoR equivalents (incl. W07 GAP-collapse cosine check, reported as WARN by design).

## 8. Known limitations / open issues

1. **Conservation is not exact end-to-end (TFV6).** Sources, all systematic: BN canonization is order-heuristic; residual additions (ResNet skips, GPT/decoder `x + f(x)`) are split by plain autograd, not an ε-rule on the sum (lrp_todo Design Issue 8 — accepted simplification); the 1/√d attention scale multiplies relevance through (finding 3.7); ε=1e-2 absorbs relevance; `F.interpolate` in `_fuse` is handled by autograd. Net residual amplification ~2×10⁷ in `fc→input`, stable across frames (CoV 0.15) and cancelled by ATOMs normalization — absolute relevance magnitudes are **not** interpretable, relative spatial/object structure is.
2. **Offline conditioning gap**: `target_point`/`acceleration` zeroed in `_make_minimal_data` → attributions can differ from the live agent's (route conditioning absent).
3. **Mode-2 seed is not LRP** in the paper sense: positive F_c *activations*, not decision relevance (documented in-code, 821–826); MODE_ANALYSIS=1/3 are the decision-faithful modes.
4. **WoR `fc→input` is architecturally uninformative** (GAP collapse) — per-node maps identical; only `output→fc` and `output→input` carry node-specific signal for WoR.
5. **Stale documentation**: `docs/design_decisions.md` still claims "no canonizer" and ε=1e-6 (finding 3.1); `CLAUDE.md` still lists "BatchNorm canonization, contrastive seeding" as *remaining open issues* although `docs/lrp_todo.md:352` marks nothing blocking and both are implemented (finding 3.2); the `lrp_transfuser.py` header claims `[B,3,384,2304]` input although offline frames are 1152 px wide (finding 2.3).
6. **Latent hazards**: NaN through `LRPSoftmax` if attention masks ever used (finding 3.6); silent activation/norm-layout mismatch if the PlanningDecoder training config changes (finding 3.8); canonizer mutates shared backbone weights inside the context (finding 3.9); WoR narrow stem conv lacks the input-layer rule (finding 3.4); zennit not version-pinned (finding 3.10).

## 9. Cross-references

- `01_architecture_overview.md` — config flags consumed here (`AGENT`, `MODE_ANALYSIS`, `PLOT_COMPARATIVE_REL`), pipeline position of profile computation.
- `02_agents.md` — TFV6/WoR architectures, F_c choice rationale, speed-bin/two-hot training target, ensemble/single-member caveat (finding 2.5), 3-vs-6-camera domain shift (finding 2.2).
- `04_atoms.md` — how `forward_relevance` outputs are consumed: node filter (p=0.25), |node_w| weighting, element selectivity, per-frame normalization, comparative-map reweighting.
- `07_distances_and_detectors.md` — consumers of the feature-extraction helpers (MDX-v1/v2, PEOC).
- `11_validation_and_testing.md` — full description of the D/L/A/W test suites referenced in §7.
- `documentation/99_bugs_and_findings.md` §"From Topic 3" — findings 3.1–3.10.
