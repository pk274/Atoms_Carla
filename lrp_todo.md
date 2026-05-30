# LRP / ATOMs Implementation Review — TODO

Review of `ATOMs_Analysis/saliency/lrp_transfuser.py` and `atoms_carla.py`
against the ATOMs paper (Beylier et al., NeurIPS 2024) and AttnLRP paper
(Achtibat et al., 2024).  Context: these files implement LRP-based saliency
and the ATOMs hierarchical-attention metric for the TFV6 (TransFuser v6) agent.

---

## Bug 1 — HIGH: LRP1 (output → FC) uses forward activations, not LRP

**File:** `ATOMs_Analysis/saliency/lrp_transfuser.py:376-386`  
**Function:** `_attribute_to_backbone`

```python
with torch.no_grad():
    pooled = self.backbone_model(rgb_x).squeeze(0)  # forward pass only
    act    = pooled.clamp(min=0)
return (act / act.sum()).cpu()
```

**Problem:** No backward pass, no zennit composite context. Returns normalised
positive forward activations as proxy node weights.

**What the paper requires (ATOMs Appendix A, Eq. 2):** LRP is backpropagated
from the action output `f(x)` to the FC layer, yielding `R_{i∈F_c}` — the
relevance of each neuron to the *decision*, not to the activation magnitude.

**Effect:** Node selection (`_relevance_filter`) and the node weights multiplied
into `self._hierarchical` in `_compute_node_level` reflect "how active is this
node" rather than "how much does this node contribute to the driving decision."
The core ATOMs two-pass logic is broken at the seam: LRP2 (pixel maps) is
correctly seeded per node, but the weights attached to those maps are wrong.

**Fix direction:** Seed LRP1 from a scalar output.  For TFV6 a reasonable
choice is the sum or max of the planning decoder output, or the logit
corresponding to the current navigation command.  Run the backward pass via
`_attribute_backbone` with the composite context, just as LRP2 already does.

---

## Bug 2 — HIGH: Attention softmax and matmul skip all LRP rules

**File:** `ATOMs_Analysis/saliency/lrp_transfuser.py:117-119`  
**Function:** `SelfAttentionExplicit.forward`

```python
scores  = torch.matmul(q, k.transpose(-2, -1)) * (hs ** -0.5)
weights = torch.softmax(scores, dim=-1)
y       = torch.matmul(weights, v)
```

**Problem:** `torch.matmul` and `torch.softmax` are raw functional ops, not
`nn.Module` instances.  Zennit registers hooks on `nn.Module.forward`, so it
never intercepts these operations.  The relevance flowing back through them uses
plain PyTorch autograd (chain-rule gradient), not any LRP rule.

**What AttnLRP requires:**
- Softmax (Proposition 3.1, Eq. 13):  
  `R^{l-1}_i = x_i (R^l_i − s_i Σ_j R^l_j)`  — Taylor decomp with bias term.
- Bi-linear matmul A·V (Proposition 3.3, Eq. 15):  
  `R^{l-1}_{ji} = Σ_p A_{ji} V_{ip} R^l_{jp} / (2·O_{jp} + ε)`  — ε + uniform rule.

**Effect:** Attribution through all 4 GPT cross-modal fusion blocks is
gradient-based, not LRP. Conservation is violated. The four GPT blocks are
the architectural core where image–LiDAR fusion happens, so this is the
most attribution-critical part of the network.

**Fix direction:** Wrap `softmax` and each `matmul` in lightweight `nn.Module`
subclasses and register custom zennit rules on them, or implement the AttnLRP
backward logic directly in `SelfAttentionExplicit` as a custom autograd Function.
The AttnLRP reference repo (`rachtibat/LRP-eXplains-Transformers`) provides
ready-made implementations that can be adapted.

---

## Bug 3 — MEDIUM: AlphaBeta applied to attention Linear layers (should be ε-rule)

**File:** `lrp_transfuser.py:295-296`

```python
(AnyLinear, AlphaBeta(alpha=self.alpha, beta=self.beta, zero_params="bias")),
```

**Problem:** `AlphaBeta(α=1, β=0)` (= z⁺-rule) is applied uniformly to **all**
`nn.Linear` layers, including the K/Q/V/proj matrices inside `SelfAttentionExplicit`.

AttnLRP recommends the ε-rule for linear layers inside the attention module.
The α-β/z⁺-rule is only recommended for CNN and FFN layers outside attention,
and only in ViTs to combat gradient shattering.

Additionally, `zero_params="bias"` silently drops bias terms from the LRP
computation.  The AttnLRP paper (Remark A.2.2) explicitly warns this can cause
sign flips in relevance scores in downstream layers.

**Fix:** Split the composite `layer_map` — use `AlphaBeta` for `nn.Conv2d` and
FFN/MLP `nn.Linear`, use `Epsilon` (with small ε) for the four attention Linear
layers (key, query, value, proj) inside each GPT block.

---

## Bug 4 — MEDIUM: Node weights are sum-normalised (deviates from paper formula)

**File:** `atoms_carla.py:124-145`, `_lrp1_nodes` (lines ~450-451)

```python
r = wide_r.abs()
return r / (r.sum() + 1e-12)   # <-- normalised to sum=1
```

Then in `_compute_node_level`:
```python
node_w = r_nodes[node_id].item()   # fractional weight, not raw R_k(x)
self._hierarchical += R_sum * node_w
```

**Problem:** The paper formula `h(og) = (1/|X|) Σ_x Σ_{k∈S} R_k(x) · R̄^k_g(x)`
uses the *raw* LRP relevance `R_k(x)`.  Normalising to sum=1 means every frame
contributes with the same total node-weight magnitude, regardless of whether
the network was highly confident or uncertain.

**Fix:** Return raw values from `_lrp1_nodes`; let `_relevance_filter` select by
absolute value; multiply raw weights into `_hierarchical`.  Final normalisation
by `get_hierarchical(normalize=True)` already handles the scale at output time.

---


## Design Issue 6 — MEDIUM: "FC layer" is backbone encoder output, not decision layer

**File:** `design_decisions.md` (Option A), `lrp_transfuser.py:13`

**Problem:** The TFV6 node space is the 512-dim globally-averaged backbone
output, produced *before* the `PlanningDecoder` (a 6-layer `nn.TransformerDecoderLayer`
that transforms backbone features into waypoints and speed predictions).

The ATOMs paper defines `F_c` as "the layer just before the output layer —
the final world-model on which the agent chooses its action."  For TFV6 this
would be the internal state of the planning decoder just before its final
linear projection, not the backbone encoder output.

**Status:** Acknowledged in `design_decisions.md` as Option A.  Option B
(256-dim target-speed query inside PlanningDecoder) is more correct but
requires wrapping `nn.TransformerDecoderLayer` for AttnLRP.

**Thesis impact:** ATOMs profiles currently describe "what the image encoder
attends to," not "what the driving decision-maker attends to."  This weakens
the claim that ATOMs detects OOD inputs via decision-relevant attention shifts.

---

## Design Issue 7 — LOW-MEDIUM: BatchNorm uses Pass without canonization

**File:** `lrp_transfuser.py:293`

```python
(nn.BatchNorm2d, Pass()),
```

**Problem:** ResNet34 has ~36 BatchNorm layers.  `Pass` (identity rule) ignores
the BN scaling factor `γ/σ`.  Any layer where `γ/σ ≠ 1` introduces a
relevance-conservation error.  Errors accumulate across the 4 ResNet stages.

**Quantitative measurement (2026-05-28, D06 diagnostic):**
- D05 (z+ through target_speed_decoder): Σ node_rel ≈ 0.983 — excellent conservation.
- D06 (backbone LRP2): Σ pixel_rel / Σ node_rel ≈ 10¹³–10¹⁷, sign-oscillating,
  CoV ≈ 2.67.  The BatchNorm Pass errors compound multiplicatively across 36 layers,
  producing relevance explosion of 13+ orders of magnitude.  This makes the
  signed Σ pixel_rel meaningless as an absolute conservation check.
  Crucially D07 (two-step consistency) passes with rel_L∞ = 0 — the
  mathematical chain is internally consistent; the explosion is inherent to
  the Pass-rule approximation, not a code bug.

**Fix:** Implement a custom canonizer that folds BatchNorm parameters into the
preceding Conv weight and bias before running LRP.  timm's ResNet34 differs
from torchvision so `zennit.torchvision.ResNetCanonizer` cannot be used
directly — it needs adaptation.

---

## Design Issue 8 — LOW: Residual connections handled by autograd

**File:** `transfuser_backbone.py:389-391`, `Block.forward`

```python
x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
```

**Problem:** For perfect LRP conservation, relevance at a residual addition
should be distributed proportionally between the skip path and the computation
path (ε-rule on the sum).  Standard autograd does chain-rule gradient instead.

**Status:** Widely accepted simplification in LRP literature; no clean zennit
fix without implementing a custom rule for elementwise addition.

---

## Design Issue 9 — LOW: `_give_element_selectivity` normalises by non-zero pixels

**File:** `atoms_carla.py:566-569`

```python
nz  = ((masks > 0) & (r_hw.unsqueeze(0) != 0)).float().flatten(1).sum(1).clamp(min=1.0)
return raw / nz
```

**Status:** This IS mathematically correct per the paper formula
(`V = |non-zero pixels in mask|`).  However, it amplifies small but intense
hotspots over large areas with diffuse coverage.  On its own it is fine; it
becomes a compounding distortion when combined with Bug 1 (wrong node weights)
and Bug 4 (normalised node weights).

---

## Priority order for fixes

1. **Bug 1** — fix LRP1 seed (highest impact, relatively small code change) ✅ DONE
2. **Bug 2** — implement AttnLRP rules for softmax/matmul (most complex) ✅ DONE
3. **Bug 3** — split composite map for attention vs non-attention Linear layers ✅ DONE
4. **Bug 4** — remove normalisation in `_lrp1_nodes` ✅ DONE
5. **Issue 6** — Option B implemented: PlanningDecoder speed_query (256-dim) as F_c ✅ DONE
6. **Bug A** — wrong seed in `beg="output", end="input"` (found 2026-05-28) ✅ DONE
7. **Bug B** — negative node weights corrupt ATOMs `_hierarchical` (found 2026-05-28) ✅ DONE
8. **Issue 7** — BatchNorm canonization ✅ DONE (2026-05-28)
9. **Bug C** — ε=1e-6 for AttentionLinear caused relevance explosion and sign oscillation ✅ DONE (2026-05-28)

**Bug C details:** ε=1e-6 in the ε-rule for Q/K/V/proj linear layers caused near-zero denominators
(LayerNorm outputs have near-zero pre-activation sums in some neurons), producing ~10¹⁴× amplification
with oscillating signs. Measured impact: pos_frac dropped to 0.45 (half of all relevance was spuriously
negative), CoV=2.67 (frame-to-frame profiles were incomparable noise). Fix: ε=1e-2. After fix:
amplification ~2×10⁷ (residual connections, systematic and cancels in ATOMs normalization),
pos_frac=0.993, CoV=0.15. ε=1e-2 is the dominant source of improvement; BN canonization reduced
amplification by an additional ~100×.

---

## Architecture decisions (discussed 2026-05-27)

### Decision A — Which layer is F_c?

| Option | Layer | Pro | Con |
|--------|-------|-----|-----|
| A (current) | 512-dim globally pooled backbone output | Simple; no AttnLRP needed for PlanningDecoder | Not the decision layer; describes image encoder, not driver |
| B | Internal state of PlanningDecoder (just before final linear projection to waypoints) | True F_c per paper; attribution explains the driving decision | Requires wrapping 6-layer nn.TransformerDecoderLayer for AttnLRP |

### Decision B — What to seed LRP1 from?

These two decisions are coupled.  The F_c choice determines what "the output"
even means for the LRP1 seed:

- **If Option A (backbone):** LRP1 seeds from the backbone output itself →
  the "seed" question collapses (it is the node activations). The current
  activation proxy becomes less wrong, but it is still not proper LRP.
  A clean seed for Option A would be e.g. the planning decoder's final output
  scalar (waypoint norm, or speed prediction) backpropagated all the way
  through PlanningDecoder + backbone to the backbone output nodes.

- **If Option B (planning decoder):** LRP1 seeds naturally from the final
  waypoint/speed output — a well-defined decision signal. This is the cleaner
  and more paper-faithful choice.

**Decision made 2026-05-27:** Option B implemented.  F_c = 256-dim speed_query
token from PlanningDecoder TransformerDecoder.  LRP1 seeds from
`target_speed_decoder(speed_query).argmax` class one-hot.

### Decision C — abs() on negative LRP1 node weights (decided 2026-05-28)

AttnLRP (softmax + matmul rules) can produce negative F_c relevances.  The
paper's formula `h(o) = Σ_x Σ_k R_k(x) · R̄^k_g(x)` was designed assuming
z+-only LRP (all R_k ≥ 0).  With AttnLRP, negative node weights fed raw into
`_hierarchical` would:

- Make ATOMs attention profiles non-monotone and potentially negative
- Break downstream detectors (Mahalanobis, GMM) that assume non-negative
  "attention distribution" inputs

**Decision:** Take `abs(node_w)` in `_compute_node_level`.  This is consistent
with `_relevance_filter` which already selects nodes by absolute mass, and
preserves non-negativity of ATOMs profiles.  A node with negative LRP1
relevance is interpreted as "actively relevant (in the inhibitory direction)"
and its pixel map is still counted with positive weight.

### Decision D — LRP1 seed: softmax distribution (decided 2026-05-28)

`target_speed_decoder` was trained with a two-hot target (energy split across
two adjacent bins).  Argmax seeding is discontinuous at bin boundaries: two
frames with essentially the same predicted speed but with mass distributed
differently across adjacent bins would get different LRP1 node relevances.

**Decision:** use `grad_outputs = softmax(speed_logits.detach())` in both
`_attribute_to_fc` and `_attribute_true_output_to_input`.  This weights each
bin by the model's confidence, gives a smooth attribution for the full
predicted speed distribution, and eliminates bin-boundary noise.

---

## Implementation summary (2026-05-27 / 2026-05-28)

Files changed:
- `ATOMs_Analysis/saliency/lrp_transfuser.py` — full rewrite + second-pass fixes:
  - `LRPSoftmax` / `LRPMatMul` custom autograd Functions (AttnLRP Props 3.1 & 3.3)
  - `AttentionLinear` marker subclass + ε-rule in composite
  - `SelfAttentionExplicit` updated to use LRP-aware ops (fixes Bug 2 for GPT blocks)
  - `MultiheadAttentionExplicit` + `TransformerDecoderLayerExplicit` (fixes Bug 2 for PlanningDecoder)
  - `TFv6FullModelForLRP` wraps backbone + PlanningDecoder for single-context LRP
  - `LRPTFv6Model` rewritten: proper LRP1/LRP2, split composite, 256-dim node space
  - `_attribute_true_output_to_input` — true output→input LRP through `target_speed_decoder` (Bug A fix)
  - `_attribute_fc_to_input` — renamed from `_attribute_output_to_input`; positive F_c seed, mode 2 only
- `ATOMs_Analysis/saliency/atoms_carla.py`:
  - `process_frame` accepts `data: Optional[dict] = None`
  - `_lrp1_nodes` returns raw relevances (no normalization)
  - `_compute_node_level`: `node_w = abs(r_nodes[node_id].item())` (Bug B fix)
- `run_analysis.py` / `run_online_analysis.py`:
  - `LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder, ...)`
- `ATOMs_Analysis/utils/tfv6_test_suite.py`:
  - Updated comments for 256-dim node space

## Bug F — HIGH: Single-pass output→input caused LRPMatMul explosion (fixed 2026-05-28)

**Symptom:** L02 ratio ~10^15, A02 contributions ~−10^11.

**Root cause:** The original `_attribute_true_output_to_input` used
`outputs=speed_logits, inputs=[rgb_x]` — a single backward through both
`target_speed_decoder` AND `transformer_decoder` AND backbone.  In the decoder
cross-attention, `LRPMatMul` divides by `2·O + ε` where `O = A·V`.  When
attention weights A are nearly uniform and V values cancel, `O ≈ 0` and
`denom ≈ ε = 1e-6`.  With 12 LRPMatMul ops (6 decoder layers × 2 matmuls),
amplification of `R / 1e-6` per op cascades to ~10^15.

The LRP2 one-hot path (`beg="fc"`) was stable because it seeds from `speed_query`
(not `speed_logits`), so the backward goes from `speed_query → rgb_x` directly —
this ALSO goes through transformer_decoder but the sparse one-hot seed doesn't
trigger the cancellation instability.  The distributed seed (from AlphaBeta
backprop through `target_speed_decoder`) does.

**Fix:** Two-step backward in `_attribute_true_output_to_input`:
1. LRP1 (stable): `speed_logits → speed_query` via `target_speed_decoder` only.
   Yields R_k ≥ 0 (AlphaBeta z+, positive softmax seed).
2. LRP2 (stable): `speed_query → rgb_x` seeded with R_k.

Because `autograd.grad` is linear in `grad_outputs`, the result is exactly
`Σ_k R_k · pixel_map_k` — ATOMs mode 1 in a single backward pass.

**L04 test threshold** also updated: z+-only thresholds (0.55/0.70) replaced with
AttnLRP thresholds (FAIL < 0.30, WARN < 0.45) since LRPSoftmax produces signed
output even from a positive seed.

---

Remaining:
- Nothing blocking. Issue 7 resolved 2026-05-28.

---

## Decision E — Contrastive seeding via PLOT_COMPARATIVE_REL (decided 2026-05-28)

`PLOT_COMPARATIVE_REL` is now supported for TFV6 via `forced_brake` /
`forced_drive` flags in `forward_relevance`.

Seed choices:
- `forced_brake=True`  → one-hot at bin 0 (0 m/s, stop)
- `forced_drive=True`  → one-hot at the highest-probability non-brake bin
                          (argmax of logits[1:])
- neither               → default softmax distribution

`is_brake` (4th return value) now reflects the model's actual argmax prediction
(bin 0 = True), independently of which forced flag is set.

**Sign note on the difference map `R_drive − R_brake`:** negative values are
correct and intended — they indicate pixels that promote braking more than
driving.  Both individual maps can themselves contain negative values with
AttnLRP (unlike z+-only LRP for WoR where maps are non-negative).

**Not supported for TFV6 mode 1 (node-level, beg="fc"):** `forced_brake/drive`
have no effect on the per-node LRP2 pixel maps for TFV6 because those seed at
a specific F_c node one-hot regardless of the output.  When `PLOT_COMPARATIVE_REL`
is active in mode 1, the stored brake and drive pixel maps will be identical to
the default map for TFV6 (three identical LRP2 backward passes are run; the
difference map will be all-zero).  For WoR the forced flags do affect the LRP2
seed (different action head seeds), so comparative maps are meaningful there.
The contrastive is most useful for TFV6 in mode 3 (beg="output", end="input"),
where brake and drive seeds produce genuinely different pixel maps.
