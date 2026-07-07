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

**Extension — GMM cluster count K also selected on val (implemented 2026-06-09):**

The original design left GMM K selected by BIC on the baseline, which is clean. However, the
intended sweep workflow (`sweep_clusters.py` + manual inspection of `summarize_results.py`) had
users picking K by test-set AUROC — leakage. Two additional problems existed:

1. `sweep_clusters.py` passed `--gmm-k {K}` to `run_analysis.py`, but `run_analysis.py` had no
   argparse, so the flag was silently ignored and every sweep run used `conf.NUM_GMM_CLUSTERS`.
2. Val profiles and `val_labeled.npz` existed on HPC but were not yet used for GMM K selection.

**Changes (2026-06-09):**
- `run_analysis.py` — added argparse for `--gmm-k`; K resolution order is now: CLI arg → `conf.NUM_GMM_CLUSTERS` → BIC. Step 9.5 extended: after GMM is fitted, scores all 5 GMM detector variants on val profiles and computes their mean AUROC (`__val_auc_gmm_avg__`), written to `summary.json`. Skipped gracefully when val files are absent.
- `summarize_results.py` — `load_summary` reads `__val_auc_gmm_avg__`; `Run` dataclass gains `val_auc_gmm_avg` field; Section 3 of SUMMARY.md gains a **Val-set K selection** table and a recommendation blockquote identifying the best K by val AUROC. The old test-set aggregate table is retained but labelled as not suitable for reporting.
- `atoms_config.py` — comment on `NUM_GMM_CLUSTERS` updated to note CLI override.

**Correct sweep workflow after these changes:**
1. Download `val_labeled.npz` + `val_profiles_{mode}.npy` from HPC to `data/TFV6/val_data/`.
2. `python sweep_clusters.py --k-values 3 5 7 9 11 13 15` — each run now computes and stores `__val_auc_gmm_avg__`.
3. `python summarize_results.py` — SUMMARY.md Section 3 shows the recommended K.
4. Report test results from the `{best_K} clusters/` snapshot folder.

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

**Update (2026-07-01): the BatchNorm gap above was closed** — `_create_composite`
now passes `canonizers=[SequentialMergeBatchNorm()]` (a generic, type-agnostic
canonizer that matches Conv/Linear→BatchNorm pairs by structural adjacency, not
by ResNet-specific class, so it works for timm's `BasicBlock` without adaptation;
see `docs/lrp_todo.md` Issue 7).

**The claim "residual additions in ResNet and GPT are handled automatically by
autograd" above was WRONG and has been retracted** — see the dedicated entry
below ("TFV6 LRP: residual/skip-connection conservation").  "Handled by
autograd" was true only in the sense that autograd computes *something*; that
something silently duplicates relevance at every residual junction rather than
conserving it, since `d(a+b)/da = d(a+b)/db = 1`.

---

## TFV6 LRP: residual/skip-connection conservation (2026-07-01)

**Problem, found while reviewing Otsuki et al. 2024 ("Layer-Wise Relevance
Propagation with Conservation Property for ResNet", arXiv:2407.09115) against
this codebase.** Every `+` in this model that joins a skip/residual branch
with a computed branch — the ResNet34 `BasicBlock` skip connections (image
*and* lidar encoders), the GPT fusion blocks' two residual streams, the
backbone's cross-modal fuse (`image_features + img_out`), and the
PlanningDecoder's self-attn/cross-attn/FFN residuals — was a raw tensor `+`,
never intercepted by any LRP rule. Standard autograd differentiates `z = a+b`
as `dz/da = dz/db = 1`, so the *full* upstream relevance is copied to *both*
branches instead of being split between them. With ~50+ such junctions
between the input and F_c, none of them conserved relevance, and the
duplication compounds across every one.

This is exactly the failure mode Otsuki et al. formalize and fix for plain
ResNets (their "Relevance Splitting" at the point where a skip connection
reconverges with its residual block). Their result: fixing it doesn't just
restore the conservation *property* — it materially changes attribution
*quality* (Insertion/Deletion/ID score in their Table 1/2), because the
duplication factor is not spatially uniform (it depends on how many
skip-hops vs. conv-hops a given feature's path takes), so it distorts the
*shape* of the map, not just its scale. Since ATOMs' whole premise is that
profile *shape* carries the OOD signal, this was a real confound, not a
cosmetic one.

**Also found while investigating:** WoR's `lrp_analysis.py` uses
`zennit.torchvision.ResNetCanonizer`, whose docstring implies it "handles
BasicBlock skip connections" — but `ResNetCanonizer` matches via
`isinstance(module, torchvision.models.resnet.BasicBlock)`, and WoR's
backbone (`pcla_agents/wor/common/resnet.py`) defines its **own** standalone
`BasicBlock`/`Bottleneck` classes (not subclassing torchvision's), so the
isinstance check silently fails and the canonizer is a no-op there. WoR is
out of scope for this fix (per user direction, 2026-07-01) but the same class
of bug is worth remembering if WoR is revisited.

**Fix — `LRPResidualAdd`** (`ATOMs_Analysis/saliency/lrp_transfuser.py`): a
`torch.autograd.Function` implementing the paper's **Ratio-Based Relevance
Splitting** (Eq. 5) directly, rather than routing through zennit's
`Sum`+`Norm`/`Epsilon` machinery:
```
forward(a, b)  = a + b                              # unchanged forward value
backward(R)    = (R * |a| / (|a|+|b|+eps),
                   R * |b| / (|a|+|b|+eps))          # R_a + R_b == R
```

**Why absolute value, not zennit's signed `Norm` rule (`a/(a+b+eps)`):**
zennit's own `ResNetCanonizer` pairs its `Sum` wrapping with the `Norm` rule
in `layer_map_base()`, which uses the *signed* forward value `z=a+b` as the
denominator. That collapses when `a` and `b` are similar magnitude but
opposite sign (`a+b ≈ 0` while `|a|+|b| ≫ 0`) — division by a near-zero
signed denominator explodes. This is the exact same failure class already
hit once in this file: `LRPMatMul.EPS` uses `2*O + eps*sign(O)` for the same
reason, and `docs/lrp_todo.md` "Bug C" documents a real ~1e14× explosion from
a near-zero ε-rule denominator. Using `|a|+|b|+eps` in the denominator can
only be small when *both* branches are near zero — never from sign
cancellation — so it inherits none of that instability. Confirmed by
diagnostic D13 (`tfv6_lrp_diagnostics.py`), which includes a
near-cancellation case (`a≈-b`, large magnitude) that a signed ratio would
blow up on (~1e6×) but `LRPResidualAdd` keeps bounded (~0.5).

**Scope of the fix — every residual junction, not just ResNet Bottlenecks.**
The paper only discusses literal ResNet Bottleneck/BasicBlock modules, but
Ratio-Based Splitting is architecture-agnostic: given any `z = a+b`, split by
`|a|`/`|b|` regardless of what the branches represent. Applied at:
1. **ResNet34 `BasicBlock`** (`_basic_block_forward_explicit`, both
   `image_encoder` and `lidar_encoder` — both are `resnet34`). timm's
   `BasicBlock.forward` is reimplemented faithfully (conv1→bn1→drop_block→
   act1→aa→conv2→bn2→se?→drop_path?→**add**→act2) with only the add
   replaced, matching timm's actual source (fetched from
   `timm/models/resnet.py`, `timm>=1.0.0` per `hpc/requirements_hpc.txt`).
2. **GPT fusion `Block`** (`_gpt_block_forward_explicit`): both
   `x + self.attn(...)` and `x + self.mlp(...)`.
3. **Backbone cross-modal fuse** (`TFv6FullModelForLRP._fuse`):
   `image_features + img_out`, `lidar_features + lid_out`.
4. **PlanningDecoder** (`TransformerDecoderLayerExplicit.forward`): the
   self-attn, cross-attn, and FFN residual adds (3 per layer × 6 layers).

**Deliberately left untouched — `pos_emb + token_embeddings` (GPT) and
`scores + attn_mask` (MultiheadAttentionExplicit).** These are not two
*competing* computed branches; one side is a fixed parameter (positional
embedding) or a constant/unused mask (never non-None in this codebase's call
sites), analogous to a Linear layer's bias — under z+/AlphaBeta, biases
already don't "compete" for relevance the way a genuine input branch does.
Splitting relevance there would misattribute mass to a dead-end that has no
path back to the input pixels anyway (irrelevant for `d(output)/d(rgb_x)`,
since autograd only traverses the subgraph needed for the requested input).

**Why `image_encoder`/`lidar_encoder` are now deep-copied.** They previously
aliased the live driving-agent backbone (`self.image_encoder =
backbone.image_encoder`, "shared, read-only" per the old comment). Patching
`BasicBlock.forward` in place would have changed inference for the live
agent, not just LRP. They are now `copy.deepcopy`d in
`TFv6FullModelForLRP.__init__` before patching — consistent with
`transformers` and `planning_decoder`, which were already deep-copied for
the same reason. The patch only changes *backward* behavior
(`LRPResidualAdd.forward` still returns `a+b` unchanged), so
`get_backbone_features()` and other no-grad forward-only paths through the
copies remain numerically identical to going through the original backbone.

**Guard against a silent no-op repeat of the WoR bug.**
`_patch_resnet_basic_blocks` returns the number of blocks it patched, and
`TFv6FullModelForLRP.__init__` asserts this is `> 0` for both encoders. A
future timm version renaming/restructuring `BasicBlock` would fail loudly
here instead of silently reproducing exactly the isinstance-mismatch bug
found in WoR's `ResNetCanonizer` usage above. The same function also raises
if it finds any `Bottleneck` block (resnet50+), since that has a different
forward structure not yet covered.

**Verification.** D13 (`tfv6_lrp_diagnostics.py`) unit-tests the
`LRPResidualAdd` formula and conservation with no model required. The
existing D06 diagnostic ("backbone amplification budget", `Σpixel_rel /
Σnode_rel`) should be re-run on the HPC after this change — it was
previously documented (`docs/lrp_todo.md` "Bug C" note) as showing a stable
but large (~2×10⁷×) amplification attributed to residual connections; this
fix is expected to reduce that substantially. Not yet confirmed against a
live model/GPU (this change was authored and reviewed on a machine without
CUDA/timm installed) — run `TFV6LRPDiagnostics.run_all_tests()` (D01–D13) on
HPC before trusting new baseline/test profiles produced with this code.

Files changed: `ATOMs_Analysis/saliency/lrp_transfuser.py` (new
`LRPResidualAdd`, `_basic_block_forward_explicit`,
`_patch_resnet_basic_blocks`, `_gpt_block_forward_explicit`;
`TFv6FullModelForLRP.__init__`, `_fuse`, `TransformerDecoderLayerExplicit.forward`
updated), `ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py` (new D13).

### HPC verification (2026-07-01) and what the numbers mean

Ran via `hpc/submit_tfv6_lrp_diagnostics.sh` on Viper (CPU-only), 8 baseline
frames. Result: **12 PASS, 1 WARN (D10), 0 FAIL.** Two numbers changed
dramatically from the pre-fix documented baseline and are worth interpreting
rather than just noting as "different":

**D06 (backbone amplification budget) — `Σpixel_rel / Σnode_rel` went from a
documented ~2×10⁷× (pre-fix, `lrp_todo.md` "Bug C" note) to ~1.07×10⁻⁴×
(post-fix), stable across 8 frames (`CoV=0.19`, comparable to the pre-fix
`CoV=0.15`).** This is interpreted as the fix working correctly, not a
regression: `LRPResidualAdd` is provably exact-conserving at each junction
(`R_a+R_b=R_z`, unit-tested by D13), so it cannot be *creating* the previous
inflation — the ~2×10⁷ number could only have come from the old bug
duplicating (not splitting) relevance at ~50+ junctions, which compounds
multiplicatively. What's visible now is very likely the *true* behavior of
z⁺/AlphaBeta LRP propagated through a network this deep (ResNet34 ×2 + 4 GPT
blocks + 6-layer decoder): these rules are known to lose relevance through
negative-activation clipping at every layer, and across dozens of layers
that compounds into severe attenuation. The old duplication bug was masking
this by continuously re-injecting relevance that should have been split, not
copied. **Net conclusion: pixel-level LRP2 maps for TFV6 carry roughly four
orders of magnitude less relevance "mass" than the F_c-level seed they were
computed from.** This does not by itself invalidate ATOMs profiles (which
are re-normalized downstream — see `NORMALIZE_BY_PIXEL_COUNT` /
`get_hierarchical(normalize=True)`), since only the *relative* spatial
distribution matters for the final profile, not the absolute scale. It does
mean the SNR of any given pixel map is much smaller than it visually
appeared to be before, and reinforces that this network's LRP is right at
the edge of vanishing under z⁺/AlphaBeta.

**D10 (per-node pixel-map diversity) — mean pairwise cosine 0.994 across 8
probed F_c nodes (WARN, not FAIL: not literally 1.0, so node routing itself
is not broken — see D10's own FAIL/WARN distinction).** A plausible
mechanism, consistent with the "shape not just scale" warning above: after
the first ReLU, every `BasicBlock`'s skip/shortcut value is already
non-negative and typically larger in magnitude than the AlphaBeta-processed
conv branch's output. Since `LRPResidualAdd` splits by `|a|`/`|b|`, more
relevance rides the skip path than the node-specific conv path at every one
of the ~16 BasicBlocks per encoder — and the skip path carries approximately
the same signal regardless of which F_c node seeded the LRP2 backward pass.
That would directly produce more self-similar per-node maps than before.
Consistent with `pos_frac_mean` also rising (was ~0.993 post-Bug-C,
pre-residual-fix; now ~0.9998): skip-path values are structurally
non-negative, so a map dominated by that path skews further toward positive.
**Caveat:** D10 only ever tests frame 0 regardless of how many frames are
loaded (`_d10_node_cosines` hardcodes `frames["wide_rgb"][0]`) — this has
not yet been checked across multiple frames or node subsets.

**Candidate levers against the attenuation (2026-07-02, untested).**
`LRPTFv6Model` gained two constructor flags, both mirroring options already
used by WoR/LBC or already partially wired but never exposed for TFV6, so
they can be A/B-tested via `--uitb`/`--zero-bias` on
`tfv6_lrp_diagnostics.py` (or `UITB=1`/`ZERO_BIAS=1` on
`hpc/submit_tfv6_lrp_diagnostics.sh`) without further code changes:

- `zero_bias=True` → `zero_params='bias'` on the `AnyLinear` AlphaBeta rule,
  matching WoR/LBC's composite (TFV6 never had this — an unintentional
  inconsistency, not a deliberate choice). Expected impact: **small**. D11
  already measures `target_speed_decoder`'s bias absorption at only 1-4%,
  and `Convolution` layers use `bias=False` by ResNet convention (standard
  practice: bias before BatchNorm is redundant), so this cannot touch the
  majority of the network at all — it only affects the GPT fusion `mlp`
  Linears, the PlanningDecoder FFN Linears, and `target_speed_decoder`.
- `uitb=True` → `AlphaBeta(2,1)` instead of `(1,0)` for `Convolution` *and*
  `AnyLinear` (existing constructor flag, already used this way for
  WoR/LBC, just never passed for TFV6). Expected impact: **plausibly
  larger**. `(1,0)` is pure z⁺ — it fully discards the negative-activation
  contribution at *every* Conv/Linear layer in the ~50+-layer backbone.
  `(2,1)` retains some of that (subject to α−β=1 for conservation), which is
  the more likely dominant lever given AlphaBeta's clipping (not bias) is
  the standing hypothesis for where the ~4-orders-of-magnitude attenuation
  actually comes from.

Bigger, riskier options *not* implemented yet, for if neither of the above
moves D06's ratio meaningfully: ε-rule for Conv/FFN instead of AlphaBeta
(closer to exact per-layer conservation, but risks reintroducing the
near-zero-denominator sign-oscillation class of bug already hit once — "Bug
C" — so would need the same kind of careful ε tuning); zennit's `Gamma` rule
as a middle ground between z⁺ and signed ε; or a depth-dependent rule mix
(γ/AlphaBeta(2,1) for early/middle backbone layers, ε/z⁺ only near F_c),
which is the general guidance in Montavon et al. 2019 ("LRP: An Overview")
for very deep networks but is a bigger redesign than a constructor flag.

**Verdict (2026-07-02, HPC-verified): neither flag fixes the attenuation —
`uitb` is actively counterproductive.** Ran the diagnostics (with the abs-
value D06 ratio added, see below) on the same 8 frames for `default`,
`zero_bias=True`, and `uitb=True`:

| Config    | signed ratio | abs ratio | pos_frac | D08 brake/drive cosine |
|-----------|--------------|-----------|----------|-------------------------|
| default   | 1.03e-4      | 1.03e-4   | 0.9998   | ~0.95 (baseline)        |
| zero_bias | ~unchanged   | ~unchanged| ~unchanged | ~unchanged            |
| uitb      | 1.98e-5      | 6.56e-5   | 0.6469   | dropped to ~0.25        |

`zero_bias` behaved exactly as predicted from its scope (§ above, ≤ a
handful of Linear layers): D05 conservation goes to exactly 1.0000, D06/D08/
D10 essentially unchanged. Safe to enable for AttnLRP-standard rigor (bias
exclusion is what WoR/LBC already does), but it was never going to be — and
isn't — a fix for the attenuation.

`uitb`'s signed ratio (1.98e-5) initially looked *worse* than default's
1.03e-4. Adding the abs-value ratio (`Σ|pixel_rel| / |Σnode_rel|`, immune to
sign cancellation) resolved the ambiguity: uitb's abs ratio is 6.56e-5 —
still lower than default's 1.03e-4, not higher. So correcting for the fact
that `pos_frac` collapsed from 0.9998 to 0.647 (i.e. `AlphaBeta(2,1)` lets
much more negative relevance survive to the pixel layer, which is what
shrank the signed number), the *true magnitude* reaching the pixel layer is
not better under `uitb` — it's roughly 1.6× worse, not better. It also comes
with two real costs beyond the raw ratio: pixel maps are now ~35%
negative-mass (closer to a cancellation-noisy saliency map than a clean
attention map), and D08's brake-vs-drive map distinctiveness collapsed from
~0.95 to ~0.25 cosine — the extra negative-branch relevance swamps the very
seed-dependent signal ATOMs needs to tell forced-brake and forced-drive
attention apart. **Conclusion: reject `uitb=True` for TFV6; keep
`alpha=1, beta=0` (pure z⁺) as the default.** `zero_bias=True` can be
adopted independently since it is free and harmless, but it is not doing any
of the load-bearing work here.

D10's node self-similarity (~0.994-0.997, confirmed stable across 4 frames)
is unmoved by either flag, ruling out AlphaBeta parameterization as its
cause — further evidence for the skip-path-dominance mechanism already
hypothesized (`LRPResidualAdd` splits by raw `|a|`/`|b|` magnitude, not by
which branch carries F_c-node-specific information, so the structurally
larger post-ReLU identity/skip branch wins regardless of the conv branch's
rule). That is a `LRPResidualAdd` question, not a composite-tuning one, and
is out of scope for this experiment.

The remaining ~1e-4 abs-ratio attenuation therefore looks like a genuine,
rule-choice-insensitive property of AlphaBeta/z⁺ propagated through a
network this deep, not fixable by either flag tested. Given `default`'s abs
ratio is stable (CoV 0.15), nearly pure-positive (pos_frac 0.9998), and
doesn't hurt D08 discriminability, the practical recommendation is to *not*
chase the bigger options (ε-rule, Gamma, depth-dependent mixing) blind —
recompute baseline/test/val profiles with the current fixed code first and
re-run the OOD-AUC sweep, and only pursue further rule redesign if that run
shows a regression traceable to pixel-level signal-to-noise.

**Practical implication — recompute, don't just trust the old data.** All
`baseline_{mode}.npz`, `test_profiles_{mode}.npy`, `val_profiles_{mode}.npy`,
etc. currently on disk were computed *before* this fix, with the previous
(duplicating) residual handling. Given the scale of the D06/D10 changes,
these should be treated as stale for any conclusion beyond "the pipeline
used to run" — re-run the baseline/test/val HPC pipelines with this code
before drawing new OOD-AUC conclusions, especially before comparing against
the WOR numbers in `ood_sweep_findings` memory.

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

### PGD hyperparameter selection: ε=4, 5 steps, target=brake
Empirically determined by running the brake-target attack across a range of ε values on
representative test frames. At ε=12 (prior setting) the attack was already converging
within 2 steps; even at ε=4 with 5 steps the attack drives P(bin 0) to ≥99.9% on
virtually all frames. Using ε=4 is preferable for the thesis because:
- The perturbation is imperceptible at ε=4/255 ≈ 0.016 per channel, which more clearly
  demonstrates that adversarial inputs are visually indistinguishable from clean ones.
- A weaker budget still achieves near-100% success, showing the model is sensitive
  rather than merely overwhelmed by a large perturbation.

The `brake` target (maximise P(speed bin 0 = 0 m/s)) was chosen over `steer_right`
because it produces a measurable scalar output (the speed distribution) that is easy to
visualise and interpret in the thesis, and because the binary "stopped/not stopped"
outcome maps cleanly onto a safety-relevant failure mode.

`run_analysis.py` now reports a **PGD success rate** (count of test frames where
softmax(speed\_logits)[0] ≥ 99.9%) printed at step 9 and saved to `summary.json` as
`__pgd_success__`. Updated in: `atoms_config.py`, `hpc/prep_test.py`,
`hpc/compute_test_chunk.py`, `hpc/array_test_task.sh`, `hpc/prep_test_task.sh`.

### Brightness scale factor: 4 → 3
`BRIGHTNESS_INTENSITY` reduced from 4 to 3 (i.e. 3× pixel multiply, clipped to 255).
At factor 4 the image was almost entirely saturated to white; factor 3 still produces a
clearly visible over-exposure artefact while retaining slightly more scene structure.
Updated in: `atoms_config.py`, `hpc/prep_test.py`, `hpc/prep_test_wor.py`,
`hpc/prep_test_task.sh`, and all documentation files.

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


---

## Alternative Same-Distribution Data Split (`EXPERIMENT_VARIANT`)

### Motivation

The original split (non-Town05 baseline, Town05 test/val) makes test frames OOD by construction due to domain shift (different road geometry, visual character) independent of any applied perturbation. This confounds evaluation: a detector that merely recognises "novel town" would score well without detecting perturbations at all.

### Design

`atoms_config.py` exposes `EXPERIMENT_VARIANT = "original" | "alternative"`. When set to `"alternative"`, all four path variables (`BASELINE_DATA_DIR`, `TEST_DATA_DIR`, `VAL_DATA_DIR`, `RESULTS_DIR`) resolve to `*_data_alt` / `results_alt` counterparts under the same `_DATA_ROOT`. No analysis scripts need changes — they read paths exclusively from config.

### Data split strategy

`migrate_lead_to_baseline.py --mode alt_split`:
1. Discovers all routes across all towns (Town05 included by default in the alt split).
2. Shuffles routes deterministically with `conf.RANDOM_SEED` via `np.random.default_rng`.
3. Splits at the **route level** (whole routes go to one split, never both) using proportional slicing: first `baseline_n / (baseline_n + test_n + val_n)` fraction → baseline, next slice → test, remainder → val. Default: 5000 / 1000 / 1000.
4. Writes all three sets in one invocation using `_build_plan_from_routes` (even sampling across the pre-assigned route list, no per-town balancing).

Route-level splitting prevents leakage from temporally-correlated frames within a route.

### Invariant

No filename overlap between `baseline_data_alt/frames/`, `test_data_alt/frames/`, and `val_data_alt/frames/` — guaranteed by construction (each shuffled route appears in exactly one slice).

---

## GMM nearest-cluster assignment: Mahalanobis for all detectors (2026-06-18)

All GMM-based detectors now determine the nearest cluster via **Mahalanobis distance**
rather than each detector's own metric (Euclidean / JSD / Wasserstein) or raw L2 to
centroids (GMM k-NN).

**Rationale:** Euclidean distance to centroids treats all dimensions equally, so it is
biased toward large-variance clusters (they look closer in raw space).  Mahalanobis
normalises by each cluster's covariance, making assignment geometry-aware: a cluster
whose variance is small in a given direction acts as a tighter attractor for nearby
points, which is the correct behaviour for a GMM.

For JSD and Wasserstein the old approach found the cluster minimising the distance
under each metric — that is self-consistent but unintuitive because the "nearest
cluster" depended on the scoring function rather than on the underlying geometry.
Decoupling assignment (Mahalanobis) from scoring (JSD / Wasserstein) makes all
detectors comparable: they all see the same cluster partition and differ only in
how they measure distance to the assigned cluster's mean.

**Changes:**
- `DistanceComputer._nearest_cluster_mahal(means, covariances, target, regularization)` —
  new private helper; returns the index of the Mahalanobis-nearest cluster.
- `compute_gmm_euclidean`, `compute_gmm_jsd`, `compute_gmm_wasserstein` — signatures
  extended with `covariances` and `regularization`; cluster selection delegated to
  `_nearest_cluster_mahal`.  Return value semantics unchanged (score against nearest mean).
- `run_analysis.py` — GMM k-NN centroid loop (test + val) replaced explicit
  `np.linalg.norm` with `_nearest_cluster_mahal`; all four call sites of
  `compute_gmm_euclidean` / `compute_gmm_jsd` updated to pass `gmm.covariances_` and
  `conf.MAHAL_RIDGE`.
- `compute_gmm_distance` (Mahalanobis-GMM) was already correct; no change.

## Brake-counterfactual profile block — `ADD_BRAKE_SEEDS` (2026-07-02)

**Problem.** The default LRP seed is `softmax(speed_logits)` — every ATOMs
profile answers "what determines how fast I should go?".  For clean driving
the honest answer is the road corridor and the lead vehicle, so relevance
almost never reaches lateral/rare classes (Pedestrian, Biker, StopSign ≈ 0 in
the baseline bar chart).  Constant profile dimensions carry no OOD signal, so
the detector effectively works with far fewer than 10 dimensions.

**Decision.** Add a second profile block from a *counterfactual* seed:
one-hot at speed bin 0 (0 m/s = stop), backpropagated output→input via the
existing stable two-step scheme (`forced_brake=True` in `forward_relevance`,
Decision E infrastructure).  Interpretation: **the evidence the model
currently sees in favor of stopping** — a standard class-conditional
attribution (in classification XAI, explaining a non-predicted class is
routine).  A perturbation corrupts the model's evidence landscape for *all*
actions, not only the chosen one; monitoring the inhibitory evidence widens
the observable decision spectrum without touching the LRP rules.  Note the
PGD attack targets exactly this logit (`PGD_TARGET="brake"`), so the brake
block is the attribution of the attacked output.

**Mechanics.**
- `ATOMsCarla` gains `profile_dim = num_classes × n_blocks` and
  `profile_names` (`class_names` + `Brake:*`).  The brake pass runs after the
  mode-specific default computation in *every* `MODE_ANALYSIS` — the
  counterfactual is defined at the output level, so it always uses
  output→input regardless of mode.
- **Block-wise normalization:** each block is normalized to sum `1/n_blocks`
  (`_normalize_profile`), so the full profile still sums to 1 (valid input
  for JSD/Wasserstein) and neither block's raw LRP magnitude can drown the
  other.  Detectors are dimension-agnostic and need no changes.
- **Cost:** +1 full LRP backward per frame (≈ one forward).  Offset by a
  same-commit optimization: the mode-2 `PLOT_COMPARATIVE_REL` forced passes
  are *provably identical* to the default map for TFV6 (fc seeds cannot
  depend on the output — Decision E), so they are now mirrored instead of
  recomputed.  Net mode-2 cost went from 3 backward passes per frame to 2
  while gaining the brake block.  Runtime stays a small constant multiple of
  a single inference — compatible with online monitoring.
- The brake pass also populates `saliency_data_wide_brake` with a genuinely
  brake-seeded map (the old mode-2 mirror could not differ from the default),
  making the comparative visualization meaningful in mode 2.
  `plot_saliency_examples.py` saves a `_brake.png` per sampled frame.

**Validation hook.** The LEAD metas contain per-frame hazard flags
(`walker_hazard`, `vehicle_hazard`, `light_hazard`, `stop_sign_hazard`) —
free ground truth to check that brake-seeded relevance concentrates on the
hazard class on hazard frames.

## Real target-point conditioning — `USE_REAL_TARGET_POINTS` (2026-07-02)

**Problem.** `_make_minimal_data` zeroed all three target-point tokens, but
the deployed TFV6 (leaderboard mode) conditions on current/previous/next TP
(`use_tp`, `use_previous_tp`, `use_next_tp` all True).  Offline attributions
therefore ran with degenerate "target at ego position" route conditioning.

**Decision.** The LEAD meta pickles turn out to contain the full
route-planner state per pop distance.  `migrate_lead_to_baseline.py` now
reproduces the **training dataloader recipe exactly** (`carla_dataset.py`):
keys `next/previous_target_points_3.25` (`TP_POP_DISTANCE = 3.25` ==
`TrainingConfig.tp_pop_distance`), duplicate-merge, prev/current/next
indexing, ego transform via `inverse_conversion_2d(point, pos_global[:2],
theta)`.  No TP augmentation — the clean `rgb/` stream has perturbation 0
(`carla_dataset.py:127`).  Verified by re-projecting the extracted ego-frame
TP back to world coordinates: exact match with the raw meta entry.

- npz schema gains `target_point/target_point_previous/target_point_next
  [N,2] float32`; the whole pipeline (BaselineDataLoader, PerturbationApplier,
  prep_test.py, all HPC chunk scripts, run_analysis step 9) passes them
  through to `_make_minimal_data`.  The PGD attack in `compute_test_chunk.py`
  now also crafts under the same TP conditioning as the profile pass.
- **cmd source fix:** when TP extraction succeeds, `cmd` comes from the
  filtered `next_commands_3.25` list (training-faithful).  The unsuffixed
  `next_commands` reflects a different pop state and can genuinely differ
  (measured: RIGHT vs LANEFOLLOW at a junction frame).
- Old npz without TP keys fall back to zeros with a one-time warning.
- First single-frame measurement (junction frame, real model): the
  speed-seeded profile changed only marginally vs zero TPs (L1 ≈ 3e-4) — the
  conditioning fix is about correctness/deployment-faithfulness; its OOD-AUC
  impact is an open empirical question.

## Waypoint-head seeding + seed-invariance of TFV6 LRP maps (2026-07-02)

**Proposal (user).** Instead of the output-level brake counterfactual, seed at
the waypoint head's F_c equivalent — the waypoint query tokens
([N_wp=8, 256], penultimate before the per-token `wp_decoder` Linear) — with
the same positive-activation rule as mode 2 uses for `speed_query`.  Faithful
to mode 2, one extra backward per frame.  Implemented as
`conf.ADD_WAYPOINT_SEEDS` (`_attribute_wp_to_input`, `beg="wp_fc"`;
`TFv6FullModelForLRP.forward(_return_wp_queries=True)` exposes the tokens).

**Measurement — the proposal's premise fails empirically.**  On 4 clean
frames from the example route (incl. junction/turn frames, real checkpoints,
real TP conditioning), the wp-seeded pixel map is near-identical to the
default speed-seeded map: **pixel cosine 0.994–0.998, class-profile cosine
≥ 0.9995**.  A follow-up prototype tested the *decision-direction* variant —
seeding the lateral coordinate (index 1; steering uses
`arctan2(aim[1], aim[0])`, `closed_loop_inference.py:162`) of the predicted
waypoints through `wp_decoder`, weighted by the prediction (Decision-D
analogue).  Even on a turning frame with pred. lateral displacement up to
3.9 m: **cosine(speed map, lateral-decision map) ≥ 0.997**.  The earlier
brake-counterfactual block shows the same signature (its class profile
matched the default to 3 decimals on the test frame).

**Conclusion: TFV6 LRP pixel maps are effectively seed-invariant at the
planning-decoder level.**  Any seed placed at or behind the decoder queries
(brake one-hot, wp activations, lateral decision direction) yields the same
input attribution up to a scale.  Plausible mechanism: all query tokens read
the same BEV context through 6 shared cross-attention layers, and the map's
spatial structure is dominated by the shared backbone path (cf. the strong
D06 attenuation); the seed only re-weights a 256-dim bottleneck whose
influence on the pixel-level *shape* has washed out by the input.  This is
the transformer analogue of the WoR GAP-collapse finding ("per-FC-node pixel
maps are identical").

**Consequences.**
- `ADD_BRAKE_SEEDS` and `ADD_WAYPOINT_SEEDS` both default to **False** — a
  second decoder-seeded block is a redundant copy of the first and would
  waste an HPC recompute (and degrade covariance conditioning with duplicate
  dimensions).  The implementations stay in the codebase, flag-gated, for
  A/B or thesis illustration.
- The single-backward "fused seed" idea (seed speed_query + wp_queries in one
  `grad_outputs`) is answered by linearity: one backward returns the *sum* of
  the per-seed maps, which — given seed-invariance — equals the default map
  up to scale anyway.
- **Thesis note:** this is a reportable negative result — it bounds what
  multi-seed ATOMs extensions can achieve for this architecture and
  motivates enrichment along *other* axes (e.g. spatial/per-camera splitting
  of the one map, or profile normalization variants), which do not require
  a second backward pass.

---

## Heat Quantization (Otsuki et al. 2024, Sec. 4.4) — deliberately not applied (2026-07-07)

**What it is.** After computing the channel-wise-summed relevance map `α_R`,
the paper quantizes it into `Q=8` bins (Eq. 8: `α = min + floor((α_R - min) /
((max - min)/Q)) * Q`) to get the final attribution map `α`. Motivation
stated in the paper: raw `α_R` "tends to excessively concentrate on
irrelevant regions," and quantizing spreads attribution more evenly —
explicitly a visualization/qualitative-explanation aid (their Sec. 5 evaluates
it via Insertion/Deletion and human-facing heatmaps).

**Decision: not implemented in this codebase.** `atoms_carla.py` feeds the
continuous, normalized pixel relevance map (`wide_r / norm_w`, i.e.
`saliency_data_wide_default`) straight into `_hierarchical`'s per-class pixel
sums — no binning step anywhere in `lrp_transfuser.py` or `atoms_carla.py`.

**Why.** ATOMs profiles are a *quantitative* signal (per-class relevance mass
feeding Mahalanobis/GMM/k-NN OOD detectors), not a *qualitative* heatmap for
human viewing. Heat Quantization would discretize the continuous relevance
values into only 8 levels before the per-class sum, throwing away exactly the
fine-grained magnitude differences the downstream statistical detectors rely
on to separate baseline vs. perturbed profile clouds — it solves a problem
(visually concentrated heatmaps) that this pipeline doesn't have, at the cost
of a problem (reduced signal resolution) that it can't afford. Revisit only if
a future qualitative-visualization deliverable (e.g. thesis figures showing
raw pixel heatmaps rather than class-level bar charts) needs the same "spread
out the hotspot" effect Otsuki et al. were targeting.
