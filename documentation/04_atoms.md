# Topic 4 — ATOMs: From LRP Heatmaps to Object-Level Attention Profiles

All claims verified against code on 2026-06-12. Line numbers refer to the current working tree.
Primary source: `ATOMs_Analysis/saliency/atoms_carla.py` (658 lines, read in full).
Reference: Beylier, Hofmann, Scherf — *Revealing the Learning Process in RL Agents Through Attention-Oriented Metrics*, NeurIPS 2024 SciForDL workshop (`papers/ATOM-paper.pdf`, arXiv:2406.14324v2).

---

## 1. Purpose & scope

This document covers the ATOMs layer: how a per-frame LRP relevance map (Topic 3) is converted into a fixed-length, per-frame **attention profile vector** — the feature on which all OOD detectors (Topic 7) operate. It documents the `ATOMsCarla` class, its analysis modes, the semantic class registries, conditioning reconstruction, all magic constants, and — thesis-critical — the exact deviations from the original Beylier et al. definitions.

Headline result up front: **only hierarchical attention h(o) is implemented. Combinatorial attention c(T) is not implemented anywhere in the repository** (see §5). The "attention profile" is therefore a per-frame vector of per-class hierarchical attention, length 29 (WoR) or 10 (TFV6).

---

## 2. ATOMs in the original paper

Beylier et al. study an A2C Pong agent (3 conv layers → linear F_c → output). ATOMs quantify what objects the F_c neurons attend to, via two metrics computed over a curated input set **X** (150 frames, filtered so all objects are present and non-overlapping, objects instance-labelled):

**Hierarchical attention** (paper §2):

```
h(o_g) = (1/|X|) Σ_{x∈X} Σ_{k∈S} R_k(x) · R̄_g^k(x)
R̄_g^k(x) = (1/V) Σ_{p∈P_og} R_p^k(x),   V = |{p ∈ o_g : R_p^k(x) ≠ 0}|
```

i.e. per neuron k and object g: the **mean relevance over the object's nonzero-relevance pixels** (R̄), weighted by the neuron's own relevance R_k from a first LRP pass (LRP1, output→F_c), summed over the selected neuron subset S and averaged over frames. S is the smallest neuron set carrying ≥ 90 % of total F_c relevance. The per-node pixel maps R^k(x) come from a second LRP pass (LRP2, one-hot at neuron k → input). LRP uses the z⁺-rule throughout, so all relevances are non-negative.

**Combinatorial attention** (paper §2):

```
c(T) = (1/|X|) Σ_{x∈X} Σ_{k∈S} R_k(x) · δ_k(x;T)
δ_k(x;T) = 1  iff  R̄_g^k(x) > β for all g∈T  and  R̄_g^k(x) = 0 for all g∉T
β = α · M^k,  α = 0.25,  M^k = max_g R̄_g^k(x)
```

i.e. the (relevance-weighted) frequency with which a neuron attends **jointly and exclusively** to the object subset T, with a per-neuron relative threshold at 25 % of that neuron's maximum object attention (paper Appendix E justifies α=0.25). The paper uses h to rank objects and c to detect object co-observation patterns over training.

---

## 3. Implementation: per-frame flow through `ATOMsCarla.process_frame`

Entry point: `ATOMsCarla.process_frame(wide_rgb, narr_rgb, seg_wide, seg_narr, cmd=None, spd=None, data=None)` (`atoms_carla.py:293-335`). Unlike the paper (offline, curated 150-frame set), processing is **online, one frame at a time, with no object-presence filtering** — absent classes simply get zero attention (`atoms_carla.py:11-14`).

Per-frame sequence (TFV6 shapes in brackets; WoR shapes per finding 2.1: wide [1,3,192,480], narr [1,3,160,384]):

1. **Defaults** (`:303-306`): `cmd=None → self.default_cmd`, `spd=None → 0.0`.
2. **Context update** (`:308-316`): dispatch on `hasattr(self.lrp, '_data_cache')` (TFV6 marker, `lrp_transfuser.py:612`). TFV6 without a full `data` dict calls `lrp.update_context(wide, narr, spd, cmd=cmd)`, which builds the conditioning dict via `_make_minimal_data` (§8). WoR calls the two-camera `update_context(wide, narr, spd)`.
3. **Masks** (`:318-321`): `seg_to_masks(seg_red, class_ids)` (`:152-176`) converts the segmentation red-channel image [H,W] uint8 (pixel value = class tag) into binary masks `[C,H,W]` float32, C = 10 (TFV6) or 29 (WoR). `seg_narr=None → masks_narr=None`.
4. **Mode dispatch** (`:325`, table at `:264-268`): runs `_compute_node_level` (mode 1), `_compute_layer_level` (mode 2), or `_compute_node_output_level` (mode 3). Each adds the frame's per-class attention to the running accumulator `self._hierarchical` [C] float64.
5. **Contribution extraction** (`:323, :327-332`): `contribution = self._hierarchical - prev` (the un-normalized per-class vector of this frame); appended to `_frame_series` along with `cmd`, `is_brake`, `wide_frac` bookkeeping.
6. **Return** (`:334-335`): `contribution / (contribution.sum() + 1e-12)` — the **per-frame profile normalized to sum 1**. This returned vector is what `BaselineComputer` (`detection/baseline_dataset.py:518-519`) and the HPC chunk scripts (`hpc/compute_baseline_chunk.py:163`, `hpc/compute_test_chunk.py:219`, `hpc/compute_live_pert_chunk.py:165`) store as the frame's profile.

Inside the mode implementations, every LRP call goes through `self.lrp.forward_relevance(...)`, which returns `(wide_rel, narr_rel, wide_frac, is_brake)`:

- **WoR** (`lrp_analysis.py:209-345`): pixel maps are **cross-normalized**: `|wide_rel|` rescaled to sum to `wide_frac`, `|narr_rel|` to `1−wide_frac` (`_cross_normalize`, `lrp_analysis.py:334-345`) — non-negative by construction, total mass 1.
- **TFV6** (`lrp_transfuser.py:680-737`): `narr_rel=None`, `wide_frac=1.0`, and the pixel map is **raw, un-normalized, and signed** (AttnLRP softmax/matmul rules produce negative relevance). No abs is applied to the map.

**Class projection** — `_give_element_selectivity` (`:629-658`): for the wide map (and the narrow map only when `narr_r is not None and not conf.WIDE_ONLY_PROFILE`, `:656`):

```
r_hw = rel.sum(dim=0)                                  # [H,W], channel sum (signed for TFV6)
masks → nearest-interpolated to r_hw shape if needed   # :643-648
raw_c = Σ_p mask_c(p) · r_hw(p)                        # per-class relevance sum
nz_c  = #{p : mask_c(p)>0 ∧ r_hw(p)≠0}, clamped ≥ 1    # :650-651
h_c   = raw_c / nz_c                                   # mean relevance over nonzero-relevance object pixels = R̄
```

This is exactly the paper's R̄ (numerator over all object pixels is equivalent since zero-relevance pixels contribute 0; `clamp(min=1)` guards the empty case where the paper's V would be 0). With `WIDE_ONLY_PROFILE=True` (`atoms_config.py:97`, current setting) the narrow camera never contributes, for WoR as well.

Note the docstring at `:634-637` ("Both maps already sum to (wide_fraction) and (narr_fraction)") is only true for WoR; TFV6 maps are raw (finding 4.6).

---

## 4. Hierarchical attention — exact formula as implemented

### Mode 1 (`_compute_node_level`, `:413-442`) — the paper procedure

1. **LRP1** `_lrp1_nodes` (`:470-480`): `forward_relevance(beg="output", end="fc")` → `r_nodes` [256] raw F_c relevances. Seed: WoR = drive/brake action-selector mask (`lrp_analysis.py:256-258, 439-521`); TFV6 = softmax distribution over the 8 speed-bin logits (`lrp_transfuser.py:743-772`, default branch).
2. **Node filter** `_relevance_filter(r_nodes, p_relevance)` (`:117-145`): sorts `|r|` descending, keeps the smallest prefix whose cumulative mass ≥ p (count strictly-below-p plus one, `:133`). Mirrors the paper's 90 %-mass selection of S. Operates on **absolute** relevance (paper: raw, but z⁺ guarantees ≥ 0 there).
3. **LRP2 per node** `_lrp2_pixels` (`:482-507`): `forward_relevance(beg="fc", end="input", node_id=k)` — one-hot seed at F_c node k → pixel map (matches paper step 2, t_k = 1).
4. **Accumulation** (`:430-436`):

```python
R_sum  = self._give_element_selectivity(wide_r, narr_r)      # R̄ per class
node_w = abs(r_nodes[node_id].item())                         # |R_k|
self._hierarchical += R_sum * node_w
```

So the per-frame contribution is `Σ_{k∈S} |R_k| · R̄_g^k` — the paper's inner double sum with `R_k → |R_k|`. The `abs()` is a deliberate deviation, documented inline (`:431-434`): AttnLRP can produce negative F_c relevances, whereas the paper formula assumes z⁺ (R_k ≥ 0). Note `abs()` is applied to the **node weight only**, not to the pixel map (relevant for TFV6, finding 4.5).

### Mode 2 (`_compute_layer_level`, `:444-452`) — the configured default (`MODE_ANALYSIS=2`)

Single LRP pass `beg="fc", end="input"` with **no node selection and no node weighting**; the class projection of that one map is the frame contribution. The seed differs by agent:

- **TFV6** (`lrp_transfuser.py:821-836`): seed = `speed_query.clamp(min=0)` — **positive F_c activations**. Explicitly activation-weighted, not decision-weighted: it never passes through `target_speed_decoder` (docstring `:823-825`).
- **WoR** (`lrp_analysis.py:299-308, 361-376`): `node_id=None → selector=None → grad_outputs = torch.ones_like(fc_output)` — a **uniform all-ones seed** over the 256 F_c nodes.

Mode 2 is therefore a different metric per agent (activation-weighted vs. uniform) and in both cases **not** the paper's h(o). It is ~|S|× cheaper than mode 1 (one backward instead of 1 + |S|).

### Mode 3 (`_compute_node_output_level`, `:454-462`)

Single `beg="output", end="input"` map. For TFV6 this is implemented as LRP1 followed by output-weighted LRP2 with **signed** R_k as seed (`lrp_transfuser.py:838-886`), mathematically `Σ_k R_k · pixel_map_k` — the closest single-pass equivalent of the paper's sum (no abs, no p-filter), modulo the per-object R̄ averaging happening once on the summed map rather than per node. Per-node R̄ then weighted sum (paper / mode 1) and weighted-sum-of-maps then R̄ (mode 3) differ because the nonzero-pixel denominator V is map-dependent.

### Mode 4 — documented but not wired

The class docstring advertises mode 4 "layer-output: output→input, single map" (`:203-207`), but the dispatch dict only contains {1, 2, 3} and its `.get(mode_analysis, self._compute_node_level)` default means **mode 4 silently runs mode 1** (`:264-268`). Finding 4.2.

### Normalization chain (raw relevance → stored profile)

1. (WoR only) pixel maps abs-normalized so total mass = 1 split as wide_frac/narr_frac (`lrp_analysis.py:334-345`). TFV6: raw scale.
2. R̄ per class = class-masked sum / nonzero-pixel count (`atoms_carla.py:649-652`).
3. (mode 1 only) × |R_k| per node, summed over S.
4. Per-frame normalization to sum 1 at `process_frame` return (`:334-335`). **This is the vector saved to disk** — baseline npz `series`, `test_profiles_*.npy`, etc.
5. The episode-level accumulator `_hierarchical` keeps un-normalized contributions; `get_hierarchical(normalize=True)` (`:337-352`) divides by the total — a relevance-mass-weighted average over frames, which matches the paper's 1/|X| dataset averaging more closely than the per-frame-normalized series. The OOD pipeline uses (4), not (5); (5) and `get_series_df`/`get_mean_df` (`:354-394`) are used by diagnostics/plots only.

Consequence of (4): the per-frame profile is a **relative** attention distribution; absolute relevance magnitude is discarded (a global relevance-scale shift under perturbation is invisible to the profile; only redistribution across classes is detectable).

---

## 5. Combinatorial attention — NOT implemented (definitive)

**c(T) does not exist in this codebase.** Evidence:

- `atoms_carla.py` read line-by-line (658 lines): no function, accumulator, or threshold logic for object subsets; the only metric computed is `_hierarchical` and its per-frame series.
- `rg -i 'combinatorial|c\(T\)|joint.{0,20}atten'` over `ATOMs_Analysis/` matches nothing except an unrelated plotting constant (`utils/viz_config.py:206`, `ARROW_ALPHA_INDIVIDUAL = 0.25`).
- The α = 0.25 threshold appears nowhere in ATOMs code.

This settles finding 1.4: `CLAUDE.md` ("Combinatorial-attention c(T): fraction of frames where a neuron attends jointly to a subset of objects T (using threshold α = 0.25) … Implemented in `ATOMs_Analysis/saliency/atoms_carla.py`") is **wrong**.

Consequences for the thesis:
- Attention profiles are **hierarchical-only**: a C-dim simplex vector per frame. All detectors, distances, GMM clustering and results operate exclusively on h.
- Joint/exclusive co-attention structure (the paper's second descriptive level) is unavailable; any claim that the OOD signal uses "ATOMs" must be qualified as *hierarchical ATOMs*.
- In modes 2/3 there is also no per-neuron map at all, so c(T) could only ever be added for mode 1.

---

## 6. Semantic class sets

Two registries, defined **twice** in the repo with identical content: `atoms_carla.py:57-103` and `utils/visualization_carla.py:57-104`. Local pipelines (`run_analysis.py:192`, `run_online_analysis.py:144`) import from `atoms_carla`; the HPC chunk scripts import from `visualization_carla` (`hpc/compute_baseline_chunk.py:127`). Duplication is a divergence risk (finding 4.9).

### CARLA_CLASSES — 29 raw CARLA 0.9.13+ tags (WoR profiles, dim 29)

| Tag | Name | Tag | Name | Tag | Name |
|----|------|----|------|----|------|
| 0 | Unlabeled | 10 | Terrain | 20 | Static |
| 1 | Roads | 11 | Sky | 21 | Dynamic |
| 2 | SideWalks | 12 | Pedestrian | 22 | Other |
| 3 | Building | 13 | Rider | 23 | Water |
| 4 | Wall | 14 | Car | 24 | RoadLine |
| 5 | Fence | 15 | Truck | 25 | Ground |
| 6 | Pole | 16 | Bus | 26 | Bridge |
| 7 | TrafficLight | 17 | Train | 27 | RailTrack |
| 8 | TrafficSign | 18 | Motorcycle | 28 | GuardRail |
| 9 | Vegetation | 19 | Bycicle [sic] | | |

The segmentation camera writes the tag into the red channel; `seg_to_masks` thresholds on equality. Note the module header (`atoms_carla.py:18`) still says "0–22 for CARLA 0.9.x" — stale; the registry correctly covers 0–28 (finding 4.10).

`REDUCED_CLASS_IDS` (`:106-107`): optional WoR-only subset `[12, 24, 1, 2, 14, 7, 9, 11]` = Pedestrian, RoadLine, Roads, SideWalks, Car, TrafficLight, Vegetation, Sky — **8** entries, though docstrings say "7 driving-relevant classes" (`:209-210`, `run_analysis.py:196`). Activated by `use_reduced=True` only when no explicit `class_map` is passed (`:256-259`); every pipeline passes `use_reduced=False`.

### TFV6_CLASSES — 10 grouped LEAD classes (TFV6 profiles, dim 10)

| ID | Name | CARLA raw tags mapped into it (verified against `pcla_agents/transfuserv6/lead/common/constants.py:273-306`, `SEMANTIC_SEGMENTATION_CONVERTER`) |
|----|------|------|
| 0 | Unlabeled | Unlabeled, SideWalks, Building, Wall, Fence, Pole, TrafficSign, Vegetation, Terrain, Sky, Train, Static, Dynamic, Other, Water, Ground, Bridge, RailTrack, GuardRail |
| 1 | Vehicle | Car, Truck, Bus, Motorcycle |
| 2 | Road | Roads |
| 3 | TrafficLight | TrafficLight |
| 4 | Pedestrian | Pedestrian |
| 5 | RoadLine | RoadLine |
| 6 | Obstacle | ConeAndTrafficWarning |
| 7 | SpecialVehicle | SpecialVehicles |
| 8 | StopSign | StopSign |
| 9 | Biker | Rider, Bicycle |

Rationale: LEAD stores segmentation PNGs with `save_grouped_semantic=True`, i.e. already collapsed by the converter above before reaching the npz files — the analysis has no access to the raw 32-tag map for TFV6 data. The grouping concentrates driving-relevant categories and dumps all background into class 0; TFV6 profiles consequently have a dominant "Unlabeled" component covering sky/buildings/vegetation/sidewalks. Profile dimensionality: **29 (WoR) vs 10 (TFV6)** — earlier "23-dim" mentions in old docs were a stale tag count (already corrected in `CLAUDE.md`).

Class-map selection: `run_analysis.py:192` / `run_online_analysis.py:144` (`TFV6_CLASSES if conf.AGENT == "TFV6" else CARLA_CLASSES`); HPC: `hpc/compute_*_chunk.py` (`CARLA_CLASSES if args.agent == "WOR" else TFV6_CLASSES`).

---

## 7. MODE_ANALYSIS 1 vs 2 — exact difference and file suffixes

| | Mode 1 (node-level) | Mode 2 (layer-level) |
|---|---|---|
| LRP passes/frame | 1× LRP1 (output→fc) + |S|× LRP2 (fc→input, one-hot node) | 1× (fc→input) |
| Node selection | `_relevance_filter(p_relevance)` on |R_k| | none |
| Weighting | Σ_k |R_k| · R̄^k | single map, unweighted |
| Seed | TFV6: softmax over speed logits; WoR: drive/brake selector (LRP1) | TFV6: ReLU(F_c activations); WoR: all-ones over F_c |
| Decision-conditioned? | yes (via LRP1 seed) | **no** (never touches the output head) |
| Paper fidelity | closest to paper h(o) | not in the paper |
| Cost | most expensive (≈10–40 LRP2 passes at p=0.9, `:230-232`) | cheapest |

Mode 3 (output→input, decision-weighted single map) exists but is not used by current configs; mode 4 is documented-only (§4).

The active mode is `conf.MODE_ANALYSIS` (`atoms_config.py:23`, currently **2**) for local runs; the HPC scripts take `--mode-analysis` (default **1**, `hpc/compute_baseline_chunk.py:37`; the array-task shell wrappers default to `${MODE_ANALYSIS:-1}`, e.g. `hpc/array_task.sh:48`). Every profile artifact is suffixed with the mode: `baseline_{mode}.npz` (`baseline_dataset.py:440`), `test_profiles_{mode}.npy` (`run_analysis.py:806`), `val_profiles_{mode}.npy`, `live_pert_profiles_{mode}.npy`. `run_analysis.py` loads whichever suffix matches `conf.MODE_ANALYSIS` — mode-1 and mode-2 profiles are never mixed, but **mode-1 HPC profiles computed before 2026-06-14 used `p_relevance=0.25` hardcoded** (`hpc/compute_baseline_chunk.py:132`, `compute_test_chunk.py:153`, `compute_live_pert_chunk.py:141`) while the config and class default say 0.9 — **fixed 2026-06-14, the workers now use 0.9** (finding 4.3); pre-fix profiles must be recomputed or reported as 0.25.

---

## 8. Conditioning reconstruction (`_make_minimal_data`) and fidelity limits

TFV6's planning decoder is conditioned on a status dict. Offline frames only store `cmd` and `speed`, so `LRPTFv6Model.update_context` (`lrp_transfuser.py:628-648`) rebuilds a minimal dict via `_make_minimal_data(spd, device, cmd=3)` (`lrp_transfuser.py:1040-1060`):

| Key | Value | Fidelity |
|-----|-------|---------|
| `speed` | `[[spd]]` from npz | exact |
| `command` | one-hot length 6 at `clamp(cmd, 0, 5)` | exact round-trip: the agent stored `cmd = argmax(command)`, the reconstruction inverts it |
| `target_point`, `target_point_previous`, `target_point_next` | zeros [1,2] | **lost** — route geometry not in npz |
| `acceleration` | zeros [1,1] | **lost** |

History (recorded in `docs/design_decisions.md:316-336`): before the `cmd` parameter existed, the command token was all-zeros (bias-only), which distorted cross-attention badly enough that ~80 % of baseline frames predicted speed bin 0. The one-hot fix resolved that; zeroed `target_point`/`acceleration` remain as accepted secondary distortion — **route conditioning is absent**, so offline attributions can differ from what the live agent would produce, identically for baseline and test frames (so it largely cancels in the OOD comparison, but limits "explains the live policy" claims). `BaselineComputer` deliberately omits `data=` for the same reason (`baseline_dataset.py:515-517`, rationale in `docs/design_decisions.md:340-354`).

Default-command caveat: when `cmd` is genuinely missing, `ATOMsCarla` falls back to `default_cmd`. The class default is 3 (`:241`, "FOLLOW_LANE"), but the pipelines pass `conf.DEFAULT_CMD = 2` (`atoms_config.py:87`) and the HPC scripts hardcode `default_cmd=2` with the comment "(FOLLOW_LANE)" — under the 0-based mapping 2 = STRAIGHT. Tracked as findings 1.7/2.12; harmless as long as every frame supplies `cmd`, which all current loaders do.

**Speed conditioning asymmetry (new, finding 4.4):** the local test-profile loop calls `atoms.process_frame(wide, narrow, seg_wide, seg_narr, cmd=cmd)` **without `spd`** (`run_analysis.py:768`) → speed token = 0.0 on every test frame, while the baseline loop passes the true speed (`baseline_dataset.py:518`). The HPC test path passes `spd` correctly (`hpc/compute_test_chunk.py:219`), as does `run_online_analysis.py:438,510`. Locally recomputed test profiles therefore carry a systematic conditioning shift relative to the baseline that is not caused by any perturbation.

### Saliency attributes & PLOT_COMPARATIVE_REL

After each frame, three pairs of attributes hold pixel maps for visualization (never fed back into profiles):

- `saliency_data_wide_default` / `_narr_default` (`:276-277`): the map from the default seed, divided by `wide_frac` (and `1−wide_frac` for narrow) so each camera's map is renormalized to unit mass (`:492-495`, `:591-594`). **This is the map that produced `_hierarchical`** (modulo the renormalization, which `_give_element_selectivity` does *not* see — it gets the raw `wide_r`).
- `saliency_data_wide_brake` / `_drive` (+ narrow): with `PLOT_COMPARATIVE_REL=False`, these merely mirror the default map into the slot matching `is_brake` (`:497-502`, `:595-601`). With `PLOT_COMPARATIVE_REL=True` (`atoms_config.py:49`, current setting):
  - modes 2/3: two extra full LRP passes with `forced_brake=True` / `forced_drive=True` seeds (`:603-620`; seed semantics in `lrp_transfuser.py:743-772`: one-hot bin 0 vs one-hot best non-brake bin).
  - mode 1: no extra LRP2 passes; instead two cheap LRP1 passes give per-node weight vectors `r_brake`, `r_drive`, and the cached per-node LRP2 maps are re-weighted `Σ_k |r_·[k]| · map_k`, then abs-sum-normalized (`_set_comparative_maps_node_level`, `:509-575`).
  - Edge case: `narr_r / (1 - norm_w)` divides by zero if `wide_frac == 1.0` while a narrow map exists (`:495, :594, :618-620`) — visualization-only (finding 4.7).
- `is_brake` bookkeeping: LRP2 returns `is_brake=False` unconditionally (TFV6 skips the forward, `lrp_transfuser.py:726-727`), so mode 1 saves/restores the LRP1 value around the node loop (`:417-419, :437`).

Consumers: `BaselineComputer` plots `default` and `drive − brake` maps every `PLOT_INTERVAL=20` frames (`baseline_dataset.py:521-552`; note `rgb_wide` is defined only inside the comparative branch but used outside it — finding 4.8).

---

## 9. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| `p_relevance` | 0.9 (class default `:241`; `conf.FC_RELEVANCE_FILTER=0.9`, `atoms_config.py:24`); HPC now also 0.9 (**fixed 2026-06-14; was 0.25 hardcoded** — `hpc/compute_baseline_chunk.py:132` etc.; profiles before that date used 0.25) | `_relevance_filter` `:420` | config locally, hardcoded on HPC | mode 1 only: F_c-relevance mass kept when selecting S |
| `mode_analysis` | 2 locally (`atoms_config.py:23`); 1 on HPC (`--mode-analysis` default) | dispatch `:264-268` | yes | metric variant (§7) |
| `default_cmd` | 3 (class), 2 (all pipelines) | `:303-304` | yes | fallback command when `cmd=None` |
| `spd` fallback | 0.0 | `:305-306` | no | conditioning speed when `spd=None` |
| `use_reduced` | False everywhere | `:256-259` | yes | WoR 8-class subset |
| `WIDE_ONLY_PROFILE` | True | `atoms_config.py:97`, used `:656` | yes | drop narrow-camera contribution from profiles |
| `PLOT_COMPARATIVE_REL` | True | `atoms_config.py:49`, used `:424, :603` | yes | extra forced-seed LRP passes for viz (2 per frame in mode 2/3; 2 LRP1 in mode 1) |
| α (combinatorial threshold) | 0.25 in paper | — | — | **absent: c(T) not implemented** |
| `1e-12` | ε against division by zero | `:335, :351, :373, :569-575` | no | profile/row normalization guards |
| `clamp(min=1.0)` | nonzero-pixel count floor | `:651` | no | R̄ denominator guard (paper V=0 case → class sum 0/1 = 0) |
| nearest interpolation | mask → relevance-map size | `:643-648` | no | only triggers on shape mismatch (e.g. WoR seg_red_narr 192 vs 160 rows, finding 2.7 — silently rescales instead of erroring) |
| float64 accumulator | `_hierarchical` dtype | `:271, :397` | no | numeric stability over long episodes |

---

## 10. Deviations from Beylier et al. (thesis-critical, explicit list)

1. **c(T) omitted entirely.** Profiles are hierarchical-only (§5). The paper's two-level description collapses to one level.
2. **Default mode is not the paper's procedure.** `MODE_ANALYSIS=2` replaces the node-level two-pass scheme with a single fc→input map whose seed is positive F_c activations (TFV6) or all-ones (WoR) — activation-weighted, not decision-relevance-weighted, no 90 %-mass neuron selection. Only mode 1 reproduces the paper's pipeline structure.
3. **|R_k| instead of R_k** as node weights in mode 1 (`:435`), and `_relevance_filter` ranks by |r| (`:125`). Necessary because AttnLRP yields signed F_c relevance; under the paper's z⁺ LRP the two coincide. Side effect: nodes that argue *against* the decision are weighted as strongly as supporting ones.
4. **Per-frame normalization instead of dataset averaging.** Paper: h = (1/|X|) Σ raw sums — frames with more total relevance weigh more. Implementation (as consumed downstream): each frame's contribution renormalized to sum 1 (`:334-335`); absolute relevance magnitude is discarded. (The episode-level `get_hierarchical` is closer to the paper but unused by the OOD pipeline.)
5. **Online, unfiltered, class-level objects.** No "all objects present & non-overlapping" filtering, no instance labelling; "objects" are semantic *classes* from the CARLA/LEAD segmentation camera (a class mask unions all instances). Absent classes get 0.
6. **LRP rules differ.** Paper: z⁺ everywhere (3-conv A2C). Here: WoR z⁺-composite over dual ResNets; TFV6 AttnLRP (ε for attention linears, AlphaBeta(1,0) conv/FFN, custom softmax/matmul) — see Topic 3. Pixel relevance can be negative for TFV6, and per-class attention entries can in principle be negative (finding 4.5).
7. **LRP1 seed.** Paper initializes R_output = f(x) (the full output vector). WoR uses a drive/brake-masked action selector at the interpolated speed bins; TFV6 uses the softmax distribution over speed logits (Decision D, `docs/lrp_todo.md`).
8. **Single RGB frame, no temporal stack.** Paper inputs are 4-frame 84×84 grayscale stacks; here one RGB frame per camera.
9. **Cross-camera mass split (WoR)** has no paper counterpart: relevance is split wide/narrow by mass fraction, and with `WIDE_ONLY_PROFILE=True` the narrow share is simply dropped before per-frame renormalization.
10. **Per-frame profiles as the analysis object.** The paper computes one h per *agent* over a 150-frame set; this project treats each frame's (normalized) contribution as a sample, enabling distributional OOD detection — an extension, not present in the paper.

---

## 11. Known limitations / open issues

- **Mode-1 hyperparameter mismatch HPC vs local** (p = 0.25 vs 0.9): **fixed 2026-06-14** — the HPC chunk workers now hardcode 0.9, matching the local default (finding 4.3). ⚠ Any mode-1 profiles computed on the cluster **before** that date used 0.25 and must be recomputed to obtain 0.9 profiles (or reported as 0.25).
- **Negative profile entries possible for TFV6** (signed maps + per-frame normalization by a signed sum). Downstream, `DistanceComputer.wasserstein` hard-fails on genuinely negative weights (`utils/distance_computer.py:358-362`) and JSD silently clips to 1e-12 (`:229-232`); apparently negatives are rare/absent in practice (Wasserstein runs through), but this is unguarded at the ATOMs layer (finding 4.5).
- **spd omitted on the local test loop** (`run_analysis.py:768`) — baseline/test conditioning asymmetry (finding 4.4).
- Mode 4 advertised but unreachable (finding 4.2).
- `process_frame` returns garbage direction if a frame's contribution sums to ≈0 (division by 1e-12 amplifies noise) — no guard, no occurrence reported.
- Route conditioning (`target_point`) lost offline (§8): attributions explain a partially-conditioned model.
- `_relevance_filter` prints two diagnostic lines per LRP1 call (`:135-144`) — log noise in mode 1 at scale, no functional impact.
- Class-level masks cannot distinguish instances (e.g. lead vehicle vs oncoming traffic both = "Vehicle").

### Stale documentation flagged

- `CLAUDE.md` ATOMs section: claims c(T) implemented (wrong, §5); the h(o) wording ("then re-normalized across objects") is accurate for the stored per-frame profiles.
- `atoms_carla.py:18`: "0–22 for CARLA 0.9.x" vs 29-tag registry; `:209-210` "7 driving-relevant classes" vs 8 IDs.
- `docs/design_decisions.md:86-92` ("ATOMs class: single shared implementation") is accurate; `:114` ("LRP1 seed: one-hot at argmax") remains stale (already finding 2.6).

---

## 12. Cross-references

- **01_architecture_overview.md** — config keys (`MODE_ANALYSIS`, `FC_RELEVANCE_FILTER`, `WIDE_ONLY_PROFILE`, `EXPERIMENT_VARIANT`), data-flow position of profiles.
- **02_agents.md** — F_c definitions (TFV6 `speed_query` [256], WoR act_head hidden layer [256]), input shapes, the 3-vs-6-camera domain shift (finding 2.2) that also affects every relevance map ATOMs consumes.
- **03_lrp.md** — `forward_relevance` modes and seeds, AttnLRP rules, cross-normalization, conservation caveats that propagate into profile noise.
- **05_dataset_creation.md** — npz frame schema feeding `process_frame`; `BaselineComputer`/loaders.
- **07_distances_and_detectors.md** — consumers of the profile vectors: `baseline_{mode}.npz` series → Mahalanobis/GMM fit; `test_profiles_{mode}.npy` → all detector scoring; JSD/Wasserstein simplex assumptions (§11).
- **10_hpc_pipeline.md** — chunked profile computation (`hpc/compute_*_chunk.py`), where the 0.25/0.9 and spd discrepancies live.
- **99_bugs_and_findings.md** — Topic 4 findings 4.1–4.10.
