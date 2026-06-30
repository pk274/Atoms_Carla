# Topic 6 — Perturbations: Registry, Adversarial Attacks, and Labeled-Set Construction

All claims verified against code on 2026-06-13. Line numbers refer to the current working tree.
Primary sources read in full: `ATOMs_Analysis/perturbation_manager.py` (1082 lines), `ATOMs_Analysis/detection/dataset.py` (586 lines), `hpc/prep_test.py` (179 lines). Cross-checked against `ATOMs_Analysis/atoms_config.py`, `run_analysis.py:685-722`, `hpc/compute_test_chunk.py:45-218`, `hpc/prep_test_wor.py`, `hpc/prep_live_pert.py`, `hpc/prep_live_pert_wor.py`, `pcla_agents/transfuserv6/lead/inference/sensor_agent_live_perturbation.py`, `docs/design_decisions.md`, `CLAUDE.md`.

---

## 1. Purpose & scope

This document covers the **OOD signal source** of the whole project: the perturbations applied to the agent's visual input. It spans three concerns:

1. **The perturbation library** — `PerturbationManager` (`perturbation_manager.py`), a registry of nine image-space perturbations plus three gradient-based adversarial methods (`pgd_attack`, `pgd_attack_tfv6`, `fgsm_attack`). For each registered perturbation we document the exact pixel transform, the intensity semantics, and the design rationale recorded in the docstrings.
2. **Labeled-set construction** — how the clean test/val frames produced in Topic 5 are mixed into a labeled OOD dataset (`PerturbationSpec` / `PerturbationApplier.apply`, `dataset.py`), the canonical **5-way 20 % mix**, the seed-42 frame-assignment algorithm, and the **TFV6 PGD deferral** (perturbed label, clean pixels).
3. **Local vs HPC parity, and offline vs live perturbation** — how the local `PerturbationApplier` and the HPC `prep_test.py` are kept bit-identical (same seed + same spec → same shuffle), how the deferred PGD attack is finally crafted on the GPU worker (`compute_test_chunk.py`), and the separate *live* perturbation path that injects perturbations mid-drive (`sensor_agent_live_perturbation.py`).

Topic 5 §3.6 already documented the labeled-set *structure* (the `*_labeled.npz` schema, that local and HPC paths must agree). This document goes deep on the perturbation *semantics* and cross-references rather than duplicates that material. The driving loop that consumes live perturbations is Topic 9; the HPC array-job mechanics are Topic 10.

Scope note: the registry functions are agent-agnostic (they operate on RGB lists / concatenated strips). The two PGD variants are agent-specific: `pgd_attack` targets the WoR dual-camera policy, `pgd_attack_tfv6` targets the TFV6 `forward(data)` graph. FGSM is implemented but **not wired into any current evaluation mix** (§2.5).

---

## 2. Key design decisions

### 2.1 OOD signal comes from synthetic perturbations, not in-sim hazards

The project deliberately uses `noScenarios` LEAD routes for clean driving and never trains/tests on accident or obstacle scenarios (`docs/design_decisions.md:310-311`, Topic 5 §2.1). The entire OOD signal is therefore injected post-hoc by perturbing the camera input. Under the **alternative split** (the currently configured `EXPERIMENT_VARIANT="alternative"`, `atoms_config.py:63`) this is the *only* OOD signal — there is no domain shift between baseline/test/val — so the perturbation design is the load-bearing methodological choice for the whole detection experiment.

### 2.2 The active 5-way 20 % mix vs the registry of extras

Of the nine registered perturbations, only **four image-space transforms plus PGD** are wired into the evaluation. The canonical labeled-set mix is:

| Entry | perturbation | fraction | intensity (TFV6) | label |
|---|---|---|---|---|
| 1 | `None` (clean) | 0.20 | — | 0 |
| 2 | `gaussian_noise` | 0.20 | `conf.NOISE_INTENSITY = 21` | 1 |
| 3 | `brightness_scale` | 0.20 | `conf.BRIGHTNESS_INTENSITY = 3` | 1 |
| 4 | `camera_loss` | 0.20 | 0 (drops camera 0) | 1 |
| 5 | `pgd` | 0.20 | `conf.PGD_EPSILON = 4.0` (ε), target=brake, 5 steps | 1 |

Defined at `run_analysis.py:696-702` (TFV6) and `run_analysis.py:704-710` (WoR), and hardcoded as `_SPEC` at `prep_test.py:35-42`. The remaining five registered perturbations (`isolate_channel`, `mirror_horizontal`, `camera_swap`, `blur`, `salt_and_pepper`, `phantom_obstacle`) are **registered but not used** by any labeled-set spec or HPC prep. They are reachable only manually via `pm.perturb_wide_image(...)` or through the *live* perturbation path (`conf.PERTURBATION` can be set to any registered name, §3.6) — so they are exercised in the online experiment but not in the offline 5-way ROC/AUC evaluation. (This contradicts `CLAUDE.md`'s "Perturbation types" table, which lists only four — finding 6.1; and the "registered-but-unused-in-offline" distinction is finding 6.2.)

**Rationale for the four chosen image perturbations** (docstrings + `docs/design_decisions.md:24-25`): they span four *qualitatively distinct* corruption modes that a real autonomous-driving sensor stack can plausibly suffer —
- `gaussian_noise` → additive sensor / ISO noise,
- `brightness_scale` → exposure / gain drift (day↔night),
- `camera_loss` → hard sensor dropout (one camera black),
- `pgd` → worst-case adversarial input (security threat model).

They also differ in *how localized* the attention shift should be: brightness/noise are global, camera_loss is a hard regional zero, PGD is a structured high-frequency mask. This breadth is what lets the per-perturbation breakdown (Topic 8 step 11) report which perturbation each detector catches.

### 2.3 Targeted, deferred PGD as the adversarial threat model

PGD (Madry et al.) was chosen over single-step FGSM because, within the same ℓ∞ budget, it is substantially stronger — each small step stays near the loss surface and accumulates curvature a single coarse step discards (`perturbation_manager.py:217-219`). The attack is **ℓ∞-projected** (`torch.clamp(δ, -ε, ε)` after every step, `:339-342`, `:484-487`), uses the **Madry step heuristic** `α = 2.5·ε/n_steps` when `step_size` is `None` (`:276`, `:429`), and **random-starts** `δ ~ U(-ε, ε)` by default (`:283-284`, `:436-437`).

The attack is **targeted**: `pgd_attack_tfv6` supports `brake` (maximise P(speed bin 0 = 0 m/s) → force a stop), `max_speed` (maximise P(bin 7 = 20 m/s)), `steer_left` / `steer_right` (push mean predicted waypoint x). The configured default is `PGD_TARGET = "brake"` (`atoms_config.py:83`). `brake`/`max_speed` craft `loss = -CE(speed_logits, target_bin)` and ascend it (i.e. reward = maximise P(target bin)); steering targets ascend `±mean(wp_x)` (`:459-478`). All four objectives were **sign-corrected on 2026-06-08** — the previous code ascended quantities that should have been minimised, driving the agent *away* from the stated target; this is documented and verified in `docs/design_decisions.md:492-509` and the in-code comments at `:310-315`, `:453-458`.

**Why PGD is deferred on TFV6** (the central design decision here): crafting a PGD attack requires forward+backward passes through the model, but the labeled-set prep step (`prep_test.py`) is deliberately **model-free** so it can run on a cheap CPU node. So for TFV6, `pgd` frames are recorded with **clean pixels + `label=1` + `perturbation="pgd"`**, and the actual adversarial image is crafted later in `compute_test_chunk.py` (GPU worker, model loaded). The frame-to-entry assignment is deterministic (seed 42), so the model-free prep and the GPU worker agree on exactly which frames are the PGD frames (`prep_test.py:30-34`, `dataset.py:296-312`, `compute_test_chunk.py:165-217`). WoR's offline PGD would similarly need a model; in practice WoR's HPC prep omits PGD entirely (§2.6).

**Clean-pixel-but-perturbed-label rationale / caveat:** because the local `test_labeled.npz` stores clean pixels for PGD frames, recomputing ATOMs profiles locally for those frames (`RECOMPUTE_TEST_ATOMS=True`) yields **non-adversarial** profiles that must not be trusted — the HPC-crafted profiles must be merged instead. Two guards warn about this (`dataset.py:300-312`, `run_analysis.py:735-748`); see `docs/design_decisions.md:521-526`. This is a deliberate but fragile arrangement (finding 6.3).

### 2.4 ℓ∞ pixel-unit budget on the 0–255 scale

Both PGD variants operate on tensors in `[0, 255]` (raw pixel units), not normalized floats, and clip the final adversarial image to `[0, 255]` (`:368-375`, `:447`, `:507`). ε is therefore directly interpretable as a per-pixel ℓ∞ budget in 8-bit intensity. The docstrings note ε=4 ≈ "virtually invisible", ε=8 ≈ "common imperceptibility threshold" (`:236-239`). The project uses larger budgets than the imperceptibility literature: **ε=14.0** (`PGD_EPSILON`, `atoms_config.py:84`) for the recorded offline label and **ε=8.0** (`EPSILON`, `atoms_config.py:79`) for the live driving attack — chosen empirically (config comment `:79`: "5 → No effect; WoR: 8; TF: 12") to actually destabilise the policy rather than to stay imperceptible. This budget proliferation across files is the source of finding 1.1 (logged in Topic 1) and a new live-vs-offline angle (finding 6.4).

### 2.5 FGSM kept as dead-but-callable code

`fgsm_attack` (`:509-646`, WoR only) and the `fgsm` branch of `PerturbationApplier.apply` (`dataset.py:293`, `:329-342`) are fully implemented but appear in **no current spec** — neither `run_analysis.py` nor any HPC prep registers an `fgsm` entry. It survives as a single-step baseline for the adversarial-attack comparison and as the historical predecessor of `pgd_attack` (the PGD docstring describes itself as "a drop-in replacement for fgsm_attack", `:266`). Documented as unused (finding 6.2). Its `steer_right`/`steer_left` targets and the WoR `pgd_attack` ones share the same "weak proxy" caveat: the raw steer-logit sum is shift-invariant under softmax, so it does not cleanly steer left/right (`:318-323`, `:599-604`; `docs/design_decisions.md:501-503`).

### 2.6 Local vs HPC parity, and the WoR 4-way exception

The local `PerturbationApplier._assign_frames` (`dataset.py:411-428`) and the HPC `prep_test.assign_frames` (`prep_test.py:85-93`) implement the **identical** counts-and-shuffle algorithm under `np.random.default_rng(seed)` (§3.5), so a TFV6 labeled set built locally and one built on HPC pick the same frames for each entry. This is what makes the deferred-PGD scheme sound. The TFV6 mix is 5-way 20 % in **both** paths.

Exception (finding 6.5): the **WoR** HPC prep `prep_test_wor.py:27-32` uses a **4-way 25 %** mix with *no PGD entry* ("25% each, no PGD on HPC"), whereas WoR's local `run_analysis.py:704-710` spec is the 5-way 20 % mix *including* `pgd`. A WoR labeled set built on HPC and one built locally therefore differ in both the entry set and the fractions, breaking the parity property for WoR.

---

## 3. Implementation details

### 3.1 The registry mechanism (`perturbation_manager.py:25-33`)

A module-level dict `_WIDE_IMAGE_REGISTRY` maps perturbation name → function. The `@_register_wide(name)` decorator (`:28`) populates it at import time. Every registered function has the signature `fn(wide_rgbs: List[np.ndarray], intensity: float, **kwargs) -> List[np.ndarray]` and **must return a list of arrays** (one per camera). `PerturbationManager.list_perturbations()` (`:722-725`) returns the sorted registry keys. Three public dispatch methods consume the registry:

- `perturb_wide_image(wide_rgbs, perturbation, intensity, camera_index=None)` (`:87-146`) — WoR multi-camera wide list. `camera_index` restricts the perturbation to one camera by perturbing a single-element list and splicing the result back (`:139-144`).
- `perturb_narrow_image(narr_rgb, perturbation, intensity)` (`:148-190`) — WoR single narrow camera; wraps the image in a one-element list (`:189`). Multi-camera perturbations (`camera_swap`, `camera_loss`) no-op or warn on a 1-element list.
- `perturb_tfv6_image(rgb_chw, perturbation, intensity, camera_index=None, n_cameras=6)` (`:648-720`) — splits a concatenated `[3, H, W_total]` TFV6 strip into `n_cameras` sub-images of `W_total // n_cameras` px each (`:702-708`), applies the registry function, and re-concatenates back to `[3, H, W_total]` uint8 (`:718-720`). This dedicated TFV6 path fixes the layout bugs that arise from passing a single concatenated strip to `perturb_wide_image` (e.g. `camera_loss` indexing, per-camera mirroring, per-camera box placement — docstring `:663-668`). In the labeled-set path the caller passes `n_cameras = W // H` (`dataset.py:371`, `prep_test.py:120`), i.e. the number of square-ish camera tiles, which is 3 for the 1152-wide LEAD frames.

### 3.2 The four active image perturbations

**`gaussian_noise`** (`:732-756`). Adds `N(0, intensity²)` noise per pixel, clips to `[0,255]`. `intensity` = standard deviation σ in pixel units (docstring range 1–50, `:744-746`). Configured σ = 21 (`NOISE_INTENSITY`, day=25 / night=21 per config comment `:27`). **Unseeded** — `np.random.normal` uses the global NumPy RNG, so the noise realisation differs every run even with a fixed labeling seed (finding 6.6).

**`brightness_scale`** (`:759-783`). Multiplies every pixel by the scalar `intensity`, clips to `[0,255]`. `<1` darkens, `>1` brightens (`:770-774`). Configured factor = 3 (`BRIGHTNESS_INTENSITY`) — a 3× multiply that strongly over-exposes the image while leaving slightly more residual detail than the former 4× setting. Deterministic.

**`camera_loss`** (`:786-824`). Replaces one camera's sub-image with zeros (a hard black-out simulating sensor dropout). The camera index is `int(intensity)`; out-of-range indices no-op with a warning (`:816-820`). In the active mix `intensity=0`, so **camera 0** (the first sub-image) is always dropped. For TFV6 this drops one of the three forward camera tiles. Deterministic.

**`pgd`** — not a registry entry; handled specially in the appliers (§3.3-3.4) and crafted by `pgd_attack_tfv6` / `pgd_attack`.

### 3.3 The five registered-but-unused extras

| name | line | transform | intensity semantics |
|---|---|---|---|
| `isolate_channel` | `:827-873` | zeroes all channels except one | channel index 0/1/2 to keep (debug tool for BGR/RGB confusion) |
| `mirror_horizontal` | `:876-900` | `np.fliplr` each camera | unused (binary flip); kept for API consistency |
| `camera_swap` | `:903-936` | swaps cameras 0 and last (centre untouched); needs ≥3 cams else no-op | ignored |
| `blur` | `:939-967` | `cv2.blur` box filter, kernel `(2k+1)²`, `k=max(1,int(intensity))` | kernel half-size (1→3×3, 5→11×11) |
| `salt_and_pepper` | `:970-1019` | overwrites random pixels with 0/255, half each | fraction of pixels corrupted [0,1] (unseeded RNG) |
| `phantom_obstacle` | `:1022-1083` | injects a near-black (value 20) rectangle anchored to the bottom-centre | box size as fraction of image dims [0,1]; offset=5 px from bottom |

`phantom_obstacle` uses fill value **20** (not pure 0) so it does not alias with a `camera_loss` black-out (`:1075-1077`) — note the docstring still says "pixel value 10" (`:1051`) while the code writes 20 (finding 6.7). `isolate_channel`/`mirror_horizontal`/`camera_swap` are debugging / spatial-confusion tools described in their docstrings; none feeds the offline ROC evaluation.

### 3.4 PGD — WoR (`pgd_attack`, `:192-377`) vs TFV6 (`pgd_attack_tfv6`, `:379-507`)

Both share the structure: optional random start, `n_steps` ascent iterations of `δ += α·sign(∇_δ loss)` projected into the ℓ∞ ε-ball, then `x_adv = clip(x + δ, 0, 255)`. Differences:

| | `pgd_attack` (WoR) | `pgd_attack_tfv6` (TFV6) |
|---|---|---|
| Model interface | `model.policy(wide, narr, cmd) → (steer, throt, brake)` logits | `net.forward(data) → pred` (uses `pred.pred_target_speed_distribution` / `pred.pred_future_waypoints`) |
| Inputs perturbed | wide and/or narrow (`apply_to_wide`/`apply_to_narrow`) | only `data["rgb"]`; LiDAR/target-point/cmd/speed held fixed (`:396-397`) |
| Ensemble | single model | **averages the gradient over all ensemble members** (`nets` list, loss = mean over nets, `:450-481`) |
| Targets | `steer_right`, `steer_left` (weak proxy), `brake` (+brake_logits), `max_steer` (+\|steer\|) | `brake`/`max_speed` (∓CE on bin 0/7), `steer_left`/`steer_right` (∓mean wp_x) |
| Default ε / n_steps | 8.0 / 15 (signature defaults `:199-200`) | 8.0 / 10 (signature defaults `:384-385`) |
| Noise cache | `last_wide_noise`, `last_narr_noise` | `last_tfv6_noise` (`:81`) |

**Frame-interval caching.** Both attacks only *recompute* δ when `frame_counter % attack_interval == 0` (`:278`, `:433`), reusing the cached δ on intervening frames. `attack_interval = FRAMES_TO_SKIP + 1` (`:78`). With `FRAMES_TO_SKIP = 0` (`atoms_config.py:78`), `attack_interval = 1` → **a fresh attack on every frame**. The inline comment "Recompute attack every 3rd frame" (`:78`) is stale (finding 1.9, logged in Topic 1). On the HPC the worker explicitly sets `pm.attack_interval = 1` to force a fresh δ per attacked frame (`compute_test_chunk.py:174`), so the interval optimisation is effectively off everywhere it matters.

**TFV6 ensemble subtlety:** `pgd_attack_tfv6` averages over `nets`, but the **live agent** drives with the 3-model ensemble (`self.closed_loop_inference.nets`, `sensor_agent_live_perturbation.py:137`), whereas the **HPC** crafts the attack against a **single** member (`nets=[tfv6_model]`, `compute_test_chunk.py:210-211`). The analyzed PGD attack is therefore crafted against a different (single-member) policy than the one the live recordings were produced with — a relevance facet of finding 2.5 (ensemble-vs-single-member; new angle 6.8).

### 3.5 `PerturbationSpec` / `PerturbationApplier` / frame assignment (`dataset.py`)

**`PerturbationEntry`** (`:59-74`): dataclass with `fraction` (float), `perturbation` (str or `None` for clean), `intensity` (float, doubles as the ε budget for PGD/FGSM), and `fgsm_target` (default `"steer_right"`, reused as the PGD target for both adversarial methods, `:338`, `:350`).

**`PerturbationSpec`** (`:77-100`): a list of entries; `__post_init__` asserts the fractions sum to 1.0 ± 1e-6 (`:91-96`).

**`PerturbationApplier.apply(spec, seed=42, max_runs=None, output_name="test_labeled")`** (`:238-409`):
1. Loads all clean frame runs (`_load_all_runs`, `:550-584`; assigns `run_id` by sorted-file order).
2. `_assign_frames(n, spec, seed)` (`:411-428`) — converts fractions to integer counts (`round(fraction·n)`, last entry absorbs the rounding remainder, `:419-421`), builds a flat index array `[0,0,…,1,1,…]`, and `rng.shuffle`s it under `np.random.default_rng(seed)`. The shuffle decorrelates perturbed/clean frames along the temporal axis (`:219-221`).
3. Per entry, perturbs the assigned frames: clean → copy; `fgsm`/non-deferred `pgd` → call the adversarial method (requires a model, raises if `model=None`, `:314-318`); image perturbation → `perturb_wide_image`+`perturb_narrow_image` (WoR) or `perturb_tfv6_image` (TFV6, `:366-373`).
4. **TFV6 PGD deferral** (`:298-312`): `is_pgd_deferred = is_pgd and not has_narr` (TFV6 has no narrow camera). Deferred frames keep clean pixels, get `label=1` and `perturbation="pgd"`, and emit a `warnings.warn`.
5. Saves `test_labeled.npz` with `wide_rgb` (perturbed), clean `seg_red_wide`/`seg_red_narr` ("always clean — seg is ground truth", `:262-263`), `cmd`, `speed`, `is_brake`, `frame_idx`, `run_id`, `label`, `perturbation` (object array), `intensity`.

**`prep_test.py`** mirrors this for the HPC: hardcoded `_SPEC` (`:35-42`), `assign_frames` (`:85-93`, byte-identical algorithm to `_assign_frames`), TFV6 image perturbations applied here, PGD pixels deferred ("pixels deferred to array job", `:152`). It also writes `test_meta.txt` with the frame count so the SLURM array can be sized (`:171`). The val set reuses the same prep path against the val frames directory (Topic 5 §3.6).

### 3.6 Live perturbation path (offline vs online)

Distinct from the offline labeled set: the **live** path injects a perturbation *mid-drive* and records both pre- and post-injection frames for the online analysis (Topic 9).

- `sensor_agent_live_perturbation.py` (TFV6) reads `conf.PERTURBATION`, `conf.INTENSITY`, `conf.CAM_INDEX`, and activates injection once `timestamp >= conf.INJECTION_TIME` (`:176-186`). Each post-injection frame is marked `is_perturbed=1`; pre-injection frames `is_perturbed=0` (`:30`).
- **Non-PGD** perturbations are applied to the uint8 image in `tick()` via `perturb_tfv6_image(..., n_cameras=_N_FORWARD_CAMS=3)` — but on the *full* model input (the comment crops only the **saved** copy to 1152 px; the model sees the full strip, `:208-220`). Any registered perturbation can be selected here, so the registered-but-unused extras (§3.3) *are* exercised in the live experiment.
- **PGD** is applied to the float tensor in `_perturb_tensor_hook` (`:131-169`) via `pgd_attack_tfv6` against the **full ensemble** (`nets=self.closed_loop_inference.nets`), using `conf.EPSILON=8.0` and `conf.PGD_N_STEPS=8` — different ε from the offline label (§2.4). The perturbed frame is recorded after the attack so the saved pixels match what the model saw.
- `hpc/prep_live_pert.py` (TFV6, wide-only) and `hpc/prep_live_pert_wor.py` (WoR, both cameras) are **pure concatenation** scripts — no model, no perturbation — that stitch the recorded `run_<pert>_live_pert_*.npz` files into one indexable NPZ for the array job and write `live_pert_meta.txt` (`prep_live_pert.py:77-86`). The actual live perturbation was already baked into the recorded pixels during driving, so these scripts apply nothing. See Topic 9 for the driving loop and Topic 10 for the array-job mechanics.

---

## 4. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| 5-way fractions (TFV6) | 0.20 each | `run_analysis.py:697-701`, `prep_test.py:37-41` | code | clean/noise/brightness/camera_loss/pgd |
| WoR HPC mix | 4-way 0.25 each (no pgd) | `prep_test_wor.py:27-32` | code | diverges from WoR local 5-way (finding 6.5) |
| labeling `seed` | 42 | `dataset.py:241`, `run_analysis.py:715`, `prep_test.py:51` | partly (CLI on HPC) | frame-to-entry shuffle; must match local↔HPC |
| `NOISE_INTENSITY` (σ) | 21 | `atoms_config.py:27` (day 25 / night 21) | config | gaussian noise std dev (pixel units) |
| `BRIGHTNESS_INTENSITY` | 3 | `atoms_config.py:28` | config | multiplicative brightness factor (3× over-exposure) |
| camera_loss intensity | 0 → drops cam 0 | `run_analysis.py:700`, `prep_test.py:40` | code | which camera tile is zeroed |
| `PGD_EPSILON` (ε, offline) | 4.0 | `atoms_config.py:84` | config | recorded ℓ∞ budget for offline pgd label |
| `prep_test.py --pgd-epsilon` | 4.0 default | `prep_test.py:54`, `compute_test_chunk.py:50` | CLI | HPC ε default — matches config |
| `EPSILON` (ε, live) | 8.0 | `atoms_config.py:79` | config | ℓ∞ budget for the **live** PGD attack |
| `PGD_N_STEPS` | 5 | `atoms_config.py:85` | config | PGD iterations (5 steps converges for brake target at ε=4) |
| `PGD_TARGET` | `"brake"` | `atoms_config.py:83` | config | PGD objective; also `compute_test_chunk.py --pgd-target` default `brake` |
| `pgd_attack` default n_steps | 15 | `perturbation_manager.py:200` | signature | WoR PGD iterations (signature default; overridden by callers) |
| `pgd_attack_tfv6` default n_steps | 10 | `perturbation_manager.py:385` | signature | TFV6 PGD iterations (overridden by callers) |
| PGD step size α | `2.5·ε/n_steps` | `:276`, `:429` | derived (Madry heuristic) | per-step magnitude when `step_size=None` |
| PGD `random_start` | True | `:202`, `:387` | signature | δ₀ ~ U(−ε, ε) |
| ℓ∞ projection range | `[−ε, ε]`; pixel clip `[0,255]` | `:339-342`, `:484-487`, `:368-375`, `:507` | fixed | budget + valid-pixel constraint |
| `FRAMES_TO_SKIP` | 0 | `atoms_config.py:78` | config | `attack_interval = FRAMES_TO_SKIP+1 = 1` (recompute δ every frame) |
| `attack_interval` (HPC override) | 1 | `compute_test_chunk.py:174` | code | force fresh δ per attacked frame |
| `INJECTION_TIME` | 10 (s) | `atoms_config.py:32` | config | live-perturbation activation time |
| `PERTURBATION` (live) | `"brightness_scale"` | `atoms_config.py:30` | config | live perturbation type (any registry name) |
| `INTENSITY` (live) | 4 | `atoms_config.py:31` | config | live perturbation strength |
| `CAM_INDEX` (live) | None (all cams) | `atoms_config.py:34` | config | live per-camera restriction |
| `_N_FORWARD_CAMS` / `_CAM_PX` | 3 / 384 | `sensor_agent_live_perturbation.py:67-68` | code | TFV6 forward-camera tiling for save crop |
| blur kernel | `(2k+1)²`, `k=max(1,int(intensity))` | `:957-958` | per-call | box-blur kernel size (unused in offline mix) |
| salt_and_pepper density | clip(intensity, 0, 1) | `:989` | per-call | fraction corrupted (unused in offline mix) |
| phantom_obstacle fill / offset | 20 / 5 px | `:1077`, `:1059` | code | box fill value (docstring says 10 — finding 6.7) |
| FGSM default ε | 8.0 | `:516` | signature | unused (no fgsm spec entry) |

---

## 5. Known limitations & open issues

- **`CLAUDE.md` lists only 4 of 9 registered perturbations** (finding 6.1) — the "Perturbation types" table omits `isolate_channel`, `mirror_horizontal`, `camera_swap`, `blur`, `salt_and_pepper`, `phantom_obstacle`. Five of the nine are unused by the offline evaluation (finding 6.2), which is thesis-relevant: only `gaussian_noise`/`brightness_scale`/`camera_loss`/`pgd` produce ROC/AUC numbers.
- **Offline ε mismatch (14.0 vs 12.0)** (finding 1.1, Topic 1) — `PGD_EPSILON=14.0` claims to match the HPC default, which is 12.0 (`prep_test.py:54`, `compute_test_chunk.py:50`). The label recorded in `test_labeled.npz` (14.0) is read back as the actual ε by `compute_test_chunk.py:195` (`eps = data["intensity"][i]`), so the crafted attack uses 14.0 when the intensity field is populated — but a run that relied on the CLI default would use 12.0. The two are reconciled only by accident.
- **Live vs offline ε divergence** (finding 6.4) — the offline label uses ε=14.0 while the live driving attack uses `EPSILON=8.0` and the live config comment claims "TF: 12". Three different PGD budgets coexist; the online and offline PGD results are not the same attack strength.
- **Deferred-PGD fragility** (finding 6.3) — the clean-pixel-but-`label=1` arrangement is correct only if HPC-crafted profiles are merged; locally recomputed PGD profiles are silently non-adversarial. Guarded by warnings (`dataset.py:300-312`, `run_analysis.py:735-748`) but not enforced.
- **WoR offline mix is inconsistent across paths** (finding 6.5) — WoR local spec is 5-way (with PGD); WoR HPC prep (`prep_test_wor.py`) is 4-way 25 % (no PGD). A WoR labeled set's composition depends on where it was built.
- **Unseeded stochastic perturbations** (finding 6.6) — `gaussian_noise` and `salt_and_pepper` use the global NumPy RNG, not the labeling seed. The *frame assignment* is reproducible (seed 42) but the *noise realisation* is not; two runs with the same seed produce different noisy pixels, so PGD-independent OOD signal is non-deterministic.
- **Ensemble-vs-single-member PGD** (finding 6.8, extends 2.5) — the live PGD attack averages over the 3-model ensemble; the HPC attack targets a single member. The analyzed adversarial perturbation is crafted against a different policy than the one that drove the live recordings.
- **`phantom_obstacle` docstring/code mismatch** (finding 6.7) — docstring says fill value 10, code writes 20. Harmless (unused offline) but inconsistent.
- **Stale "every 3rd frame" comment** (finding 1.9, Topic 1) — `perturbation_manager.py:78` comment contradicts `attack_interval=1`.
- **FGSM and the registered extras are dead in the offline pipeline** — they raise the maintenance question of whether to delete or document them; the thesis should describe the feature as "four image corruptions plus targeted PGD" to match what produced the results.

---

## 6. Cross-references

- **01_architecture_overview.md** — `atoms_config.py` as single source of truth for `NOISE_INTENSITY`, `BRIGHTNESS_INTENSITY`, `PGD_EPSILON`, `EPSILON`, `PGD_TARGET`, `PGD_N_STEPS`, `FRAMES_TO_SKIP`; the recompute flags (`REAPPLY_PERTURBATIONS`, `RECOMPUTE_TEST_ATOMS`) that gate when `test_labeled.npz` is rebuilt; `EXPERIMENT_VARIANT` (perturbation is the *only* OOD signal under the alternative split).
- **02_agents.md** — TFV6 `pred_target_speed_distribution` (8 speed bins, two-hot) and `pred_future_waypoints` are the PGD attack surfaces; the WoR `policy` head logits; the 3-vs-6-camera tiling (`perturb_tfv6_image` `n_cameras`) and the ensemble (finding 2.5) that PGD targets.
- **04_atoms.md** — the perturbed `wide_rgb` is what `ATOMsCarla.process_frame` attributes; segmentation masks stay clean so the attention shift is measured against ground-truth object pixels.
- **05_dataset_creation.md** — §3.6 documents the `*_labeled.npz` structure and the local↔HPC parity requirement that this topic's §2.6 / §3.5 detail; the clean test/val frames consumed here are produced there.
- **07_distances_and_detectors.md** — the detectors scored against the `label` produced here; per-perturbation breakdown.
- **08_offline_analysis.md** — `run_analysis.py` steps 7–11: spec definition (`:696-710`), `PerturbationApplier.apply`, per-perturbation evaluation; the deferred-PGD guard at step 9.
- **09_online_analysis.md** — the live perturbation injection (`sensor_agent_live_perturbation.py`), `INJECTION_TIME`, the live PGD ensemble attack, and the `prep_live_pert*.py` concatenation scripts.
- **10_hpc_pipeline.md** — `prep_test.py` / `prep_test_wor.py` (model-free labeling), `compute_test_chunk.py` (GPU PGD crafting), `compute_live_pert_chunk.py`, array-job sizing via `test_meta.txt` / `live_pert_meta.txt`.
- **99_bugs_and_findings.md** — Topic 6 findings 6.1–6.8; cross-references 1.1 (ε mismatch), 1.9 (stale interval comment), 2.5 (ensemble-vs-single-member).
