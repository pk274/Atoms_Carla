# Code Review — ATOMs / LRP OOD-detection pipeline

**Date:** 2026-06-08
**Scope:** `ATOMs_Analysis/` (saliency, detection, utils) + root pipeline scripts (`run_analysis.py`, `run_online_analysis.py`, `summarize_results.py`, `sweep_clusters.py`). TFV6 primary, WOR secondary. PCLA data-collection infra not reviewed.
**Focus (as requested):** scientific correctness, doc consistency, logical bugs, file-overwrite issues, and whether results are presentable / show exactly what they claim.
**Method:** static reading cross-checked against the five papers in `papers/`, plus inspection of generated JSONs/PNGs and `results_summary/SUMMARY.md`. The CARLA pipeline was **not** re-run (not possible in this environment). **No code was modified.**

Severity legend: **CRIT** = invalidates a claim / silently wrong · **HIGH** = real bug or validity risk · **MED** = correctness-relevant but guarded/edge · **LOW** = cleanliness · **DOC** = documentation drift · **OVR** = file-overwrite.

---

## 1. Headline — read this first

The LRP/ATOMs *mathematics* and the *visualizations* are in good shape. The risks that could affect what the thesis can honestly claim are concentrated in three places:

1. **Hyperparameter selection on the test set** (`run_analysis.py`). The headline k-NN, k-NN-GMM and "best-K" GMM AUCs are obtained by choosing the neighbour count *k* and cluster count *K* that maximise AUC **on the evaluation set**. These are optimistic upper bounds, not honest operating-point results.
2. **The PGD attack does the opposite of its stated target** (`perturbation_manager.py`). Both PGD implementations take gradient-**ascent** steps on losses written to *reach* the target, so "brake"/"max-speed"/"max-steer" attacks push *away* from those behaviours. Detection AUC is still valid as "detect an ε-bounded gradient perturbation", but any claim about the attack's *objective* is unsupported.
3. **A stale-cache / overwrite design** that can silently pair the wrong profiles with the wrong labels, and filenames that don't encode Town/weather/perturbation identity.

Everything else is secondary. A detailed, verified list of what is **correct** is in §7 so you can cite it with confidence.

---

## 2. Scientific-validity issues (affect what you can claim)

### 2.1 [CRIT] Hyperparameters are selected on the test set
`run_analysis.py:1241` `best_k = max(results_knn_by_k, key=…["auc"])` and `:1256` `best_k_gmm` pick the *k* with the highest **test** AUC. The GMM cluster count *K* is swept (`sweep_clusters.py`) and `summarize_results.py` reports the **"best achievable over K"** per perturbation (`SUMMARY.md` §1 and the per-perturbation section). So the reported best numbers (e.g. *k-NN-GMM 0.7158 @K=4*) are oracle-selected on the data they're evaluated on.

*Why it matters:* this inflates the best detectors and is not a fair comparison against the fixed-hyperparameter baselines (Mahalanobis, MDX, PEOC) in the same tables.
*Fix direction:* select *k* and *K* on a held-out validation split (or fix them a priori), then report on the test set; or clearly label these rows as oracle/upper-bound.

### 2.2 [HIGH] PGD attacks are sign-inverted vs their stated targets
`perturbation_manager.py`:
- `pgd_attack_tfv6` (l.437–471): `brake` uses `loss = CE(logits, bin0)` and then **ascends** it (`delta + step·grad.sign()`), which *increases* the cross-entropy to bin 0 → pushes the model *away* from braking. `max_speed` likewise moves away from bin 7. This inversion is unambiguous for the cross-entropy targets.
- `pgd_attack` (WOR, l.311–325): `brake = -brake_logits` ascended → *less* brake; `max_steer = -|steer_logits|` ascended → toward *straight*. The `steer_left/right` objectives use `±sum(steer_logits)`, but softmax is shift-invariant, so a uniform logit shift barely changes the decoded steering — a weak proxy for "steer right/left".

*Why it matters:* the data's `pgd` frames were generated with these functions. The OOD experiment is still valid as "detect a gradient-based ε-bounded perturbation", but the thesis cannot describe the attack as *forcing* a specific behaviour.
*Fix direction:* use gradient **descent** (subtract) for targeted losses, and target the decoded action value (`steers·softmax(logits)`) rather than raw logit sums.

### 2.3 [MED] PGD may be dominated by its random start
Both attacks use `random_start = U(−ε, ε)` with ε = 12 (TFV6) / 8 (WOR), i.e. the perturbation begins at full magnitude. Combined with a mis-directed/weak gradient (§2.2), the result may be close to uniform ±ε noise. Worth empirically confirming that `pgd` differs from `gaussian_noise` in detectability before framing it as "adversarial".

### 2.4 [MED] TFV6 MDX uses a degenerate action proxy
`run_analysis.py:348` builds the MDX action as `[steer=0, throttle=min(spd/25,1), brake=spd<0.5]`. Steer is constant 0, so all samples fall in one steering bin → only throttle×brake (≤4 of the 12 classes) are informative. This is documented and explains the weak MDX AUC (≈0.61); just disclose it when comparing MDX to the WOR MDX (which uses the real action distribution, ≈0.67).

### 2.5 [NOTE] PEOC is below chance — report it honestly
PEOC / speed-logit entropy scores AUC ≈ 0.43 (< 0.5) for TFV6, i.e. entropy *anti-correlates* with perturbation. The code and figures show this honestly (the ROC dips below the diagonal). Keep it that way; don't present 0.43 as if it were a positive result.

---

## 3. Logical bugs

### 3.1 [MED] `MahalanobisDetector.score` double-sqrt
`detectors.py:201-204`: `dist2 = DistanceComputer.compute_mahalanobis(...)` already returns the Mahalanobis **distance** (it does the `sqrt` internally), but `score` then returns `sqrt(max(dist2, 0))` → it returns √(Mahalanobis distance). The variable name `dist2` reflects the wrong assumption.
*Scope:* the main AUC path and the score-distribution plots use `DistanceComputer.compute_mahalanobis` **directly** (`run_analysis.py:987`), so **no reported AUC or figure is affected**. The bug only affects the `MahalanobisDetector` class itself and the threshold saved in `mahal_detector.npz` (which is on the √-distance scale and inconsistent with the GMM path, which returns the plain distance). Fix: `return DistanceComputer.compute_mahalanobis(...)`.

### 3.2 [MED] Step 10 (trajectory analysis) is dead code but documented as active
`run_analysis.py:815-977`: all of 10.a–10.f is commented out; only the `print` statements run (no crash, since the figure calls are commented too). But `CLAUDE.md` (Step 8.5) and the `run_analysis.py` module docstring describe trajectory analysis as an active step producing `trajectory_analysis/*` figures. A stale `trajectory_analysis/` folder exists under `data/WOR/results/atoms_analysis/` from before it was disabled. If the thesis cites attention-trajectory / displacement figures, they are **not regenerated by current code** and the on-disk ones are stale. The imports (`plot_pca_perturbation_trajectories`, `compute_perturbation_displacement_stats`, …) are now unused.

### 3.3 [LOW] `results_knn` dict mutation aliases the per-k result
`run_analysis.py:1243` mutates `results_knn["detector_name"]`, which is the *same object* stored in `results_knn_by_k[best_k]`. Step 14 then writes `results_knn_k{best_k}.json` with the `"(k=…, best)"` name instead of `"(k={k})"`. Cosmetic mislabel of one per-k JSON.

### 3.4 [LOW] `MahalanobisDetector._precision` computed but never used
`detectors.py:194` pre-inverts the covariance in `fit()`, but `score()` recomputes a `pinv` via `DistanceComputer` every call. Also `fit()` uses `np.linalg.inv` while `DistanceComputer` uses `pinv` — harmless but inconsistent.

---

## 4. File-overwrite hazards (you flagged this)

The data tree is keyed only by **AGENT** (the `data/<AGENT>/` subfolder) and, for some files, **MODE_ANALYSIS** (a `_1`/`_2` suffix). Nothing encodes **Town, weather, speed mode, or perturbation spec**.

| File | Keyed by | Risk |
|---|---|---|
| `baseline_data/baseline_{mode}.npz` | agent, mode | [OVR] Re-running baseline in a different Town/weather overwrites the previous one silently. |
| `test_data/test_labeled.npz` | agent only | [OVR] No mode/Town/perturbation suffix. Re-applying a different perturbation spec overwrites it. |
| `test_data/attention/test_profiles_{mode}.npy` | agent, mode | [OVR] Overwritten on recompute; no Town/spec identity. |

### 4.1 [CRIT] Silent stale-cache / label mismatch
The recompute flags (`RECOMPUTE_BASELINE`, `REAPPLY_PERTURBATIONS`, `RECOMPUTE_TEST_ATOMS`) are independent booleans. If you regenerate `test_labeled.npz` (new seed/spec/Town) but leave `RECOMPUTE_TEST_ATOMS=False`, the loaded `test_profiles_{mode}.npy` can be stale. The **only** guard is a length check (`run_analysis.py:765`). If the new set has the *same length* but different content, the mismatch is silent → profiles paired with the wrong labels → invalid AUC. No config fingerprint is stored in the `.npz` to detect this.
*Fix direction:* store a small metadata header (agent, town, weather, mode, seed, spec hash, n_frames) in each `.npz` and assert it matches on load.

### 4.2 [HIGH] TFV6 PGD frames carry clean pixels locally
For TFV6, `PerturbationApplier` records `pgd` frames with **clean pixels but `label=1`** (`dataset.py:297,340`); the adversarial image is crafted on HPC. If the TFV6 pipeline is ever run end-to-end locally with `RECOMPUTE_TEST_ATOMS=True` *without* merging the HPC-computed profiles, the `pgd`-labeled frames are actually clean → the PGD result is meaningless and there is no in-code guard that flags it.

### 4.3 [MED] HPC reproducibility depends on load order
`_assign_frames` (`dataset.py:397`) is positional and deterministic given `(seed, n)`. The "same seed + spec → same shuffle as HPC" assumption additionally requires identical frame **load order** (`_load_all_runs` file sort) and matching NumPy RNG version between local and HPC. If either differs, HPC-crafted PGD profiles align to the wrong frames silently.

---

## 5. Documentation drift

- [DOC] **Pipeline length/order.** `CLAUDE.md` and the `run_analysis.py` module docstring describe **12 steps** in an old order; the code has **14 steps** (3=MDX, 7=viz baseline, 8=perturb, 9=test ATOMs, 10=trajectory[disabled], 11=score, 12=eval, 13=per-pert, 14=viz).
- [DOC] **Output dir.** `CLAUDE.md` says results go to `atoms_analysis/`; actual is `atoms_analysis_mode_{N}/`.
- [DOC] **Profile dimensionality.** `CLAUDE.md` repeatedly says "23-dim attention vector / all 23". Actual: **WOR = 29** (`CARLA_CLASSES` 0–28), **TFV6 = 10** (`TFV6_CLASSES`). Verified from saved series shapes `(1100,29)` and `(3000,10)`. The "23" is stale (old CARLA tag count). Comments like "set None for all 23" recur in `run_analysis.py`.
- [DOC] **Hierarchical attention wording.** `CLAUDE.md` calls `h(o)` "fraction of total relevance falling on object o". Per the ATOM paper it is the **mean** relevance over the object's nonzero-relevance pixels (R̄ᵏ, normalised by `V` = nonzero-pixel count). The code matches the paper; reword the doc.
- [DOC] **"FOLLOW_LANE = 3".** Wrong for TFV6 — stored `cmd` is in CARLA RoadOption space where LANEFOLLOW = 4 (the data contains `cmd ∈ {4,5}`). Appears in `run_analysis.py:204`, `atoms_carla` docstring, and `_make_minimal_data`. Also the defaults disagree: `conf.DEFAULT_CMD = 2`, `ATOMsCarla.default_cmd = 3`, `_make_minimal_data(cmd=3)`. Low impact (only used for missing-cmd frames) but actively misleading.
- [DOC] **Stale "zero command vector" warning.** `CLAUDE.md` warns that `_make_minimal_data` "uses a zero command vector which distorts LRP attributions" — no longer true; the command one-hot is now built from `cmd`. `baseline_dataset.py:515` deliberately relies on this.
- [DOC] **LRP rule constant.** `lrp_transfuser.py` top docstring (l.27) says `AttentionLinear: Epsilon(ε=1e-6)`; the code uses `Epsilon(epsilon=1e-2)` (l.635).
- [DOC] **Reduced class count.** Comments say "7 driving-relevant classes" but `REDUCED_CLASS_IDS` has 8 entries.
- [DOC] **Primary agent.** `atoms_config.py` currently has `AGENT = "WOR"`, while CLAUDE.md states TFV6 is primary (state, not a bug — just check it's set correctly before a TFV6 run).

---

## 6. Cleanliness / dead code

- [MED] **Parallel, unused detector classes.** `run_analysis.py` scores via `DistanceComputer.compute_knn_distance` (k-th NN), `compute_wasserstein` (W1 to baseline mean), `compute_jsd` (JSD to baseline mean). The classes `KNNDetector` (mean-of-k), `WassersteinDetector` (sliced-W1), `JensenShannonDetector` (KDE sliced-JSD) in `detectors.py` are **not wired into the pipeline**, yet their docstrings describe the methods in detail — a thesis reader could misattribute methodology. Either remove or mark clearly as unused.
- [MED] **Wasserstein over nominal classes.** `compute_wasserstein` places attention mass at integer class indices `[0..K-1]`, so the cost of moving mass depends on the arbitrary class **ordering** (`Unlabeled=0, Vehicle=1, …`). W1 over nominal categories is not a meaningful metric; caveat any "Wasserstein between attention distributions" claim.
- [LOW] `baseline_dataset.py:503` `data["seg_red_narr"] is not NotImplementedError` is always True (comparison to the exception class); the real guard is the second clause.
- [LOW] ATOMs mode dispatch implements modes 1–3; the docstring advertises a mode 4 that silently falls back to mode 1. Config only uses 1 & 2, so no live impact.
- [MED→presentability] `_make_minimal_data` zeros `target_point` and `acceleration`; the comment calls this "equivalent for vision-only/LTF mode", which overstates it — `target_point` is route conditioning independent of the LiDAR (LTF) setting, so attributions are computed without route conditioning. Documented tradeoff (the npz lacks TP), but soften the "equivalent" claim.

---

## 7. Verified correct (cite with confidence)

**LRP / AttnLRP (`lrp_transfuser.py`)**
- Softmax rule `x·(R − s·ΣR)` = AttnLRP Prop 3.1. ✓
- Bilinear matmul rule (`denom = 2O + ε·sign`, `R_A=(R/denom)@Bᵀ·A`, `R_B=Aᵀ@(R/denom)·B`) = AttnLRP Prop 3.3. ✓
- Two-pass `output→input` (logits→query, then query→pixels seeded by node relevance) is mathematically identical to a single pass, because every backward rule is linear in the incoming relevance; the stability rationale is sound. ✓
- Speed-query index = `num_route(10) + num_waypoints(8) = 18`, exactly matching `PlanningDecoder.forward` (`target_speed_query = queries[:,18]`). ✓
- Command one-hot round-trips correctly (agent stores `cmd = argmax(onehot)`; `_make_minimal_data` rebuilds `onehot[cmd]=1`) — **no off-by-one**. ✓
- Speed bins `[0,4,8,10,13.888,16,17.777,20]` match the agent config. ✓

**ATOMs (`atoms_carla.py`)**
- `_give_element_selectivity` (relevance summed over object pixels ÷ nonzero-pixel count) matches the ATOM paper's R̄ᵏ exactly; node-level weighting `Σ_k |Rₖ|·R̄ᵏ` matches the paper (the `abs` is the documented AttnLRP adaptation). ✓

**Detectors (`detectors.py`, `distance_computer.py`, `clustering.py`)**
- Mahalanobis (`pinv`, ridge), Euclidean, k-th-NN (+L2 normalize, Sun et al.), JSD (`0.5·(KL(p‖m)+KL(q‖m))`), GMM nearest/weighted — all correct. ✓
- MDX = PCA→50 + per-class Gaussian + min **squared** Mahalanobis (Zhang Eq. 5) + conformal threshold on a held-out split. ✓
- `DetectorEvaluator`: sklearn `roc_auc_score`/`roc_curve`, Youden J = TPR−FPR argmax. ✓
- `GMMClustering`: sklearn GMM (`n_init=5`), BIC/AIC sweep argmin, nearest-cluster Mahalanobis. ✓

**Pipeline / aggregation**
- Label alignment (profiles/scores/labels all index `test_data` in order). ✓
- Per-perturbation eval builds `clean ∪ this_perturbation` correctly. ✓
- `SUMMARY.md` numbers match the underlying JSONs exactly (spot-checked Mahalanobis constant-across-K, k-NN, k-NN-GMM, MDX, PEOC, Mahal-GMM@K=13). ✓
- Figures render what they claim (ROC legend AUCs match JSONs; PCA-OOD honestly shows weak PGD separation; attention bar shows the 10 TFV6 classes). ✓

**WOR (`lrp_analysis.py`)**
- z⁺ rule (AlphaBeta(1,0)) for Conv+Linear, WSquare first layer; joint dual-camera backward + cross-normalisation; command/speed-conditioned brake/drive selector. Consistent with docs. ✓

---

## 8. Suggested fix priority (no changes made yet)

1. **Validity:** move k/K selection to a validation split (or label as oracle) — §2.1.
2. **Validity:** fix the PGD ascent/descent sign and the steer objective, or re-label the attack — §2.2.
3. **Data integrity:** add a config fingerprint to the `.npz` files and assert on load — §4.1; add a guard that TFV6 `pgd` profiles came from adversarial images — §4.2.
4. **Bug:** remove the double-sqrt in `MahalanobisDetector.score` — §3.1.
5. **Docs:** re-enable or remove Step 10 and reconcile CLAUDE.md (steps, output dir, 23→29/10 dims, FOLLOW_LANE, hierarchical-attention wording) — §3.2, §5.
6. **Cleanliness:** remove/relabel the unused sliced detector classes — §6.

---

*Update 2026-06-08:* the easy fixes were applied — §2.2 (PGD sign), §3.1 (Mahalanobis double-sqrt), §4.2 (deferred-PGD guard) and §3.3 (profile↔label key check) — plus the documentation sync. See `docs/design_decisions.md` → "Code-review fixes — 2026-06-08". Still open for discussion: §2.1 (validation split), §2.4 (MDX binning), the WoR steer objective, and dead-code removal.
