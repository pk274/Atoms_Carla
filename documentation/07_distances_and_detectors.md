# Topic 7 — Distances & Detectors: Metrics, OOD Scorers, GMM Clustering, Thresholds

All claims verified against code on 2026-06-13. Line numbers refer to the current working tree.
Primary sources read in full: `ATOMs_Analysis/detection/detectors.py` (1128 lines), `ATOMs_Analysis/detection/clustering.py` (370 lines), `ATOMs_Analysis/utils/distance_computer.py` (380 lines), `run_analysis.py` steps 2.5/4/5/9/9.5/12 (repo root). Cross-checked against `ATOMs_Analysis/atoms_config.py`, `papers/Zhang A Simple Unified Framework for AD in DRL.pdf` (TMLR 10/2024), `papers/uncertainty based müller sedlmeier.pdf` (Sedlmeier et al. 2020), `CLAUDE.md`, `docs/design_decisions.md`.

---

## 1. Purpose & scope

This document covers the **scoring layer** of the pipeline: how a per-frame feature vector is converted into a scalar anomaly score, how those scores are thresholded, and how the optional GMM baseline is fitted and selected. It spans three modules:

1. **`DistanceComputer`** (`distance_computer.py`) — a stateless library of static distance/divergence functions (Mahalanobis, Euclidean, k-NN, GMM-nearest, JSD, Wasserstein, and their GMM-min variants). These are the numerical kernels.
2. **`detectors.py`** — stateful detector classes that wrap a fitted baseline and expose the common `fit / score / save / load` `BaseDetector` interface: `MahalanobisDetector`, `EuclideanDetector`, `KNNDetector`, `WassersteinDetector`, `JensenShannonDetector`, `MDXDetector` (Zhang et al. 2024), `ActionEntropyDetector` (which **is** the PEOC scorer of Sedlmeier et al. 2020 when fed logits), plus `DetectorEvaluator` (ROC / AUC / Youden-J).
3. **`clustering.py`** — `GMMClustering`, the BIC/AIC sweep, component selection, and per-cluster Mahalanobis scoring used by every "GMM" detector variant.

The crucial fact threaded through this whole topic is **which input space each detector consumes**. The detectors are *not* a homogeneous family scoring one feature type — they split into three input spaces:

| Input space | Dim | Source | Detectors that consume it |
|---|---|---|---|
| **ATOMs attention profile** | 10 (TFV6 grouped) / 29 (WoR) | Topic 4 `process_frame` | Mahalanobis(single+GMM), Euclidean(single+GMM), k-NN(single+GMM), JSD(single+GMM), Wasserstein (disabled) |
| **Backbone-512 / penultimate** | 512 (TFV6 GAP) / model penultimate (WoR) | `get_backbone_features` / `model.get_features` | MDX-v1 (and MDX-v2 with the *current* config flag) |
| **F_c-256 `speed_query`** | 256 | `get_fc_features` / `get_planning_action_and_features` | MDX-v2 *intended* (but currently disabled at config + scoring) |
| **8-bin speed logits / WoR action logits** | 8 (TFV6) / 28 joint (WoR) | `get_speed_logits` / `get_action_logits` | PEOC (TFV6) / Action-entropy (WoR) |

The clean baseline profiles and the labeled test/val sets that these detectors consume are produced in Topics 4–6; this topic describes only what happens *after* a profile exists. The orchestration that calls all of these in sequence is `run_analysis.py` (Topic 8); here we document the detector contracts and the selection logic, not the full step ordering.

Scope note: `compute_wasserstein` / `WassersteinDetector` and the entire MDX-v2 *test-scoring* path are **implemented but commented out / disabled** in the current `run_analysis.py` (§5.6, §3.10). They are documented because the thesis story references them and because the code is live; we flag their disabled status explicitly.

---

## 2. Key design decisions

### 2.1 Stateless metrics (`DistanceComputer`) vs stateful detectors

`DistanceComputer` is a pure-function namespace: every method is `@staticmethod`, all parameters are passed explicitly, and nothing is stored (`distance_computer.py:21-25`). The `detectors.py` classes are the opposite — they hold a fitted baseline (`mean`/`cov`/`_precision`, BallTree, projections, per-class Gaussians) and implement `fit/save/load`. This is a deliberate split recorded in `CLAUDE.md` ("`DistanceComputer` is stateless (static methods); detectors … are stateful (fit/save/load)").

**Rationale and consequence:** the split lets `run_analysis.py` score GMM variants *without instantiating a detector object* — e.g. `scores_mahal_gmm` is computed by calling `DistanceComputer.compute_gmm_distance(gmm.means_, …)` directly in a list comprehension (`run_analysis.py:1169-1180`), reusing the `GMMClustering` parameters rather than wrapping each cluster in a `MahalanobisDetector`. The single-Gaussian path, by contrast, *does* use the stateful `MahalanobisDetector` (it owns the 99th-percentile threshold and is saved to disk). The result is a subtle asymmetry: the same Mahalanobis kernel (`compute_mahalanobis`) backs both, but the single path goes through the detector object while every GMM variant goes through the stateless functions. The two share the kernel, so they are numerically consistent (the historical double-sqrt bug that broke this consistency was fixed 2026-06-08, `detectors.py:203-207`).

### 2.2 Why this particular detector roster

The roster is organised as a **ladder of distributional assumptions** over the ATOMs profile, plus two model-internal baselines from the literature:

- **Euclidean** (`EuclideanDetector`) — L2 from the baseline mean. Parameter-free; the sanity floor that the richer detectors must beat (docstring `detectors.py:481-486`).
- **Mahalanobis single-Gaussian** (`MahalanobisDetector`) — adds the baseline covariance, so it is scale/correlation-aware. This is the *primary* ATOMs detector (`run_analysis.py:425`).
- **Mahalanobis GMM** — relaxes the single-Gaussian assumption to a mixture, scoring the distance to the *nearest* cluster (§2.3).
- **k-NN** (`KNNDetector` / `compute_knn_distance`) — non-parametric, naturally multi-modal; mean (detector) or k-th (stateless) L2 distance to neighbours. Grounded in Sun et al. ICML 2022 (`distance_computer.py:98-99`).
- **JSD** (`JensenShannonDetector` and `compute_jsd`) — treats the profile as a probability distribution and measures divergence; bounded, symmetric.
- **Wasserstein** (`WassersteinDetector` / `compute_wasserstein`) — optimal-transport distance over the object axis; **currently disabled** (§2.6, §5.6).
- **MDX** (Zhang et al. 2024) — class-conditional Mahalanobis in the *model's* feature space, not the ATOMs space (§2.4).
- **PEOC / Action-entropy** (Sedlmeier et al. 2020) — policy-entropy baseline on the *logit* space (§2.5).

Most detectors come in a **single** and a **GMM** flavour. The GMM flavour replaces "distance to the global baseline" with "min distance to the nearest of K baseline clusters", on the hypothesis that clean driving is multi-modal (intersection / highway / narrow street produce distinct attention profiles, `clustering.py:6-9`). Whether the GMM flavour actually helps is an empirical question the thesis answers via the per-detector AUC table (Topic 8).

### 2.3 Single-Gaussian vs GMM Mahalanobis

The single-Gaussian detector fits one `(μ, Σ)` over all baseline profiles and scores `√((x−μ)ᵀ(Σ+λI)⁻¹(x−μ))` (`MahalanobisDetector.fit/score`, `detectors.py:174-207`). The GMM detector fits K Gaussians and scores `min_k √((x−μ_k)ᵀ(Σ_k+λI)⁻¹(x−μ_k))` (`compute_gmm_distance` mode `"nearest"`, `distance_computer.py:162-176`; also `GMMClustering.score`, `clustering.py:175-197`). The nearest-cluster mode is the recommended/used one; a `"weighted"` mode (softmax-weighted average over clusters with a Gaussian likelihood weighting, `distance_computer.py:178-188`) exists but is not used by `run_analysis.py`.

**Design subtlety — the GMM Mahalanobis is a *distance*, the GMM density is not used for scoring.** `compute_gmm_distance` does not evaluate the GMM log-likelihood; it computes K independent Mahalanobis distances and takes the min. The GMM's mixture weights `weights_` are passed in but used only in the unused `"weighted"` mode. So the "GMM detector" is really "nearest-cluster single-Gaussian", which is why it shares the exact `compute_mahalanobis` kernel and the same ridge.

### 2.4 MDX-v1 vs MDX-v2: two feature spaces, two binning strategies

MDX (Zhang et al. 2024, §6 below) is instantiated **twice** with different constructor arguments to probe two questions: *which feature space* and *which discretisation* best separate clean from perturbed.

| | MDX-v1 | MDX-v2 (intended per CLAUDE.md) | MDX-v2 (current config) |
|---|---|---|---|
| Constructor | `MDXDetector(n_pca_components=50)` `run_analysis.py:351` | `MDXDetector(n_pca_components=50, bin_strategy="quantile")` | `MDXDetector(n_pca_components=50, bin_strategy="quantile")` `:412` |
| Feature space | 512-d backbone GAP (`get_backbone_features`) | 256-d F_c `speed_query` | **512-d backbone** (because `MDX2_USE_FC_FEATURES=False`) |
| Action proxy | speed-derived `[0, min(spd/25,1), 1 if spd<0.5]` `:346` | waypoint-steer + throttle/brake proxy from `get_planning_action_and_features` `:398` | F_c path taken only when the flag is True; with the flag False it falls back to the same speed proxy `:402-404` |
| Binning | `"equal-width"` (default) | `"quantile"` | `"quantile"` (`MDX2_USE_QUANTILE_BINNING=True`) |
| Save target | `mdx_parameters.pkl` | `mdx_v2_parameters.pkl` | `mdx_v2_parameters.pkl` |

The CLAUDE.md description ("v2 = 256-d F_c + quantile") describes the *intended* ablation, but the **live config disables the F_c half**: `MDX2_USE_FC_FEATURES = False` (`atoms_config.py:45`), so MDX-v2 as currently configured differs from MDX-v1 *only* in `bin_strategy` (quantile vs equal-width), both on 512-d backbone features. The steer proxy is therefore also always 0.0 in the current config (the diagnostic print at `run_analysis.py:409-410` would report `steer std 0.0`), collapsing it to the same proxy as v1 (finding 7.6).

Worse, **MDX-v2 is not actually scored at all** in the current `run_analysis.py`: the entire test-scoring block (`:1350-1371`) and its evaluation (`:1552-1556`) are commented out, and it is omitted from the sanity-check roster (`:1387`). MDX-v2 is fitted and saved (`:412-415`) but produces no AUC. So in practice only **MDX-v1** contributes a number (finding 7.7).

**Rationale for two MDX variants** (per CLAUDE.md / pipeline): MDX-v1 uses generic backbone features (closest to Zhang's "penultimate layer") with equal-width bins; MDX-v2 was meant to test whether the task-specific F_c token plus quantile bins (which keep classes balanced when steer/throttle are near-constant) improves separation. The equal-width binning of v1 is fragile for TFV6 because the steer proxy is constant 0.0, collapsing all steer bins into one (§3.10).

### 2.5 PEOC as a logit-entropy baseline (Sedlmeier et al. 2020)

PEOC (Policy Entropy OOD Classifier) uses the Shannon entropy of the policy's action distribution as a one-class OOD score: clean (in-distribution) states should yield confident, low-entropy policies; perturbed states should raise entropy (paper Eq. 2-3, §6.2 below). The implementation is **the same class as the WoR action-entropy detector** — `ActionEntropyDetector(from_logits=True, cmd=None)` — distinguished only by which logit vector it is fed:

- **TFV6 PEOC** (`run_analysis.py:1377-1378`): scores `test_speed_logits`, the **8-bin** `target_speed_decoder` distribution (`get_speed_logits`, computed at `:784`). Enabled by `speed_logits_available=True` (set only for TFV6, `:160`).
- **WoR Action-entropy** (`run_analysis.py:1311-1312`): scores `test_logits_all`, the **28-dim** joint steer×throttle+brake distribution at the interpolated speed (`get_action_logits`, `:775`; the comment at `:771-773` calls this "the true π(a|s)"). Enabled by `action_logits_available=True` (WoR only, `:186`).

They are mutually exclusive by agent. PEOC requires no baseline fit (`ActionEntropyDetector.fit` is a no-op, `detectors.py:316-318`).

**Deviation from the original.** Sedlmeier's PEOC entropy is computed over a *PPO discrete-action policy* `π(a|s)`. The TFV6 implementation substitutes the **8-bin speed distribution** for `π(a|s)` — TFV6 has no single discrete action head, so the speed bins are used as a proxy policy. This is a defensible adaptation but it means TFV6 "PEOC" measures speed-prediction confidence, not full policy entropy (no steering/waypoint uncertainty enters the score). Documented as a paper deviation (finding 7.9).

### 2.6 Threshold policy: 99th-percentile (hard decision) vs Youden-J (reported operating point)

Two distinct thresholding mechanisms coexist, for two distinct purposes:

- **99th-percentile baseline threshold** (`MahalanobisDetector.fit_threshold`, `detectors.py:213-234`; called at `run_analysis.py:438-441` with `percentile=99.0`). This sets a *hard runtime decision boundary* from in-distribution scores alone — ~1% of clean frames are flagged. It is stored on the detector and saved to disk, and is the only threshold that would be available *online* (no labels needed). Only the single-Gaussian Mahalanobis detector sets it; no other detector calls `fit_threshold`.
- **Youden-J optimal threshold** (`DetectorEvaluator.evaluate`, `detectors.py:419-426`). This is computed *post-hoc on the labeled test set* by maximising `J = TPR − FPR` over the ROC curve. It is a *reporting* threshold (the operating point quoted alongside AUC), not a deployable one — it requires ground-truth labels. Every detector gets a Youden-J point because every detector is run through `DetectorEvaluator`.
- **MDX adds two more** (Zhang's framework): a `chi2_threshold` (χ²_p(1−α), Prop 1, `detectors.py:1087-1095`) and a `conformal_threshold` calibrated on a held-out 20% split of the baseline (`detectors.py:1035-1046`). Neither is used for the ROC/AUC numbers — MDX is scored continuously and thresholded by Youden-J like the others — but both are computed at fit time and printed (§3.9).

The 99th-percentile choice is the only label-free threshold and is therefore the honest "what you could deploy" boundary; Youden-J is the "best achievable given an oracle" boundary used to compare detectors fairly. The AUC itself is threshold-free and is the headline metric.

### 2.7 Val-AUC-based k/K selection to avoid test leakage

`KNNDetector`/k-NN-GMM have a hyperparameter k that strongly affects AUC, and the GMM has K. To avoid choosing these on the test labels (leakage), the pipeline selects k on the **validation set** when one is present:

- k-NN k and GMM-k-NN k are swept over `KNN_K_VALUES = [1, 5, 10, 25, 50, 100, 250]` (`run_analysis.py:1134`); the best k is the one maximising **val** AUC (`:1486-1489`, `:1517-1520`). When no val set is present, k falls back to the **test**-AUC argmax with an explicit `WARNING: k selected on test — leakage` print (`:1491-1493`, `:1522-1525`).
- The GMM component count K is *not* selected on val AUC at runtime — it is resolved by the precedence in §3.8. Instead, the pipeline reports a **mean val AUC across the five GMM detector variants** (`__val_auc_gmm_avg__`, `:1531-1544`) so an external sweep (`sweep_clusters.py`) can pick K per the val signal offline (this is the "val-set K-selection recommendation" of `summarize_results.py`, Topic 8).

**Rationale:** k/K are the only tuned hyperparameters in the detection layer; tuning them on test would inflate AUC. The val set (disjoint Town05 routes, Topic 5 §2.5) exists precisely for this. The fallback-to-test path is a leakage hazard that is loudly warned about but not prevented (finding 7.8).

---

## 3. Implementation details

### 3.1 `BaseDetector` interface (`detectors.py:80-138`)

ABC with abstract `fit(data[N,D])`, `score(x[D])→float`, `save`, `load`; concrete `score_batch` loops `score` over rows (`:119-131`); `_check_fitted` raises if `fit` was not called (`:133-138`). Higher score = more anomalous, by convention, for every detector.

### 3.2 `MahalanobisDetector` (`detectors.py:145-271`) — ATOMs profile (or any fixed vector)

- **Input space:** ATOMs profile [10 TFV6 / 29 WoR]. The docstring also lists 512-d / 256-d uses (`:150-153`) but in the current pipeline only the profile path is used (MDX handles the deep-feature Mahalanobis separately).
- **fit** (`:174-195`): `mean = data.mean(0)`, `cov = np.cov(data.T)`, precision `= inv(cov + ridge·I)`. Scalar `D=1` is reshaped to `[[·]]` (`:190-191`). Ridge default in the constructor is `1e-6` (`:163`) but `run_analysis.py:431` passes `conf.MAHAL_RIDGE = 0.01`.
- **score** (`:201-207`): delegates to `DistanceComputer.compute_mahalanobis(mean, cov, x, ridge)`, which returns the **distance** (sqrt taken internally). The comment documents the 2026-06-08 fix that removed a second sqrt so single and GMM paths share scale.
- **fit_threshold** (`:213-234`): 99th-percentile of baseline scores; **is_anomalous** (`:236-241`) compares score to the stored threshold.
- **save/load** (`:247-271`): `np.savez_compressed` of mean/cov/ridge/threshold; load re-inverts the precision.

### 3.3 `EuclideanDetector` (`detectors.py:480-517`) — ATOMs profile

`fit` stores `mean`; `score = ‖x−mean‖₂` (`:504-506`). No covariance, no threshold helper. Save/load is just the mean.

### 3.4 `KNNDetector` (`detectors.py:524-584`) — ATOMs profile

- `k` default 50 (`:542`). `fit` builds an sklearn `BallTree` (euclidean) over the baseline; raises if `N < k` (`:553-557`).
- `score` (`:562-566`) returns the **mean** L2 distance to the k nearest neighbours.
- **Note the divergence from the stateless k-NN used in `run_analysis.py`.** The pipeline does *not* use `KNNDetector`; it calls `DistanceComputer.compute_knn_distance` (`:1136-1146`), which returns the **k-th** neighbour distance (`np.sort(distances)[k-1]`, `distance_computer.py:131`) with optional L2 normalisation (`normalize=True` is passed). So the detector class (mean-of-k, no normalisation) and the actually-used stateless function (k-th, L2-normalised) implement *different* k-NN scores. `KNNDetector` is effectively dead in the offline pipeline (finding 7.4).

### 3.5 `WassersteinDetector` (`detectors.py:591-662`) — ATOMs profile [DISABLED in pipeline]

Sliced Wasserstein-1: projects training data and the test point onto `n_projections=200` random unit vectors (`:615`, seeded `random_state=42`), and along each 1-D projection returns the mean absolute deviation `(1/N)Σ|vᵀx − vᵀx_i|`, averaged over projections (`:634-642`). Distribution-free, rotation-invariant. **Not instantiated anywhere**; the related `compute_wasserstein`/`compute_gmm_wasserstein` calls in `run_analysis.py` are all commented out (`:1156-1162`, `:1200-1206`, `:1236-1242`, `:1455-1465`). Disabled because TFV6 profiles are signed (negative entries) and `compute_wasserstein` *raises* on genuinely negative weights (§5.5; cross-ref finding 4.5).

### 3.6 `JensenShannonDetector` (`detectors.py:669-793`) — ATOMs profile

Sliced JSD: `n_projections=50` random directions (`:697`); per projection, the training data is a Gaussian **KDE** (Scott's-rule bandwidth `h = N^(−1/5)·std`, `:719`, floored at 1e-8) and the test point is a narrow Gaussian (`σ = bw/4`, `:763`); JSD is integrated on an `n_grid=1000` grid (`:758`) via `_js_divergence` (`:733-745`, natural-log, bounded `[0, ln2]`). This is a *different* JSD from the pipeline's actual JSD scorer.

**The pipeline uses `DistanceComputer.compute_jsd` instead** (`run_analysis.py:1148-1154`), which treats the **profile vector itself** as a discrete probability distribution and computes JSD against the **baseline mean profile** directly (`distance_computer.py:194-250`) — no projections, no KDE. The GMM JSD variant (`compute_gmm_jsd`, `:274-298`) takes the min JSD over cluster-mean profiles. So `JensenShannonDetector` (sliced/KDE) is also dead in the offline pipeline; the live JSD is the simple discrete-distribution JSD (finding 7.4).

### 3.7 `ActionEntropyDetector` / PEOC (`detectors.py:278-370`) — logit space

- **Input space:** raw logits or probabilities. TFV6: 8-bin speed logits. WoR: 28-dim joint action logits (optionally sliced to one command via `cmd`/`num_steers`/`num_throts`, `:335-338`, unused in the current runs which pass `cmd=None`).
- **score** (`:320-349`): softmax (with max-subtraction for stability) then Shannon entropy `H = −Σ p log p` in nats (`:347-349`, clipped at 1e-12). Higher entropy = more anomalous.
- `fit` is a no-op; save/load is a small JSON of the four config fields.

### 3.8 `GMMClustering` — BIC/AIC sweep and K-resolution (`clustering.py`)

- **fit** (`:82-130`): sklearn `GaussianMixture(n_components=K, covariance_type=conf.GMM_COV_TYPE, random_state=self.random_state, max_iter=200, n_init=5)`. **Seed caveat:** the `random_state` is the *constructor* argument — the production GMM in `run_analysis.py:494` is built with `random_state=conf.RANDOM_SEED = 17`, so the fitted model uses **seed 17**, not the class default 42. The `select_n_components` BIC/AIC sweep (`:236-277`), by contrast, is called *without* a `random_state` and so uses its own default **42** (Topic 8 §… / finding 8.5). The two stages therefore seed differently. Stores `means_`, `weights_`, and full `[K,C,C]` covariances via `_expand_covariances` (`:324-342`, which densifies diag/spherical/tied). Note: the `ridge_mat` computed at fit (`:122-123`) is **unused** — it is computed but never applied (the ridge is applied later inside `compute_mahalanobis` at score time; finding 7.5).
- **predict / predict_proba / score / score_per_cluster** (`:136-220`): hard label, soft membership, min-Mahalanobis distance to clusters, and per-cluster distance vector.
- **select_n_components** (`:236-277`): fits a fresh GMM for `K = 1..max_components` (`n_init=5`) and returns `argmin(BIC or AIC) + 1`. Called twice in `run_analysis.py` (`:458` BIC, `:464` AIC) over `K = 1..GMM_MAX_K`.
- **K-resolution precedence** (verified at `run_analysis.py:475-479`, in assignment order):
  1. `N_COMPONENTS = best_k_bic` (BIC sweep winner) — `:475`
  2. overwritten by `conf.NUM_GMM_CLUSTERS` **if not None** — `:476-477`
  3. overwritten by `_cli.gmm_k` **if not None** — `:478-479`

  Because each line overwrites the previous, the **effective precedence is CLI `--gmm-k` > `conf.NUM_GMM_CLUSTERS` > BIC**. This **matches** CLAUDE.md's claim ("(1) `--gmm-k`, (2) `conf.NUM_GMM_CLUSTERS`, (3) BIC"). With the live config `NUM_GMM_CLUSTERS = 10` (`atoms_config.py:21`), the BIC result is always discarded unless the config is set to `None` or the CLI overrides it. `sweep_clusters.py` drives K via `--gmm-k`.
- **save/load** (`:283-317`): persists means/covariances/weights/K/ridge/cov-type. Load reconstructs a usable sklearn GMM via `_reconstruct_gmm` (`:344-366`), which does a throwaway `fit` on the means themselves then overwrites the parameters and recomputes `precisions_cholesky_` — an inelegant workaround acknowledged in the code comments (`:355-357`).

### 3.9 `MDXDetector` (`detectors.py:800-1127`) — deep feature space (Zhang et al. 2024)

- **Input space:** 512-d backbone (v1, and v2 with current config) or 256-d F_c (v2 intended). Raw, pre-PCA.
- **fit(data[N,D], actions[N,3])** (`:942-1052`):
  1. **Held-out calibration split** — `calibration_split=0.2` of the data is set aside (seeded `default_rng(0)`, `:982`) for the conformal threshold and never seen by PCA/covariance (`:979-990`). *This is an addition beyond Zhang's vanilla MD; it implements the §5.2 conformal calibration.*
  2. **PCA** to `min(n_pca_components, D, N)` dims (`:992-1002`); warns and shrinks if data is too small.
  3. **Bin edges** from training-action ranges via `_build_bin_edges` (`:882-910`) — equal-width (`np.linspace(min,max,nb+1)`) or quantile (`np.unique(np.quantile(...))`, collapsing constant dims to a single bin).
  4. **Discretise** each action into `class = steer_bin·(n_throt·n_brake) + throt_bin·n_brake + brake_bin` (`discretise_action`, `:912-936`); default `3×2×2 = 12` classes (`:866`).
  5. **Per-class Gaussian** μ_c, Σ_c (+ ridge·I) in PCA space; classes with <2 samples are skipped (`:1016-1031`). This is exactly Zhang Eq. 4 with **per-class (quadratic-discriminant) covariance**, not a tied covariance.
- **score(x)** (`:1058-1085`): `min_c (x_pca−μ_c)ᵀ Σ_c⁻¹ (x_pca−μ_c)` — the **squared** Detection Mahalanobis Distance M(s), Zhang Eq. 5. (Note: this is squared and unrooted, unlike `MahalanobisDetector.score` which returns the rooted distance — the two Mahalanobis-style detectors are *not* on the same scale, but that is fine since each is thresholded/AUC'd independently.)
- **chi2_threshold** (`:1087-1095`): `χ²_{n_pca}(1−α)`, Prop 1. **conformal_threshold** (`:1097-1107`): the `ceil((1−α)(n+1))/n` quantile of the calibration scores (`:1038-1040`). Neither is used for AUC.
- **save/load** (`:1113-1127`): pickles the whole object (PCA, class means, precisions). Saved as `mdx_parameters.pkl` / `mdx_v2_parameters.pkl`.

### 3.10 `DistanceComputer` static metrics (`distance_computer.py`)

| Method | Line | Formula / behaviour | Numerical guards |
|---|---|---|---|
| `compute_mahalanobis` | 28 | `√((Δ)ᵀ(Σ+λI)⁺(Δ))` using **pinv** (not inv) | NaN/Inf/negative → clamp to `[0, 1e6]` (`:58-63`) |
| `compute_euclidean` | 67 | `‖Δ‖₂` | non-finite → 1e6 |
| `compute_knn_distance` | 89 | **k-th** sorted L2 distance; optional L2-normalise | raises if `k>N`; non-finite → 1e6 |
| `compute_gmm_distance` | 137 | min (`"nearest"`) or softmax-weighted (`"weighted"`) Mahalanobis over K clusters | inherits `compute_mahalanobis` guards |
| `compute_jsd` | 194 | renormalise p,q to sum 1; `0.5(KL(p‖m)+KL(q‖m))`, `m=½(p+q)`; nats by default | clip to ≥1e-12; negative/non-finite → 0.0 (`:247-248`) |
| `compute_gmm_euclidean` | 254 | min L2 to any cluster mean | — |
| `compute_gmm_jsd` | 274 | min `compute_jsd` over cluster means | — |
| `compute_gmm_wasserstein` | 301 | min `compute_wasserstein` over cluster means | inherits W guards |
| `compute_wasserstein` | 327 | scipy W1 over positions `[0..K-1]`, weights p,q | **raises** if `min < −1e-5`; else `max(·,0)` then renormalise; non-finite → 1e6 |

Key behavioural notes: `compute_mahalanobis` uses **`np.linalg.pinv`** (pseudo-inverse, `:55`) whereas `MahalanobisDetector` and the GMM precompute use **`np.linalg.inv`** — pinv is more forgiving of singular covariances, but the two code paths differ. The default `regularization` in the signature is **0.01** (`:32`), but every caller passes `conf.MAHAL_RIDGE` explicitly (also 0.01), so the stale `MahalanobisDetector(ridge=1e-6)` default is never the one used in the pipeline (cross-ref finding 1.14).

### 3.11 `DetectorEvaluator` — ROC / AUC / Youden-J (`detectors.py:377-473`)

`evaluate(scores, labels, name)` (`:391-434`) computes `roc_auc_score`, `roc_curve`, then `J = TPR − FPR` and reports the threshold/TPR/FPR/J at `argmax(J)` plus the full ROC arrays and `n_samples`/`n_anomalous`. `evaluate_from_detector` (`:436-452`) is a convenience wrapper that scores then evaluates. `save_results` writes JSON; `compare` prints a sorted table. The per-perturbation breakdown (Topic 8 step 11) reuses `evaluate` on masked score/label subsets.

---

## 4. Parameters & magic constants

| Constant | Value | Where | Configurable? | Effect |
|---|---|---|---|---|
| `MAHAL_RIDGE` | 0.01 | `atoms_config.py:88`; passed at `run_analysis.py:431,1121,1176,1218` | config | covariance ridge for all Mahalanobis (single + GMM) |
| `MahalanobisDetector` ridge default | 1e-6 | `detectors.py:163` | code default | never used (config 0.01 always passed); stale comment (finding 1.14) |
| `compute_mahalanobis` reg default | 0.01 | `distance_computer.py:32` | signature | unused default (callers pass explicitly) |
| Mahalanobis threshold percentile | 99.0 | `run_analysis.py:440`, `detectors.py:216` | code | label-free hard decision boundary (~1% FPR) |
| `NUM_GMM_CLUSTERS` | 10 | `atoms_config.py:21` | config | forced K; overrides BIC (None → use BIC) |
| `GMM_MAX_K` | 10 | `atoms_config.py:89`, `run_analysis.py:456` | config | upper K in BIC/AIC sweep |
| `GMM_COV_TYPE` | "full" | `atoms_config.py:90`, `run_analysis.py:462` | config | GMM covariance type (full/diag/tied/spherical) |
| GMM `random_state` | **17** (production fit) / 42 (sweep) | `run_analysis.py:494` (=`conf.RANDOM_SEED`) vs `clustering.py:242` default | config / code | production GMM seeds with `RANDOM_SEED=17`; BIC/AIC sweep uses default 42 (finding 8.5) |
| GMM `max_iter` / `n_init` | 200 / 5 | `clustering.py:109-110,269` | code | EM iterations / restarts |
| GMM `ridge` | 1e-6 | `clustering.py:64` | constructor | per-cluster ridge default (but `run_analysis.py` GMM uses `compute_gmm_distance` with `conf.MAHAL_RIDGE=0.01`, not this) |
| `KNN_K_VALUES` | `[1,5,10,25,50,100,250]` | `run_analysis.py:1134` | code | k sweep for k-NN and GMM-k-NN |
| `KNNDetector.k` default | 50 | `detectors.py:542` | constructor | unused (detector class not used in pipeline) |
| k-NN `normalize` | True | `run_analysis.py:1142,1269,1283,1303` | code | L2-normalise before k-NN distance |
| k-NN return | k-th distance | `distance_computer.py:131` | code | pipeline k-NN (≠ KNNDetector's mean-of-k) |
| `compute_knn_distance` default k | 100 | `distance_computer.py:92` | signature | unused (callers pass k) |
| `WassersteinDetector` `n_projections` | 200 | `detectors.py:615` | constructor | sliced-W projections (detector disabled) |
| `JensenShannonDetector` `n_projections` / `n_grid` | 50 / 1000 | `detectors.py:697-698` | constructor | sliced-JSD (detector unused; pipeline uses discrete JSD) |
| JSD KDE bandwidth | Scott `N^(−1/5)·std`, floor 1e-8 | `detectors.py:719,722` | code | sliced-JSD KDE bw (unused path) |
| sliced detectors `random_state` | 42 | `detectors.py:615,699` | constructor | projection seed (unused paths) |
| `compute_jsd` base | `np.e` (nats) | `distance_computer.py:197` | signature | JSD log base; bound `[0, ln2]` |
| `MDXDetector.n_pca_components` | 50 | `run_analysis.py:305,351,367,412`; default `detectors.py:846` | code/constructor | PCA dim (Zhang's value) |
| MDX bins | 3 steer × 2 throt × 2 brake = 12 | `detectors.py:847-849,866` | constructor | action-class count |
| MDX `ridge` | 1e-6 | `detectors.py:850` | constructor | per-class covariance ridge |
| MDX `alpha` | 0.05 | `detectors.py:851` | constructor | χ²/conformal significance |
| MDX `calibration_split` | 0.2 | `detectors.py:852` | constructor | held-out fraction for conformal threshold |
| MDX calibration seed | 0 | `detectors.py:982` | code | calibration-split RNG (≠ other seeds) |
| MDX-v1 `bin_strategy` | "equal-width" | `detectors.py:853` default | code | MDX-v1 binning |
| MDX-v2 `bin_strategy` | "quantile" | `run_analysis.py:411-412`, `MDX2_USE_QUANTILE_BINNING=True` `atoms_config.py:46` | config | MDX-v2 binning |
| `MDX2_USE_FC_FEATURES` | **False** | `atoms_config.py:45` | config | False → MDX-v2 uses 512-d backbone, not 256-d F_c (finding 7.6) |
| TFV6 MDX speed proxy | `[0, min(spd/25,1), 1 if spd<0.5 else 0]` | `run_analysis.py:346,404` | code | steer always 0.0; speed→throttle proxy; brake from low speed |
| PEOC input dim | 8 (TFV6 speed) / 28 (WoR action) | `run_analysis.py:1376-1378` / `:1310-1312` | derived | entropy feature space |
| Youden-J | `argmax(TPR−FPR)` | `detectors.py:420-421` | code | reported operating point (needs labels) |

---

## 5. Known limitations & open issues

- **GMM K forced outside the BIC range, now equal to the cap** (cross-ref finding 1.10) — `NUM_GMM_CLUSTERS = 10` (`atoms_config.py:21`) now *equals* `GMM_MAX_K = 10`, so the earlier "12 > 10" mismatch is gone, but the BIC/AIC sweep result is still always discarded in favour of the forced K=10 (which is the boundary of the sweep, the most-components end). The sweep is computed and plotted but never drives the live K unless the config is set to `None` (finding 7.1).
- **Stale Mahalanobis ridge comments** (cross-ref finding 1.14) — `detectors.py:160` ("Default 1e-6 … increase to 1e-4"), `:163` default `1e-6`, and `distance_computer.py:32` default `0.01` all coexist; the *effective* ridge everywhere is `conf.MAHAL_RIDGE = 0.01`, passed explicitly. The constructor default and the docstring are dead/misleading (finding 7.2).
- **Signed TFV6 profiles break Wasserstein, are tolerated by JSD** (cross-ref finding 4.5) — TFV6 ATOMs profiles can contain negative entries (signed AttnLRP relevance, Topic 4 §10). `compute_wasserstein` *raises* on `min < −1e-5` (`distance_computer.py:358-362`); `compute_jsd` silently renormalises a possibly-mixed-sign vector by its sum and clips to ≥1e-12 (`:222,230-232`), which is mathematically dubious if the profile sum is near zero or the negatives are large. This is the direct cause of the Wasserstein detector being disabled (§3.5) and a latent correctness concern for JSD on signed profiles (finding 7.3).
- **Simplex/normalisation assumptions in JSD and Wasserstein** — both `compute_jsd` and `compute_wasserstein` *assume* their inputs are (after renormalisation) probability distributions on a shared support. For ATOMs profiles this holds only loosely: the profile is per-frame renormalised in `process_frame`, but JSD against the *baseline mean* (`run_analysis.py:1150`) and `compute_gmm_jsd` against cluster means assume both vectors live on the same object simplex — defensible for non-negative WoR profiles, shaky for signed TFV6 ones (finding 7.3).
- **`KNNDetector` and `JensenShannonDetector`/`WassersteinDetector` are dead in the offline pipeline** — the stateful detector classes are bypassed: the pipeline uses `DistanceComputer.compute_knn_distance` (k-th, normalised) and `compute_jsd` (discrete) directly, which differ from the class implementations (mean-of-k; sliced-KDE JSD; sliced-W). The classes are still tested/saveable but produce *different scores* than what is reported, a maintenance and reproducibility hazard (finding 7.4).
- **MDX-v2 feature ablation is off and MDX-v2 is not scored** — `MDX2_USE_FC_FEATURES=False` collapses MDX-v2 to "MDX-v1 with quantile bins" on 512-d backbone features (finding 7.6); and the MDX-v2 *test-scoring + evaluation* blocks are commented out (`run_analysis.py:1350-1371,1552-1556`), so MDX-v2 contributes no AUC at all in the current pipeline (finding 7.7). The thesis should report MDX-v1 only, or re-enable v2.
- **k/K selection falls back to test labels when val is absent** — both k-NN selections explicitly fall back to test-AUC argmax with a `WARNING: … leakage` print (`run_analysis.py:1491-1493,1522-1525`). The warning is not enforced; a run without val files silently selects k on the test set (finding 7.8).
- **PEOC deviates from Sedlmeier** — TFV6 PEOC substitutes the 8-bin speed distribution for the policy `π(a|s)`; it measures speed-prediction confidence, not full policy entropy (no steering/waypoint uncertainty). Documented as an intentional adaptation (finding 7.9).
- **MDX conformal calibration is global, not class-conditional** — Zhang Eq. 8 defines a *per-action-class* quantile `Q^c_{1−α}`; `MDXDetector.fit` computes a single global quantile over all calibration scores (`detectors.py:1036-1040`). Harmless for the AUC numbers (conformal threshold is not used for ROC) but a deviation from the paper's distribution-free guarantee (finding 7.10).
- **Unused `ridge_mat` in `GMMClustering.fit`** — `ridge_mat` is computed at `:122-123` and `:312` but never applied; the ridge is instead applied at score time inside `compute_mahalanobis`. Dead code (finding 7.5).
- **`compute_mahalanobis` uses pinv, the detectors use inv** — `distance_computer.py:55` (`pinv`) vs `MahalanobisDetector`/GMM precompute (`inv`). Both ridge-regularise first, so divergence is unlikely, but the two paths are inconsistent (minor, noted for completeness).
- **Numerical clamping masks failures** — `compute_mahalanobis`/`euclidean`/`knn`/`wasserstein` all silently clamp non-finite results to `1e6` and JSD to `0.0`. A degenerate covariance or empty pool therefore produces a finite-but-meaningless score rather than an error, which can quietly bias AUC (finding 7.11).

---

## 6. Original methods (paper grounding)

### 6.1 MDX — Zhang et al., "A Simple Unified Framework for Anomaly Detection in Deep RL" (TMLR 10/2024)

The MDX framework feeds a state `s` into the pretrained policy network and extracts the **penultimate-layer feature** `f(s)`; states are categorised by the policy's discrete action class `c ~ π(·|s)` (paper §4, Fig. 2). For each class it estimates a class-conditional Gaussian `(μ_c, Σ_c)` (Eq. 4) — using **per-class (quadratic-discriminant) covariance**, explicitly rejecting the "tied covariance" of Lee et al. 2018 as implausible (p. 4, p. 7). The detection score is the **minimum Detection Mahalanobis Distance** to any class centroid:
`M(s) = min_c (f(s)−μ_c)ᵀ Σ_c⁻¹ (f(s)−μ_c)` (Eq. 5), which under the Gaussian assumption is χ²_p distributed (Prop 1), giving a hard threshold `Θ = χ²_p(1−α)`. The paper adds a **Robust MD** variant (MCD estimator, Eq. 6-7) and a **distribution-free conformal** variant (split conformal calibration, per-class quantile `Q^c_{1−α}`, Eq. 8) — and notes (footnote 2) that for **continuous** action spaces one should "discretise the actions into several bins and then follow the same detection pipeline".

**Implementation mapping & deviations.** `MDXDetector` implements vanilla MD + PCA-50 + the conformal split: Eq. 4 = `fit` per-class μ/Σ, Eq. 5 = `score` (squared, unrooted), Prop 1 = `chi2_threshold`, Eq. 8 = `conformal_threshold`. Deviations: (i) the continuous-action discretisation (footnote 2) is realised as the 3×2×2=12 steer/throttle/brake bins; (ii) for TFV6 the "action" is a *speed-derived proxy* with **constant steer**, so steer bins collapse — far from a real discrete policy class; (iii) Robust MD (MCD) is **not** implemented; (iv) the conformal quantile is **global, not per-class** (finding 7.10); (v) the feature is the 512-d backbone GAP (or 256-d F_c), TFV6's nearest analogue to a "penultimate layer".

### 6.2 PEOC — Sedlmeier et al., "Policy Entropy for Out-of-Distribution Classification" (2020)

PEOC frames OOD detection as **one-class classification** and uses the policy entropy `H(π(s_t)) = −Σᵢ π(aᵢ|s_t) log π(aᵢ|s_t)` (Eq. 2) as the classification score, on the hypothesis that successful training *reduces* entropy on in-distribution states, so OOD states have higher entropy (Eq. 3). It is evaluated by ROC/AUC as a binary classifier (paper §2.3) and is built on PPO/actor-critic *discrete-action* policies.

**Implementation mapping & deviations.** `ActionEntropyDetector(from_logits=True)` implements Eq. 2 exactly (softmax → Shannon entropy in nats). The deviation is the **proxy policy**: TFV6 has no single discrete action head, so the **8-bin speed distribution** stands in for `π(a|s)` (finding 7.9); WoR uses the genuine 28-dim joint action distribution. Evaluation via `DetectorEvaluator` ROC/AUC matches the paper's protocol.

---

## 7. Cross-references

- **01_architecture_overview.md** — `MAHAL_RIDGE`, `NUM_GMM_CLUSTERS`, `GMM_MAX_K`, `GMM_COV_TYPE`, `RANDOM_SEED`, `MDX2_USE_FC_FEATURES`, `MDX2_USE_QUANTILE_BINNING`, `RECOMPUTE_MDX_*` recompute flags as the config single-source-of-truth; the `--gmm-k` CLI override; finding 1.10 (K vs MAX_K) and 1.14 (ridge comment).
- **02_agents.md** — the feature extractors these detectors consume: `get_backbone_features` (512-d, MDX-v1), `get_fc_features`/`get_planning_action_and_features` (256-d F_c, MDX-v2), `get_speed_logits` (8-bin, PEOC), WoR `get_action_logits`/`model.get_features`; the speed bins and two-hot decoder (finding 2.4) behind the PEOC proxy; the single-member-vs-ensemble caveat (finding 2.5) — detectors score whichever policy produced the profiles.
- **04_atoms.md** — the ATOMs profile (dim 10 TFV6 / 29 WoR) that the profile-space detectors consume; the **signed** TFV6 profiles (finding 4.5) that disable Wasserstein and stress JSD (§5).
- **05_dataset_creation.md** — the baseline profiles fitted here come from `baseline_data`; the val set (disjoint Town05 routes, §2.5 there) is what k/K selection uses to avoid test leakage (§2.7 here); the test/val split that defines the `label` field.
- **06_perturbations.md** — the `label` (0 clean / 1 perturbed) and `perturbation` fields these detectors are scored against; the per-perturbation breakdown reuses `DetectorEvaluator.evaluate` on masked subsets; the deferred-PGD profiles (finding 6.3) feed these scorers.
- **08_offline_analysis.md** — `run_analysis.py` step ordering (steps 2.5/2.5-v2/4/5/6/9/9.5/12), the full detector roster table, `sweep_clusters.py` K-sweep driven by `--gmm-k` + `__val_auc_gmm_avg__`, `summarize_results.py` val-set K recommendation.
- **09_online_analysis.md** — `run_online_analysis.py` scores live frames with the same detectors; only the 99th-percentile (label-free) threshold is deployable there.
- **10_hpc_pipeline.md** — `mdx_features.npz` precomputation on HPC that feeds MDX-v1 fit; chunked profile computation that produces the baseline/test/val profiles scored here.
- **99_bugs_and_findings.md** — Topic 7 findings 7.1–7.11; cross-references 1.10 (K vs MAX_K), 1.14 (ridge comment), 4.5 (signed profiles), 2.4/2.5 (speed bins / single-member).
