# Implementation Plan — Validation Set (§2.1) & MDX Rework (§2.4)

**Status:** Part A — proposed, not yet implemented. Part B — **implemented as MDX-v2 (additive; MDX-v1 unchanged).**
**Companion docs:** `docs/code_review.md` (§2.1, §2.4), `docs/design_decisions.md`.
**Author note:** these are the two items deferred from the 2026-06-08 code-review fix
session because they change the experimental protocol / feature extraction and need a
deliberate decision before coding.

Both changes will require **regenerating result JSONs** (and, for some options, the
test profiles). Plan to run them together with the PGD-fix regeneration so the data is
only rebuilt once.

---

# Part A — Validation set for hyperparameter selection

## A.1 Problem definition

`run_analysis.py` chooses the two free hyperparameters of the detector suite by
maximising AUC **on the same labelled set it then reports**:

- `run_analysis.py:1282` `best_k = max(results_knn_by_k, key=lambda k: results_knn_by_k[k]["auc"])`
- `run_analysis.py:1297` `best_k_gmm = max(results_knn_gmm_by_k, key=…["auc"])`
- the GMM cluster count **K** is swept by `sweep_clusters.py` and
  `summarize_results.py` reports the **best AUC achieved over K** (`summarize_results.py:24`,
  `extract_best_k` at `:94`, headline tables in `results_summary/SUMMARY.md`).

This is selection on the evaluation set: the reported k-NN / k-NN-GMM / GMM numbers are
**oracle upper bounds**, and the comparison against the fixed-hyperparameter baselines
(Mahalanobis-single, Euclidean, JSD, Wasserstein, MDX, PEOC) is unfair.

## A.2 Goal / acceptance criteria

1. Three **disjoint, town-separated** sets exist and are documented:
   - **baseline** (clean) → fits every detector. *(all towns except Town05 and the val town)*
   - **validation** (labelled: clean+perturbed) → selects `k`, `k_gmm`, `K`. *(a reserved town)*
   - **test** (labelled: clean+perturbed) → the ONLY set whose AUC is reported. *(Town05, unchanged)*
2. No hyperparameter is ever chosen using test-set AUC.
3. The validation set is a **new** set produced by `migrate_lead_to_baseline.py` from a
   town that appears in **neither** the baseline nor the test set.
4. Every headline number in `SUMMARY.md` is a test AUC produced by a
   validation-selected hyperparameter (or a fixed one).
5. `k`-vs-AUC and `K`-vs-AUC sensitivity figures plot the **validation** curve used for
   selection (optionally overlaying the test curve for transparency).

## A.3 Design decision — reserved validation town

The migrate script already partitions LEAD data by town: `migrate()` writes the baseline
from all towns except Town05, and `migrate_testset()` writes the test set from Town05.
Extend this with a third, reserved town for validation, so the three sets are disjoint by
construction (a **cross-town holdout** — stronger than a random split):

| Set | Town(s) | Output dir |
|---|---|---|
| baseline | all LEAD towns **except** Town05 **and** `VAL_TOWN` | `conf.BASELINE_DATA_DIR/frames` |
| validation | `VAL_TOWN` (e.g. `Town03`) | `conf.VAL_DATA_DIR/frames` |
| test | Town05 | `conf.TEST_DATA_DIR/frames` |

> Pick `VAL_TOWN` from a town present in your LEAD data with enough routes, not equal to
> Town05. It must be **removed from the baseline** so baseline/val stay disjoint.

## A.4 Exact instructions

### Step 1 — config: add the validation town + dir (`ATOMs_Analysis/atoms_config.py`)
After the existing `*_DATA_DIR` definitions (`atoms_config.py:53-55`):

```python
VAL_TOWN     = "Town03"                 # reserved for validation; must be in LEAD data, != Town05
VAL_DATA_DIR = _DATA_ROOT / "val_data"  # mirrors TEST_DATA_DIR layout
```

### Step 2 — migrate: add `migrate_valset()` + exclude the val town from baseline
In `migrate_lead_to_baseline.py`:

a) Add a validation builder mirroring `migrate_testset` (`:361`):
```python
def migrate_valset(lead_dir: Path, n_frames: int = 500,
                   include_towns: Optional[List[str]] = None) -> None:
    """Convert LEAD routes to clean VALIDATION npz files (default: VAL_TOWN only)."""
    if include_towns is None:
        include_towns = [conf.VAL_TOWN]
    routes = discover_routes(lead_dir)
    if not routes:
        raise FileNotFoundError(f"No valid routes found under {lead_dir}")
    plan    = build_sampling_plan(routes, n_frames, exclude_towns=[], include_towns=include_towns)
    out_dir = Path(conf.VAL_DATA_DIR) / "frames"
    total   = _write_plan(plan, out_dir)
    LOG.info("Done — %d validation frames written to %s", total, out_dir)
```

b) Make the baseline exclude **both** Town05 and the val town. Change the `migrate()`
default (`:344-345`) and the CLI default for `--exclude_towns` (`:421`) to
`["Town05", conf.VAL_TOWN]` (or pass it explicitly on the command line, see below).

c) Add a CLI mode `valset` (extend `--mode` choices at `:413` and the dispatch at `:434`):
```python
parser.add_argument("--valset_n_frames", type=int, default=500)
parser.add_argument("--valset_towns", nargs="*", default=None)   # default → [conf.VAL_TOWN]
...
if args.mode in ("valset", "all"):
    migrate_valset(args.lead_dir, args.valset_n_frames, args.valset_towns)
```

Build commands (run once):
```bash
python migrate_lead_to_baseline.py --lead_dir <LEAD> --mode baseline --exclude_towns Town05 Town03
python migrate_lead_to_baseline.py --lead_dir <LEAD> --mode valset            # Town03 → val_data/frames
python migrate_lead_to_baseline.py --lead_dir <LEAD> --mode testset           # Town05 → test_data/frames
```

### Step 3 — PerturbationApplier: allow a custom data dir
`ATOMs_Analysis/detection/dataset.py`, `PerturbationApplier.__init__` (`:232-236`) hardcodes
`self._data_dir = conf.TEST_DATA_DIR`. Add an optional override so it can read the val
frames and write `val_labeled.npz` into `conf.VAL_DATA_DIR`:

```python
def __init__(self, perturbation_manager, model=None, data_dir=None):
    self._data_dir = Path(data_dir) if data_dir is not None else conf.TEST_DATA_DIR
    self._pm    = perturbation_manager
    self._model = model
    self._out_path = self._data_dir / "test_labeled.npz"   # overwritten in apply()
```

`apply()` already reads `self._data_dir / "frames"` (`:273`) and writes
`self._data_dir / f"{output_name}.npz"` (`:385`), so no other change is needed there.

### Step 4 — run_analysis: build + profile the validation set
Mirror Step 8 (perturb) and Step 9 (ATOMs) for the validation data. After the test-set
blocks:

```python
# --- Build labelled validation set (same spec, different seed) ---
applier_val = PerturbationApplier(pm, model, data_dir=conf.VAL_DATA_DIR)
applier_val.apply(spec=spec, seed=43, output_name="val_labeled")   # seed != test's 42
val_data   = LabeledTestLoader.load_path(conf.VAL_DATA_DIR / "val_labeled.npz")
val_labels = val_data["label"].astype(np.int32)

# --- ATOMs profiles on validation (mirror Step 9, incl. the §3.3 key guard) ---
VAL_ATT = Path(conf.VAL_DATA_DIR) / "attention"; VAL_ATT.mkdir(parents=True, exist_ok=True)
val_profiles = <compute exactly as test_profiles, looping val_data frames>
np.save(VAL_ATT / f"val_profiles_{_mode}.npy", val_profiles)
# + save val_profiles_{_mode}.keys.npy and verify on load, same as §3.3
```

> **PGD deferral note:** for TFV6 the `pgd` frames in `val_labeled.npz` store clean pixels
> (crafted on HPC), exactly like the test set. Either (a) run the val set through the same
> HPC PGD/profile flow, or (b) select hyperparameters using only the non-PGD validation
> perturbations. Either way selection stays independent of the test set. (a) is preferred
> for consistency.

### Step 5 — select on validation, report on test
Compute each detector's scores for **both** `val_profiles` and `test_profiles`, then:

```python
# k-NN: pick k on val, report on test
best_k = max(KNN_K_VALUES, key=lambda k: evaluator.evaluate(
    knn_scores_val[k], val_labels)["auc"])
results_knn = evaluator.evaluate(knn_scores_test[best_k], test_labels,
                                 f"ATOMs-k-NN (k={best_k}, val-selected)")
```
- Do the same for `best_k_gmm` (replaces `:1282`/`:1297`).
- For **K**: loop a K grid, fit `GMMClustering(K)` on `baseline_series`, score the val set,
  pick `K* = argmax_K val_AUC` (primary = Mahalanobis-GMM); refit at `K*` and report all
  GMM-* detectors on **test**. This replaces the `conf.NUM_GMM_CLUSTERS` / external-sweep
  headline path.
- Fixed detectors (Mahalanobis-single, Euclidean, JSD, Wasserstein, MDX, PEOC): report on
  the test set as before (no val needed).
- Per-perturbation breakdown (`:1348`–`1433`) stays on the test set.

### Step 6 — reporting / docs
- `summary.json` / `results_*.json`: store `auc` (= test AUC) plus `selected_k`/`selected_K`
  and `auc_val` for transparency.
- `summarize_results.py`: drop the "best achievable over K" path (`:24`, `extract_best_k`,
  the per-perturbation max); report the single val-selected `K*` and its test AUC.
- Document the three-town split (towns, frame counts, seeds) in `CLAUDE.md`, `SUMMARY.md`,
  and `docs/design_decisions.md`. Add `VAL_DATA_DIR` to the Data-Layout section of `CLAUDE.md`.

## A.5 Verification
- Assert the town sets are disjoint: no `run_<town>_*` name appears in more than one of
  baseline/val/test `frames/` dirs (the `run_id`/run filenames encode the town).
- Assert val and test each contain both classes (`0 < labels.sum() < len(labels)`).
- Confirm the val build is deterministic (fixed sampling plan + seed) across two runs.
- Sanity: any fixed detector's **test** AUC must not change when `VAL_TOWN` / val size
  changes (it never touches val).
- Apply the §3.3 key guard to `val_profiles` too (alignment with `val_labeled.npz`).

## A.6 Risks / notes
- `VAL_TOWN` must exist in the LEAD dump with enough routes; reserving it shrinks the
  baseline fit set slightly — check baseline frame count stays adequate (≫ profile dim).
- Cross-town selection means `k`/`K` are chosen to generalise across towns; if `VAL_TOWN`
  differs a lot from Town05 the selection may be slightly conservative — this is honest and
  acceptable (and arguably a stronger result than same-distribution selection).
- The val set is subject to the same file-overwrite caveats as the test set; the `_mode`
  suffix on `val_profiles_{mode}.npy` and the key companion file handle the main risks.
- PGD on the val set is HPC-deferred for TFV6 (see Step 4 note).

---

# Part B — MDX rework for TFV6 (§2.4) — IMPLEMENTED AS MDX-v2

## B.1 Problem definition

`MDXDetector` (Zhang et al. 2024) builds class-conditional Gaussians over
`n_steer × n_throt × n_brake = 12` discretised action classes and scores by the
minimum class Mahalanobis distance. For TFV6 the action proxy is degenerate
(`run_analysis.py:339`):

```python
actions_list.append([0.0, min(spd / 25.0, 1.0), 1.0 if spd < 0.5 else 0.0])
#                     ^steer is ALWAYS 0
```

With steer constant, `_build_bin_edges` (`detectors.py:887`) makes all three steer bins
collapse to one, so only throttle×brake vary → ≤4 of the 12 classes are ever populated.
This starves MDX of structure and explains its weak AUC (≈0.61 vs WOR's ≈0.67, where a
real action distribution is available).

## B.2 Goal / acceptance criteria — STATUS

All four criteria were met by the MDX-v2 implementation (2026-06-08):

1. **Non-degenerate steer** — `get_planning_action_and_features` uses mean lateral
   waypoint offset (`pred_future_waypoints[..., 0].mean()`) as the steer proxy. ✓
2. **Balanced classes** — quantile binning via `bin_strategy="quantile"` in `MDXDetector`.
   Expected `len(_class_means) > 4` after fit. ✓ (verify by running with `RECOMPUTE_MDX_V2_BASELINE=True`)
3. **MDX-v2 re-evaluated** — reported alongside MDX-v1 in `run_analysis.py` Step 10/11. ✓
4. **Agent-safe** — MDX-v1 code path untouched; WOR path unaffected. `bin_strategy`
   defaults to `"equal-width"` preserving all existing behaviour. ✓

**Implementation note:** rather than modifying MDX-v1, the fix was realised as a
parallel **MDX-v2** detector. This preserves the MDX-v1 result as a baseline for
comparison while also reporting the improved detector. See `docs/design_decisions.md`
(MDX-v2 section) for the full design rationale.

## B.3 Exact instructions

### Step 1 — Derive a real steering signal for TFV6
The TFV6 model predicts waypoints; the lateral offset of the predicted path is a valid
steering proxy and is exactly the quantity the (now-fixed) PGD steer target uses
(`pred.pred_future_waypoints[..., 0]`).

Add a helper to `LRPTFv6Model` (`ATOMs_Analysis/saliency/lrp_transfuser.py`, next to
`get_backbone_features` at `:899`):

```python
def get_planning_action_proxy(self, wide_rgb, cmd: int = 4, spd: float = 0.0):
    """Forward pass → (steer_proxy, throttle_proxy, brake_proxy) for MDX.

    steer_proxy  : mean lateral (x) offset of predicted future waypoints
                   (same signal as the PGD steer target).
    throttle/brake: derived from the decoded target speed.
    """
    import numpy as np
    wide_t = wide_rgb.float().to(self.device)
    data   = _make_minimal_data(float(spd), self.device, cmd=int(cmd))
    with torch.no_grad():
        # full_model returns the speed_query; we also need waypoints + speed.
        # Easiest: run the underlying agent model forward instead (see note).
        ...
    return float(steer), float(throttle), float(brake)
```

> **Note on access:** `LRPTFv6Model.full_model` only returns the `speed_query` token, not
> the waypoint head. Two clean options:
> - (preferred) In `run_analysis.py` Step 3 TFV6 branch, you already hold the real
>   `model` (`TFv6`). Call `pred = model(_make_minimal_data(...))` (or the agent's
>   forward) once per baseline frame to read `pred.pred_future_waypoints` and
>   `pred.pred_target_speed_distribution`, and build the action there — no new method
>   needed. This mirrors how `pgd_attack_tfv6` already reads `pred.*`.
> - (alt) extend `TFv6FullModelForLRP.forward` to also return waypoints.

Concretely, replace `run_analysis.py:339` (TFV6 baseline action extraction) with:

```python
spd  = float(runs_dict["speed"][i])
cmd  = int(runs_dict["cmd"][i])
data = _make_minimal_data(spd, device, cmd=cmd)          # import from lrp_transfuser
with torch.no_grad():
    pred = model(data if model_takes_data_dict else {**data, "rgb": wide_t})
steer    = float(pred.pred_future_waypoints[..., 0].mean())   # lateral offset proxy
speeds   = conf.<speed_bins>                                  # [0,4,8,10,13.89,16,17.78,20]
tgt_v    = float((torch.softmax(pred.pred_target_speed_distribution, -1)
                  * torch.tensor(speeds)).sum())
throttle = min(tgt_v / 20.0, 1.0)
brake    = 1.0 if tgt_v < 0.5 else 0.0
actions_list.append([steer, throttle, brake])
```

(Use the decoded expected speed for throttle/brake instead of the raw ego speed — it
reflects the policy's intent, consistent with MDX operating on policy outputs.)

The **same** `(steer, throttle, brake)` must also be produced for the **test** frames
when MDX scores them (`run_analysis.py` Step 11 MDX block) — but note MDX scores from
**features**, not actions, so only the baseline `fit` needs actions. No test-time action
change is required.

### Step 2 — Quantile binning (balanced classes)
Modify `MDXDetector._build_bin_edges` (`detectors.py:880`) to support quantile edges and
make it the default:

```python
def __init__(self, ..., bin_strategy: str = "quantile"):   # add param
    ...
    self.bin_strategy = bin_strategy

def _build_bin_edges(self, steers, throts, brakes):
    def edges(v, nb):
        v = np.asarray(v, float)
        if self.bin_strategy == "quantile":
            e = np.unique(np.quantile(v, np.linspace(0, 1, nb + 1)))
            if e.size < 2:                          # degenerate (constant) dim
                e = np.array([v.min() - 1e-6, v.max() + 1e-6])
        else:                                       # legacy equal-width
            e = np.linspace(v.min(), v.max(), nb + 1)
        e[0] -= 1e-6; e[-1] += 1e-6
        return e
    self._steer_edges = edges(steers, self.n_steer_bins)
    self._throt_edges = edges(throts, self.n_throt_bins)
    self._brake_edges = edges(brakes, self.n_brake_bins)
```

`discretise_action` (`detectors.py:895`) already uses `np.digitize` + `np.clip`, which
works with unequal/quantile edges unchanged. Quantile edges with duplicate values
(e.g. brake ∈ {0,1}) collapse via `np.unique`, yielding fewer but populated bins.

### Step 3 (optional) — Better MDX feature
MDX currently uses 512-d pooled backbone features (`get_backbone_features`, `:899`).
The 256-d `speed_query` (the F_c node, closest to the decision) may be a stronger
"penultimate policy feature" in the spirit of Zhang et al. Add `get_fc_features` that
returns the `speed_query` and try MDX on it; keep the backbone-feature variant for
comparison. Treat this as an A/B experiment, not a guaranteed win.

### Step 4 — Re-fit, re-evaluate, document
- Set `RECOMPUTE_MDX_BASELINE=True` once to rebuild `mdx_parameters` with the new
  actions/binning; then back to False.
- Report the new TFV6 MDX AUC next to the old 0.61 and the WOR 0.67.
- Record the change in `docs/design_decisions.md`.

## B.4 Verification
- After `fit`, assert `len(self._class_means) > 4` (more than the old throttle×brake
  count) and that the steer dimension uses >1 bin (`np.unique(class_labels // (n_throt*n_brake)).size > 1`).
- Print the per-class population counts; confirm no class has the entire dataset.
- Confirm the steer proxy is non-constant: `np.std(actions[:,0]) > 0`.
- Re-run the full evaluation and compare MDX AUC (val-selected nothing here — MDX has no
  tunable hyperparameter besides PCA dim, which stays at 50).

## B.5 Risks / notes
- The waypoint lateral offset has the model's own coordinate convention; binning is
  scale-free (quantiles), so absolute sign/scale does not matter for MDX.
- Running `model.forward` per baseline frame is slower than backbone-only; this only
  affects the one-off MDX fit, and can be done on the HPC alongside feature extraction
  (`hpc/compute_baseline_chunk.py`) where `mdx_features.npz` is produced — update that
  script to also store the new 3-D action.
- Keep WOR's MDX action extraction (`run_analysis.py:294`–`302`) untouched; it already
  uses the real steer/throttle/brake distribution.

---

# Sequencing

1. Land Part A and Part B behind the existing recompute flags.
2. Regenerate together with the PGD-fix data refresh (one rebuild):
   `REAPPLY_PERTURBATIONS` (if needed) → HPC PGD/profiles → `RECOMPUTE_MDX_BASELINE` →
   run `run_analysis.py` (now val/test-aware) → `summarize_results.py`.
3. Update `CLAUDE.md`, `SUMMARY.md`, and `docs/design_decisions.md` to describe the
   three-town baseline/validation/test separation and the new MDX action.
