# Topic 11 — Validation & Testing: The LRP/ATOMs Correctness Harness (property-based suites + mathematical diagnostics), and why it is dead-by-default

All claims verified against code on 2026-06-14. Line numbers refer to the current working tree.
Primary sources read in full: `ATOMs_Analysis/utils/lrp_test_suite.py` (1367), `ATOMs_Analysis/utils/tfv6_test_suite.py` (1023), `ATOMs_Analysis/utils/atoms_test_suite.py` (1313), `ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py` (1213), `ATOMs_Analysis/utils/wor_lrp_diagnostics.py` (882). Wiring verified against `ATOMs_Analysis/detection/baseline_dataset.py` (`:56,59,417-496`). Repo-wide caller search performed (`run_all_tests`, `*TestSuite(`, `*Diagnostics(`). PCLA smoke test skimmed (`pcla_functions/test_all_agents.py`, 251). Absence of pytest/CI confirmed by Glob (`pytest.ini`/`conftest.py`/`tox.ini`/`setup.cfg`/`pyproject.toml`/`.github/workflows/*` — none). Cross-checked against `documentation/03_lrp.md` (§7, §8), `documentation/04_atoms.md`, `documentation/05_dataset_creation.md` (§ which already flags the dead block at `:476-496`), `documentation/10_hpc_pipeline.md`, `CLAUDE.md`.

---

## 1. Purpose & scope

This document covers the **correctness-validation layer** of the project: the five standalone harnesses that test whether the LRP backends (Topic 3) and the ATOMs metric (Topic 4) compute what they claim to. This is the material a thesis "validation of the implementation" chapter is mined from — for a method where there is **no ground-truth relevance map** to compare against, the only available evidence of correctness is that the implementation satisfies the *mathematical properties* the rules are supposed to satisfy (conservation, the closed-form AttnLRP backward equations, determinism, node-routing, sum-normalization). The harnesses encode exactly those properties as PASS/WARN/FAIL assertions.

The layer has **two kinds** of artefact, with a deliberate division of labour:

1. **Property-based / output-sanity test suites** (`lrp_test_suite.py`, `tfv6_test_suite.py`, `atoms_test_suite.py`): black-box checks on the *outputs* of the public interface — "is the relevance finite, spatially non-trivial, conserved-enough, normalized, node-distinct, cmd-sensitive?" They assert sanity bands, not exact equalities, because most properties are only approximately true for the chosen rule set.
2. **Mathematical diagnostics** (`tfv6_lrp_diagnostics.py`, `wor_lrp_diagnostics.py`): white-box checks on the *internals* — they re-implement each AttnLRP equation by hand and compare it to the custom autograd `Function`'s backward to a tight tolerance (1e-5/1e-4), unit-test the seed constructor, verify the two-step decomposition is exact, measure the amplification budget, etc. The diagnostics are the closest thing the project has to a formal proof-of-correctness for the LRP rules.

The single most important structural fact, established by code search below: **none of these harnesses is invoked by any live pipeline.** The only place they were ever wired in — a block inside `BaselineComputer.compute()` — is entirely commented out (`baseline_dataset.py:476-496`); the only live reference is one import. There is no pytest, no `conftest.py`, no CI workflow, and no `RUN_TESTS` config flag. The validation harness is therefore **dead-by-default**: it can only be run by manually un-commenting the block or by a one-off script the operator writes by hand. For the thesis, the suites are a *source of evidence the author ran at some point* (the WARN-thresholds and decision history in their docstrings are clearly calibrated against real runs — e.g. L02's "CoV 2.67→0.15 after the ε change" lineage in Topic 3 §3.3), not an automated regression gate.

This topic does **not** re-derive the LRP rules or the ATOMs metric — it documents what each test *checks*, the criterion and tolerance, what implementation claim it validates, and the true invocation story. Where a test's pass/fail behaviour interacts with a known finding from an earlier topic (esp. 4.5 signed TFV6 profiles, 3.6 latent NaN, 3.7 attention-scale attenuation), that interaction is flagged in §5.

---

## 2. Key design decisions

### 2.1 Property-based / metamorphic testing because ground truth is unavailable

There is no reference relevance map for any frame; LRP is itself the definition of the attribution. The suites therefore test **invariants the method must satisfy regardless of input**, not "is the output correct":

- *Closed-form identity* (diagnostics D01/D02/D03, W04, W05): the few quantities that *do* have a known correct value — the AttnLRP softmax and matmul backward formulas (Prop 3.1/3.3), the seed constructor, the cross-normalization sum, the brake-selector slot positions — are tested against their hand-computed reference to ≤1e-4. These are genuine unit tests.
- *Metamorphic / relational* (L05/A05/A07, D08/D10, W06/W07, A08, D07): "different `node_id` → different map", "`forced_brake` ≠ `forced_drive`", "different `cmd` → different attention", "two-step ≡ one-pass". The expected *relationship* is known even though the absolute value is not. These catch the project's most important historical regression — Bug #1, where `node_id` was accepted but never passed to the backward seed, so every per-node map was identical (A07 is explicitly the regression test for it, `atoms_test_suite.py:622-740`).
- *Sanity-band* (L01/L03/L04/L06, T01/T03/T07/T08/T13): finiteness, spatial Gini/entropy in a plausible range, positive-fraction dominance, activation diversity, logit non-degeneracy. These are heuristic guards against the catastrophic failure modes (NaN/Inf explosion, all-zero maps, single-pixel spikes, dead network), with thresholds calibrated empirically.

### 2.2 PASS / WARN / FAIL, with WARN reserved for known-non-strict properties

All five harnesses share a three-state verdict (`PASS`/`WARN`/`FAIL`, plus `ERROR` for an exception inside the test). The crucial design choice is **WARN as a first-class state for properties that are only approximately true under the chosen rules**:

- **Conservation is a *stability* check, not an equality check, for TFV6** (L02, D06). AttnLRP on TFV6 is *not* conservative end-to-end (Topic 3 §8): the BN canonization is order-heuristic, residual additions are split by plain autograd, the 1/√d attention scale multiplies relevance through (finding 3.7), ε=1e-2 absorbs relevance, and `F.interpolate` is handled by autograd — a net ~2×10⁷ residual amplification in `fc→input`. So L02 explicitly does **not** assert `pixel_sum ≈ node_sum`; it asserts the *ratio* `pixel_sum/node_sum` is **stable across frames** (CoV < 0.2), with a docstring spelling out that "the absolute mean_ratio is not interpretable as a conservation check for TFV6" (`tfv6_test_suite.py:304-309`). The thesis claim it validates is the weaker but sufficient one: *relevance is conserved-enough and consistent-enough across frames that ATOMs profiles are comparable*. WoR's T02 is stricter (CoV < 0.10 PASS, < 0.30 WARN) because WoR's WSquare+AlphaBeta composite is closer to conservative.
- **Positive dominance thresholds are AttnLRP-adjusted** (L04). The WoR/z⁺ suite expects >70% positive pixels (T08), because z⁺ only propagates positive contributions. The TFV6 suite deliberately **lowers** the bar to FAIL<0.30 / WARN<0.45 (`tfv6_test_suite.py:471-472`) with a docstring noting "LRPSoftmax produces x·(R − s·ΣR) which can be negative … z⁺-only thresholds do not apply here." This is the test layer acknowledging the signed-relevance reality that finding 4.5 flags downstream.
- **GAP-collapse is reported as WARN-by-design for WoR** (W07): identical per-node maps are *expected* for WoR because the global average pool destroys spatial specificity before the FC layers (the architectural motivation for moving to TFV6, Topic 3 §4). W07 reports cosine≈1.0 as WARN with a note distinguishing the architectural limit from a genuine routing bug; the corresponding TFV6 test (D10/L05) treats identical maps as a hard FAIL.

### 2.3 Suites vs diagnostics — black-box vs white-box

The **suites** call only the *public* interface (`forward_relevance`, `process_frame`, `reset`, `get_series_df`, `get_mean_df`, `_relevance_filter`, `seg_to_masks`) and synthesize segmentation if absent — so they can run on any frame batch without touching internals. The **diagnostics** reach into *private* methods (`_attribute_to_fc`, `_attribute_backbone`, `_attribute_true_output_to_input`, `_make_speed_seed`, `_make_lidar`, `_attribute_to_concat`, `_build_drive_brake_selector`, `_prepare_input`) and even rebuild a second zennit composite (W09) to isolate a single mechanism. The split mirrors Topic 3's own §7 "Validation hooks" inventory (D-series diagnostics, L/A-series suite, W-series WoR diagnostics).

### 2.4 Per-agent harnesses, not one unified harness — and the mismatch it creates

The suites are **agent-specific and not interchangeable**, but this is implicit, not enforced:

- `lrp_test_suite.py` (`LRPTestSuite`, T01–T16) is **WoR-only**: it reads `lrp.num_steers/num_throts/num_speeds/num_cmds` in `__init__` (`:141-146`), processes `narr_rgb` every frame, and calls WoR-only methods (`_build_drive_brake_selector`, `_lerp_bins`, `_attribute_to_concat`, `model_lrp(wide, narr)`). Run against a `LRPTFv6Model` it would crash in `__init__`.
- `atoms_test_suite.py` (`ATOMsTestSuite`, A01–A12) is **WoR-flavoured**: it imports `CARLA_CLASSES` (the 29-class WoR set), passes `narr_rgb`/`seg_narr` to every `process_frame`, and A12 calls `lrp._model_eval.policy(...)` and `lrp._lerp_bins` (WoR-only).
- `tfv6_test_suite.py` (`TFV6TestSuite`, L01–L07 + A01–A05) is the **only TFV6-targeted suite**: wide-only, no WoR methods, asserts `narr` output is always `None` (L07).
- `tfv6_lrp_diagnostics.py` (D01–D12) targets `LRPTFv6Model`; `wor_lrp_diagnostics.py` (W01–W09) targets `LRPCameraModel`.

The dead invocation block (§3.6) constructs **both** a WoR suite *and* the TFV6 suite/diagnostics against the *same* `BaselineComputer.self.lrp` — which is typed `LRPCameraModel` (WoR, `baseline_dataset.py:56,434`). Even if un-commented, the TFV6 lines would ERROR on a WoR instance (finding 11.3). The author evidently swapped this block between agents by hand.

### 2.5 Why standalone/manual rather than CI-wired — and the consequence

The harnesses are slow (they each run a full LRP backward per frame, the same cost the whole HPC layer exists to offload, Topic 10 §2.1) and require a loaded model + checkpoint, so wiring them into every `run_analysis.py` run would add minutes per experiment for no per-run benefit. That justifies *not* running them on every analysis. But the chosen alternative — a commented-out block plus docstring usage examples — means there is **no path that runs them at all** without manual edits, **no record of when they last passed**, and **no protection against regressions**. A change to `lrp_transfuser.py` or `atoms_carla.py` that breaks node-routing or sum-normalization would be caught by A07/A04 *only if someone remembers to un-comment and run them*. For the thesis this is the honest framing: the suites document the properties the author verified during development, but they are not a living regression gate, and at least one of them is now stale enough that it cannot pass as written (T12, §5).

---

## 3. Implementation details

### 3.1 The shared `TestResult` model and `run_all_tests` flow

All five files independently define an identical dataclass and harness skeleton (copy-paste, not shared — finding 11.6):

```
@dataclass
class TestResult:
    name: str; status: str; summary: str
    metrics: Dict[str, Any]; per_frame: Optional[np.ndarray]
    notes: List[str]; exception: Optional[str]
```

`run_all_tests(testframes)` iterates a hardcoded `[(name, method), …]` list, wraps each via `_safe_run` (catches any exception → `ERROR` status + traceback string, so one broken test never aborts the battery), records `wall_time_s`, prints a one-line `✓/✗/△/!` verdict, and returns an ordered `Dict[name → TestResult]`. `print_report` dumps metrics+notes; `save_report(out_dir)` writes a `*_report.txt` and one `<name>_per_frame.npy` per test that populated `per_frame`. **No file ever aggregates the verdicts into an overall pass/fail exit code** — the summary is a printed PASS/WARN/FAIL count, not a return value, so even a manual run cannot be used as a shell gate without extra code.

`testframes` is a **dict-of-lists** (`{"wide_rgb": [...], "narr_rgb": [...], "speed": [...], "cmd": [...], optional "seg_wide"/"seg_narr"/"is_brake"/"data": [...]}`) — the same in-memory `data` dict that `BaselineDataLoader.load_all_runs` returns (Topic 5), which is why the dead block could pass `data` straight in. The suites cap how many frames they touch: `LRPTestSuite.max_checks = 50` (`:148`), `ATOMsTestSuite.max_frames = 20` (`:143`), `TFV6TestSuite.max_frames = 10` (`:198`); the diagnostics use per-test caps of 4–8 frames (e.g. D05 `min(N, 8)`, D07 `min(N, 4)`).

### 3.2 Catalogue — `lrp_test_suite.py` (WoR, T01–T16)

`LRPTestSuite(atoms, lrp, device='cpu', mode=2)`. Black-box on the WoR `LRPCameraModel`. (ID | property | criterion/tolerance | semantics | file:line.)

| ID | Property validated | Criterion / tolerance | Verdict logic | Line |
|---|---|---|---|---|
| T01 | No NaN/Inf in `fc→input` relevance | any NaN or Inf | FAIL if any, else PASS | 210 |
| T02 | Conservation **stability**: pixel_sum/output_sum ratio | CoV < 0.10 / 0.30 | PASS/WARN/FAIL | 260 |
| T03 | Amplification **stability**: |R|_pixel sum across frames | CoV < 0.15 / 0.40 | PASS/WARN/FAIL | 336 |
| T04 | Brake-mode classification rate is non-zero | rate>0 / ≥0.02 | FAIL if 0%, WARN<2% | 384 |
| T05 | Brake logit can reach the brake-prob>0.5 threshold (≈ln 54) | ever > ln(num_steers·num_throts·2) | PASS if reachable, else WARN | 474 |
| T06 | wide/narr split not stuck at 0 or 1 | degenerate-frame frac >0.8 / >0.2 | FAIL/WARN/PASS | 547 |
| T07 | Spatial coherence (Gini/entropy/top-1% mass) | top1%>0.90 spike, entropy<0.5 flat | FAIL if >20% spike | 608 |
| T08 | Sign ratio (z⁺ → mostly positive pixels) | mean pos-frac > 0.70 / 0.40 | PASS/WARN/FAIL | 676 |
| T09 | Drive/brake selector sums to 1.0 | \|sum−1\|>1e-4 | FAIL if any bad | 727 |
| T10 | `_lerp_bins` valid (x0,x1∈range, w∈[0,1]) | out-of-range bins; >10% clamped | FAIL/WARN | 788 |
| T11 | Node-selection coverage (mode 1): not 0, not all 256 | 0 nodes; >50% select all 256 | FAIL/WARN | 837 |
| T12 | `update_context` deterministic + frame-sensitive | same<1e-6 det.; diff>1e-4 sens. | **see §5 — reads a nonexistent attribute → WARN always** | 909 |
| T13 | Logit range sanity (not dead, not exploding) | std<1e-5 dead; \|logit\|>20 explode | FAIL if >10% dead or any explode | 985 |
| T14 | Mode 2 vs mode 3 wide_frac consistency | Pearson r > 0.8 / 0.5 | PASS/WARN/FAIL | 1054 |
| T15 | GT-brake alignment (precision/recall vs `is_brake`) | recall<0.3 FAIL, <0.6 WARN | needs GT labels else WARN | 1120 |
| T16 | Concat-level (576-d) vs pixel-level narr split + conservation | total rel∈[0.6,1.4]; correction gap>0.02 | FAIL on conservation break or gap | 1270 |

Notes: T02/T03 are the WoR conservation/stability validation that L02 mirrors for TFV6. T05's `print(exec)` on the error path (`:197`) is a typo (`exec` builtin, not `exc`) — harmless but evidence the error branch was never exercised (finding 11.5). T16 contains a misspelling in a FAIL message ("relevbance"), again a never-hit path.

### 3.3 Catalogue — `tfv6_test_suite.py` (TFV6, L01–L07 + A01–A05)

`TFV6TestSuite(lrp, atoms, device='cpu')`. The header docstring (`:48-61`) lists the catalogue. L-series = LRP output sanity (wide-only, no WoR methods); A-series = ATOMs integration via the shared `ATOMsCarla` API with `narr=None`.

| ID | Property validated | Criterion / tolerance | Verdict logic | Line |
|---|---|---|---|---|
| L01 | No NaN/Inf in all 3 modes (`output→input`, `output→fc`, `fc→input[0]`) | any NaN/Inf | FAIL/PASS | 260 |
| L02 | **Conservation stability** pixel_sum/node_sum | **CoV < 0.2**; FAIL if mean≈0 | PASS/WARN/FAIL; abs magnitude explicitly *not* interpreted | 299 |
| L03 | Non-trivial spatial map | Gini∈(0.05,0.99), entropy≥2 bits | FAIL if >½ frames degenerate | 367 |
| L04 | Positive dominance (**AttnLRP-adjusted**) | FAIL<0.30, WARN<0.45 (not z⁺ 0.55/0.70) | min<0.30 FAIL, mean<0.45 WARN | 431 |
| L05 | Per-node `fc→input` map distinctiveness | rel_diff<1e-5 identical → FAIL; <0.01 WARN | probes nodes at sorted-magnitude 0,n/4,n/2,3n/4 | 503 |
| L06 | 256-d LRP1 activation diversity | not all-zero, inter-frame std>1e-8, Gini≥0.1 | FAIL on any | 577 |
| L07 | `narr` output always `None` (wide-only invariant) | any non-None | FAIL/PASS | 646 |
| A01 | `process_frame(narr=None)` returns sum∈[0.99,1.01] or 0.0 | \|s−1\|>0.01 and \|s\|>0.01 | FAIL on any bad | 679 |
| A02 | Per-frame `_hierarchical` increment ≥ 0 | any value < −1e-8 | FAIL on any negative | 729 |
| A03 | Accumulation exactness: Σ`_frame_series` == `_hierarchical` | err<1e-8 (series), <1e-10 (`get_hierarchical`) | FAIL otherwise | 773 |
| A04 | `reset()` clears all state | `_hierarchical`/`_frame_series`/`_frame_cmds`/`_n_frames`/`_current_masks_wide` | FAIL on any non-clear | 820 |
| A05 | Node-routing via ATOMs (`_lrp2_pixels` per node) | rel_diff<1e-5 identical → FAIL; <0.01 WARN | needs ≥2 selected nodes else WARN | 865 |

L02 (CoV<0.2) ↔ Topic 3's "relevance is conserved-enough to be meaningful"; A01/A03 (sum-normalization + accumulation) ↔ "profiles are comparable across frames and the cumulative profile is the exact sum of per-frame contributions"; L05/A05 (node distinctiveness) ↔ Bug #1 fix; A04 (reset) ↔ the `atoms.reset()`-between-datasets contract (Topic 4).

### 3.4 Catalogue — `atoms_test_suite.py` (ATOMs metric, A01–A12)

`ATOMsTestSuite(atoms, device='cpu')`. Validates the metric layer *on top of* LRP (Topic 4). A01–A03 are **pure unit tests** (no model/frames); A04–A12 are integration tests. WoR-flavoured (imports `CARLA_CLASSES`, passes `narr`/`seg_narr`).

| ID | Property validated | Criterion / tolerance | Verdict logic | Line |
|---|---|---|---|---|
| A01 | `seg_to_masks`: shape [C,H,W], binary {0,1}, correct pixels, **non-overlap** (≤1 class/pixel), order preserved | exact equality | FAIL on any | 197 |
| A02 | `_relevance_filter`: coverage ≥ p, **minimality**, descending order, `[]` for all-zero, `[0]` single-elem, **abs of negatives** | coverage ≥ p−1e-5 | FAIL on any | 277 |
| A03 | `_give_element_selectivity` V = **non-zero-relevance pixels**, not object area (paper R̄ denominator; regression for fix #5) | result 2.5 (correct) vs 1.25 (area) within 1e-4 | FAIL if area-V or unexpected | 362 |
| A04 | `process_frame` returns sum≈1.0 (or 0.0 degenerate) | \|s−1\|>0.01 and \|s\|>0.01 | FAIL on any | 444 |
| A05 | Hierarchical accumulation exactness | err<1e-8 / <1e-10 | FAIL otherwise | 504 |
| A06 | `reset()` zeros all state incl. `_frame_brake`, `_frame_wide_frac` | any non-clear | FAIL on any | 565 |
| A07 | **★ Node-map diversity** (Bug #1 regression) — up to 5 pairs of top-nodes | rel_diff<1e-5 identical → FAIL; <0.01 WARN | one distinct pair ⇒ routing alive | 632 |
| A08 | Command conditioning sensitivity (cmd 3 vs 1, same frame) | rel_diff<1e-6 FAIL, <0.01 WARN | FAIL if identical | 746 |
| A09 | `get_series_df` integrity: n rows, columns=class_names+[cmd,wide_frac], wide_frac∈[0,1], class rows sum≈1, cmd integer | per-check | FAIL on any | 822 |
| A10 | Per-frame contributions non-negative (every class) | any < −1e-8 | FAIL on any | 910 |
| A11 | `get_mean_df` groupby: one row/cmd, rows sum≈1, non-empty | \|sum−1\|>0.02 | FAIL on any | 969 |
| A12 | Command-LRP-routing diagnostic (logit L1 / LRP1 cosine / mode-3 rel vs cmd-agnostic baseline) | routing<3× baseline FAIL; cos>0.98 WARN | layered diagnosis | 1045 |

**Relation to finding 4.5 (signed TFV6 profiles):** A10 (and the TFV6 mirror A02 in §3.3) explicitly **asserts every per-class contribution is ≥ 0** and A04/A11 assume profile rows sum to ≈1.0 on a shared simplex. Both assumptions hold for WoR (cross-normalize takes abs, Topic 3 §4) but are **violated by TFV6**, where pixel maps are raw and signed and per-class sums can be negative (finding 4.5). So if A10 were ever run on TFV6 profiles it should FAIL, and A04/A11 could fail or pass spuriously on a near-zero signed total — see §5. A02's check "negative values treated as absolute" (`:338-342`) is the one place the metric layer is verified to abs-fold the *node* relevances (Topic 3 Decision C / `atoms_carla.py:431-435`), which is the only abs that TFV6 actually applies.

### 3.5 The two diagnostics scripts (white-box internals)

**`tfv6_lrp_diagnostics.py` — `TFV6LRPDiagnostics(lrp, device=...)`, D01–D12.** Reports/outputs: printed PASS/WARN/FAIL with per-test metrics; `save_report` writes `tfv6_diagnostics_report.txt` + `diag_<name>_per_frame.npy`. No figures. Tests:

| ID | What it computes | FAIL criterion | Line |
|---|---|---|---|
| D01 | `LRPSoftmax.backward` vs hand-computed Prop 3.1 `x·(R−s·ΣR)` on 4 cases | max err > 1e-5 | 187 |
| D02 | `LRPMatMul.backward` vs Prop 3.3 `R_A/R_B` + approximate conservation Σ(R_A)+Σ(R_B)≈Σ(R_O) | per-grad err > 1e-4; conservation err > 5% (skipped in ε-dominated regime, \|O_mean\|<100·EPS) | 241 |
| D03 | `_make_speed_seed` 3 modes + is_brake = argmax==0 | any assertion (sum≠1, wrong one-hot bin, is_brake≠argmax) | 329 |
| D04 | `_make_lidar` requires_grad=False, deterministic, 2-channel, ramps∈[0,1] | grad path, non-determinism, wrong chans | 425 |
| D05 | LRP1 z⁺ conservation 0<Σnode_rel≤1 | Σ>1.01 (created rel) or mean<0.05; WARN if >80% frames Σ<0.2 | 494 |
| D06 | Backbone amplification ratio pixel_sum/node_sum + pos-frac | \|ratio\|>50; WARN if CoV>1.0 | 586 |
| D07 | Two-step (`_attribute_true_output_to_input`) ≡ manual LRP1+LRP2 | rel L∞ > 1e-3 | 687 |
| D08 | forced_brake vs forced_drive node_rel distinct | cosine>0.9999 on **all** frames | 766 |
| D09 | is_brake independent of forced flag | differs across the 3 calls | 840 |
| D10 | Per-node pixel-map pairwise cosine (top-K≤8) | min cosine>0.9999 (Bug #1); WARN mean>0.90 | 903 |
| D11 | Bias fraction in `target_speed_decoder` z⁺ denominator | mean>0.50 FAIL, >0.30 WARN | 992 |
| D12 | LRP output determinism (3 modes, repeat call) | max diff > 1e-6 | 1093 |

D01/D02/D03 are self-contained unit tests of the custom autograd Functions and seed — they validate the exact claims of Topic 3 §3.2 (LRPSoftmax/LRPMatMul implement Prop 3.1/3.3) and §3.6 (softmax seed, is_brake semantics). D07 validates Topic 3's "two-step output→input is exact by linearity" (Bug F). D05/D06/D11 quantify the non-conservation budget Topic 3 §8 describes. `_get_data` (`:1143`) feeds per-frame conditioning via `update_context`, falling back to `_make_minimal_data` when `frames["data"]` is absent — so unless the caller supplies a `data` list, the diagnostics run with the zero target_point/acceleration conditioning gap of finding 3.x/4 (the offline limitation Topic 4 documents).

**`wor_lrp_diagnostics.py` — `WoRLRPDiagnostics(lrp, device=...)`, W01–W09.** Same report format → `wor_diagnostics_report.txt`. Tests:

| ID | What it computes | FAIL criterion | Line |
|---|---|---|---|
| W01 | `JointCameraForLRP` logits ≡ `CameraModel` logits | max diff > 1e-3 | 167 |
| W02 | `JointCameraToFC` ≡ `act_head[:4]` (256-d FC) | max diff > 1e-4 | 228 |
| W03 | `undo_resnet_amplification`: pixel wide_frac ≈ concat wide_frac | gap > 0.02 | 289 |
| W04 | Cross-normalization: wide.sum()+narr.sum() = 1.0 exactly | err > 1e-5 | 365 |
| W05 | Brake-selector mass only at correct logit slots, weights (1−w,w) | wrong slots or weights | 427 |
| W06 | forced_brake vs forced_drive selector masks distinct | cosine>0.9999 on **all** frames | 522 |
| W07 | FC-node pixel-map cosine — **WARN-by-design (GAP collapse)** | min cos>0.9999 → WARN (not FAIL) | 584 |
| W08 | LRP determinism (3 modes × 2 cameras) | max diff > 1e-6 | 680 |
| W09 | `zero_params='bias'` actually changes LRP (builds a 2nd composite without it) | rel_diff<1e-4 FAIL, <0.01 WARN | 735 |

W01/W02 are the WoR analogue of D01–D03 in spirit: they prove the *re-wrapped* model attributes the **same function** as the deployed `CameraModel` — the single most important WoR correctness claim (a mismatch means "all LRP attributions are for the wrong function"). W07 is the explicit GAP-collapse acknowledgement (Topic 3 §4, finding from `design_decisions.md`); W09 builds a second `SpecialFirstLayerMapComposite` *without* `zero_params='bias'` and diffs the maps to confirm the bias-exclusion convention (Topic 3 §4) is active.

### 3.6 The dead invocation block (`baseline_dataset.py:476-496`)

The **only** wiring of the harness into a pipeline, verbatim shape:

```
# line 59 (LIVE):
from ATOMs_Analysis.utils.lrp_test_suite import LRPTestSuite

# lines 476-496 inside BaselineComputer.compute() (ALL COMMENTED):
# TESTING ----------------------------------------------------------------------
#self.narr_tester = LRPTestSuite(atoms=self.atoms, lrp=self.lrp)
#self.narr_tester.run_all_tests(data)
#from ATOMs_Analysis.utils.atoms_test_suite import ATOMsTestSuite
#self.atoms_tester = ATOMsTestSuite(atoms=self.atoms)
#self.atoms_tester.run_all_tests(data)
#from ATOMs_Analysis.utils.tfv6_test_suite import TFV6TestSuite
#suite  = TFV6TestSuite(self.lrp, self.atoms); report = suite.run_all_tests(data); suite.print_report(report)
#from ATOMs_Analysis.utils.tfv6_lrp_diagnostics import TFV6LRPDiagnostics
#diag = TFV6LRPDiagnostics(self.lrp); report = diag.run_all_tests(data); diag.print_report(report)
#from ATOMs_Analysis.utils.wor_lrp_diagnostics import WoRLRPDiagnostics
#diag = WoRLRPDiagnostics(self.lrp); report = diag.run_all_tests(data); diag.print_report(report)
# END OF TESTING ---------------------------------------------------------------
```

**Definitive wiring status:** the `LRPTestSuite` import (`:59`) is the *only live reference anywhere in the repo*; it is imported but never used (its constructor at `:477` is commented). Every instantiation and every `run_all_tests`/`print_report` call is commented. A repo-wide search for `run_all_tests` / `*TestSuite(` / `*Diagnostics(` returns **only** these commented lines plus the harness files' own docstring usage-examples and method definitions — i.e. **no external caller exists**. Three further problems with the block even if un-commented (findings 11.3): (a) `BaselineComputer.self.lrp` is a WoR `LRPCameraModel` (`:56,434`), so the `TFV6TestSuite`/`TFV6LRPDiagnostics` lines would ERROR; (b) `data` is a dict-of-arrays whereas `ATOMsTestSuite`/`TFV6TestSuite` index it like `frames["wide_rgb"][i]` (works for ndarray) but `LRPTestSuite.run_all_tests` types `testframes: List[Dict]` in its signature (`:154`) — a stale type hint, it actually consumes the dict-of-lists; (c) it would run the full battery **inside the baseline compute loop**, multiplying its already-large cost.

### 3.7 What is NOT part of this layer

`pcla_functions/test_all_agents.py` is a **PCLA agent smoke test**: it loads every agent in `agents.json`, spawns a vehicle in Town02, runs 20 CARLA frames per agent via `pcla.get_action()`, and writes pass/fail + tracebacks to `documents/agent_test_results.txt` (`:204-236`). It exercises the *deployment* path (does each pretrained agent initialise and step without crashing), requires a live CARLA server, and touches **nothing** in the LRP/ATOMs validation story. It is the only file in the repo with "test" in a runnable sense, but it is not a unit/property test and not part of the implementation-validation chapter. There is **no pytest, no `conftest.py`, no `tox.ini`/`setup.cfg`/`pyproject.toml`, and no `.github/workflows`** (Glob-confirmed), and **no test toggle in `atoms_config.py`** (the `TEST_*` config keys are all dataset-sizing, e.g. `MAX_TEST_SIZE`, `TEST_SAMPLE_INTERVAL`).

---

## 4. Parameters & magic constants

All hardcoded in the harness source (none configurable; no harness imports `atoms_config`).

| Constant | Value | Where | Meaning |
|---|---|---|---|
| Frame cap (WoR suite) | `max_checks = 50` | `lrp_test_suite.py:148` | frames scanned per test |
| Frame cap (ATOMs suite) | `max_frames = 20` | `atoms_test_suite.py:143` | per integration test |
| Frame cap (TFV6 suite) | `max_frames = 10` | `tfv6_test_suite.py:198` | per test |
| Diagnostics caps | 4–8 (per test) | e.g. D05 `min(N,8)`, D07 `min(N,4)` | per-test frame slices |
| **TFV6 conservation CoV** | **< 0.2** | `tfv6_test_suite.py:346` | L02 PASS band (stability, not equality) |
| WoR conservation CoV | < 0.10 / 0.30 | `lrp_test_suite.py:305` | T02 PASS/WARN |
| WoR amplification CoV | < 0.15 / 0.40 | `lrp_test_suite.py:359` | T03 PASS/WARN |
| TFV6 positive-frac thresholds | FAIL 0.30 / WARN 0.45 | `tfv6_test_suite.py:471-472` | L04 (AttnLRP-adjusted) |
| WoR positive-frac threshold | mean > 0.70 / 0.40 | `lrp_test_suite.py:702` | T08 (z⁺) |
| Gini band (TFV6 L03) | (0.05, 0.99) | `tfv6_test_suite.py:395-398` | spatial sanity |
| Entropy floor (TFV6 L03) | ≥ 2.0 bits | `tfv6_test_suite.py:399` | spatial sanity |
| Node-map "identical" tol | rel_diff < 1e-5 | L05/A05/A07/D10/W07 | FAIL/WARN routing |
| Node-map "similar" WARN | rel_diff < 0.01 (suites), cosine>0.90 (diag) | L05/A05/A07; D10/W07 | low diversity |
| Forced-seed identical tol | cosine > 0.9999 | D08/W06 | comparative-map zero |
| Determinism tol | 1e-6 | D12/W08, T12 (same-frame) | repeat-call equality |
| Accumulation tol | 1e-8 (series), 1e-10 (`get_hierarchical`) | A03/A05 | exactness |
| Profile sum band | sum ∈ [0.99, 1.01] or 0.0 | A01/A04 | normalization |
| Row-sum band (DataFrames) | \|sum−1\| ≤ 0.02 | A09/A11 | normalized rows |
| Non-negativity tol | < −1e-8 | A02/A10 | signed-leak guard |
| Softmax-backward tol (D01) | 1e-5 | `tfv6_lrp_diagnostics.py:219` | Prop 3.1 |
| Matmul-backward tol (D02) | 1e-4 grad; 5% conservation | `tfv6_lrp_diagnostics.py:290,307` | Prop 3.3 |
| `LRPMatMul.EPS` (used by D02) | 1e-6 | read from `LRPMatMul.EPS` | matmul stabilizer (Topic 3 §3.2) |
| ε-dominated skip threshold | \|O_mean\| < 100·EPS | `tfv6_lrp_diagnostics.py:300` | conservation check waived |
| LRP1 conservation bounds (D05) | 0 < Σ ≤ 1.01; mean ≥ 0.05; 80% < 0.2 | `tfv6_lrp_diagnostics.py:526-551` | z⁺ rule |
| Amplification cap (D06) | \|ratio\| ≤ 50; CoV ≤ 1.0 | `tfv6_lrp_diagnostics.py:640,654` | budget |
| Two-step consistency (D07) | rel L∞ ≤ 1e-3 | `tfv6_lrp_diagnostics.py:737` | exactness |
| Bias-fraction thresholds (D11) | 0.50 FAIL / 0.30 WARN | `tfv6_lrp_diagnostics.py:1060,1065` | bias absorption |
| WoR forward-match tol (W01/W02) | 1e-3 / 1e-4 | `wor_lrp_diagnostics.py:202,262` | same-function proof |
| WoR cross-norm tol (W04) | 1e-5 | `wor_lrp_diagnostics.py:401` | sum=1 by construction |
| WoR undo-amp gap (W03) | 0.02 | `wor_lrp_diagnostics.py:332` | correction effectiveness |
| WoR bias-exclusion (W09) | 1e-4 FAIL / 0.01 WARN | `wor_lrp_diagnostics.py:796,802` | zero_params effect |
| `p_relevance` for node probes (D10/W07) | 0.9 (hardcoded literal) | `tfv6_lrp_diagnostics.py:924`, `wor_lrp_diagnostics.py:619` | node selection in diag (≠ HPC 0.25 / config default; finding 4.3 context) |
| Synthetic-seg RNG seed | 42 | all suites `_synthetic_seg` | reproducible random seg when absent |
| Probe `cmd` pairs (A08/A12) | 3 vs 1 (A08), ref 3 (A12) | `atoms_test_suite.py:771,1083` | "FOLLOW_LANE" assumption (WoR-indexed; cf. finding 1.7 cmd inconsistency) |

---

## 5. Known limitations & open issues

- **The entire validation harness is dead-by-default / not in CI (11.1).** Only the `LRPTestSuite` import is live; every invocation is commented (`baseline_dataset.py:476-496`). No pytest, no `conftest.py`, no CI workflow, no `RUN_TESTS` config flag, no aggregate exit code. Regressions in `lrp_transfuser.py`/`lrp_analysis.py`/`atoms_carla.py` are caught only if the author manually un-comments and runs the suite. This is the central finding; it is consistent with the broader "feature exists but is commented out / dead" pattern (2.9 destroy()-flush, 7.7 MDX-v2 scoring, 8.3 trajectory step, 8.4 Wasserstein-GMM).
- **`ATOMsTestSuite` non-negativity tests would FAIL on TFV6 profiles (11.2; ties to 4.5).** A10 asserts every per-class contribution ≥ 0 and the TFV6 mirror A02 asserts per-frame `_hierarchical` increments ≥ 0; A04/A11/A09 assume rows sum to ≈1 on a non-negative simplex. TFV6 maps are raw/signed (finding 4.5, Topic 3 §3.5), so per-class sums can be negative and the normalized sum can be off a near-zero signed total. The suites encode the *WoR-correct* invariant; they do not tolerate the signed TFV6 profiles the detectors (and finding 4.5) flag. The TFV6 suite's own L04 *acknowledges* signed maps (AttnLRP-adjusted thresholds) but its A02 (non-negative contributions) still asserts ≥ 0 — internally inconsistent: L04 says "signed is expected", A02 says "contributions must be ≥ 0". On TFV6 A02 should fire whenever a signed map drives a class sum negative.
- **`LRPTestSuite` T12 is stale and cannot pass as written (11.4).** It reads `self.lrp.act_fc_model_lrp.head.fixed_context` (`:931`); neither `act_fc_model_lrp` nor `fixed_context` exists on `LRPCameraModel` (grep-confirmed — the live class exposes `fc_model_lrp`/`model_lrp`, no frozen-context buffer). The `AttributeError` is swallowed and T12 always returns WARN "Could not access fixed_context buffer". The determinism/sensitivity property is therefore **never actually tested** for WoR — a silently-dead test.
- **Suites are agent-specific but the binding is unenforced; the dead block mixes them against one WoR instance (11.3).** `LRPTestSuite`/`ATOMsTestSuite` are WoR-only (`num_steers`, `_build_drive_brake_selector`, `_model_eval.policy`, `CARLA_CLASSES`, mandatory `narr`); `TFV6TestSuite`/`TFV6LRPDiagnostics` are TFV6-only. The commented block constructs the TFV6 harnesses against `BaselineComputer.self.lrp` (a WoR `LRPCameraModel`), which would ERROR. No agent-type assertion guards either suite.
- **Never-exercised error paths contain bugs (11.5).** `lrp_test_suite.py:197` prints `exec` (the builtin) instead of the traceback `exc` on a test-function exception; T16's FAIL message misspells "relevbance" (`:1325`). Both are in branches that have evidently never run — circumstantial confirmation the suite is not routinely executed.
- **Five copy-pasted `TestResult`/`_safe_run`/reporting blocks (11.6, low).** The dataclass, `_safe_run`, `_to_tensor`, `_gini`/`_entropy_bits`, and the `print_report`/`save_report` machinery are duplicated verbatim across all five files (the TFV6 suite even comments "mirrored from lrp_test_suite.py"). A shared `test_common.py` would remove ~300 lines of drift-prone duplication.
- **Tolerances calibrated to looseness for known non-conservation, with the inherent caveat.** L02/T02/D06's CoV bands and D05/D06's amplification caps are wide enough to PASS the systematic ~2×10⁷ TFV6 amplification (Topic 3 §8) — which is correct (the design only needs *stability*), but it also means these checks would **not** flag a moderate regression in the conservation budget as long as it stayed *stable* across frames. They validate "comparable across frames", not "physically conserved". The thesis should state the conservation evidence is stability-of-ratio, not equality (mirroring Topic 3 §8).
- **Conditioning gap propagates into the diagnostics.** Unless the caller supplies a per-frame `data` dict, `TFV6LRPDiagnostics._get_data` falls back to `_make_minimal_data` (zero target_point/acceleration), so D05/D06/D07/D08/D11 validate the *offline-conditioned* attribution, not the live-agent one (the same limitation as finding 4.4 / Topic 3 §8 issue 2). A passing diagnostic suite does not certify the live-agent attribution path.
- **No coverage of several live behaviours.** There is no test for: the perturbation transforms (Topic 6), the detector math (Topic 7), the HPC chunk workers' hardcoded `p_relevance=0.25` (finding 4.3 — a suite run with the default 0.9 would not reproduce HPC profiles), MODE_ANALYSIS=2/3 dispatch on TFV6, or the `_make_minimal_data` cmd-default inconsistency (1.7/2.12). The harness validates the LRP+ATOMs core only.

---

## 6. Cross-references

- **01_architecture_overview.md** — the suites deliberately do not import `atoms_config`; their probe constants (p=0.9 for node selection, cmd=3/1) are hardcoded literals, decoupled from `FC_RELEVANCE_FILTER`/`DEFAULT_CMD` (findings 4.3, 1.7).
- **02_agents.md** — W01/W02 prove the re-wrapped WoR model attributes the *same function* as the deployed `CameraModel`; the suites read agent-specific structure (`num_steers/throts/speeds/cmds`, `speed_query`/256-d F_c).
- **03_lrp.md** — this topic is the detailed companion to §7 (validation hooks inventory) and §8 (non-conservation budget). D01/D02/D03 unit-test the custom autograd Functions (§3.2) and seed (§3.6); D05/D06/D11 quantify the §8 budget; D07 validates the two-step exactness (Bug F); L02/T02 implement conservation as a *stability* check exactly because of §8; L04's AttnLRP-adjusted positive-fraction thresholds track §3.5's signed maps; D06 references the 1/√d attenuation (finding 3.7); the latent NaN of finding 3.6 is *not* covered by any test (the masks are never passed in any harness either).
- **04_atoms.md** — A01–A12 validate the metric layer: `seg_to_masks` non-overlap, `_relevance_filter` coverage/minimality, the R̄ non-zero-pixel denominator (A03 = fix #5), profile sum-normalization, `get_series_df`/`get_mean_df` integrity, node-routing (A07 = Bug #1), command sensitivity (A08/A12). The non-negativity assertions (A02/A10) collide with the signed-TFV6-profile finding 4.5 (§5).
- **05_dataset_creation.md** — already flags this dead block at `:476-496`; the suites consume the same `BaselineDataLoader` dict-of-arrays (`data`) the migration produces.
- **06_perturbations.md / 07_distances_and_detectors.md** — out of scope for the harness (no perturbation or detector tests exist); a gap noted in §5.
- **08_offline_analysis.md / 09_online_analysis.md** — neither `run_analysis.py` nor `run_online_analysis.py` invokes any suite; the validation layer is orthogonal to the analysis pipelines.
- **10_hpc_pipeline.md** — the HPC chunk workers (which actually produce the committed profiles) run with `p_relevance=0.25` (finding 4.3), whereas the diagnostics probe nodes at p=0.9; the suites would not reproduce HPC profile semantics. The harness validates the *local* LRP+ATOMs code, not the HPC-produced artefacts.
- **12_visualization.md** — the harnesses emit `*_report.txt` + `*_per_frame.npy`, not figures; the one QC figure in the HPC chain (`visualize_perturb.py`, finding 10.6) is unrelated.
- **99_bugs_and_findings.md** §"From Topic 11" — findings 11.1–11.6; cross-references 2.9, 3.6, 3.7, 4.3, 4.5, 7.7, 8.3, 8.4.
