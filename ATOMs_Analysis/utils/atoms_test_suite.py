"""
atoms_test_suite.py
===================
Correctness test suite for the ATOMsCarla semantic-attribution pipeline.

Tests the ATOMs logic that sits *on top of* the LRP layer:
segmentation mask generation, relevance filtering, per-node map diversity,
frame accumulation, DataFrame integrity, command conditioning, and
non-negativity of contributions.

Pair this with LRPTestSuite, which covers the underlying LRP mechanics
(relevance conservation, sign ratio, selector normalization, speed bins,
node count, etc.).  The suites are complementary — this one does not
repeat those tests.

Usage
-----
    suite  = ATOMsTestSuite(atoms_instance)
    report = suite.run_all_tests(testframes)
    suite.print_report(report)
    suite.save_report(report, "out/")

Testframe format  (same dict-of-lists as LRPTestSuite)
-------------------------------------------------------
Each key is a list of length n_frames:
    wide_rgb  : np.ndarray uint8 [H,W,3]  or torch.Tensor [1,3,H,W]
    narr_rgb  : np.ndarray uint8 [H,W,3]  or torch.Tensor [1,3,H,W]
    speed     : float     — current speed in m/s
    cmd       : int       — navigation command index (0-based)

Optional:
    seg_wide  : np.ndarray uint8 [H,W]  — red channel of wide CARLA sem-seg
    seg_narr  : np.ndarray uint8 [H,W]  — red channel of narr CARLA sem-seg
    frame_id  : str       — for labelling

If seg_wide / seg_narr are absent, synthetic random segmentations are
generated automatically.  Tests A01–A03 are pure unit tests and never
need frames or a live model.

Test inventory
--------------
  A01  seg_to_masks correctness         — shape, binary, pixel coverage, non-overlap
  A02  _relevance_filter coverage       — coverage ≥ p, minimality, ordering, edge cases
  A03  _give_element_selectivity V      — nonzero-pixel denominator (regression fix #5)
  A04  process_frame output normalized  — returned attention sums to ≈ 1 per frame
  A05  hierarchical accumulation        — _hierarchical == sum(_frame_series) exactly
  A06  reset clears all state           — every accumulator zeros; series/cmds empty
  A07  node-map diversity  ★            — CRITICAL regression for fix #1:
                                          two different node_ids must yield different maps
  A08  command conditioning sensitivity — different cmds → different attention
  A09  get_series_df integrity          — shape, columns, dtypes, wide_frac ∈ [0,1]
  A10  contributions non-negative       — no negative per-class contribution after abs/normalize
  A11  get_mean_df groupby              — one row per command; rows sum to ≈ 1
"""

from __future__ import annotations

import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Import ATOMs helpers — adjust the import path to match your project layout
from ATOMs_Analysis.saliency.atoms_carla import (
    ATOMsCarla,
    CARLA_CLASSES,
    _relevance_filter,
    seg_to_masks,
)


# ---------------------------------------------------------------------------
# Shared constants and data structures  (mirrors lrp_test_suite.py style)
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
ERROR = "ERROR"


@dataclass
class TestResult:
    name:      str
    status:    str                        # PASS / FAIL / WARN / ERROR
    summary:   str                        # one-line verdict
    metrics:   Dict[str, Any]            = field(default_factory=dict)
    per_frame: Optional[np.ndarray]      = None   # scalar-per-frame where applicable
    notes:     List[str]                 = field(default_factory=list)
    exception: Optional[str]            = None


def _safe_run(fn, *args, **kwargs):
    """Run fn; return (result, None) or (None, traceback_str) on exception."""
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc()


def _to_tensor(rgb, device) -> torch.Tensor:
    """Accept [H,W,3] ndarray or [1,3,H,W] tensor; always return [1,3,H,W] float32."""
    if isinstance(rgb, np.ndarray):
        t = torch.from_numpy(rgb).unsqueeze(0).float()
    else:
        t = rgb.float()
    return t.to(device)


def _synthetic_seg(H: int, W: int, class_ids: List[int]) -> np.ndarray:
    """Random CARLA semantic-seg red channel using the given class_ids."""
    rng = np.random.default_rng(42)
    return rng.choice(class_ids, size=(H, W)).astype(np.uint8)


# ---------------------------------------------------------------------------
# ATOMsTestSuite
# ---------------------------------------------------------------------------

class ATOMsTestSuite:
    """
    Parameters
    ----------
    atoms   : ATOMsCarla  — already initialised with a valid LRPCameraModel.
    device  : str         — torch device string (default 'cpu').

    Notes
    -----
    Tests A01–A03 are pure unit tests; they run without testframes.
    Tests A04–A11 are integration tests; they require testframes.
    All integration tests call atoms.reset() before and after to avoid
    cross-contamination between tests.
    """

    def __init__(self, atoms: ATOMsCarla, device: str = "cpu"):
        self.atoms      = atoms
        self.device     = torch.device(device)
        self.max_frames = 20   # cap per integration test for speed

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all_tests(self, testframes: Optional[Dict] = None) -> Dict[str, TestResult]:
        n = len(testframes["cmd"]) if testframes is not None else 0
        print(f"\n{'='*70}")
        print(f"  ATOMs Test Suite  —  {n} integration frames + 3 unit tests")
        print(f"{'='*70}\n")

        results: Dict[str, TestResult] = {}

        tests = [
            ("A01_seg_to_masks_correctness",        self._a01_seg_to_masks),
            ("A02_relevance_filter_coverage",        self._a02_relevance_filter),
            ("A03_v_normalization_nonzero_pixels",   self._a03_v_normalization),
            ("A04_process_frame_output_normalized",  self._a04_process_frame_normalized),
            ("A05_hierarchical_accumulation",        self._a05_hierarchical_accumulation),
            ("A06_reset_clears_state",               self._a06_reset),
            ("A07_node_map_diversity",               self._a07_node_map_diversity),
            ("A08_command_conditioning_sensitivity", self._a08_command_sensitivity),
            ("A09_series_df_integrity",              self._a09_series_df_integrity),
            ("A10_contributions_non_negative",       self._a10_contributions_nonneg),
            ("A11_mean_df_grouped_by_command",       self._a11_mean_df_groupby),
            ("A12_command_lrp_routing_diagnostic", self._a12_command_lrp_routing),
        ]

        for name, fn in tests:
            t0 = time.time()
            print(f"  Running {name} ...", end="", flush=True)
            result, exc = _safe_run(fn, testframes)
            elapsed = time.time() - t0
            if exc is not None:
                result = TestResult(
                    name=name,
                    status=ERROR,
                    summary="Test raised an exception.",
                    exception=exc,
                )
            result.metrics["wall_time_s"] = round(elapsed, 2)
            results[name] = result
            tag = {PASS: "✓", FAIL: "✗", WARN: "△", ERROR: "!"}[result.status]
            print(f"  {tag}  [{elapsed:.1f}s]  {result.summary}")

        print()
        return results

    # ------------------------------------------------------------------
    # A01 — seg_to_masks: shape, binary values, correct pixels, non-overlap
    # Pure unit test — no model or testframes needed.
    # ------------------------------------------------------------------

    def _a01_seg_to_masks(self, frames) -> TestResult:
        """
        seg_to_masks must produce:
          - shape exactly [num_classes, H, W]
          - values in {0.0, 1.0}  (binary, not probabilistic)
          - mask[k] == 1 exactly at pixels where seg == class_ids[k]
          - mask[k] == 0 for classes absent in the image
          - at most one class active per pixel  (CARLA classes are mutually exclusive)
          - return type is torch.Tensor
        """
        failures = []

        # --- Scenario 1: two-class image ---------------------------------
        H, W = 32, 32
        seg = np.zeros((H, W), dtype=np.uint8)
        seg[:H//2, :] = 14   # Car
        seg[H//2:, :] = 1    # Road
        class_ids = [14, 1, 12]   # Car, Road, Pedestrian

        masks = seg_to_masks(seg, class_ids)

        if not isinstance(masks, torch.Tensor):
            failures.append("Return type must be torch.Tensor")

        if masks.shape != (3, H, W):
            failures.append(f"Wrong shape: {masks.shape}, expected (3, {H}, {W})")

        vals = set(masks.numpy().flatten().tolist())
        if not vals.issubset({0.0, 1.0}):
            failures.append(f"Non-binary mask values found: {vals - {0.0, 1.0}}")

        if not (masks[0, :H//2, :] == 1.0).all():
            failures.append("Car mask wrong in top half (expected all 1)")
        if not (masks[0, H//2:, :] == 0.0).all():
            failures.append("Car mask wrong in bottom half (expected all 0)")
        if not (masks[1, :H//2, :] == 0.0).all():
            failures.append("Road mask wrong in top half (expected all 0)")
        if not (masks[1, H//2:, :] == 1.0).all():
            failures.append("Road mask wrong in bottom half (expected all 1)")
        if not (masks[2] == 0.0).all():
            failures.append("Pedestrian mask should be all-zero (class absent from image)")

        overlap_max = float(masks.sum(dim=0).max().item())
        if overlap_max > 1.0:
            failures.append(
                f"Masks overlap: max simultaneous classes per pixel = {overlap_max:.0f} "
                f"(CARLA classes are mutually exclusive)"
            )

        # --- Scenario 2: all-same-class image ----------------------------
        seg2 = np.full((8, 8), 9, dtype=np.uint8)   # all Vegetation
        masks2 = seg_to_masks(seg2, [9, 14])
        if not (masks2[0] == 1.0).all():
            failures.append("All-same-class image: Vegetation mask should be all-1")
        if not (masks2[1] == 0.0).all():
            failures.append("All-same-class image: Car mask should be all-0")

        # --- Scenario 3: class_ids order preserved -----------------------
        seg3 = np.array([[1, 14], [9, 0]], dtype=np.uint8)
        masks3 = seg_to_masks(seg3, [14, 1])   # Car first, Road second
        if masks3[0, 0, 1] != 1.0 or masks3[1, 0, 0] != 1.0:
            failures.append("Class ID ordering not preserved in output masks")

        status  = FAIL if failures else PASS
        summary = ("All checks passed." if not failures else
                   f"{len(failures)} failure(s): " + "; ".join(failures[:2]))

        return TestResult(
            name="A01_seg_to_masks_correctness",
            status=status,
            summary=summary,
            metrics={"n_failures": len(failures)},
            notes=failures,
        )

    # ------------------------------------------------------------------
    # A02 — _relevance_filter: cumulative coverage, minimality, ordering
    # Pure unit test — no model or testframes needed.
    # ------------------------------------------------------------------

    def _a02_relevance_filter(self, frames) -> TestResult:
        """
        For a relevance vector r and threshold p, _relevance_filter must:
          1. Return indices covering ≥ p of total absolute mass.
          2. Be minimal: removing the last index drops coverage below p.
          3. Return indices in descending order of relevance.
          4. Return [] for an all-zero input.
          5. Return [0] for a single-element input.
        """
        failures = []

        # Three relevance profiles that stress different parts of the logic
        r_exp   = torch.exp(-torch.arange(256, dtype=torch.float32) * 0.05)
        r_exp   = r_exp / r_exp.sum()

        r_uni   = torch.ones(256, dtype=torch.float32) / 256.0   # uniform

        r_spike = torch.zeros(256, dtype=torch.float32)
        r_spike[42] = 0.95   # single dominant neuron
        r_spike[1]  = 0.05

        profiles = [("exp-decay", r_exp), ("uniform", r_uni), ("spike", r_spike)]

        for tag, r in profiles:
            for p in [0.5, 0.9, 1.0]:
                selected = _relevance_filter(r, p)

                if not selected and r.sum() > 1e-12:
                    failures.append(f"[{tag} p={p}] empty list for non-zero relevance")
                    continue

                covered = r[selected].sum().item()

                # 1. Coverage ≥ p
                if covered < p - 1e-5:
                    failures.append(
                        f"[{tag} p={p}] coverage={covered:.5f} < {p} — under-selects"
                    )

                # 2. Minimality
                if len(selected) > 1:
                    cov_without_last = r[selected[:-1]].sum().item()
                    if cov_without_last >= p:
                        failures.append(
                            f"[{tag} p={p}] removing last index still covers "
                            f"{cov_without_last:.5f} ≥ {p} — over-selects"
                        )

                # 3. Descending order
                rel_vals = [r[i].item() for i in selected]
                if rel_vals != sorted(rel_vals, reverse=True):
                    failures.append(f"[{tag} p={p}] indices not in descending relevance order")

        # 4. All-zero → empty
        if _relevance_filter(torch.zeros(256), 0.9) != []:
            failures.append("All-zero input should return []")

        # 5. Single element → [0]
        if _relevance_filter(torch.tensor([1.0]), 0.9) != [0]:
            failures.append("Single-element input should return [0]")

        # 6. Negative values treated as absolute (filter uses abs internally)
        r_neg = torch.tensor([-0.9, 0.05, 0.05], dtype=torch.float32)
        sel_neg = _relevance_filter(r_neg, 0.9)
        if sel_neg and sel_neg[0] != 0:
            failures.append("Largest absolute value (index 0, value=-0.9) should be first")

        status  = FAIL if failures else PASS
        summary = ("All coverage / minimality / ordering checks passed." if not failures else
                   f"{len(failures)} failure(s): " + "; ".join(failures[:2]))

        return TestResult(
            name="A02_relevance_filter_coverage",
            status=status,
            summary=summary,
            metrics={"n_failures": len(failures)},
            notes=failures,
        )

    # ------------------------------------------------------------------
    # A03 — _give_element_selectivity: V = non-zero relevance pixels
    # Regression for fix #5 (V was object area, not nonzero-relevance count).
    # Pure unit test — no model or testframes needed (calls atoms method directly).
    # ------------------------------------------------------------------

    def _a03_v_normalization(self, frames) -> TestResult:
        """
        Paper eq.: R̄_g^k = (1/V) Σ_{p ∈ o_g} R_p^k,
                   V = |{p ∈ o_g : R_p^k ≠ 0}|   (non-zero relevance pixels only)

        Setup: 2×2 object mask (4 pixels); only 2 of the 4 have non-zero relevance
               (values 2.0 and 3.0, rest 0.0).
          - Correct   (nonzero V = 2): result = (2+3)/2 = 2.5
          - Incorrect (area    V = 4): result = (2+3)/4 = 1.25

        The test distinguishes these two by numerical value and fails if the
        wrong denominator is being used.
        """
        H, W = 6, 6

        # Single-class mask: top-left 2×2 block
        masks = torch.zeros(1, H, W)
        masks[0, :2, :2] = 1.0     # 4 pixels masked

        # Relevance: only top row of the block is non-zero
        r_hw = torch.zeros(H, W)
        r_hw[0, 0] = 2.0           # pixel (0,0): in mask, non-zero
        r_hw[0, 1] = 3.0           # pixel (0,1): in mask, non-zero
        # pixels (1,0) and (1,1): in mask, zero relevance

        # wide_r shape expected by _give_element_selectivity: [1, C, H, W]
        wide_r = r_hw.unsqueeze(0).unsqueeze(0)   # [1, 1, H, W]

        expected_nonzero_v = 5.0 / 2   # = 2.5   ← paper-correct
        expected_area_v    = 5.0 / 4   # = 1.25  ← wrong (fix #5 not applied)

        # Temporarily install synthetic masks and restore afterwards
        prev_wide = self.atoms._current_masks_wide
        prev_narr = self.atoms._current_masks_narr
        self.atoms._current_masks_wide = masks
        self.atoms._current_masks_narr = masks

        result = self.atoms._give_element_selectivity(wide_r, narr_r=None)
        actual = float(result[0].item())

        self.atoms._current_masks_wide = prev_wide
        self.atoms._current_masks_narr = prev_narr

        tol = 1e-4
        if abs(actual - expected_nonzero_v) < tol:
            status  = PASS
            summary = (f"V = nonzero pixels: result = {actual:.4f} ✓  "
                       f"(expected {expected_nonzero_v:.4f})")
        elif abs(actual - expected_area_v) < tol:
            status  = FAIL
            summary = (
                f"V = object area (WRONG): result = {actual:.4f}. "
                f"Fix #5 not applied — expected {expected_nonzero_v:.4f} "
                f"(nonzero V=2), got {expected_area_v:.4f} (area V=4)."
            )
        else:
            status  = FAIL
            summary = (
                f"Unexpected result = {actual:.4f}. "
                f"Expected {expected_nonzero_v:.4f} (correct) or {expected_area_v:.4f} (pre-fix)."
            )

        return TestResult(
            name="A03_v_normalization_nonzero_pixels",
            status=status,
            summary=summary,
            metrics={
                "actual":              actual,
                "expected_nonzero_v":  expected_nonzero_v,
                "expected_area_v":     expected_area_v,
            },
            notes=[
                "Paper: V = |{p ∈ o_g : R_p^k(x) ≠ 0}| — non-zero relevance pixels.",
                "FAIL means _give_element_selectivity divides by mask area instead.",
                "The fix: replace (masks > 0) with (masks > 0) & (r_hw != 0) in V computation.",
            ],
        )

    # ------------------------------------------------------------------
    # A04 — process_frame return value sums to ≈ 1 per frame
    # ------------------------------------------------------------------

    def _a04_process_frame_normalized(self, frames) -> TestResult:
        """
        process_frame() returns the per-frame normalized attention vector.
        It must sum to ≈ 1.0.  A sum of 0.0 is acceptable for degenerate
        frames (empty masks, zero relevance); any other value is a bug in
        the normalization step inside process_frame.
        """
        if frames is None:
            return TestResult("A04_process_frame_output_normalized", WARN,
                              "No testframes provided — skipped.", {})

        sums = []
        bad  = []

        for i in range(min(len(frames["cmd"]), self.max_frames)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]

            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            seg_n = (frames["seg_narr"][i] if "seg_narr" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))

            self.atoms.reset()
            att = self.atoms.process_frame(
                wide, narr, seg_wide=seg_w, seg_narr=seg_n, cmd=cmd, spd=spd
            )
            s = float(att.sum())
            sums.append(s)
            # 0.0 is acceptable (degenerate frame); anything else outside [0.99, 1.01] is bad
            if abs(s - 1.0) > 0.01 and abs(s) > 0.01:
                bad.append((i, s))

        self.atoms.reset()
        sums_arr = np.array(sums)
        status   = FAIL if bad else PASS
        summary  = (f"All {len(sums)} frames: return sums ≈ 1.0." if not bad else
                    f"{len(bad)}/{len(sums)} frames with bad return sum: "
                    + str([(fi, f"{s:.4f}") for fi, s in bad[:3]]))

        return TestResult(
            name="A04_process_frame_output_normalized",
            status=status,
            summary=summary,
            metrics={
                "return_sum_mean":  float(sums_arr.mean()),
                "return_sum_std":   float(sums_arr.std()),
                "return_sum_min":   float(sums_arr.min()),
                "return_sum_max":   float(sums_arr.max()),
                "n_bad_frames":     len(bad),
            },
            notes=["sum=0.0 is acceptable for degenerate frames (all-zero masks or zero relevance)."],
        )

    # ------------------------------------------------------------------
    # A05 — Hierarchical accumulation: _hierarchical == Σ frame_series exactly
    # ------------------------------------------------------------------

    def _a05_hierarchical_accumulation(self, frames) -> TestResult:
        """
        _hierarchical must equal the element-wise cumulative sum of all
        contributions stored in _frame_series.  Any discrepancy indicates
        that process_frame() is not correctly accumulating, or that
        get_hierarchical(normalize=False) is doing something unexpected.
        """
        if frames is None:
            return TestResult("A05_hierarchical_accumulation", WARN,
                              "No testframes — skipped.", {})

        N = min(len(frames["cmd"]), self.max_frames)
        self.atoms.reset()

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            seg_n = (frames["seg_narr"][i] if "seg_narr" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            self.atoms.process_frame(wide, narr, seg_wide=seg_w, seg_narr=seg_n,
                                     cmd=cmd, spd=spd)

        series_sum = sum(self.atoms._frame_series)          # Σ frame contributions
        cumulative  = self.atoms._hierarchical              # internal accumulator
        raw_h       = self.atoms.get_hierarchical(normalize=False)

        err_series  = float(np.abs(series_sum - cumulative).max())
        err_get_h   = float(np.abs(raw_h - cumulative).max())

        status  = PASS if err_series < 1e-8 and err_get_h < 1e-10 else FAIL
        summary = (
            f"series_sum vs _hierarchical: max_err={err_series:.2e}  "
            f"get_hierarchical(normalize=False) err={err_get_h:.2e}"
        )

        self.atoms.reset()

        return TestResult(
            name="A05_hierarchical_accumulation",
            status=status,
            summary=summary,
            metrics={
                "max_abs_err_vs_series": err_series,
                "max_abs_err_get_h":     err_get_h,
                "n_frames":              N,
                "cumulative_sum":        float(cumulative.sum()),
            },
            notes=[
                "err > 1e-8 means _hierarchical and _frame_series diverge — accumulation bug.",
                "err_get_h > 1e-10 means get_hierarchical(normalize=False) wraps incorrectly.",
            ],
        )

    # ------------------------------------------------------------------
    # A06 — reset() zeros every piece of state
    # ------------------------------------------------------------------

    def _a06_reset(self, frames) -> TestResult:
        """
        After reset(), every accumulator and series list must be back to its
        initial value:
          - _hierarchical    all-zero ndarray
          - _frame_series    empty list
          - _frame_cmds      empty list
          - _frame_brake     empty list
          - _frame_wide_frac empty list
          - _n_frames        == 0
          - _current_masks_* None
          - get_hierarchical() returns all-zero vector
        """
        # Dirty the state with at least one frame
        if frames is not None and len(frames["cmd"]) > 0:
            wide = _to_tensor(frames["wide_rgb"][0], self.device)
            narr = _to_tensor(frames["narr_rgb"][0], self.device)
            spd, cmd = frames["speed"][0], frames["cmd"][0]
            H, W = wide.shape[-2], wide.shape[-1]
            seg  = _synthetic_seg(H, W, self.atoms.class_ids)
            self.atoms.process_frame(wide, narr, seg_wide=seg, seg_narr=seg, cmd=cmd, spd=spd)

        self.atoms.reset()
        failures = []

        if self.atoms._hierarchical.sum() != 0.0:
            failures.append(f"_hierarchical not zero: sum={self.atoms._hierarchical.sum():.4e}")
        if not isinstance(self.atoms._hierarchical, np.ndarray):
            failures.append("_hierarchical should be np.ndarray after reset")
        if self.atoms._frame_series:
            failures.append(f"_frame_series not empty: {len(self.atoms._frame_series)} entries")
        if self.atoms._frame_cmds:
            failures.append(f"_frame_cmds not empty: {len(self.atoms._frame_cmds)} entries")
        if self.atoms._frame_brake:
            failures.append(f"_frame_brake not empty: {len(self.atoms._frame_brake)} entries")
        if self.atoms._frame_wide_frac:
            failures.append(f"_frame_wide_frac not empty: {len(self.atoms._frame_wide_frac)} entries")
        if self.atoms._n_frames != 0:
            failures.append(f"_n_frames not 0: {self.atoms._n_frames}")
        if self.atoms._current_masks_wide is not None:
            failures.append("_current_masks_wide not None after reset")
        if self.atoms._current_masks_narr is not None:
            failures.append("_current_masks_narr not None after reset")
        if self.atoms.get_hierarchical().sum() != 0.0:
            failures.append("get_hierarchical() returns non-zero after reset")

        status  = FAIL if failures else PASS
        summary = "All state correctly zeroed." if not failures else "; ".join(failures[:2])

        return TestResult(
            name="A06_reset_clears_state",
            status=status,
            summary=summary,
            metrics={"n_failures": len(failures)},
            notes=failures,
        )

    # ------------------------------------------------------------------
    # A07 — Node-level: different node_ids produce DIFFERENT pixel maps
    #
    # ★ Critical regression for Bug #1 ★
    # In the original code, _lrp2_pixels received node_id but never passed it
    # to forward_relevance.  The fc→input branch then fell back to
    # grad_outputs=ones (all 256 nodes), so every iteration of the node loop
    # produced the IDENTICAL relevance map.  This test catches that regression.
    # ------------------------------------------------------------------

    def _a07_node_map_diversity(self, frames) -> TestResult:
        """
        Strategy
        --------
        1. Run LRP1 (_lrp1_nodes) to obtain per-neuron relevances.
        2. Pick the top-2 nodes via _relevance_filter.
        3. Call _lrp2_pixels with each node_id and compare wide_r maps.
        4. FAIL if any two maps are pixel-identical (rel_diff < 1e-5).
           That means node_id never reached forward_relevance.

        For robustness, up to 5 pairs are tested.  If even one pair is
        non-identical, the routing is working.
        """
        if frames is None:
            return TestResult("A07_node_map_diversity", WARN,
                              "No testframes — skipped.", {})

        wide = _to_tensor(frames["wide_rgb"][0], self.device)
        narr = _to_tensor(frames["narr_rgb"][0], self.device)
        spd, cmd = frames["speed"][0], frames["cmd"][0]
        H, W = wide.shape[-2], wide.shape[-1]

        # Set up the atoms context for a single frame
        self.atoms.lrp.update_context(wide, narr, spd)
        self.atoms._current_spd = spd
        seg = _synthetic_seg(H, W, self.atoms.class_ids)
        self.atoms._current_masks_wide = seg_to_masks(seg, self.atoms.class_ids)
        self.atoms._current_masks_narr = seg_to_masks(seg, self.atoms.class_ids)

        # LRP1: node relevances
        r_nodes  = self.atoms._lrp1_nodes(wide, narr, cmd)   # [256]
        node_ids = _relevance_filter(r_nodes, self.atoms.p_relevance)

        if len(node_ids) < 2:
            return TestResult(
                "A07_node_map_diversity", WARN,
                f"Only {len(node_ids)} node(s) selected at p={self.atoms.p_relevance}; "
                "need ≥ 2 to test diversity.  Try a different frame or lower p_relevance.",
                {"n_nodes_selected": len(node_ids)},
            )

        # LRP2 for the first node (reference map)
        wide_r0, _ = self.atoms._lrp2_pixels(wide, narr, node_id=node_ids[0], cmd=cmd)
        arr0 = (wide_r0.numpy() if isinstance(wide_r0, torch.Tensor)
                else np.array(wide_r0)).flatten()
        mean_abs = float(np.abs(arr0).mean()) + 1e-12

        # Compare against subsequent nodes
        n_pairs = min(5, len(node_ids) - 1)
        pair_diffs = []
        identical_pairs = []

        for k in range(1, n_pairs + 1):
            wide_rk, _ = self.atoms._lrp2_pixels(wide, narr, node_id=node_ids[k], cmd=cmd)
            arrk = (wide_rk.numpy() if isinstance(wide_rk, torch.Tensor)
                    else np.array(wide_rk)).flatten()

            max_diff = float(np.abs(arr0 - arrk).max())
            rel_diff = max_diff / mean_abs
            pair_diffs.append(rel_diff)

            if rel_diff < 1e-5:
                identical_pairs.append((node_ids[0], node_ids[k], rel_diff))

        min_rel_diff = float(min(pair_diffs))

        if identical_pairs:
            status  = FAIL
            summary = (
                f"Node maps IDENTICAL for {len(identical_pairs)}/{n_pairs} pair(s) "
                f"(min rel_diff={min_rel_diff:.2e}). "
                f"node_id is NOT being passed to forward_relevance — Bug #1 present."
            )
        elif min_rel_diff < 0.01:
            status  = WARN
            summary = (
                f"Node maps very similar across all pairs (min rel_diff={min_rel_diff:.4f}). "
                f"Verify node_id routing; check if many nodes share near-identical weights."
            )
        else:
            status  = PASS
            summary = (
                f"Per-node maps are distinct across {n_pairs} pair(s) "
                f"(min rel_diff={min_rel_diff:.4f}). Bug #1 fix confirmed."
            )

        self.atoms.reset()

        return TestResult(
            name="A07_node_map_diversity",
            status=status,
            summary=summary,
            metrics={
                "node_id_reference":       int(node_ids[0]),
                "n_nodes_selected":        len(node_ids),
                "n_pairs_tested":          n_pairs,
                "min_rel_diff":            min_rel_diff,
                "max_rel_diff":            float(max(pair_diffs)),
                "mean_rel_diff":           float(np.mean(pair_diffs)),
                "n_identical_pairs":       len(identical_pairs),
            },
            notes=[
                "FAIL: _lrp2_pixels receives node_id but does not pass it to "
                "forward_relevance → all maps are the all-neuron-seeded map.",
                "Fix: add node_id=node_id to both forward_relevance calls in _lrp2_pixels.",
                "WARN with small diff: could be genuine near-identical nodes — "
                "check another frame pair before concluding.",
            ],
        )

    # ------------------------------------------------------------------
    # A08 — Command conditioning: different cmds → measurably different attention
    # ------------------------------------------------------------------

    def _a08_command_sensitivity(self, frames) -> TestResult:
        """
        The LRP selector is conditioned on the navigation command.  Processing
        the same frame with two different commands must produce different
        hierarchical attention vectors.  Identical output means the command
        index is not actually reaching the selector.

        Note: the model may genuinely produce very similar attention for some
        command pairs; a single WARN is not conclusive — check across frames.
        """
        if frames is None:
            return TestResult("A08_command_conditioning_sensitivity", WARN,
                              "No testframes — skipped.", {})

        frame_idx = 7
        wide = _to_tensor(frames["wide_rgb"][frame_idx], self.device)
        narr = _to_tensor(frames["narr_rgb"][frame_idx], self.device)
        spd  = frames["speed"][frame_idx]
        H, W = wide.shape[-2], wide.shape[-1]
        seg_w = (frames["seg_wide"][frame_idx] if "seg_wide" in frames
                 else _synthetic_seg(H, W, self.atoms.class_ids))
        seg_n = (frames["seg_narr"][frame_idx] if "seg_narr" in frames
                 else _synthetic_seg(H, W, self.atoms.class_ids))

        # Default: FOLLOW_LANE (3) vs RIGHT_TURN (1) — standard World-on-Rails indices
        cmd_a, cmd_b = 3, 1

        self.atoms.reset()
        att_a = self.atoms.process_frame(
            wide, narr, seg_wide=seg_w, seg_narr=seg_n, cmd=cmd_a, spd=spd
        )

        self.atoms.reset()
        att_b = self.atoms.process_frame(
            wide, narr, seg_wide=seg_w, seg_narr=seg_n, cmd=cmd_b, spd=spd
        )

        l1_diff  = float(np.abs(att_a - att_b).sum())
        denom    = float(np.abs(att_a).sum()) + 1e-12
        rel_diff = l1_diff / denom

        if rel_diff < 1e-6:
            status  = FAIL
            summary = (f"cmd={cmd_a} and cmd={cmd_b} produce IDENTICAL output "
                       f"(rel_diff={rel_diff:.2e}) — command not reaching selector.")
        elif rel_diff < 0.01:
            status  = WARN
            summary = (f"cmd={cmd_a} vs cmd={cmd_b} very similar "
                       f"(rel_diff={rel_diff:.4f}).  May be genuine or a routing issue.")
        else:
            status  = PASS
            summary = (f"cmd={cmd_a} vs cmd={cmd_b}: L1={l1_diff:.4f} rel_diff={rel_diff:.4f}.")

        self.atoms.reset()

        return TestResult(
            name="A08_command_conditioning_sensitivity",
            status=status,
            summary=summary,
            metrics={
                "l1_diff":  l1_diff,
                "rel_diff": rel_diff,
                "cmd_a":    cmd_a,
                "cmd_b":    cmd_b,
            },
            notes=[
                "Commands 3 (FOLLOW_LANE) and 4 (RIGHT) used by default.",
                "Persistent FAIL across multiple frames is a strong signal of a bug.",
                "If your model has fewer commands, adjust cmd_a/cmd_b accordingly.",
            ],
        )

    # ------------------------------------------------------------------
    # A09 — get_series_df: shape, columns, dtypes, wide_frac ∈ [0,1], row sums
    # ------------------------------------------------------------------

    def _a09_series_df_integrity(self, frames) -> TestResult:
        """
        get_series_df(normalize_rows=True) must return a DataFrame with:
          - Exactly n_frames rows
          - Columns: class_names + ['cmd', 'wide_frac']
          - wide_frac ∈ [0.0, 1.0] for every frame
          - Each row (class columns only) sums to ≈ 1.0 for non-degenerate frames
        """
        if frames is None:
            return TestResult("A09_series_df_integrity", WARN,
                              "No testframes — skipped.", {})

        N = min(len(frames["cmd"]), self.max_frames)
        self.atoms.reset()

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            seg_n = (frames["seg_narr"][i] if "seg_narr" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            self.atoms.process_frame(wide, narr, seg_wide=seg_w, seg_narr=seg_n,
                                     cmd=cmd, spd=spd)

        df = self.atoms.get_series_df(normalize_rows=True)
        failures = []

        # Row count
        if len(df) != N:
            failures.append(f"Expected {N} rows, got {len(df)}")

        # Columns
        expected = set(self.atoms.class_names + ["cmd", "wide_frac"])
        missing  = expected - set(df.columns)
        extra    = set(df.columns) - expected
        if missing:
            failures.append(f"Missing columns: {sorted(missing)}")
        if extra:
            failures.append(f"Unexpected columns: {sorted(extra)}")

        # wide_frac in [0, 1]
        if "wide_frac" in df.columns:
            wf = df["wide_frac"].values
            if not np.all((wf >= -1e-6) & (wf <= 1.0 + 1e-6)):
                failures.append(f"wide_frac out of [0,1]: min={wf.min():.4f} max={wf.max():.4f}")

        # Row sums for class columns
        class_cols = [c for c in df.columns if c in self.atoms.class_names]
        if class_cols:
            row_sums  = df[class_cols].sum(axis=1).values
            zero_rows = int((np.abs(row_sums) < 1e-8).sum())   # degenerate, acceptable
            bad_rows  = int((np.abs(row_sums - 1.0) > 0.02).sum()) - zero_rows
            if bad_rows > 0:
                failures.append(
                    f"{bad_rows} non-degenerate row(s) do not sum to ≈1.0 after normalize_rows."
                )

        # cmd column is integer-like
        if "cmd" in df.columns:
            if not np.issubdtype(df["cmd"].dtype, np.integer):
                failures.append(f"'cmd' column dtype is {df['cmd'].dtype}, expected integer.")

        status  = FAIL if failures else PASS
        summary = (f"DataFrame OK: {N} rows × {len(df.columns)} cols." if not failures
                   else "; ".join(failures[:2]))

        self.atoms.reset()

        return TestResult(
            name="A09_series_df_integrity",
            status=status,
            summary=summary,
            metrics={
                "n_rows":    len(df),
                "n_cols":    len(df.columns),
                "n_frames":  N,
                "n_failures": len(failures),
            },
            notes=failures,
        )

    # ------------------------------------------------------------------
    # A10 — Per-frame contributions are non-negative
    # ------------------------------------------------------------------

    def _a10_contributions_nonneg(self, frames) -> TestResult:
        """
        After abs() and cross-normalisation in the LRP pipeline, all pixel
        relevance values are ≥ 0.  The mask-weighted class sums
        (_give_element_selectivity) and therefore the per-frame increment to
        _hierarchical must also be ≥ 0 for every class.

        A negative contribution means signed (raw) relevance leaked through
        without being abs'd, or a mask is somehow negative.
        """
        if frames is None:
            return TestResult("A10_contributions_non_negative", WARN,
                              "No testframes — skipped.", {})

        N = min(len(frames["cmd"]), self.max_frames)
        self.atoms.reset()
        neg_frames = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            seg_n = (frames["seg_narr"][i] if "seg_narr" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))

            prev = self.atoms._hierarchical.copy()
            self.atoms.process_frame(wide, narr, seg_wide=seg_w, seg_narr=seg_n,
                                     cmd=cmd, spd=spd)
            contrib = self.atoms._hierarchical - prev

            n_neg = int((contrib < -1e-8).sum())
            if n_neg > 0:
                neg_frames.append((i, n_neg, float(contrib.min())))

        self.atoms.reset()
        status  = FAIL if neg_frames else PASS
        summary = (f"All {N} frames: contributions ≥ 0 for every class." if not neg_frames else
                   f"{len(neg_frames)} frame(s) have negative contributions.")

        return TestResult(
            name="A10_contributions_non_negative",
            status=status,
            summary=summary,
            metrics={"n_neg_frames": len(neg_frames), "n_total_frames": N},
            notes=(
                [f"Frame {fi}: {nn} negative-class values, min={mn:.4e}"
                 for fi, nn, mn in neg_frames[:5]]
                + ["Negative values → raw (signed) relevance leaked through "
                   "abs/cross_normalize, or masks have negative entries."]
            ),
        )

    # ------------------------------------------------------------------
    # A11 — get_mean_df: one row per command, rows sum to ≈ 1
    # ------------------------------------------------------------------

    def _a11_mean_df_groupby(self, frames) -> TestResult:
        """
        get_mean_df() groups the per-frame series by navigation command and
        returns the mean normalized attention per command.

        Checks:
          - One row per unique command seen during processing
          - Each row (class columns) sums to ≈ 1.0  (mean of normalized rows is normalized)
          - DataFrame is not empty after processing multiple frames
        """
        if frames is None:
            return TestResult("A11_mean_df_grouped_by_command", WARN,
                              "No testframes — skipped.", {})

        N = min(len(frames["cmd"]), self.max_frames)
        self.atoms.reset()
        seen_cmds = set()

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            seg_n = (frames["seg_narr"][i] if "seg_narr" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            self.atoms.process_frame(wide, narr, seg_wide=seg_w, seg_narr=seg_n,
                                     cmd=cmd, spd=spd)
            seen_cmds.add(cmd)

        mean_df  = self.atoms.get_mean_df()
        failures = []

        if mean_df.empty:
            failures.append("get_mean_df() returned empty DataFrame after processing frames")
        else:
            present_cmds = set(mean_df.index.tolist())
            if present_cmds != seen_cmds:
                failures.append(
                    f"Command row mismatch: seen={sorted(seen_cmds)} "
                    f"present={sorted(present_cmds)}"
                )

            class_cols = [c for c in mean_df.columns if c in self.atoms.class_names]
            if class_cols:
                row_sums = mean_df[class_cols].sum(axis=1).values
                bad_rows = int((np.abs(row_sums - 1.0) > 0.02).sum())
                if bad_rows:
                    failures.append(
                        f"{bad_rows} command row(s) do not sum to ≈1.0: {row_sums}"
                    )

        status  = FAIL if failures else PASS
        summary = (f"mean_df: {len(mean_df)} command row(s), each sums to ≈1.0." if not failures
                   else "; ".join(failures[:2]))

        self.atoms.reset()

        return TestResult(
            name="A11_mean_df_grouped_by_command",
            status=status,
            summary=summary,
            metrics={
                "n_command_rows": len(mean_df) if not mean_df.empty else 0,
                "n_seen_cmds":    len(seen_cmds),
                "n_failures":     len(failures),
            },
            notes=failures + [
                "get_mean_df groups get_series_df(normalize_rows=True) by cmd index.",
                "If only one command is present in testframes, the test has only one row to check.",
            ],
        )
    


    def _a12_command_lrp_routing(self, frames) -> TestResult:
        """
        Layered diagnostic for command sensitivity.

        Three measurements, each isolating one component:

          [1] LOGIT LEVEL
              L1 distance between speed-lerped steer logits for cmd_ref vs cmd_b.
              If ≈ 0: model is cmd-invariant at the output level.

          [2] LRP1 LEVEL  (output → 256-dim FC)
              Cosine similarity of FC-neuron relevances for cmd_ref vs cmd_b.
              cos ≈ 1 means act_head[4] weight columns are proportional across
              cmd blocks → LRP1 gives proportional (identical after normalisation)
              FC patterns regardless of cmd. This explains why pixel maps are
              identical even when logits differ.

          [3] MODE-3 LEVEL  (output → input, full path)
              Relative L1 of pixel maps after cross-normalisation.
              Compared against the cmd-AGNOSTIC BASELINE: two identical
              fc→input(selector=None) calls whose ≈0.0001 difference is pure
              floating-point noise. If mode-3 rel > 10× baseline → routing
              IS working, model is just cmd-invariant visually.
              If mode-3 rel ≤ 3× baseline → routing broken (indistinguishable
              from a None selector).

        MODE-2 DESIGN NOTE
        ------------------
        fc→input LRP uses selector=None (ones) → cmd-agnostic BY DESIGN.
        A08's ≈0 diff for mode 2 is EXPECTED.  For cmd-conditioned maps
        use mode 1 or mode 3.
        """
        if frames is None:
            return TestResult("A12_command_lrp_routing_diagnostic", WARN,
                              "No testframes — skipped.", {})

        lrp = self.atoms.lrp

        cmd_ref   = 3
        other_cmds = [c for c in range(lrp.num_cmds) if c != cmd_ref]
        if not other_cmds:
            return TestResult("A12_command_lrp_routing_diagnostic", WARN,
                              "Only one command — cannot compare.", {})

        logit_L1s      = []
        lrp1_cos_sims  = []   # cosine similarity of FC relevances
        mode3_rels     = []
        pair_labels    = []

        # ── CMD-AGNOSTIC BASELINE ────────────────────────────────────────────────
        # Two identical fc→input calls with selector=None — pure floating-point
        # noise. mode-3 rel_diff must be meaningfully larger than this to confirm
        # the selector is actually reaching the backward pass.
        baseline_diffs = []
        N_base = min(3, len(frames["cmd"]))
        for i in range(N_base):
            wide_b = _to_tensor(frames["wide_rgb"][i], self.device)
            narr_b = _to_tensor(frames["narr_rgb"][i], self.device)
            spd_b  = frames["speed"][i]
            lrp.update_context(wide_b, narr_b, spd_b)
            r1, _, _, _ = lrp.forward_relevance(wide_b, narr_b, beg="fc", end="input",
                                                cmd=cmd_ref, spd=spd_b)
            r2, _, _, _ = lrp.forward_relevance(wide_b, narr_b, beg="fc", end="input",
                                                cmd=cmd_ref, spd=spd_b)
            arr1 = r1.numpy() if isinstance(r1, torch.Tensor) else np.array(r1)
            arr2 = r2.numpy() if isinstance(r2, torch.Tensor) else np.array(r2)
            baseline_diffs.append(
                float(np.abs(arr1 - arr2).sum() / (np.abs(arr1).sum() + 1e-12))
            )
        cmd_agnostic_baseline = float(np.mean(baseline_diffs)) if baseline_diffs else 1e-4

        # ── PER-FRAME / PER-PAIR MEASUREMENTS ───────────────────────────────────
        N = min(len(frames["cmd"]), self.max_frames)

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd  = frames["speed"][i]
            lrp.update_context(wide, narr, spd)

            x0, x1, w = lrp._lerp_bins(spd, lrp.min_speeds, lrp.max_speeds, lrp.num_speeds)

            with torch.no_grad():
                sl_ref, _, _ = lrp._model_eval.policy(wide, narr, cmd_ref)
                steer_ref    = (1 - w) * sl_ref[x0] + w * sl_ref[x1]

            for cmd_b in other_cmds:
                label = f"cmd{cmd_ref}vs{cmd_b}"

                # [1] Logit L1
                with torch.no_grad():
                    sl_b, _, _ = lrp._model_eval.policy(wide, narr, cmd_b)
                    steer_b    = (1 - w) * sl_b[x0] + w * sl_b[x1]
                logit_L1s.append((label, float((steer_ref - steer_b).abs().sum().item())))

                # [2] LRP1 cosine similarity of FC neuron relevances
                sel_ref, _ = lrp._build_drive_brake_selector(wide, narr, cmd_ref, spd)
                sel_b,   _ = lrp._build_drive_brake_selector(wide, narr, cmd_b,   spd)
                r1_ref = lrp._attribute_to_fc(wide, narr, sel_ref).float()  # [256]
                r1_b   = lrp._attribute_to_fc(wide, narr, sel_b).float()
                cos_sim = float(
                    torch.nn.functional.cosine_similarity(
                        r1_ref.unsqueeze(0), r1_b.unsqueeze(0)
                    ).item()
                )
                lrp1_cos_sims.append((label, cos_sim))

                # [3] Mode-3 full-path relative L1
                wide_r_ref, _, _, _ = lrp.forward_relevance(
                    wide, narr, beg="output", end="input", cmd=cmd_ref, spd=spd
                )
                wide_r_b, _, _, _ = lrp.forward_relevance(
                    wide, narr, beg="output", end="input", cmd=cmd_b,   spd=spd
                )
                a_ref = (wide_r_ref.numpy() if isinstance(wide_r_ref, torch.Tensor)
                         else np.array(wide_r_ref))
                a_b   = (wide_r_b.numpy()   if isinstance(wide_r_b,   torch.Tensor)
                         else np.array(wide_r_b))
                mode3_rels.append((label, float(
                    np.abs(a_ref - a_b).sum() / (np.abs(a_ref).sum() + 1e-12)
                )))

                if i == 0:
                    pair_labels.append(label)

        # ── Aggregate ────────────────────────────────────────────────────────────
        logit_mean  = float(np.mean([v for _, v in logit_L1s]))
        cos_mean    = float(np.mean([v for _, v in lrp1_cos_sims]))
        mode3_mean  = float(np.mean([v for _, v in mode3_rels]))

        routing_ratio = mode3_mean / (cmd_agnostic_baseline + 1e-12)

        per_pair_logit  = {p: float(np.mean([v for lb, v in logit_L1s      if lb == p]))
                           for p in pair_labels}
        per_pair_cos    = {p: float(np.mean([v for lb, v in lrp1_cos_sims  if lb == p]))
                           for p in pair_labels}
        per_pair_mode3  = {p: float(np.mean([v for lb, v in mode3_rels     if lb == p]))
                           for p in pair_labels}

        # ── Diagnosis ────────────────────────────────────────────────────────────
        routing_broken     = routing_ratio < 3.0   # mode-3 indistinguishable from None-selector
        weight_proportional = cos_mean > 0.98      # act_head[4] cols nearly parallel across cmds
        model_invariant    = logit_mean < 1.0      # logits nearly equal

        if routing_broken:
            status = FAIL
            diagnosis = (
                f"ROUTING BUG: mode-3 rel_diff ({mode3_mean:.4f}) is only "
                f"{routing_ratio:.1f}× the cmd-agnostic baseline ({cmd_agnostic_baseline:.4f}). "
                f"The selector is not reaching the LRP backward pass — "
                f"maps are indistinguishable from selector=None."
            )
        elif weight_proportional:
            status = WARN
            diagnosis = (
                f"MODEL CMD-INVARIANT VISUAL FEATURES: routing confirmed working "
                f"(mode-3 rel={mode3_mean:.4f}, {routing_ratio:.0f}× baseline). "
                f"LRP1 cosine similarity={cos_mean:.4f} ≈ 1 → act_head[4] weight "
                f"columns are nearly parallel across cmd blocks → LRP gives "
                f"proportional (≈identical after normalisation) pixel maps for all cmds. "
                f"logit_L1={logit_mean:.3f} (magnitudes differ, directions don't). "
                f"Mode-1 & mode-3 won't show cmd sensitivity for this model."
            )
        elif model_invariant:
            status = WARN
            diagnosis = (
                f"MODEL LOGIT-INVARIANT: logit_L1={logit_mean:.4f} — all cmd heads "
                f"produce near-identical logits. Routing confirmed "
                f"({routing_ratio:.0f}× baseline). Model quality issue."
            )
        else:
            status = PASS
            diagnosis = (
                f"Routing confirmed working ({routing_ratio:.0f}× baseline). "
                f"LRP1 cos_sim={cos_mean:.4f}  mode-3 rel={mode3_mean:.4f}  "
                f"logit_L1={logit_mean:.3f}."
            )

        return TestResult(
            name="A12_command_lrp_routing_diagnostic",
            status=status,
            summary=diagnosis,
            metrics={
                "cmd_agnostic_baseline":        cmd_agnostic_baseline,
                "routing_ratio_vs_baseline":    routing_ratio,
                "logit_L1_mean":                logit_mean,
                "lrp1_cosine_similarity_mean":  cos_mean,
                "mode3_full_path_rel_diff_mean": mode3_mean,
                **{f"logit_L1_{p}":   v for p, v in per_pair_logit.items()},
                **{f"lrp1_cos_{p}":   v for p, v in per_pair_cos.items()},
                **{f"mode3_rel_{p}":  v for p, v in per_pair_mode3.items()},
                "n_frames":    N,
                "n_cmd_pairs": len(pair_labels),
            },
            notes=[
                "ROUTING BUG threshold: mode-3 rel < 3× cmd-agnostic baseline.",
                "CMD-INVARIANT VISUAL threshold: LRP1 cosine similarity > 0.98.",
                f"Your cmd-agnostic baseline ≈ {cmd_agnostic_baseline:.4f} (pure noise from "
                f"two identical fc→input passes — represents the 'routing broken' floor).",
                "MODE-2 IS CMD-AGNOSTIC BY DESIGN: fc→input uses selector=None. "
                "A08 ≈0 for mode 2 is expected and correct.",
                "For cmd-conditioned maps: use mode 1 (LRP1 node weighting) or mode 3.",
            ],
        )

    # ------------------------------------------------------------------
    # Reporting  (mirrors lrp_test_suite.py style)
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, TestResult]) -> None:
        status_sym = {PASS: "✓ PASS", FAIL: "✗ FAIL", WARN: "△ WARN", ERROR: "! ERROR"}
        sep = "─" * 70

        print(f"\n{'='*70}")
        print("  ATOMs DETAILED TEST REPORT")
        print(f"{'='*70}")

        for name, r in results.items():
            print(f"\n{sep}")
            print(f"  {status_sym[r.status]}  |  {name}")
            print(f"  {r.summary}")
            if r.metrics:
                print("  Metrics:")
                for k, v in r.metrics.items():
                    if k == "wall_time_s":
                        continue
                    if isinstance(v, float):
                        print(f"    {k:<48s} {v:.6g}")
                    else:
                        print(f"    {k:<48s} {v}")
            if r.notes:
                print("  Notes:")
                for n in r.notes:
                    print(f"    ▸ {n}")
            if r.exception:
                print("  Exception:")
                for line in r.exception.strip().split("\n"):
                    print(f"    {line}")
            print(f"  Wall time: {r.metrics.get('wall_time_s', '?')}s")

        print(f"\n{'='*70}")
        print("  SUMMARY")
        print(f"{'='*70}")
        counts = defaultdict(int)
        for r in results.values():
            counts[r.status] += 1
        for s in [PASS, WARN, FAIL, ERROR]:
            if counts[s]:
                print(f"  {status_sym[s]}: {counts[s]}")
        print(f"{'='*70}\n")

    def save_report(self, results: Dict[str, TestResult], out_dir: str) -> None:
        """Save a human-readable .txt and per-test .npy arrays to out_dir."""
        import contextlib, io
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, "atoms_test_report.txt")
        with open(txt_path, "w") as fh:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.print_report(results)
            fh.write(buf.getvalue())

        for name, r in results.items():
            if r.per_frame is not None:
                np.save(os.path.join(out_dir, f"{name}_per_frame.npy"), r.per_frame)

        print(f"Report saved to {out_dir}")
