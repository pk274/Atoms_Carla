"""
lrp_test_suite.py
=================
Diagnostic and correctness test suite for the LRP / ATOMs pipeline.

Usage
-----
    suite  = LRPTestSuite(atoms_instance, lrp_analyzer_instance)
    report = suite.run_all_tests(testframes)
    suite.print_report(report)          # human-readable
    suite.save_report(report, "out/")   # saves .txt + per-test .npy arrays

Testframe format (list of dicts)
---------------------------------
Each element must contain:
    wide_rgb  : np.ndarray  uint8  [H, W, 3]  or torch.Tensor [1, 3, H, W]
    narr_rgb  : np.ndarray  uint8  [H, W, 3]  or torch.Tensor [1, 3, H, W]
    spd       : float       current speed in m/s
    cmd       : int         command index (0-based)

Optional keys:
    ground_truth_brake : bool   whether the agent was known to have brake=1
    frame_id           : str    for labelling output
"""

from __future__ import annotations

import math
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
ERROR = "ERROR"


@dataclass
class TestResult:
    name: str
    status: str                     # PASS / FAIL / WARN / ERROR
    summary: str                    # one-line verdict
    metrics: Dict[str, Any] = field(default_factory=dict)
    per_frame: Optional[np.ndarray] = None   # scalar per frame where applicable
    notes: List[str] = field(default_factory=list)
    exception: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_tensor(rgb, device) -> torch.Tensor:
    """Accept [H,W,3] ndarray or [1,3,H,W] tensor; always return [1,3,H,W] float32."""
    if isinstance(rgb, np.ndarray):
        t = torch.from_numpy(rgb).unsqueeze(0).float()
    else:
        t = rgb.float()
    if t.shape[1] == 3:
        pass
    else:
        raise ValueError(f"Expected 3-channel image, got shape {t.shape}")
    return t.to(device)


def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1-D array (0=uniform, 1=maximally concentrated)."""
    arr = np.abs(arr).flatten()
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n


def _entropy_bits(arr: np.ndarray) -> float:
    """Shannon entropy (bits) of the absolute-value distribution of arr."""
    arr = np.abs(arr).flatten()
    s = arr.sum()
    if s == 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _top_k_mass(arr: np.ndarray, k_frac: float = 0.01) -> float:
    """Fraction of total |relevance| held by the top k_frac of pixels."""
    arr = np.abs(arr).flatten()
    s = arr.sum()
    if s == 0:
        return 0.0
    arr_sorted = np.sort(arr)[::-1]
    k = max(1, int(len(arr) * k_frac))
    return float(arr_sorted[:k].sum() / s)


def _safe_run(fn, *args, **kwargs):
    """Run fn; return (result, None) or (None, traceback_str) on exception."""
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Main test suite
# ---------------------------------------------------------------------------

class LRPTestSuite:
    """
    Parameters
    ----------
    atoms   : ATOMSCarla instance (has .lrp attribute, .mode_analysis, etc.)
    lrp     : LRPAnalyzer instance (atoms.lrp)
    device  : torch device string  (default 'cpu')
    mode    : int  mode_analysis to use for tests that need a full forward pass (default 2)
    """

    def __init__(self, atoms, lrp, device: str = "cpu", mode: int = 2):
        self.atoms  = atoms
        self.lrp    = lrp
        self.device = torch.device(device)
        self.mode   = mode

        # introspect numeric constants directly from the LRP object
        self.num_steers = lrp.num_steers
        self.num_throts = lrp.num_throts
        self.num_speeds = lrp.num_speeds
        self.num_cmds   = lrp.num_cmds
        self.min_speeds = lrp.min_speeds
        self.max_speeds = lrp.max_speeds

        self.max_checks = 50

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run_all_tests(self, testframes: List[Dict]) -> Dict[str, TestResult]:
        """
        Run the full battery on testframes.  Returns an ordered dict of
        TestResult objects keyed by test name.
        """
        n = len(testframes["cmd"])
        print(f"\n{'='*70}")
        print(f"  LRP / ATOMs Test Suite  —  {n} frames")
        print(f"{'='*70}\n")

        results: Dict[str, TestResult] = {}

        tests = [
            ("T01_nan_inf_guard",               self._t01_nan_inf_guard),
            ("T02_relevance_conservation",       self._t02_relevance_conservation),
            ("T03_amplification_stability",      self._t03_amplification_stability),
            ("T04_brake_mode_detection",         self._t04_brake_mode_detection),
            ("T05_brake_logit_distribution",     self._t05_brake_logit_distribution),
            ("T06_wide_narrow_split",            self._t06_wide_narrow_split),
            ("T07_relevance_spatial_coherence",  self._t07_spatial_coherence),
            ("T08_relevance_sign_ratio",         self._t08_sign_ratio),
            ("T09_selector_normalization",       self._t09_selector_normalization),
            ("T10_speed_interpolation",          self._t10_speed_interpolation),
            ("T11_node_selection_coverage",      self._t11_node_selection_coverage),
            ("T12_context_update_sensitivity",   self._t12_context_update_sensitivity),
            ("T13_logit_range_sanity",           self._t13_logit_range_sanity),
            ("T14_mode_consistency",             self._t14_mode_consistency),
            ("T15_ground_truth_brake_alignment", self._t15_gt_brake_alignment),
            ("T16_concat_level_split",           self._t16_concat_level_split),
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
                    summary="Test function itself raised an exception.",
                    exception=exc,
                )
                print(exec)
            result.metrics["wall_time_s"] = round(elapsed, 2)
            results[name] = result
            status_tag = {PASS: "✓", FAIL: "✗", WARN: "△", ERROR: "!"}[result.status]
            print(f"  {status_tag}  [{elapsed:.1f}s]  {result.summary}")

        print()
        return results

    # -----------------------------------------------------------------------
    # T01 — NaN / Inf guard
    # -----------------------------------------------------------------------

    def _t01_nan_inf_guard(self, frames) -> TestResult:
        """
        Every relevance tensor returned by forward_relevance must be finite.
        Hard failure on first occurrence; also counts total bad frames.
        """
        bad_frames   = []
        nan_counts   = []
        inf_counts   = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            wide_r, narr_r, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            for tag, tensor in [("wide_r", wide_r), ("narr_r", narr_r)]:
                arr = tensor.cpu().numpy() if isinstance(tensor, torch.Tensor) else np.array(tensor)
                nc = int(np.isnan(arr).sum())
                ic = int(np.isinf(arr).sum())
                nan_counts.append(nc)
                inf_counts.append(ic)
                if nc > 0 or ic > 0:
                    bad_frames.append((i, tag, nc, ic))

        total_nan = sum(nan_counts)
        total_inf = sum(inf_counts)
        status    = FAIL if (total_nan + total_inf) > 0 else PASS

        return TestResult(
            name="T01_nan_inf_guard",
            status=status,
            summary=(
                f"All finite." if status == PASS else
                f"{len(bad_frames)} bad tensors — NaN:{total_nan} Inf:{total_inf}"
            ),
            metrics={
                "total_nan_values": total_nan,
                "total_inf_values": total_inf,
                "bad_frame_indices": [b[0] for b in bad_frames],
            },
            notes=[f"Frame {b[0]}, {b[1]}: NaN={b[2]} Inf={b[3]}" for b in bad_frames[:10]],
        )

    # -----------------------------------------------------------------------
    # T02 — Relevance conservation
    # -----------------------------------------------------------------------

    def _t02_relevance_conservation(self, frames) -> TestResult:
        """
        With a unit selector, relevance injected at the output should ideally
        equal relevance that arrives at the pixel input.

        Expected:
          - output_sum  ≈ 1.0  (selector sums to 1 by construction)
          - pixel_sum   ≈ 1.0  (perfect conservation)

        In practice WSquare + AlphaBeta breaks exact conservation, so we track
        the ratio pixel_sum / output_sum.  A stable ratio (low CoV) across
        frames is what we care about even if the absolute value ≠ 1.
        """
        output_sums = []
        pixel_sums  = []
        ratios      = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            _num_acts = self.num_cmds * self.num_speeds * (self.num_steers + self.num_throts + 1)
            _dummy    = torch.zeros(1, _num_acts, device=self.device)
            selector, _ = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)

            # output-level sum
            out_sum = float(selector(_dummy).sum().item()) if selector is not None else 1.0
            output_sums.append(out_sum)

            # pixel-level sum (wide + narr combined)
            wide_r, narr_r, wide_frac, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd, raw=True
            )
            w_arr = wide_r.cpu().numpy() if isinstance(wide_r, torch.Tensor) else np.array(wide_r)
            n_arr = narr_r.cpu().numpy() if isinstance(narr_r, torch.Tensor) else np.array(narr_r)
            pix_sum = float(w_arr.sum() + n_arr.sum())
            pixel_sums.append(pix_sum)
            if abs(out_sum) > 1e-9:
                ratios.append(pix_sum / out_sum)

        ratios = np.array(ratios)
        cov    = float(ratios.std() / (abs(ratios.mean()) + 1e-12))

        status = PASS if cov < 0.10 else (WARN if cov < 0.30 else FAIL)

        return TestResult(
            name="T02_relevance_conservation",
            status=status,
            summary=(
                f"pixel_sum/output_sum: mean={ratios.mean():.3f} "
                f"std={ratios.std():.3f} CoV={cov:.3f}"
            ),
            metrics={
                "ratio_mean":      float(ratios.mean()),
                "ratio_std":       float(ratios.std()),
                "ratio_min":       float(ratios.min()),
                "ratio_max":       float(ratios.max()),
                "ratio_coeff_var": cov,
                "output_sum_mean": float(np.mean(output_sums)),
                "pixel_sum_mean":  float(np.mean(pixel_sums)),
            },
            per_frame=ratios,
            notes=[
                "CoV < 0.10 → PASS: ratio is stable even if amplified.",
                "Ratio ≠ 1.0 is expected with WSquare + AlphaBeta composite.",
                "A drifting ratio (high CoV) means LRP behaviour is input-dependent "
                "in a way that undermines cross-frame comparability of ATOMs.",
            ],
        )

    # -----------------------------------------------------------------------
    # T03 — Amplification ratio stability
    # -----------------------------------------------------------------------

    def _t03_amplification_stability(self, frames) -> TestResult:
        """
        Absolute pixel relevance sum should be roughly constant across frames.
        A large CoV here means heatmaps are not comparable between frames.
        """
        pixel_abs_sums = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            wide_r, narr_r, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd, raw=True
            )
            w_arr = wide_r.cpu().numpy() if isinstance(wide_r, torch.Tensor) else np.array(wide_r)
            n_arr = narr_r.cpu().numpy() if isinstance(narr_r, torch.Tensor) else np.array(narr_r)
            pixel_abs_sums.append(float(np.abs(w_arr).sum() + np.abs(n_arr).sum()))

        arr = np.array(pixel_abs_sums)
        cov = float(arr.std() / (arr.mean() + 1e-12))

        status = PASS if cov < 0.15 else (WARN if cov < 0.40 else FAIL)

        return TestResult(
            name="T03_amplification_stability",
            status=status,
            summary=f"|R|_pixel: mean={arr.mean():.2f} std={arr.std():.2f} CoV={cov:.3f}",
            metrics={
                "abs_pixel_sum_mean": float(arr.mean()),
                "abs_pixel_sum_std":  float(arr.std()),
                "abs_pixel_sum_min":  float(arr.min()),
                "abs_pixel_sum_max":  float(arr.max()),
                "coeff_var":          cov,
            },
            per_frame=arr,
            notes=[
                "The ~79x amplification from WSquare is expected and fine.",
                "What must be stable is the ratio across frames (low CoV).",
                "High CoV means you cannot compare ATOMs values between frames.",
            ],
        )

    # -----------------------------------------------------------------------
    # T04 — Brake mode detection
    # -----------------------------------------------------------------------

    def _t04_brake_mode_detection(self, frames) -> TestResult:
        """
        Brake-mode classification rate.  If zero frames are classified as brake,
        something is wrong regardless of the test set.
        """
        is_brake_list  = []
        brake_prob_list = []
        steer_logit_means = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            selector, is_brake = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)
            is_brake_list.append(bool(is_brake))

            # also compute raw brake_prob for diagnostics
            with torch.no_grad():
                raw    = self.lrp.model_lrp(wide, narr)
                logits = raw.view(1, self.num_cmds, self.num_speeds,
                                  self.num_steers + self.num_throts + 1)
            x0, x1, w = self.lrp._lerp_bins(spd, self.min_speeds,
                                              self.max_speeds, self.num_speeds)
            sl_x0 = logits[0, cmd, x0, :self.num_steers]
            sl_x1 = logits[0, cmd, x1, :self.num_steers]
            tl_x0 = logits[0, cmd, x0, self.num_steers:self.num_steers + self.num_throts]
            tl_x1 = logits[0, cmd, x1, self.num_steers:self.num_steers + self.num_throts]
            bl_x0 = logits[0, cmd, x0, -1]
            bl_x1 = logits[0, cmd, x1, -1]

            steer_l = (1 - w) * sl_x0 + w * sl_x1
            throt_l = (1 - w) * tl_x0 + w * tl_x1
            brake_l = float((1 - w) * bl_x0 + w * bl_x1)

            combined = torch.cat([
                steer_l.repeat(self.num_throts),
                throt_l.repeat_interleave(self.num_steers),
                torch.tensor([brake_l]),
            ])
            bp = float(torch.softmax(combined, dim=0)[-1].item())
            brake_prob_list.append(bp)
            steer_logit_means.append(float(steer_l.mean().item()))

        brake_probs = np.array(brake_prob_list)
        brake_rate  = float(np.mean(is_brake_list))

        if brake_rate == 0.0:
            status  = FAIL
            summary = "0% brake frames — is_brake is NEVER True."
        elif brake_rate < 0.02:
            status  = WARN
            summary = f"Only {brake_rate*100:.1f}% brake frames — suspiciously low."
        else:
            status  = PASS
            summary = f"{brake_rate*100:.1f}% frames classified as brake."

        percentiles = np.percentile(brake_probs, [5, 25, 50, 75, 95])

        return TestResult(
            name="T04_brake_mode_detection",
            status=status,
            summary=summary,
            metrics={
                "brake_rate":           brake_rate,
                "brake_prob_mean":      float(brake_probs.mean()),
                "brake_prob_std":       float(brake_probs.std()),
                "brake_prob_p05":       float(percentiles[0]),
                "brake_prob_p25":       float(percentiles[1]),
                "brake_prob_p50":       float(percentiles[2]),
                "brake_prob_p75":       float(percentiles[3]),
                "brake_prob_p95":       float(percentiles[4]),
                "n_brake_frames":       int(sum(is_brake_list)),
                "n_frames":             len(frames["cmd"]),
            },
            per_frame=brake_probs,
            notes=[
                "Brake threshold is brake_prob > 0.5 (agent's post_process rule).",
                "With 55-term softmax, brake logit must be >ln(54)≈4 above mean "
                "for brake_prob to exceed 0.5.",
                "If brake_rate=0 and your test set includes clear braking frames, "
                "re-check action_prob formula (concat vs add).",
            ],
        )

    # -----------------------------------------------------------------------
    # T05 — Brake logit distribution vs steer/throt
    # -----------------------------------------------------------------------

    def _t05_brake_logit_distribution(self, frames) -> TestResult:
        """
        Diagnostic: how large are the brake logits relative to steer/throt logits?
        For braking to ever trigger, brake_logit must reach ~+4 at times.
        Reports percentiles of all three logit types across frames.
        """
        brake_logits = []
        steer_logits = []
        throt_logits = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            with torch.no_grad():
                raw    = self.lrp.model_lrp(wide, narr)
                logits = raw.view(1, self.num_cmds, self.num_speeds,
                                  self.num_steers + self.num_throts + 1)
            x0, x1, w = self.lrp._lerp_bins(spd, self.min_speeds,
                                              self.max_speeds, self.num_speeds)
            sl = ((1 - w) * logits[0, cmd, x0, :self.num_steers]
                  + w * logits[0, cmd, x1, :self.num_steers])
            tl = ((1 - w) * logits[0, cmd, x0, self.num_steers:self.num_steers + self.num_throts]
                  + w * logits[0, cmd, x1, self.num_steers:self.num_steers + self.num_throts])
            bl = float((1 - w) * logits[0, cmd, x0, -1]
                       + w      * logits[0, cmd, x1, -1])
            brake_logits.append(bl)
            steer_logits.extend(sl.tolist())
            throt_logits.extend(tl.tolist())

        bla = np.array(brake_logits)
        sla = np.array(steer_logits)
        tla = np.array(throt_logits)

        # Does brake_logit ever reach the threshold needed for brake_prob > 0.5?
        threshold     = math.log(self.num_steers * self.num_throts * 2)  # ≈ ln(54)
        ever_reachable = bool((bla > threshold).any())

        status = PASS if ever_reachable else WARN

        return TestResult(
            name="T05_brake_logit_distribution",
            status=status,
            summary=(
                f"Brake logit max={bla.max():.2f} (threshold≈{threshold:.2f}) — "
                + ("threshold reachable." if ever_reachable else "threshold NEVER reached.")
            ),
            metrics={
                "brake_logit_mean":    float(bla.mean()),
                "brake_logit_std":     float(bla.std()),
                "brake_logit_max":     float(bla.max()),
                "brake_logit_min":     float(bla.min()),
                "brake_logit_p95":     float(np.percentile(bla, 95)),
                "steer_logit_mean":    float(sla.mean()),
                "throt_logit_mean":    float(tla.mean()),
                "threshold_ln54":      threshold,
                "n_above_threshold":   int((bla > threshold).sum()),
            },
            per_frame=bla,
            notes=[
                f"Threshold ln({self.num_steers*self.num_throts*2})≈{threshold:.2f} "
                f"is the minimum brake logit for brake_prob>0.5 when all "
                f"other {self.num_steers*self.num_throts*2} logits are exactly 0.",
                "In practice the bar is higher because steer/throt logits are positive.",
            ],
        )

    # -----------------------------------------------------------------------
    # T06 — Wide / narrow relevance split
    # -----------------------------------------------------------------------

    def _t06_wide_narrow_split(self, frames) -> TestResult:
        """
        wide_frac should be strictly in (0, 1) and show meaningful variation
        across frames.  A value stuck at 0 or 1 every frame means one camera's
        relevance is always being discarded.
        """
        wide_fracs = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            _, _, wide_frac, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            wide_fracs.append(float(wide_frac))

        wf = np.array(wide_fracs)
        degenerate_zero = int((wf < 0.01).sum())
        degenerate_one  = int((wf > 0.99).sum())
        deg_frac        = (degenerate_zero + degenerate_one) / len(wf)

        if deg_frac > 0.8:
            status = FAIL
        elif deg_frac > 0.2:
            status = WARN
        else:
            status = PASS

        return TestResult(
            name="T06_wide_narrow_split",
            status=status,
            summary=(
                f"wide_frac: mean={wf.mean():.3f} std={wf.std():.3f} "
                f"— {deg_frac*100:.0f}% degenerate frames (≈0 or ≈1)."
            ),
            metrics={
                "wide_frac_mean":           float(wf.mean()),
                "wide_frac_std":            float(wf.std()),
                "wide_frac_min":            float(wf.min()),
                "wide_frac_max":            float(wf.max()),
                "wide_frac_p05":            float(np.percentile(wf, 5)),
                "wide_frac_p95":            float(np.percentile(wf, 95)),
                "n_degenerate_zero":        degenerate_zero,
                "n_degenerate_one":         degenerate_one,
                "degenerate_frame_frac":    deg_frac,
            },
            per_frame=wf,
            notes=[
                "wide_frac = |R_wide| / (|R_wide| + |R_narr|) after cross-normalisation.",
                "Permanently stuck at 0 or 1 suggests _cross_normalize or the "
                "bottleneck_narr is not receiving/passing relevance.",
            ],
        )

    # -----------------------------------------------------------------------
    # T07 — Spatial coherence of relevance maps
    # -----------------------------------------------------------------------

    def _t07_spatial_coherence(self, frames) -> TestResult:
        """
        A healthy heatmap has moderate spatial spread.
        - Gini coefficient ≈ 0.5–0.9  (some concentration but not pathological)
        - Shannon entropy > 1 bit     (not collapsed to one pixel)
        - Top-1% pixel mass < 0.90    (not a single-point spike)
        """
        gini_vals      = []
        entropy_vals   = []
        top1pct_vals   = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            wide_r, narr_r, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            for r in [wide_r, narr_r]:
                arr = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.array(r)
                gini_vals.append(_gini(arr))
                entropy_vals.append(_entropy_bits(arr))
                top1pct_vals.append(_top_k_mass(arr, 0.01))

        g = np.array(gini_vals)
        e = np.array(entropy_vals)
        t = np.array(top1pct_vals)

        degenerate_spike  = int((t > 0.90).sum())
        degenerate_flat   = int((e < 0.5).sum())

        status = PASS
        if degenerate_spike > len(frames["cmd"]) * 0.2:
            status = FAIL
        elif degenerate_spike > len(frames["cmd"]) * 0.05 or degenerate_flat > len(frames) * 0.1:
            status = WARN

        return TestResult(
            name="T07_relevance_spatial_coherence",
            status=status,
            summary=(
                f"Gini={g.mean():.3f}  Entropy={e.mean():.2f}bits  "
                f"Top1%_mass={t.mean():.3f}  "
                f"({degenerate_spike} spike / {degenerate_flat} flat tensors)"
            ),
            metrics={
                "gini_mean":             float(g.mean()),
                "gini_std":              float(g.std()),
                "entropy_bits_mean":     float(e.mean()),
                "entropy_bits_min":      float(e.min()),
                "top1pct_mass_mean":     float(t.mean()),
                "n_spike_tensors":       degenerate_spike,
                "n_flat_tensors":        degenerate_flat,
            },
            notes=[
                "spike: top 1% of pixels hold >90% of |relevance| — suggests "
                "LRP collapsed to a single spatial location.",
                "flat: Shannon entropy < 0.5 bits — relevance is nearly uniform, "
                "LRP carries no spatial information.",
            ],
        )

    # -----------------------------------------------------------------------
    # T08 — Relevance sign ratio
    # -----------------------------------------------------------------------

    def _t08_sign_ratio(self, frames) -> TestResult:
        """
        Most relevance should be positive (z+ / AlphaBeta(1,0) only propagates
        positive contributions).  A high fraction of negative relevance suggests
        a misconfigured rule or the WSquare layer injecting signed values.
        """
        pos_fracs = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            wide_r, narr_r, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            for r in [wide_r, narr_r]:
                arr = r.cpu().numpy() if isinstance(r, torch.Tensor) else np.array(r)
                total = arr.size
                if total > 0:
                    pos_fracs.append(float((arr > 0).sum() / total))

        pf = np.array(pos_fracs)

        # With pure z+ we expect >80% positive pixels
        status = PASS if pf.mean() > 0.70 else (WARN if pf.mean() > 0.40 else FAIL)

        return TestResult(
            name="T08_relevance_sign_ratio",
            status=status,
            summary=f"Positive-pixel fraction: mean={pf.mean():.3f} std={pf.std():.3f}",
            metrics={
                "pos_frac_mean": float(pf.mean()),
                "pos_frac_std":  float(pf.std()),
                "pos_frac_min":  float(pf.min()),
                "pos_frac_p25":  float(np.percentile(pf, 25)),
            },
            per_frame=pf,
            notes=[
                "AlphaBeta(1,0) = z+ only propagates positive weight contributions.",
                "WSquare at conv1 uses squared weights → purely positive.",
                "Negative relevance can appear at spatial pooling boundaries but "
                "should be a small fraction.",
            ],
        )

    # -----------------------------------------------------------------------
    # T09 — Selector (mask) normalization
    # -----------------------------------------------------------------------

    def _t09_selector_normalization(self, frames) -> TestResult:
        """
        The drive/brake selector passed to the attributor should sum to exactly 1.0
        (it represents a probability distribution over output logits).
        """
        selector_sums = []
        bad_frames    = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            _num_acts = self.num_cmds * self.num_speeds * (self.num_steers + self.num_throts + 1)
            _dummy    = torch.zeros(1, _num_acts, device=self.device)
            selector, _ = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)

            if selector is not None:
                s = float(selector(_dummy).sum().item())
                selector_sums.append(s)
                if abs(s - 1.0) > 1e-4:
                    bad_frames.append((i, s))

        if not selector_sums:
            return TestResult(
                name="T09_selector_normalization",
                status=WARN,
                summary="No selectors returned (all None) — cannot test.",
            )

        ss = np.array(selector_sums)
        status = FAIL if bad_frames else PASS

        return TestResult(
            name="T09_selector_normalization",
            status=status,
            summary=(
                f"Selector sum: mean={ss.mean():.6f} max_dev={max(abs(s-1.0) for s in ss):.2e}"
                + (f"  — {len(bad_frames)} frames not summing to 1" if bad_frames else "")
            ),
            metrics={
                "selector_sum_mean":    float(ss.mean()),
                "selector_sum_std":     float(ss.std()),
                "n_bad_frames":         len(bad_frames),
                "bad_frame_indices":    [b[0] for b in bad_frames[:10]],
            },
            notes=[
                "The selector initialises relevance at the output.  If it sums to "
                "something other than 1, the conserved relevance quantity is not 1 "
                "and T02 results are ambiguous.",
                "Brake selector: lerp weights (1-w) and w already sum to 1 by construction.",
                "Drive selector: softmax weights sum to 1 by construction — "
                "but the lerp interpolation must preserve this.",
            ],
        )

    # -----------------------------------------------------------------------
    # T10 — Speed interpolation validity
    # -----------------------------------------------------------------------

    def _t10_speed_interpolation(self, frames) -> TestResult:
        """
        Check that _lerp_bins produces valid outputs for all speeds in the test set:
        - x0, x1  in [0, num_speeds-1]
        - w        in [0.0, 1.0]
        - speeds outside [min_speeds, max_speeds] are clamped (warn if many)
        """
        issues      = []
        clamp_count = 0

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            spd = frames["speed"][i]

            x0, x1, w = self.lrp._lerp_bins(
                spd, self.min_speeds, self.max_speeds, self.num_speeds
            )
            if not (0 <= x0 <= self.num_speeds - 1):
                issues.append(f"Frame {i}: x0={x0} out of range")
            if not (0 <= x1 <= self.num_speeds - 1):
                issues.append(f"Frame {i}: x1={x1} out of range")
            if not (0.0 <= w <= 1.0):
                issues.append(f"Frame {i}: w={w:.4f} out of [0,1]")
            if spd < self.min_speeds or spd > self.max_speeds:
                clamp_count += 1

        speeds = [frames["speed"][i] for i in range(min(len(frames["cmd"]), self.max_checks))]
        status = FAIL if issues else (WARN if clamp_count > len(frames["cmd"]) * 0.1 else PASS)

        return TestResult(
            name="T10_speed_interpolation",
            status=status,
            summary=(
                "All bins valid." if not issues else
                f"{len(issues)} invalid bin values."
            ) + (f"  {clamp_count} clamped speed values." if clamp_count else ""),
            metrics={
                "n_invalid_bins":   len(issues),
                "n_clamped_speeds": clamp_count,
                "speed_min":        float(min(speeds)),
                "speed_max":        float(max(speeds)),
                "speed_mean":       float(np.mean(speeds)),
            },
            notes=issues[:10],
        )

    # -----------------------------------------------------------------------
    # T11 — Node selection coverage (mode 1 only)
    # -----------------------------------------------------------------------

    def _t11_node_selection_coverage(self, frames) -> TestResult:
        """
        In mode 1, _relevance_filter selects the top-p% nodes from the 256-dim
        FC layer.  We check:
        - n_selected is not 0 (no nodes = no LRP2 ever runs)
        - n_selected is not 256 (all nodes = threshold is too permissive)
        - Distribution of n_selected is stable across frames
        """
        n_selected_list = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            selector, _ = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)

            # Reproduce the LRP1 pass to get FC relevances
            wide_rel = self.lrp._attribute_to_fc(wide, narr, selector)
            arr      = wide_rel.cpu().numpy() if isinstance(wide_rel, torch.Tensor) else np.array(wide_rel)

            # Reproduce _relevance_filter logic
            abs_r   = np.abs(arr.flatten())
            total   = abs_r.sum()
            if total < 1e-12:
                n_selected_list.append(0)
                continue
            order   = np.argsort(abs_r)[::-1]
            cumsum  = np.cumsum(abs_r[order])
            n_sel   = int((cumsum <= self.atoms.p_relevance * total).sum()) + 1
            n_sel   = min(n_sel, len(abs_r))
            n_selected_list.append(n_sel)

        ns = np.array(n_selected_list, dtype=float)
        total_nodes = 256

        status = PASS
        if (ns == 0).any():
            status = FAIL
        elif (ns == total_nodes).mean() > 0.5:
            status = WARN

        return TestResult(
            name="T11_node_selection_coverage",
            status=status,
            summary=(
                f"Nodes selected: mean={ns.mean():.1f}/{total_nodes} "
                f"min={ns.min():.0f} max={ns.max():.0f}"
            ),
            metrics={
                "n_selected_mean":   float(ns.mean()),
                "n_selected_std":    float(ns.std()),
                "n_selected_min":    float(ns.min()),
                "n_selected_max":    float(ns.max()),
                "n_zero_nodes":      int((ns == 0).sum()),
                "n_all_nodes":       int((ns == total_nodes).sum()),
                "p_relevance":       getattr(self.atoms, "p_relevance", "unknown"),
                "total_fc_nodes":    total_nodes,
            },
            per_frame=ns,
            notes=[
                "p_relevance controls the cumulative-mass threshold.",
                "0 nodes selected → LRP2 never runs → heatmap is zeros.",
                "All nodes selected → threshold too permissive, wastes compute.",
            ],
        )

    # -----------------------------------------------------------------------
    # T12 — Context update sensitivity
    # -----------------------------------------------------------------------

    def _t12_context_update_sensitivity(self, frames) -> TestResult:
        """
        Verify that update_context() actually changes the frozen narrow context
        when a different narr_rgb is provided.  Also check that re-running with
        the same narr_rgb gives identical context (determinism).
        """
        if len(frames["cmd"]) < 2:
            return TestResult(
                name="T12_context_update_sensitivity",
                status=WARN,
                summary="Need ≥ 2 frames to test context sensitivity.",
            )

        f0 = 0
        f1 = 1

        def get_ctx(i):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            self.lrp.update_context(wide, narr, frames["speed"][i])
            # Access the frozen context buffer
            try:
                ctx = self.lrp.act_fc_model_lrp.head.fixed_context
            except AttributeError:
                ctx = None
            return ctx

        ctx0a = get_ctx(f0)
        ctx0b = get_ctx(f0)  # same frame, second time
        ctx1  = get_ctx(f1)

        if ctx0a is None:
            return TestResult(
                name="T12_context_update_sensitivity",
                status=WARN,
                summary="Could not access fixed_context buffer — check attribute path.",
            )

        same_frame_diff = float((ctx0a - ctx0b).abs().max().item())
        diff_frame_diff = float((ctx0a - ctx1).abs().max().item())

        deterministic   = same_frame_diff < 1e-6
        sensitive       = diff_frame_diff > 1e-4

        if not deterministic:
            status  = FAIL
            summary = f"update_context is NOT deterministic! same-frame diff={same_frame_diff:.2e}"
        elif not sensitive:
            status  = WARN
            summary = (
                f"Context does not change between frames (diff={diff_frame_diff:.2e}). "
                f"Are both test frames identical?"
            )
        else:
            status  = PASS
            summary = (
                f"Deterministic (Δ={same_frame_diff:.2e}) and "
                f"frame-sensitive (Δ={diff_frame_diff:.2e})."
            )

        return TestResult(
            name="T12_context_update_sensitivity",
            status=status,
            summary=summary,
            metrics={
                "same_frame_context_diff": same_frame_diff,
                "diff_frame_context_diff": diff_frame_diff,
                "deterministic":           deterministic,
                "sensitive_to_narr":       sensitive,
            },
        )

    # -----------------------------------------------------------------------
    # T13 — Logit range sanity
    # -----------------------------------------------------------------------

    def _t13_logit_range_sanity(self, frames) -> TestResult:
        """
        Raw act_head logits should be in a finite, non-degenerate range.
        Flags:
          - All logits identical (constant output, dead network)
          - Logits outside ±20 (exploding weights)
          - All logits near zero (untrained / collapsed)
        """
        all_logit_stds  = []
        all_logit_maxabs = []
        dead_frames     = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            with torch.no_grad():
                raw = self.lrp.model_lrp(wide, narr)
                logits = raw.view(1, self.num_cmds, self.num_speeds,
                                  self.num_steers + self.num_throts + 1)
            logits_cmd = logits[0, cmd].cpu().numpy()   # [num_speeds, 13]
            std    = float(logits_cmd.std())
            maxabs = float(np.abs(logits_cmd).max())
            all_logit_stds.append(std)
            all_logit_maxabs.append(maxabs)
            if std < 1e-5:
                dead_frames.append(i)

        stds = np.array(all_logit_stds)
        mabs = np.array(all_logit_maxabs)

        exploding = int((mabs > 20).sum())
        dead      = len(dead_frames)

        status = PASS
        if dead > len(frames["cmd"]) * 0.1 or exploding > 0:
            status = FAIL
        elif dead > 0:
            status = WARN

        return TestResult(
            name="T13_logit_range_sanity",
            status=status,
            summary=(
                f"Logit std: mean={stds.mean():.3f}  maxabs: mean={mabs.mean():.2f}  "
                f"dead={dead}  exploding={exploding}"
            ),
            metrics={
                "logit_std_mean":    float(stds.mean()),
                "logit_std_min":     float(stds.min()),
                "logit_maxabs_mean": float(mabs.mean()),
                "logit_maxabs_max":  float(mabs.max()),
                "n_dead_frames":     dead,
                "n_exploding_frames":exploding,
            },
            per_frame=stds,
            notes=[
                "Dead frame: all logits for this cmd have std < 1e-5 — "
                "the network is effectively outputting a constant.",
                "Exploding: |logit| > 20 may destabilise softmax / LRP.",
            ],
        )

    # -----------------------------------------------------------------------
    # T14 — Mode consistency (modes 2 and 3)
    # -----------------------------------------------------------------------

    def _t14_mode_consistency(self, frames) -> TestResult:
        """
        Modes 2 (layer-level, fc→input) and 3 (output→input single-pass) should
        produce qualitatively consistent wide_frac values.  A large systematic
        difference suggests one mode has a bug in how it initialises relevance.
        """
        if len(frames["cmd"]) < 5:
            return TestResult(
                name="T14_mode_consistency",
                status=WARN,
                summary="Need ≥ 5 frames for meaningful consistency check.",
            )

        fracs_m2 = []
        fracs_m3 = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)

            # mode 2 — fc→input
            _, _, wf2, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            fracs_m2.append(float(wf2))

            # mode 3 / 4 — output→input
            _, _, wf3, _ = self.lrp.forward_relevance(
                wide, narr, beg="output", end="input", cmd=cmd, spd=spd
            )
            fracs_m3.append(float(wf3))

        m2 = np.array(fracs_m2)
        m3 = np.array(fracs_m3)
        corr     = float(np.corrcoef(m2, m3)[0, 1]) if m2.std() > 1e-6 and m3.std() > 1e-6 else float("nan")
        mean_abs_diff = float(np.abs(m2 - m3).mean())

        status = PASS if corr > 0.8 else (WARN if corr > 0.5 else FAIL)

        return TestResult(
            name="T14_mode_consistency",
            status=status,
            summary=(
                f"wide_frac Pearson r(mode2, mode3)={corr:.3f}  "
                f"mean|diff|={mean_abs_diff:.3f}"
            ),
            metrics={
                "pearson_r":           corr,
                "mean_abs_diff":       mean_abs_diff,
                "mode2_wide_frac_mean":float(m2.mean()),
                "mode3_wide_frac_mean":float(m3.mean()),
            },
            notes=[
                "Low correlation does NOT necessarily mean a bug — modes 2 and 3 "
                "initialise relevance differently (FC-level vs output-level).",
                "But a systematic constant offset between modes should be explainable.",
            ],
        )

    # -----------------------------------------------------------------------
    # T15 — Ground-truth brake alignment (optional)
    # -----------------------------------------------------------------------

    def _t15_gt_brake_alignment(self, frames) -> TestResult:
        """
        If test frames include 'ground_truth_brake' (bool, was the agent outputting
        brake=1 during data collection?), compare against is_brake from LRP.
        Reports precision, recall, and confusion matrix.
        """
        gt_frames = []
        for i in range(min(len(frames["is_brake"]), self.max_checks)):
            if frames["is_brake"][i]:
                gt_frames.append(i)
        if not gt_frames:
            return TestResult(
                name="T15_ground_truth_brake_alignment",
                status=WARN,
                summary="No ground_truth_brake labels in test frames — skipped.",
                notes=["Add 'ground_truth_brake': True/False to each frame dict "
                       "to enable this test."],
            )

        tp = fp = tn = fn = 0
        for i in gt_frames:
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            _, is_brake = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)
            pred = bool(is_brake)
            gt   = bool(frames["is_brake"][i])

            if pred and gt:     tp += 1
            elif pred and not gt: fp += 1
            elif not pred and gt: fn += 1
            else:               tn += 1

        n       = tp + fp + tn + fn
        prec    = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec     = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        acc     = (tp + tn) / n
        n_gt_pos = tp + fn

        if n_gt_pos == 0:
            status  = WARN
            summary = "No positive (brake=True) ground truth samples."
        elif math.isnan(prec):
            status  = FAIL
            summary = f"LRP never predicts brake (recall={rec:.2f})."
        elif rec < 0.3:
            status  = FAIL
            summary = f"Very low recall={rec:.2f} — missing most brake frames."
        elif rec < 0.6 or prec < 0.5:
            status  = WARN
            summary = f"Precision={prec:.2f} Recall={rec:.2f} — needs improvement."
        else:
            status  = PASS
            summary = f"Precision={prec:.2f} Recall={rec:.2f} Accuracy={acc:.2f}"

        return TestResult(
            name="T15_ground_truth_brake_alignment",
            status=status,
            summary=summary,
            metrics={
                "precision":     prec,
                "recall":        rec,
                "accuracy":      acc,
                "true_positive": tp,
                "false_positive":fp,
                "true_negative": tn,
                "false_negative":fn,
                "n_gt_brake":    n_gt_pos,
                "n_gt_drive":    tn + fp,
            },
            notes=[
                "GT brake=True means the agent's post_process returned brake=1 "
                "during the original data collection episode.",
                "This requires the agent's brake_prob > 0.5 at that specific frame.",
                f"Note: with 55-term softmax, this bar is very high (logit > ln(54) ≈ "
                f"{math.log(self.num_steers*self.num_throts*2):.1f}).",
            ],
        )

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def print_report(self, results: Dict[str, TestResult]) -> None:
        status_sym = {PASS: "✓ PASS", FAIL: "✗ FAIL", WARN: "△ WARN", ERROR: "! ERROR"}
        sep = "─" * 70

        print(f"\n{'='*70}")
        print("  DETAILED TEST REPORT")
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
                        print(f"    {k:<40s} {v:.6g}")
                    elif isinstance(v, list) and len(v) > 5:
                        print(f"    {k:<40s} [{len(v)} items]")
                    else:
                        print(f"    {k:<40s} {v}")
            if r.notes:
                print("  Notes:")
                for n in r.notes:
                    print(f"    ▸ {n}")
            if r.exception:
                print("  Exception:")
                for line in r.exception.strip().split("\n"):
                    print(f"    {line}")
            print(f"  Wall time: {r.metrics.get('wall_time_s', '?')}s")

        # Summary table
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
        os.makedirs(out_dir, exist_ok=True)

        txt_path = os.path.join(out_dir, "lrp_test_report.txt")
        with open(txt_path, "w") as fh:
            import io, contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.print_report(results)
            fh.write(buf.getvalue())

        for name, r in results.items():
            if r.per_frame is not None:
                npy_path = os.path.join(out_dir, f"{name}_per_frame.npy")
                np.save(npy_path, r.per_frame)

        print(f"Report saved to {out_dir}")


    def _t16_concat_level_split(self, frames) -> TestResult:
        """
        Measures wide/narr relevance ratio at the 576-dim concatenation point,
        before the ResNet.  This isolates whether degeneration is caused by:
          (a) act_head[0] weights favouring wide features  → concat ratio << 8:1
          (b) ResNet amplification post-hoc               → concat ratio ≈ 8:1
                                                              but pixel ratio → 0
        Expected narr_frac ≈ 64/576 = 0.111 if purely dimensional.
        """
        narr_fracs_concat = []
        narr_fracs_pixel  = []
        concat_relevance_sums = []

        for i in range(min(len(frames["cmd"]), self.max_checks)):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]

            self.lrp.update_context(wide, narr, spd)
            selector, _ = self.lrp._build_drive_brake_selector(wide, narr, cmd, spd)

            # Concat-level
            wide_rc, narr_rc, narr_frac_c = self.lrp._attribute_to_concat(wide, narr, selector)
            narr_fracs_concat.append(narr_frac_c)

            # Pixel-level (raw, before cross-normalize)
            wide_r, narr_r, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            w_c = wide_rc.cpu().numpy() if isinstance(wide_rc, torch.Tensor) else np.array(wide_rc)
            n_c = narr_rc.cpu().numpy() if isinstance(narr_rc, torch.Tensor) else np.array(narr_rc)
            total_rel_c = w_c.sum() + n_c.sum()
            concat_relevance_sums.append(total_rel_c)

            w_arr = wide_r.cpu().numpy() if isinstance(wide_r, torch.Tensor) else np.array(wide_r)
            n_arr = narr_r.cpu().numpy() if isinstance(narr_r, torch.Tensor) else np.array(narr_r)
            total = np.abs(w_arr).sum() + np.abs(n_arr).sum() + 1e-12
            narr_fracs_pixel.append(float(np.abs(n_arr).sum() / total))

        nc  = np.array(narr_fracs_concat)
        np_ = np.array(narr_fracs_pixel)
        dim_expected = 64 / 576  # ≈ 0.111

        min_rel, max_rel = np.min(total_rel_c), np.max(total_rel_c)

        # Diagnosis
        # Absolute gap between pixel and concat level — this is what the correction fixes
        correction_gap  = float(abs(nc - np_).mean())
        concat_vs_dim   = float(nc.mean()) / dim_expected  # how much network deviates from pure dim

        if min_rel < 1 - 0.4 or max_rel > 1 + 0.4:
            # Relevance conservation is not given
            status    = FAIL
            diagnosis = (
                f"relevbance conservation not given! "
                f"Min total relevance at concat layer: {min_rel}"
                f"Max total relevance at concat layer: {max_rel}"
            )
        elif correction_gap > 0.02:
            # Pixel and concat still disagree — correction not applied or broken
            status    = FAIL
            diagnosis = (
                f"Pixel/concat mismatch gap={correction_gap:.4f} — "
                f"post-hoc correction not applied or incorrect."
            )
        elif nc.mean() < 0.005:
            # Concat itself is degenerate — weights are pathological
            status    = FAIL
            diagnosis = "act_head[0] weights assign near-zero relevance to narrow — pathological."
        else:
            status    = PASS
            diagnosis = (
                f"Pixel matches concat (gap={correction_gap:.4f}). "
                f"Narrow fraction {nc.mean():.4f} = {concat_vs_dim:.1%} of dimensional baseline "
                f"— reflects learned weight structure."
            )
        return TestResult(
            name="T16_concat_level_split",
            status=status,
            summary=(
                f"narr_frac @ concat: mean={nc.mean():.4f} (expected≈{dim_expected:.3f})  "
                f"@ pixel: mean={np_.mean():.4f}  — {diagnosis}"
            ),
            metrics={
                "narr_frac_concat_mean":    float(nc.mean()),
                "narr_frac_concat_std":     float(nc.std()),
                "narr_frac_pixel_mean":     float(np_.mean()),
                "narr_frac_pixel_std":      float(np_.std()),
                "dimensional_expected":     dim_expected,
            },
            per_frame=nc,
            notes=[
                f"Dimensional baseline: 64/576 = {dim_expected:.3f}.",
                "If narr_frac_concat << 0.111, the act_head[0] weight structure is the problem.",
                "If narr_frac_concat ≈ 0.111 but narr_frac_pixel ≈ 0, apply the post-hoc "
                "rescaling in _saliency_map to restore the correct ratio.",
            ],
        )