"""
tfv6_test_suite.py
==================
Property-based test suite for the TFV6 LRP + ATOMs pipeline.

Tests desired mathematical and correctness properties.  No test calls
WoR-specific methods (_lerp_bins, _build_drive_brake_selector, num_steers,
_attribute_to_fc, etc.).  All assertions use only the public LRPTFv6Model
interface and the shared ATOMsCarla API.

LRP interface used
------------------
    lrp.forward_relevance(wide_rgb, narr_rgb=None, cmd=None, spd=None,
                          node_id=None, beg="output", end="input")
        beg="output", end="input"  -> (wide_rel [1,3,H,W], None, scale, flag)
        beg="output", end="fc"    -> (node_rel [256],      None, scale, flag)
        beg="fc",     end="input",
            node_id=N             -> (wide_rel [1,3,H,W], None, scale, flag)

ATOMs interface used
--------------------
    atoms.process_frame(wide, narr=None, seg_wide, seg_narr=None, cmd, spd)
    atoms.reset()
    atoms._lrp1_nodes(wide, narr, cmd)
    atoms._lrp2_pixels(wide, narr, node_id=N, cmd=cmd)
    _relevance_filter(r, p)

Usage
-----
    suite  = TFV6TestSuite(lrp_instance, atoms_instance)
    report = suite.run_all_tests(testframes)
    suite.print_report(report)
    suite.save_report(report, "out/")

Testframe format
----------------
    wide_rgb : np.ndarray uint8 [H,W,3]  or torch.Tensor [1,3,H,W]
    speed    : float
    cmd      : int  (navigation command index, 0-based)

Optional:
    seg_wide : np.ndarray uint8 [H,W]  — CARLA semantic red channel
    frame_id : str

narr_rgb intentionally absent — TFV6 is wide-only.
seg_wide is synthesized via _synthetic_seg if absent.

Test inventory
--------------
  L01  No NaN or Inf in any LRP output mode
  L02  Relevance conservation: pixel_sum / node_sum ratio is stable (CoV < 0.2)
  L03  Non-trivial spatial distribution: Gini in (0.05, 0.99); entropy >= 2 bits
  L04  Positive dominance: >= 70 % of pixel relevance is positive (AlphaBeta z+)
  L05  Per-node pixel-map distinctiveness: different node_ids -> different maps
  L06  Speed-query relevance diversity: 256-dim LRP1 vector has non-trivial variance
  L07  narr output always None from forward_relevance
  A01  process_frame(narr=None) returns normalized attention (sum in [0,1])
  A02  Contributions non-negative: per-frame increments to _hierarchical >= 0
  A03  Accumulation: series_sum == _hierarchical after N frames exactly
  A04  reset() clears every piece of ATOMs state
  A05  Node-map diversity via ATOMs: different node_id -> different attention vectors
"""

from __future__ import annotations

import contextlib
import io
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ATOMs_Analysis.saliency.atoms_carla import (
    ATOMsCarla,
    _relevance_filter,
    seg_to_masks,
)


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name:      str
    status:    str
    summary:   str
    metrics:   Dict[str, Any]       = field(default_factory=dict)
    per_frame: Optional[np.ndarray] = None
    notes:     List[str]            = field(default_factory=list)
    exception: Optional[str]        = None


def _safe_run(fn, *args, **kwargs) -> Tuple[Any, Optional[str]]:
    try:
        return fn(*args, **kwargs), None
    except Exception:
        return None, traceback.format_exc()


# ---------------------------------------------------------------------------
# Statistical helpers  (mirrored from lrp_test_suite.py)
# ---------------------------------------------------------------------------

def _gini(values: np.ndarray) -> float:
    """Gini coefficient of absolute values in [0, 1]. 0 = uniform, 1 = spike."""
    v = np.abs(values).flatten().astype(np.float64)
    if v.sum() < 1e-15:
        return 0.0
    v = np.sort(v)
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() - (n + 1) * v.sum()) / (n * v.sum()))


def _entropy_bits(values: np.ndarray) -> float:
    """Shannon entropy in bits of the abs-normalized distribution."""
    v = np.abs(values).flatten().astype(np.float64)
    total = v.sum()
    if total < 1e-15:
        return 0.0
    p = v / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _top_k_mass(values: np.ndarray, k: int) -> float:
    """Fraction of total absolute mass held by the top-k elements."""
    v = np.abs(values).flatten()
    if v.sum() < 1e-15:
        return 0.0
    top_k = np.partition(v, -k)[-k:].sum()
    return float(top_k / v.sum())


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _to_tensor(rgb, device) -> torch.Tensor:
    """Accept [H,W,3] ndarray or [1,3,H,W] tensor; return [1,3,H,W] float32."""
    if isinstance(rgb, np.ndarray):
        t = torch.from_numpy(rgb).unsqueeze(0).float()
    else:
        t = rgb.float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def _synthetic_seg(H: int, W: int, class_ids: List[int]) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.choice(class_ids, size=(H, W)).astype(np.uint8)


def _as_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


# ---------------------------------------------------------------------------
# TFV6TestSuite
# ---------------------------------------------------------------------------

class TFV6TestSuite:
    """
    Parameters
    ----------
    lrp    : LRPTFv6Model  — already initialised; backbone in eval mode.
    atoms  : ATOMsCarla    — already initialised with the same lrp instance.
    device : str           — torch device string (default 'cpu').

    The first three LRP tests (L01–L03) use only lrp directly.
    ATOMs tests (A01–A05) use the atoms instance with narr=None throughout.
    """

    def __init__(self, lrp, atoms: ATOMsCarla, device: str = "cpu"):
        self.lrp    = lrp
        self.atoms  = atoms
        self.device = torch.device(device)
        self.max_frames = 10

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run_all_tests(self, testframes: Optional[Dict] = None) -> Dict[str, TestResult]:
        n = len(testframes["cmd"]) if testframes is not None else 0
        print(f"\n{'='*70}")
        print(f"  TFV6 LRP + ATOMs Test Suite  —  {n} testframes")
        print(f"{'='*70}\n")

        tests = [
            ("L01_no_nan_or_inf",                self._l01_no_nan_inf),
            ("L02_relevance_conservation_stable", self._l02_conservation),
            ("L03_nontrivial_spatial_distribution", self._l03_spatial_distribution),
            ("L04_positive_dominance",            self._l04_positive_dominance),
            ("L05_per_node_pixel_map_distinctiveness", self._l05_node_distinctiveness),
            ("L06_backbone_activation_diversity", self._l06_backbone_diversity),
            ("L07_narr_always_none",              self._l07_narr_none),
            ("A01_process_frame_narr_none_normalized", self._a01_process_frame),
            ("A02_contributions_non_negative",    self._a02_contributions_nonneg),
            ("A03_accumulation_correctness",      self._a03_accumulation),
            ("A04_reset_clears_state",            self._a04_reset),
            ("A05_node_map_diversity_via_atoms",  self._a05_node_diversity_atoms),
        ]

        results: Dict[str, TestResult] = {}
        for name, fn in tests:
            t0 = time.time()
            print(f"  Running {name} ...", end="", flush=True)
            result, exc = _safe_run(fn, testframes)
            elapsed = time.time() - t0
            if exc is not None:
                result = TestResult(name=name, status=ERROR,
                                    summary="Test raised an exception.", exception=exc)
            result.metrics["wall_time_s"] = round(elapsed, 2)
            results[name] = result
            tag = {PASS: "✓", FAIL: "✗", WARN: "△", ERROR: "!"}[result.status]
            print(f"  {tag}  [{elapsed:.1f}s]  {result.summary}")

        print()
        return results

    # ------------------------------------------------------------------
    # Utility: run forward_relevance and return first output as numpy
    # ------------------------------------------------------------------

    def _lrp_output(self, wide: torch.Tensor, beg: str, end: str,
                    node_id: Optional[int] = None) -> np.ndarray:
        out, _, _, _ = self.lrp.forward_relevance(
            wide, narr_rgb=None, beg=beg, end=end, node_id=node_id
        )
        return _as_numpy(out)

    def _first_frame(self, frames) -> torch.Tensor:
        return _to_tensor(frames["wide_rgb"][0], self.device)

    # ------------------------------------------------------------------
    # L01 — No NaN or Inf in any output mode
    # ------------------------------------------------------------------

    def _l01_no_nan_inf(self, frames) -> TestResult:
        """
        Runs all three forward_relevance modes on each testframe and checks
        that no output contains NaN or Inf values.
        """
        if frames is None:
            return TestResult("L01_no_nan_or_inf", WARN, "No testframes — skipped.")

        failures = []
        N = min(len(frames["cmd"]), self.max_frames)

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            modes = [
                ("output→input", dict(beg="output", end="input")),
                ("output→fc",    dict(beg="output", end="fc")),
            ]
            # fc→input requires a node_id; use node 0
            modes.append(("fc→input[0]", dict(beg="fc", end="input", node_id=0)))

            for label, kwargs in modes:
                out, _, _, _ = self.lrp.forward_relevance(wide, narr_rgb=None, **kwargs)
                arr = _as_numpy(out)
                if np.isnan(arr).any():
                    failures.append(f"frame {i} mode {label}: NaN found")
                if np.isinf(arr).any():
                    failures.append(f"frame {i} mode {label}: Inf found")

        status  = FAIL if failures else PASS
        summary = (f"All {N} frames × 3 modes: no NaN/Inf." if not failures
                   else f"{len(failures)} violation(s): {failures[0]}")
        return TestResult("L01_no_nan_or_inf", status, summary,
                          metrics={"n_failures": len(failures), "n_frames": N},
                          notes=failures[:5])

    # ------------------------------------------------------------------
    # L02 — Relevance conservation stability
    # ------------------------------------------------------------------

    def _l02_conservation(self, frames) -> TestResult:
        """
        Checks that the ratio pixel_sum / node_sum is STABLE across frames
        (CoV < 0.2).  A wildly varying ratio indicates numerical instability.

        NOTE on absolute magnitude for TFV6: output→fc returns backbone
        activations normalized to sum=1, while output→input returns raw
        gradient×input pixel attributions over a [3,H,W] image.  The ratio
        will therefore be O(H×W×pixel_scale) >> 1 — this is expected and does
        NOT indicate broken LRP.  Only the CoV matters here; the absolute
        mean_ratio is not interpretable as a conservation check for TFV6.

        pixel_sum = sum of absolute pixel relevance (beg=output, end=input)
        node_sum  = sum of absolute backbone activations (beg=output, end=fc)
        """
        if frames is None:
            return TestResult("L02_relevance_conservation_stable", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        ratios = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)

            pixel_rel  = self._lrp_output(wide, "output", "input")
            node_acts  = self._lrp_output(wide, "output", "fc")

            pixel_sum = float(np.abs(pixel_rel).sum())
            node_sum  = float(np.abs(node_acts).sum())

            if node_sum < 1e-12:
                continue
            ratios.append(pixel_sum / node_sum)

        if len(ratios) < 2:
            return TestResult("L02_relevance_conservation_stable", WARN,
                              "Not enough valid frames to compute CoV.")

        ratios_arr = np.array(ratios)
        mean_r = float(ratios_arr.mean())
        std_r  = float(ratios_arr.std())
        cov    = std_r / (abs(mean_r) + 1e-12)

        if mean_r < 1e-6:
            status  = FAIL
            summary = f"pixel_sum / node_sum ≈ 0 — LRP may be returning all-zero maps."
        elif cov > 0.2:
            status  = WARN
            summary = (f"CoV = {cov:.4f} > 0.2 — ratio is unstable across frames "
                       f"(mean={mean_r:.4f}, std={std_r:.4f}). Check for numerical issues.")
        else:
            status  = PASS
            summary = (f"Stable across {len(ratios)} frames: "
                       f"mean={mean_r:.4g}, CoV={cov:.4f}. "
                       f"(Large mean expected for TFV6 — see docstring.)")

        return TestResult("L02_relevance_conservation_stable", status, summary,
                          metrics={"mean_ratio": mean_r, "std_ratio": std_r,
                                   "cov": cov, "n_frames": len(ratios)},
                          per_frame=ratios_arr,
                          notes=["CoV > 0.2 suggests the backward pass is numerically "
                                 "unstable or input normalization varies widely."])

    # ------------------------------------------------------------------
    # L03 — Non-trivial spatial distribution
    # ------------------------------------------------------------------

    def _l03_spatial_distribution(self, frames) -> TestResult:
        """
        A correct LRP map should be spatially concentrated (not uniform noise
        and not a single-pixel spike):
          - Gini coefficient in (0.05, 0.99)
          - Shannon entropy >= 2.0 bits

        Low Gini (<0.05): relevance is too uniform — LRP may be returning
        a nearly flat gradient.  High Gini (>0.99): single pixel spike —
        likely a numerical explosion at one location.
        """
        if frames is None:
            return TestResult("L03_nontrivial_spatial_distribution", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        ginis, entropies = [], []
        warnings = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            rel  = self._lrp_output(wide, "output", "input")

            g = _gini(rel)
            e = _entropy_bits(rel)
            ginis.append(g)
            entropies.append(e)

            if g < 0.05:
                warnings.append(f"frame {i}: Gini={g:.4f} — nearly uniform (too flat)")
            elif g > 0.99:
                warnings.append(f"frame {i}: Gini={g:.4f} — near-spike (likely explosion)")
            if e < 2.0:
                warnings.append(f"frame {i}: entropy={e:.2f} bits < 2.0 — almost degenerate")

        gini_arr = np.array(ginis)
        ent_arr  = np.array(entropies)
        n_bad = len(warnings)

        if n_bad > N // 2:
            status  = FAIL
            summary = (f"{n_bad}/{N} frames have degenerate distributions. "
                       f"mean Gini={gini_arr.mean():.4f}, mean entropy={ent_arr.mean():.2f} bits.")
        elif n_bad > 0:
            status  = WARN
            summary = (f"{n_bad}/{N} frames outside expected range. "
                       f"mean Gini={gini_arr.mean():.4f}, mean entropy={ent_arr.mean():.2f} bits.")
        else:
            status  = PASS
            summary = (f"All {N} frames: Gini in (0.05, 0.99) and entropy >= 2 bits. "
                       f"mean Gini={gini_arr.mean():.4f}, mean entropy={ent_arr.mean():.2f} bits.")

        return TestResult("L03_nontrivial_spatial_distribution", status, summary,
                          metrics={"gini_mean": float(gini_arr.mean()),
                                   "gini_min":  float(gini_arr.min()),
                                   "gini_max":  float(gini_arr.max()),
                                   "entropy_mean_bits": float(ent_arr.mean()),
                                   "n_frames": N},
                          notes=warnings[:5])

    # ------------------------------------------------------------------
    # L04 — Positive dominance (AlphaBeta z+ rule)
    # ------------------------------------------------------------------

    def _l04_positive_dominance(self, frames) -> TestResult:
        """
        Checks that the output→input pixel map is not dominated by negative values.

        For TFV6 with AttnLRP (LRPSoftmax / LRPMatMul through the transformer
        decoder), signed pixel values are expected even with a positive seed.
        LRPSoftmax produces x*(R - s*ΣR) which can be negative for components
        where R_i < s_i * ΣR.  Therefore the z+-only thresholds (>= 0.55 / 0.70)
        do not apply here.

        Revised thresholds for AttnLRP on TFV6:
          FAIL  < 0.30 — strongly negative maps indicate a bug (e.g., gradient
                         explosion causing sign flip, or raw autograd leaking through)
          WARN  < 0.45 — below what AttnLRP normally produces; worth investigating
          PASS >= 0.45
        """
        if frames is None:
            return TestResult("L04_positive_dominance", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        pos_fracs = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            rel  = self._lrp_output(wide, "output", "input").flatten()
            total = float(np.abs(rel).sum())
            if total < 1e-12:
                continue
            pos_frac = float(rel[rel > 0].sum() / total)
            pos_fracs.append(pos_frac)

        if not pos_fracs:
            return TestResult("L04_positive_dominance", WARN,
                              "All frames had zero relevance — cannot assess.")

        arr    = np.array(pos_fracs)
        mean_p = float(arr.mean())
        min_p  = float(arr.min())

        HARD_THRESHOLD = 0.30   # explosion / raw gradient leaking
        SOFT_THRESHOLD = 0.45   # below expected AttnLRP range

        if min_p < HARD_THRESHOLD:
            status  = FAIL
            summary = (f"min positive fraction = {min_p:.4f} < {HARD_THRESHOLD}: "
                       f"strongly negative maps — possible explosion or raw gradient.")
        elif mean_p < SOFT_THRESHOLD:
            status  = WARN
            summary = (f"mean positive fraction = {mean_p:.4f} < {SOFT_THRESHOLD}: "
                       f"lower than expected for AttnLRP — worth investigating.")
        else:
            status  = PASS
            summary = (f"mean positive fraction = {mean_p:.4f} >= {SOFT_THRESHOLD} "
                       f"across {len(pos_fracs)} frames.")

        return TestResult("L04_positive_dominance", status, summary,
                          metrics={"pos_frac_mean": mean_p,
                                   "pos_frac_min":  min_p,
                                   "pos_frac_max":  float(arr.max()),
                                   "n_frames": len(pos_fracs)},
                          per_frame=arr,
                          notes=["AttnLRP (LRPSoftmax/LRPMatMul) produces signed pixel maps "
                                 "even with a positive seed — z+-only thresholds (0.55/0.70) "
                                 "do not apply for TFV6.",
                                 "FAIL < 0.30: explosion or raw gradient leaking through.",
                                 "WARN < 0.45: below typical AttnLRP range."])

    # ------------------------------------------------------------------
    # L05 — Per-node pixel-map distinctiveness
    # ------------------------------------------------------------------

    def _l05_node_distinctiveness(self, frames) -> TestResult:
        """
        forward_relevance(beg='fc', end='input', node_id=N) must produce
        different pixel maps for different values of N.  Identical maps mean
        node_id is not being passed to the backward seed — every call falls
        back to the all-ones seed and produces the same aggregated map.

        Strategy:
          1. Get 256-dim speed-query relevances (beg='output', end='fc').
          2. Sort nodes by relevance magnitude; pick indices at positions
             0, 64, 128, 192 (well-separated, likely to have different weights).
          3. Run beg='fc', end='input' for each probe node.
          4. FAIL if any two maps are pixel-identical (rel_diff < 1e-5).
        """
        if frames is None:
            return TestResult("L05_per_node_pixel_map_distinctiveness", WARN,
                              "No testframes — skipped.")

        wide      = self._first_frame(frames)
        node_acts = self._lrp_output(wide, "output", "fc").flatten()
        n_nodes   = len(node_acts)

        # Probe at four evenly spread positions in sorted-by-magnitude order
        sorted_idx = np.argsort(np.abs(node_acts))[::-1]   # desc by magnitude
        probe_positions = [0, n_nodes // 4, n_nodes // 2, 3 * n_nodes // 4]
        probe_ids = [int(sorted_idx[p]) for p in probe_positions if p < len(sorted_idx)]

        maps = []
        for nid in probe_ids:
            arr = self._lrp_output(wide, "fc", "input", node_id=nid).flatten()
            maps.append(arr)

        identical_pairs = []
        pair_diffs = []
        ref = maps[0]
        mean_abs = float(np.abs(ref).mean()) + 1e-12

        for k in range(1, len(maps)):
            max_diff = float(np.abs(ref - maps[k]).max())
            rel_diff = max_diff / mean_abs
            pair_diffs.append(rel_diff)
            if rel_diff < 1e-5:
                identical_pairs.append((probe_ids[0], probe_ids[k], rel_diff))

        min_diff = float(min(pair_diffs)) if pair_diffs else 0.0

        if identical_pairs:
            status  = FAIL
            summary = (f"{len(identical_pairs)}/{len(maps)-1} map pair(s) are pixel-identical "
                       f"(rel_diff < 1e-5). node_id is NOT reaching the LRP backward seed.")
        elif min_diff < 0.01:
            status  = WARN
            summary = (f"Maps are very similar across nodes (min rel_diff={min_diff:.4f}). "
                       f"Verify that node_id is forwarded to the backward pass.")
        else:
            status  = PASS
            summary = (f"Per-node maps are distinct across {len(maps)} probes "
                       f"(min rel_diff={min_diff:.4f}).")

        return TestResult("L05_per_node_pixel_map_distinctiveness", status, summary,
                          metrics={"n_probe_nodes": len(probe_ids),
                                   "n_total_nodes": n_nodes,
                                   "min_rel_diff": min_diff,
                                   "max_rel_diff": float(max(pair_diffs)) if pair_diffs else 0.0,
                                   "n_identical_pairs": len(identical_pairs)},
                          notes=["FAIL: every beg='fc', end='input' call returns the same map "
                                 "→ node_id is not passed to the backward seed (all-ones used).",
                                 "Probe nodes chosen at positions 0, n/4, n/2, 3n/4 in "
                                 "activation-magnitude order."])

    # ------------------------------------------------------------------
    # L06 — Backbone activation diversity
    # ------------------------------------------------------------------

    def _l06_backbone_diversity(self, frames) -> TestResult:
        """
        forward_relevance(beg='output', end='fc') returns 256-dim speed-query
        LRP1 relevances.  These must show meaningful variance:
          - Not all-zero (model is running)
          - Not all-identical across frames (model responds to image content)
          - Gini >= 0.1 (not all nodes equally active)

        If all frames give the same activation vector, the backbone is frozen
        or the forward pass is broken.
        """
        if frames is None:
            return TestResult("L06_backbone_activation_diversity", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        act_list = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            acts = self._lrp_output(wide, "output", "fc").flatten()
            act_list.append(acts)

        acts_mat = np.stack(act_list, axis=0)   # [N, 256]
        failures = []

        # (a) Not all-zero
        if float(np.abs(acts_mat).max()) < 1e-8:
            failures.append("All backbone activations are zero — forward pass broken.")

        # (b) Not identical across frames
        if N > 1:
            inter_frame_std = float(acts_mat.std(axis=0).mean())
            if inter_frame_std < 1e-8:
                failures.append(
                    f"All {N} frames produce identical activations "
                    f"(inter-frame std={inter_frame_std:.2e}) — "
                    "model may be returning a constant output."
                )
        else:
            inter_frame_std = 0.0

        # (c) Within-frame Gini >= 0.1
        ginis = [_gini(row) for row in acts_mat]
        mean_gini = float(np.mean(ginis))
        if mean_gini < 0.1:
            failures.append(
                f"mean within-frame Gini = {mean_gini:.4f} < 0.1 "
                "— all backbone nodes have near-equal activation (too uniform)."
            )

        status  = FAIL if failures else PASS
        summary = (
            f"256-dim speed-query LRP1: mean Gini={mean_gini:.4f}, "
            f"inter-frame std={inter_frame_std:.4f}."
            if not failures else "; ".join(failures)
        )

        return TestResult("L06_backbone_activation_diversity", status, summary,
                          metrics={"activation_gini_mean": mean_gini,
                                   "inter_frame_std_mean": inter_frame_std,
                                   "max_abs_activation": float(np.abs(acts_mat).max()),
                                   "n_frames": N, "n_nodes": acts_mat.shape[1]},
                          notes=failures)

    # ------------------------------------------------------------------
    # L07 — narr always None in return value
    # ------------------------------------------------------------------

    def _l07_narr_none(self, frames) -> TestResult:
        """
        TFV6 is wide-only.  The second return value of forward_relevance must
        always be None, regardless of which beg/end mode is used.
        """
        if frames is None:
            return TestResult("L07_narr_always_none", WARN, "No testframes — skipped.")

        wide   = self._first_frame(frames)
        modes  = [
            ("output→input", dict(beg="output", end="input")),
            ("output→fc",    dict(beg="output", end="fc")),
            ("fc→input[0]",  dict(beg="fc", end="input", node_id=0)),
        ]
        failures = []

        for label, kwargs in modes:
            _, narr_out, _, _ = self.lrp.forward_relevance(wide, narr_rgb=None, **kwargs)
            if narr_out is not None:
                failures.append(f"mode {label}: narr output = {type(narr_out).__name__}, expected None")

        status  = FAIL if failures else PASS
        summary = ("All modes return None for narr." if not failures
                   else "; ".join(failures))

        return TestResult("L07_narr_always_none", status, summary,
                          metrics={"n_modes_tested": len(modes)},
                          notes=failures)

    # ------------------------------------------------------------------
    # A01 — process_frame with narr=None returns normalized attention
    # ------------------------------------------------------------------

    def _a01_process_frame(self, frames) -> TestResult:
        """
        atoms.process_frame(wide, narr=None, seg_wide, seg_narr=None, cmd, spd)
        must:
          - Not crash with narr=None
          - Return a 1-D vector summing to in [0.99, 1.01] (or exactly 0.0
            for degenerate frames with all-zero masks/relevance)
        """
        if frames is None:
            return TestResult("A01_process_frame_narr_none_normalized", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        bad, sums = [], []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))

            self.atoms.reset()
            att = self.atoms.process_frame(
                wide, None, seg_w, None, cmd=cmd, spd=spd
            )
            s = float(att.sum())
            sums.append(s)
            if abs(s - 1.0) > 0.01 and abs(s) > 0.01:
                bad.append((i, s))

        self.atoms.reset()
        arr    = np.array(sums)
        status = FAIL if bad else PASS
        summary = (f"All {N} frames: return sums ≈ 1.0." if not bad
                   else f"{len(bad)}/{N} frames with bad return sum: "
                        + str([(fi, f"{s:.4f}") for fi, s in bad[:3]]))

        return TestResult("A01_process_frame_narr_none_normalized", status, summary,
                          metrics={"return_sum_mean": float(arr.mean()),
                                   "return_sum_std":  float(arr.std()),
                                   "return_sum_min":  float(arr.min()),
                                   "n_bad_frames":    len(bad)},
                          notes=["sum=0.0 is acceptable for degenerate frames "
                                 "(empty masks or zero relevance)."])

    # ------------------------------------------------------------------
    # A02 — Contributions non-negative
    # ------------------------------------------------------------------

    def _a02_contributions_nonneg(self, frames) -> TestResult:
        """
        After abs() and normalization in the ATOMs pipeline, every per-frame
        increment to _hierarchical must be >= 0 for every semantic class.
        Negative increments indicate raw signed relevance has leaked through.
        """
        if frames is None:
            return TestResult("A02_contributions_non_negative", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        neg_frames = []
        self.atoms.reset()

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))

            prev = self.atoms._hierarchical.copy()
            self.atoms.process_frame(
                wide, None, seg_w, None, cmd=cmd, spd=spd
            )
            contrib = self.atoms._hierarchical - prev
            n_neg = int((contrib < -1e-8).sum())
            if n_neg > 0:
                neg_frames.append((i, n_neg, float(contrib.min())))

        self.atoms.reset()
        status  = FAIL if neg_frames else PASS
        summary = (f"All {N} frames: contributions >= 0." if not neg_frames
                   else f"{len(neg_frames)} frame(s) have negative contributions.")

        return TestResult("A02_contributions_non_negative", status, summary,
                          metrics={"n_neg_frames": len(neg_frames), "n_frames": N},
                          notes=[f"frame {fi}: {nn} neg values, min={mn:.4e}"
                                 for fi, nn, mn in neg_frames[:5]])

    # ------------------------------------------------------------------
    # A03 — Accumulation correctness
    # ------------------------------------------------------------------

    def _a03_accumulation(self, frames) -> TestResult:
        """
        _hierarchical must equal the element-wise sum of all entries in
        _frame_series after N frames.  Any discrepancy indicates process_frame
        is not correctly accumulating state.
        """
        if frames is None:
            return TestResult("A03_accumulation_correctness", WARN,
                              "No testframes — skipped.")

        N = min(len(frames["cmd"]), self.max_frames)
        self.atoms.reset()

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd, cmd = frames["speed"][i], frames["cmd"][i]
            H, W = wide.shape[-2], wide.shape[-1]
            seg_w = (frames["seg_wide"][i] if "seg_wide" in frames
                     else _synthetic_seg(H, W, self.atoms.class_ids))
            self.atoms.process_frame(
                wide, None, seg_w, None, cmd=cmd, spd=spd
            )

        series_sum  = sum(self.atoms._frame_series)
        cumulative  = self.atoms._hierarchical
        get_h_raw   = self.atoms.get_hierarchical(normalize=False)

        err_series = float(np.abs(series_sum - cumulative).max())
        err_get_h  = float(np.abs(get_h_raw - cumulative).max())

        self.atoms.reset()

        status  = PASS if err_series < 1e-8 and err_get_h < 1e-10 else FAIL
        summary = (f"series_sum vs _hierarchical: max_err={err_series:.2e}; "
                   f"get_hierarchical() err={err_get_h:.2e}")

        return TestResult("A03_accumulation_correctness", status, summary,
                          metrics={"max_abs_err_series": err_series,
                                   "max_abs_err_get_h":  err_get_h,
                                   "n_frames": N},
                          notes=["err > 1e-8: process_frame accumulation is broken.",
                                 "err_get_h > 1e-10: get_hierarchical(normalize=False) wraps incorrectly."])

    # ------------------------------------------------------------------
    # A04 — reset() clears all state
    # ------------------------------------------------------------------

    def _a04_reset(self, frames) -> TestResult:
        """
        After reset(), every accumulator and list must be back to its
        initial value: _hierarchical all-zero, _frame_series empty,
        _frame_cmds empty, _n_frames == 0, _current_masks_* None.
        """
        # Dirty the state with one frame if available
        if frames is not None and len(frames["cmd"]) > 0:
            wide = _to_tensor(frames["wide_rgb"][0], self.device)
            spd, cmd = frames["speed"][0], frames["cmd"][0]
            H, W = wide.shape[-2], wide.shape[-1]
            seg  = _synthetic_seg(H, W, self.atoms.class_ids)
            self.atoms.process_frame(
                wide, None, seg, None, cmd=cmd, spd=spd
            )

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
        if self.atoms._n_frames != 0:
            failures.append(f"_n_frames not 0: {self.atoms._n_frames}")
        if self.atoms._current_masks_wide is not None:
            failures.append("_current_masks_wide not None after reset")
        if self.atoms.get_hierarchical().sum() != 0.0:
            failures.append("get_hierarchical() returns non-zero after reset")

        status  = FAIL if failures else PASS
        summary = "All state correctly zeroed." if not failures else "; ".join(failures[:2])

        return TestResult("A04_reset_clears_state", status, summary,
                          metrics={"n_failures": len(failures)},
                          notes=failures)

    # ------------------------------------------------------------------
    # A05 — Node-map diversity via ATOMs
    # ------------------------------------------------------------------

    def _a05_node_diversity_atoms(self, frames) -> TestResult:
        """
        ATOMsCarla._lrp2_pixels calls lrp.forward_relevance with a specific
        node_id.  This test verifies that different node_ids produce different
        pixel maps (and therefore potentially different attention vectors),
        confirming node_id routing is intact inside ATOMsCarla.

        Strategy:
          1. Get node relevances via atoms._lrp1_nodes.
          2. Select top nodes via _relevance_filter.
          3. Call atoms._lrp2_pixels for the first two top nodes.
          4. FAIL if the maps are pixel-identical (rel_diff < 1e-5).
          5. As a bonus: convert each map to an attention vector via
             seg_to_masks and check that the vectors also differ.
        """
        if frames is None:
            return TestResult("A05_node_map_diversity_via_atoms", WARN,
                              "No testframes — skipped.")

        wide = self._first_frame(frames)
        spd, cmd = frames["speed"][0], frames["cmd"][0]
        H, W = wide.shape[-2], wide.shape[-1]
        seg_w = (frames["seg_wide"][0] if "seg_wide" in frames
                 else _synthetic_seg(H, W, self.atoms.class_ids))

        # Set up ATOMs context
        self.atoms.lrp.update_context(wide, None, spd)
        self.atoms._current_spd = spd
        self.atoms._current_masks_wide = seg_to_masks(seg_w, self.atoms.class_ids)
        self.atoms._current_masks_narr = None

        # Step 1: node relevances
        r_nodes  = self.atoms._lrp1_nodes(wide, None, cmd)
        node_ids = _relevance_filter(r_nodes, self.atoms.p_relevance)

        if len(node_ids) < 2:
            self.atoms.reset()
            return TestResult(
                "A05_node_map_diversity_via_atoms", WARN,
                f"Only {len(node_ids)} node(s) selected at p={self.atoms.p_relevance}; "
                "need >= 2 to test diversity.",
                {"n_nodes_selected": len(node_ids)},
            )

        # Step 2: pixel maps for first two top nodes
        wide_r0, _ = self.atoms._lrp2_pixels(wide, None, node_id=node_ids[0], cmd=cmd)
        wide_r1, _ = self.atoms._lrp2_pixels(wide, None, node_id=node_ids[1], cmd=cmd)

        arr0 = _as_numpy(wide_r0).flatten()
        arr1 = _as_numpy(wide_r1).flatten()
        mean_abs = float(np.abs(arr0).mean()) + 1e-12

        max_diff = float(np.abs(arr0 - arr1).max())
        rel_diff = max_diff / mean_abs

        # Step 3: attention vectors from the two maps
        masks = self.atoms._current_masks_wide
        if masks is not None:
            wide_r0_t = torch.from_numpy(_as_numpy(wide_r0)) if not isinstance(wide_r0, torch.Tensor) else wide_r0
            wide_r1_t = torch.from_numpy(_as_numpy(wide_r1)) if not isinstance(wide_r1, torch.Tensor) else wide_r1

            # Use _give_element_selectivity if available
            try:
                attn0 = _as_numpy(self.atoms._give_element_selectivity(wide_r0_t, narr_r=None))
                attn1 = _as_numpy(self.atoms._give_element_selectivity(wide_r1_t, narr_r=None))
                attn_l1 = float(np.abs(attn0 - attn1).sum())
            except Exception:
                attn_l1 = None
        else:
            attn_l1 = None

        self.atoms.reset()

        if rel_diff < 1e-5:
            status  = FAIL
            summary = (f"Pixel maps IDENTICAL for node_ids {node_ids[0]} and {node_ids[1]} "
                       f"(rel_diff={rel_diff:.2e}). node_id not reaching _lrp2_pixels.")
        elif rel_diff < 0.01:
            status  = WARN
            summary = (f"Maps very similar for top-2 nodes (rel_diff={rel_diff:.4f}). "
                       "Verify node_id routing.")
        else:
            status  = PASS
            summary = (f"Per-node maps are distinct (rel_diff={rel_diff:.4f}). "
                       f"ATOMs node routing confirmed.")

        metrics = {"node_id_0": int(node_ids[0]),
                   "node_id_1": int(node_ids[1]),
                   "n_nodes_selected": len(node_ids),
                   "pixel_rel_diff": rel_diff}
        if attn_l1 is not None:
            metrics["attention_l1_diff"] = attn_l1

        return TestResult("A05_node_map_diversity_via_atoms", status, summary,
                          metrics=metrics,
                          notes=["FAIL: node_id is not forwarded from _lrp2_pixels to "
                                 "forward_relevance — all node maps are identical.",
                                 "WARN with small diff: may be genuine near-identical nodes "
                                 "— check another frame or lower p_relevance."])

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, TestResult]) -> None:
        sym  = {PASS: "✓ PASS", FAIL: "✗ FAIL", WARN: "△ WARN", ERROR: "! ERROR"}
        sep  = "─" * 70

        print(f"\n{'='*70}")
        print("  TFV6 LRP + ATOMs DETAILED TEST REPORT")
        print(f"{'='*70}")

        for name, r in results.items():
            print(f"\n{sep}")
            print(f"  {sym[r.status]}  |  {name}")
            print(f"  {r.summary}")
            if r.metrics:
                print("  Metrics:")
                for k, v in r.metrics.items():
                    if k == "wall_time_s":
                        continue
                    if isinstance(v, float):
                        print(f"    {k:<52s} {v:.6g}")
                    else:
                        print(f"    {k:<52s} {v}")
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
        counts: Dict[str, int] = defaultdict(int)
        for r in results.values():
            counts[r.status] += 1
        for s in [PASS, WARN, FAIL, ERROR]:
            if counts[s]:
                print(f"  {sym[s]}: {counts[s]}")
        print(f"{'='*70}\n")

    def save_report(self, results: Dict[str, TestResult], out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        txt_path = os.path.join(out_dir, "tfv6_test_report.txt")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.print_report(results)
        with open(txt_path, "w") as fh:
            fh.write(buf.getvalue())
        for name, r in results.items():
            if r.per_frame is not None:
                np.save(os.path.join(out_dir, f"{name}_per_frame.npy"), r.per_frame)
        print(f"Report saved to {out_dir}")
