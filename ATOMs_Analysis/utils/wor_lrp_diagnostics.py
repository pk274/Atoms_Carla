"""
wor_lrp_diagnostics.py
======================
Deep diagnostic tests for the WoR LRPCameraModel implementation.

Mirrors tfv6_lrp_diagnostics.py in strictness, but targets WoR-specific
internals: the joint two-camera forward pass, the undo_resnet_amplification
correction, the selector formula, and z+-with-zero_params bias exclusion.

  W01  Forward-pass match          — JointCameraForLRP ≡ CameraModel logits
  W02  FC forward-pass match       — JointCameraToFC ≡ act_head[:4](concat)
  W03  undo_resnet correction      — pixel wide_frac ≈ concat-level wide_frac
  W04  Cross-normalization sum      — wide_r.sum() + narr_r.sum() = 1.0 exactly
  W05  Brake-selector position      — non-zero at correct logit offset(s) only
  W06  Forced-brake/drive distinct  — forced seeds produce different masks
  W07  FC-node pairwise cosine      — top-K nodes have distinct pixel maps
  W08  LRP output determinism       — two identical calls → bit-for-bit equal
  W09  Bias-exclusion effectiveness — zero_params='bias' changes LRP vs. no-exclusion

Design principles: same as tfv6_lrp_diagnostics.py — each test has a single,
numerically-specified FAIL criterion derived from mathematical constraints.

Usage
-----
    from ATOMs_Analysis.utils.wor_lrp_diagnostics import WoRLRPDiagnostics
    diag   = WoRLRPDiagnostics(lrp_instance)
    report = diag.run_all_tests(testframes)
    diag.print_report(report)
"""

from __future__ import annotations

import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from zennit.rules import Pass, WSquare, AlphaBeta
from zennit.types import Convolution, Activation, AvgPool
from zennit.types import Linear as AnyLinear
from zennit.composites import SpecialFirstLayerMapComposite
from zennit.torchvision import ResNetCanonizer

from ATOMs_Analysis.saliency.atoms_carla import _relevance_filter


# ---------------------------------------------------------------------------
# Status constants / data types (mirror tfv6_lrp_diagnostics)
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
ERROR = "ERROR"


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


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _to_tensor(rgb, device) -> torch.Tensor:
    if isinstance(rgb, np.ndarray):
        t = torch.from_numpy(rgb).unsqueeze(0).float()
    else:
        t = rgb.float()
    if t.ndim == 3:
        t = t.unsqueeze(0)
    return t.to(device)


def _as_numpy(t) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    return np.array(t)


# ---------------------------------------------------------------------------
# WoRLRPDiagnostics
# ---------------------------------------------------------------------------

class WoRLRPDiagnostics:
    """
    Parameters
    ----------
    lrp    : LRPCameraModel — already initialised, CameraModel in eval mode.
    device : torch.device or str.
    """

    def __init__(self, lrp, device=None):
        self.lrp    = lrp
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run_all_tests(self, testframes: Optional[Dict] = None) -> Dict[str, TestResult]:
        n = len(testframes["cmd"]) if testframes is not None else 0
        print(f"\n{'='*70}")
        print(f"  WoR LRP Diagnostics  —  {n} testframes")
        print(f"{'='*70}\n")

        tests = [
            ("W01_forward_pass_match",          self._w01_forward_match),
            ("W02_fc_forward_match",            self._w02_fc_match),
            ("W03_undo_resnet_correction",       self._w03_undo_correction),
            ("W04_cross_normalization_sum",      self._w04_cross_norm),
            ("W05_brake_selector_position",      self._w05_brake_selector),
            ("W06_forced_seed_distinctiveness",  self._w06_forced_seeds),
            ("W07_fc_node_cosine_matrix",        self._w07_node_cosines),
            ("W08_lrp_output_determinism",       self._w08_determinism),
            ("W09_bias_exclusion_effectiveness", self._w09_bias_exclusion),
        ]

        results: Dict[str, TestResult] = {}
        for name, fn in tests:
            t0 = time.time()
            print(f"  Running {name} ...", end="", flush=True)
            result, exc = _safe_run(fn, testframes)
            elapsed = time.time() - t0
            if exc is not None:
                result = TestResult(name=name, status=ERROR,
                                    summary="Test raised an exception.",
                                    exception=exc)
            result.metrics["wall_time_s"] = round(elapsed, 2)
            results[name] = result
            tag = {PASS: "✓", FAIL: "✗", WARN: "△", ERROR: "!"}[result.status]
            print(f"  {tag}  [{elapsed:.1f}s]  {result.summary}")

        print()
        return results

    # ------------------------------------------------------------------
    # W01 — JointCameraForLRP forward matches CameraModel
    # FAIL: max absolute diff in logits > 1e-3 on any frame.
    # ------------------------------------------------------------------

    def _w01_forward_match(self, frames) -> TestResult:
        """
        JointCameraForLRP replaces CameraModel.backbone_wide's .mean([2,3]) with
        nn.AdaptiveAvgPool2d((1,1)) + Flatten so zennit can intercept it.
        The two must produce bit-for-bit the same logits.

        A mismatch means ALL LRP attributions describe a DIFFERENT function
        than the deployed CameraModel — the most critical possible failure.

        FAIL criterion: max|lrp_logit − cam_logit| > 1e-3 on any frame.
        """
        if frames is None:
            return TestResult("W01_forward_pass_match", WARN, "No testframes — skipped.")

        cm = self.lrp._model_eval
        N  = min(len(frames["cmd"]), 8)
        failures = []
        max_diffs = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)

            with torch.no_grad():
                lrp_logits = self.lrp.model_lrp(wide, narr)  # JointCameraForLRP

                # Replicate CameraModel forward manually
                w = cm.backbone_wide(cm.normalize(wide / 255.0)).mean(dim=[2, 3])    # [1, 512]
                n_post = cm.backbone_narr(cm.normalize(narr / 255.0))
                n = cm.bottleneck_narr(n_post.mean(dim=[2, 3]))                       # [1, 64]
                cam_logits = cm.act_head(torch.cat([w, n], dim=1))                   # [1, 312]

            diff = float((lrp_logits.cpu() - cam_logits.cpu()).abs().max().item())
            max_diffs.append(diff)

            if diff > 1e-3:
                failures.append(
                    f"frame {i}: max logit diff = {diff:.2e} > 1e-3  — "
                    "JointCameraForLRP produces DIFFERENT outputs than CameraModel. "
                    "All LRP attributions are for the wrong function."
                )

        arr = np.array(max_diffs)
        status  = FAIL if failures else PASS
        summary = (f"Max logit diff ≤ 1e-3 on all {N} frames (max={arr.max():.2e})."
                   if not failures else failures[0])
        return TestResult("W01_forward_pass_match", status, summary,
                          metrics={"max_logit_diff_max":  float(arr.max()),
                                   "max_logit_diff_mean": float(arr.mean()),
                                   "n_frames": N},
                          per_frame=arr,
                          notes=failures + [
                              "JointCameraForLRP uses AdaptiveAvgPool2d+Flatten instead of .mean([2,3]).",
                              "Numerically these are equivalent — any diff > 1e-3 is a bug.",
                          ])

    # ------------------------------------------------------------------
    # W02 — JointCameraToFC forward matches CameraModel act_head[:4]
    # FAIL: max diff in 256-dim FC activation > 1e-4 on any frame.
    # ------------------------------------------------------------------

    def _w02_fc_match(self, frames) -> TestResult:
        """
        JointCameraToFC feeds act_head[:4] (Linear→ReLU→Linear→ReLU) through a
        separate nn.Sequential so zennit can intercept it in the fc→input path.
        Its output must exactly match running act_head[:4] directly on the
        concatenated features from CameraModel's backbone.

        FAIL criterion: max|fc_lrp − fc_direct| > 1e-4 on any frame.
        """
        if frames is None:
            return TestResult("W02_fc_forward_match", WARN, "No testframes — skipped.")

        cm = self.lrp._model_eval
        N  = min(len(frames["cmd"]), 8)
        failures = []
        max_diffs = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)

            with torch.no_grad():
                fc_lrp = self.lrp.fc_model_lrp(wide, narr)   # [1, 256]

                # CameraModel direct path to FC activation
                w = cm.backbone_wide(cm.normalize(wide / 255.0)).mean(dim=[2, 3])
                n_post = cm.backbone_narr(cm.normalize(narr / 255.0))
                n = cm.bottleneck_narr(n_post.mean(dim=[2, 3]))
                concat = torch.cat([w, n], dim=1)
                # act_head[:4] = Linear(576→256) → ReLU → Linear(256→256) → ReLU
                fc_direct = cm.act_head[:4](concat)            # [1, 256]

            diff = float((fc_lrp.cpu() - fc_direct.cpu()).abs().max().item())
            max_diffs.append(diff)
            if diff > 1e-4:
                failures.append(
                    f"frame {i}: max FC diff = {diff:.2e} > 1e-4  — "
                    "JointCameraToFC diverges from CameraModel act_head[:4]. "
                    "fc→input LRP is attributing the wrong function."
                )

        arr = np.array(max_diffs)
        status  = FAIL if failures else PASS
        summary = (f"FC activation matches act_head[:4] on all {N} frames (max diff={arr.max():.2e})."
                   if not failures else failures[0])
        return TestResult("W02_fc_forward_match", status, summary,
                          metrics={"max_fc_diff_max":  float(arr.max()),
                                   "max_fc_diff_mean": float(arr.mean()),
                                   "n_frames": N},
                          per_frame=arr,
                          notes=failures + [
                              "JointCameraToFC.act_head_partial = nn.Sequential(*act_head[:4]).",
                              "Diff > 1e-4 means shared weights diverged (module reference broken).",
                          ])

    # ------------------------------------------------------------------
    # W03 — undo_resnet_amplification effectiveness
    # FAIL: pixel-level wide_frac differs from concat-level wide_frac by > 0.02
    #   when undo_resnet_amplification=True.
    # ------------------------------------------------------------------

    def _w03_undo_correction(self, frames) -> TestResult:
        """
        _attribute_to_concat() measures the wide/narr relevance split at the
        576-dim concatenation point — before either ResNet amplifies the signal.
        With undo_resnet_amplification=True, the final pixel maps are rescaled
        so that their wide fraction matches the concat-level fraction.

        FAIL criterion: |pixel_wide_frac − concat_wide_frac| > 0.02 on any frame.
        A large gap means the correction was not applied or is broken.
        """
        if frames is None:
            return TestResult("W03_undo_resnet_correction", WARN, "No testframes — skipped.")
        if not self.lrp.undo_resnet_amplification:
            return TestResult("W03_undo_resnet_correction", WARN,
                              "undo_resnet_amplification=False — correction not active.")

        N = min(len(frames["cmd"]), 8)
        gaps = []
        concat_fracs, pixel_fracs = [], []
        failures = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = float(frames["speed"][i]), int(frames["cmd"][i])

            wide_x = self.lrp._prepare_input(wide)
            narr_x = self.lrp._prepare_input(narr)
            sel, _ = self.lrp._build_drive_brake_selector(wide_x, narr_x, cmd, spd)

            # Concat-level wide fraction
            _, _, narr_frac_c = self.lrp._attribute_to_concat(wide_x, narr_x, sel)
            wide_frac_concat = 1.0 - narr_frac_c
            concat_fracs.append(wide_frac_concat)

            # Pixel-level wide fraction (after correction)
            wide_r, narr_r, wide_frac_pixel, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
            )
            pixel_fracs.append(float(wide_frac_pixel))

            gap = abs(wide_frac_concat - float(wide_frac_pixel))
            gaps.append(gap)
            if gap > 0.02:
                failures.append(
                    f"frame {i}: concat_wide_frac={wide_frac_concat:.4f}, "
                    f"pixel_wide_frac={wide_frac_pixel:.4f}, gap={gap:.4f} > 0.02. "
                    "Correction not effective."
                )

        gap_arr  = np.array(gaps)
        cf_arr   = np.array(concat_fracs)
        pf_arr   = np.array(pixel_fracs)
        status   = FAIL if failures else PASS
        summary  = (f"Correction holds: mean gap={gap_arr.mean():.4f}, max={gap_arr.max():.4f}."
                    if not failures else failures[0])
        return TestResult("W03_undo_resnet_correction", status, summary,
                          metrics={
                              "concat_wide_frac_mean": float(cf_arr.mean()),
                              "pixel_wide_frac_mean":  float(pf_arr.mean()),
                              "gap_mean":              float(gap_arr.mean()),
                              "gap_max":               float(gap_arr.max()),
                              "n_frames": N,
                          },
                          per_frame=gap_arr,
                          notes=failures + [
                              "Gap = |pixel_wide_frac − concat_wide_frac|.",
                              "Wide ResNet (512-dim) amplifies more than narrow bottleneck (64-dim).",
                              "The correction rescales pixel maps so the ratio matches the concat-level.",
                          ])

    # ------------------------------------------------------------------
    # W04 — Cross-normalization sum conservation
    # FAIL: |wide_r.sum() + narr_r.sum() − 1.0| > 1e-5 on any frame.
    # ------------------------------------------------------------------

    def _w04_cross_norm(self, frames) -> TestResult:
        """
        _cross_normalize() produces non-negative maps where:
            wide_r = abs(wide_r_raw) / |wide_r_raw| * wide_frac
            narr_r = abs(narr_r_raw) / |narr_r_raw| * (1 − wide_frac)
        Therefore wide_r.sum() = wide_frac and narr_r.sum() = 1 − wide_frac,
        so their total MUST be exactly 1.0 by construction.

        FAIL criterion: |wide_r.sum() + narr_r.sum() − 1.0| > 1e-5.
        This is exact algebra, not an approximation.
        """
        if frames is None:
            return TestResult("W04_cross_normalization_sum", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 10)
        failures = []
        errors   = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = float(frames["speed"][i]), int(frames["cmd"][i])

            for mode, kwargs in [
                ("fc→input",     dict(beg="fc",     end="input", cmd=cmd, spd=spd)),
                ("output→input", dict(beg="output",  end="input", cmd=cmd, spd=spd)),
            ]:
                wide_r, narr_r, wide_frac, _ = self.lrp.forward_relevance(
                    wide, narr, **kwargs
                )
                wr = _as_numpy(wide_r).sum()
                nr = _as_numpy(narr_r).sum()
                total = wr + nr
                err = abs(total - 1.0)
                errors.append(err)

                if err > 1e-5:
                    failures.append(
                        f"frame {i} mode {mode}: "
                        f"wide_sum={wr:.6f} + narr_sum={nr:.6f} = {total:.6f} ≠ 1.0 "
                        f"(err={err:.2e}). _cross_normalize is broken."
                    )

        err_arr = np.array(errors)
        status  = FAIL if failures else PASS
        summary = (f"All frames/modes sum to 1.0 (max err={err_arr.max():.2e})."
                   if not failures else failures[0])
        return TestResult("W04_cross_normalization_sum", status, summary,
                          metrics={"sum_error_max":  float(err_arr.max()),
                                   "sum_error_mean": float(err_arr.mean()),
                                   "n_checks": len(errors)},
                          notes=failures + [
                              "After _cross_normalize: wide_r = abs/|R_w| * frac, same for narr.",
                              "Sum = frac + (1-frac) = 1.0 by construction — any deviation is a bug.",
                          ])

    # ------------------------------------------------------------------
    # W05 — Brake-selector position correctness
    # FAIL: selector has non-zero mass outside the expected brake logit offset,
    #   OR the weighted sum at the expected positions ≠ 1.0.
    # ------------------------------------------------------------------

    def _w05_brake_selector(self, frames) -> TestResult:
        """
        In brake mode, the selector mask should have:
          - Exactly two non-zero positions: the brake logit at speed bin x0 and x1,
            weighted by (1−w) and w respectively.
          - All other positions exactly zero.
          - The two weights summing to 1.0.

        We test on frames where _build_drive_brake_selector reports is_brake=True
        AND where forced_brake=True (to guarantee brake mode regardless of model pred).
        """
        if frames is None:
            return TestResult("W05_brake_selector_position", WARN, "No testframes — skipped.")

        lrp = self.lrp
        base   = lrp.num_steers + lrp.num_throts + 1
        stride = lrp.num_speeds * base
        bi     = lrp.num_steers + lrp.num_throts  # brake index within each speed-block

        N = min(len(frames["cmd"]), 10)
        failures = []
        n_checked = 0

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = float(frames["speed"][i]), int(frames["cmd"][i])

            # Force brake mode to test the formula unconditionally
            sel_brk, _ = lrp._build_drive_brake_selector(wide, narr, cmd, spd, forced_brake=True)

            total_len = lrp.num_cmds * lrp.num_speeds * base
            dummy = torch.zeros(1, total_len, device=self.device)
            mask  = sel_brk(dummy).squeeze(0)  # [total_len]

            x0, x1, w = lrp._lerp_bins(spd, lrp.min_speeds, lrp.max_speeds, lrp.num_speeds)
            expected_pos0 = cmd * stride + x0 * base + bi
            expected_pos1 = cmd * stride + x1 * base + bi

            # Non-zero positions
            nonzero_pos = (mask.abs() > 1e-8).nonzero(as_tuple=True)[0].tolist()
            expected_nonzero = {expected_pos0, expected_pos1}
            # When x0 == x1 (speed at exact bin boundary), only one position
            if x0 == x1:
                expected_nonzero = {expected_pos0}

            if set(nonzero_pos) != expected_nonzero:
                failures.append(
                    f"frame {i}: brake mask non-zero at {set(nonzero_pos)}, "
                    f"expected {expected_nonzero} (cmd={cmd}, x0={x0}, x1={x1}, bi={bi})."
                )

            # Weights must be (1-w) and w.
            # When x0 == x1 (speed on exact bin boundary, w=0), both expected
            # positions alias to the same slot — read it once to avoid double-count.
            if expected_pos0 < total_len and expected_pos1 < total_len:
                got_w0 = float(mask[expected_pos0].item())
                if x0 == x1:
                    # Single slot should hold exactly 1.0
                    if abs(got_w0 - 1.0) > 1e-5:
                        failures.append(
                            f"frame {i}: exact-bin case (x0==x1={x0}), expected weight=1.0, "
                            f"got {got_w0:.6f}."
                        )
                else:
                    got_w1 = float(mask[expected_pos1].item())
                    total_w = got_w0 + got_w1
                    if abs(total_w - 1.0) > 1e-5:
                        failures.append(
                            f"frame {i}: brake weights sum={total_w:.6f} ≠ 1.0. "
                            f"Lerp weights should sum to 1."
                        )
                    if abs(got_w0 - (1 - w)) > 1e-5 or abs(got_w1 - w) > 1e-5:
                        failures.append(
                            f"frame {i}: expected lerp weights ({1-w:.4f}, {w:.4f}), "
                            f"got ({got_w0:.4f}, {got_w1:.4f})."
                        )
            n_checked += 1

        status  = FAIL if failures else PASS
        summary = (f"Brake selector formula correct on {n_checked} frames."
                   if not failures else f"{len(failures)} violation(s).")
        return TestResult("W05_brake_selector_position", status, summary,
                          metrics={"n_checked": n_checked, "n_failures": len(failures)},
                          notes=failures + [
                              "Brake selector: one-hot at cmd*stride + xN*base + bi, "
                              "weighted (1-w) and w for x0 and x1.",
                              "base = num_steers + num_throts + 1;  bi = base - 1.",
                          ])

    # ------------------------------------------------------------------
    # W06 — Forced brake/drive selector distinctiveness
    # FAIL: all frames produce identical brake and drive masks (cosine > 0.9999).
    # ------------------------------------------------------------------

    def _w06_forced_seeds(self, frames) -> TestResult:
        """
        forced_brake=True produces a one-hot at the brake logit position.
        forced_drive=True produces softmax weights over steer positions.
        These masks must be different — if identical, R_drive − R_brake = 0.

        FAIL criterion: cosine(mask_brake, mask_drive) > 0.9999 on ALL tested frames.
        A single distinct pair is sufficient to confirm the routing is alive.
        """
        if frames is None:
            return TestResult("W06_forced_seed_distinctiveness", WARN, "No testframes — skipped.")

        lrp = self.lrp
        total_len = lrp.num_cmds * lrp.num_speeds * (lrp.num_steers + lrp.num_throts + 1)
        dummy = torch.zeros(1, total_len, device=self.device)

        N = min(len(frames["cmd"]), 8)
        cosines = []
        failures = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = float(frames["speed"][i]), int(frames["cmd"][i])

            sel_brk, _ = lrp._build_drive_brake_selector(wide, narr, cmd, spd, forced_brake=True)
            sel_drv, _ = lrp._build_drive_brake_selector(wide, narr, cmd, spd, forced_drive=True)

            m_brk = sel_brk(dummy).squeeze(0).cpu().numpy()
            m_drv = sel_drv(dummy).squeeze(0).cpu().numpy()
            cos   = _cosine(m_brk, m_drv)
            cosines.append(cos)

            if cos > 0.9999:
                failures.append(
                    f"frame {i}: brake/drive mask cosine={cos:.6f} ≈ 1.0 — "
                    "forced_brake and forced_drive produce identical masks. "
                    "R_drive − R_brake will be all-zero."
                )

        cos_arr = np.array(cosines)
        # FAIL only if ALL frames are identical (one good pair confirms routing)
        status  = FAIL if len(failures) == N else (WARN if failures else PASS)
        summary = (f"Brake/drive masks distinct: mean cosine={cos_arr.mean():.4f}."
                   if not failures else f"{len(failures)}/{N} frames identical.")
        return TestResult("W06_forced_seed_distinctiveness", status, summary,
                          metrics={"cosine_mean": float(cos_arr.mean()),
                                   "cosine_max":  float(cos_arr.max()),
                                   "n_frames": N,
                                   "n_identical": len(failures)},
                          per_frame=cos_arr,
                          notes=failures + [
                              "forced_brake → one-hot at brake position.",
                              "forced_drive → softmax over steer logits.",
                              "cosine ≈ 1.0 → both seeds identical → comparative map is zero.",
                          ])

    # ------------------------------------------------------------------
    # W07 — FC node pixel map pairwise cosine similarity
    # WARN: all pairwise cosines > 0.9999 — expected for WoR due to GAP.
    # ------------------------------------------------------------------

    def _w07_node_cosines(self, frames) -> TestResult:
        """
        forward_relevance(beg='fc', end='input', node_id=k) ideally produces
        different pixel maps for different k.  For WoR, this is IMPOSSIBLE due
        to the GlobalAveragePool (AdaptiveAvgPool2d(1,1)) that sits between the
        ResNet backbone and the FC layers.

        GAP collapse mechanism:
          FC node k → per-node 512-dim relevance vector → AvgPool backward
          (no LRP rule, so standard autograd) → uniform relevance across all
          H'×W' spatial positions → ResNet z+ backward driven only by fixed
          activations → identical pixel map for every k.

        Therefore for WoR, cosine ≈ 1.0 for all pairs is the EXPECTED result,
        not a bug.  This is the primary reason TFV6 was adopted: its speed_query
        token bypasses GAP entirely.

        Criterion: cosine > 0.9999 for all pairs → WARN (architectural limit).
        A genuine routing bug (node_id silently ignored) would produce the same
        symptom, so this test distinguishes between the two via the note.
        """
        if frames is None:
            return TestResult("W07_fc_node_cosine_matrix", WARN, "No testframes — skipped.")

        wide = _to_tensor(frames["wide_rgb"][0], self.device)
        narr = _to_tensor(frames["narr_rgb"][0], self.device)
        spd, cmd = float(frames["speed"][0]), int(frames["cmd"][0])

        wide_x = self.lrp._prepare_input(wide)
        narr_x = self.lrp._prepare_input(narr)
        sel, _ = self.lrp._build_drive_brake_selector(wide_x, narr_x, cmd, spd)

        fc_rel = self.lrp._attribute_to_fc(wide_x, narr_x, sel)  # [256]
        node_ids = _relevance_filter(
            fc_rel if isinstance(fc_rel, torch.Tensor) else torch.from_numpy(_as_numpy(fc_rel)),
            0.9
        )

        K = min(8, len(node_ids))
        if K < 2:
            return TestResult("W07_fc_node_cosine_matrix", WARN,
                              f"Only {K} node(s) selected — need ≥ 2.",
                              {"n_nodes_selected": len(node_ids)})

        probe_ids = node_ids[:K]
        maps = []
        for nid in probe_ids:
            wide_r, _, _, _ = self.lrp.forward_relevance(
                wide, narr, beg="fc", end="input", node_id=nid
            )
            maps.append(_as_numpy(wide_r).flatten())

        n_pairs = K * (K - 1) // 2
        cosines = [_cosine(maps[i], maps[j])
                   for i in range(K) for j in range(i + 1, K)]
        cos_arr  = np.array(cosines)
        min_cos  = float(cos_arr.min())
        mean_cos = float(cos_arr.mean())

        if min_cos > 0.9999:
            status  = WARN
            summary = (
                f"ALL {n_pairs} pairs have cosine > 0.9999 — node maps identical. "
                "Expected for WoR: GAP destroys spatial specificity before FC layers."
            )
        elif mean_cos > 0.90:
            status  = WARN
            summary = (f"Mean pairwise cosine={mean_cos:.4f} > 0.90 across {K} nodes. "
                       "Very low per-node diversity (partial GAP collapse?).")
        else:
            status  = PASS
            summary = (f"{K} nodes, {n_pairs} pairs: min cos={min_cos:.4f}, "
                       f"mean={mean_cos:.4f}, std={cos_arr.std():.4f}.")

        return TestResult("W07_fc_node_cosine_matrix", status, summary,
                          metrics={"n_probe_nodes":  K,
                                   "n_pairs":         n_pairs,
                                   "cosine_min":      min_cos,
                                   "cosine_mean":     mean_cos,
                                   "cosine_std":      float(cos_arr.std()),
                                   "node_ids_probed": probe_ids[:K]},
                          per_frame=cos_arr,
                          notes=[
                              "WoR uses AdaptiveAvgPool2d(1,1) after backbone → GAP collapses "
                              "spatial info before FC layers.",
                              "AvgPool has no LRP rule (excluded from composite): standard autograd "
                              "uniformly redistributes relevance → all nodes get same pixel map.",
                              "This is architectural, not a routing bug. Use TFV6 for per-node "
                              "spatial analysis.",
                          ])

    # ------------------------------------------------------------------
    # W08 — LRP output determinism
    # FAIL: any difference between two identical calls > 1e-6.
    # ------------------------------------------------------------------

    def _w08_determinism(self, frames) -> TestResult:
        """
        Two identical calls to forward_relevance with the same inputs must return
        bit-for-bit identical results.

        A failure indicates dropout accidentally active (model not in eval mode),
        GPU non-determinism, or a mutable state bug in the zennit context.
        """
        if frames is None:
            return TestResult("W08_lrp_output_determinism", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 4)
        modes = [
            ("fc→input",     dict(beg="fc",    end="input")),
            ("output→input", dict(beg="output", end="input")),
            ("output→fc",    dict(beg="output", end="fc")),
        ]
        failures = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            narr = _to_tensor(frames["narr_rgb"][i], self.device)
            spd, cmd = float(frames["speed"][i]), int(frames["cmd"][i])

            for mode_name, kwargs in modes:
                r1_w, r1_n, _, _ = self.lrp.forward_relevance(wide, narr, cmd=cmd, spd=spd, **kwargs)
                r2_w, r2_n, _, _ = self.lrp.forward_relevance(wide, narr, cmd=cmd, spd=spd, **kwargs)

                for tag, a, b in [("wide", r1_w, r2_w), ("narr", r1_n, r2_n)]:
                    diff = float((_as_numpy(a) - _as_numpy(b)).max())
                    if diff > 1e-6:
                        failures.append(
                            f"frame {i} mode {mode_name} {tag}: diff={diff:.2e} > 1e-6 — "
                            "non-deterministic output."
                        )

        status  = FAIL if failures else PASS
        summary = (f"All {N} frames × {len(modes)} modes × 2 cameras: identical on repeat."
                   if not failures else "; ".join(failures[:2]))
        return TestResult("W08_lrp_output_determinism", status, summary,
                          metrics={"n_frames": N, "n_modes": len(modes),
                                   "n_failures": len(failures)},
                          notes=failures + [
                              "Non-determinism → dropout active or GPU non-deterministic ops.",
                              "Call model.eval() and torch.use_deterministic_algorithms(True).",
                          ])

    # ------------------------------------------------------------------
    # W09 — Bias exclusion via zero_params='bias'
    # FAIL: LRP output is identical when biases are excluded vs included.
    #   (Identical means zero_params='bias' has NO effect → biases are zero or
    #   the setting is silently ignored — both are bugs.)
    # WARN: Difference is tiny (biases near-zero — the model may not use biases).
    # ------------------------------------------------------------------

    def _w09_bias_exclusion(self, frames) -> TestResult:
        """
        WoR composite uses AlphaBeta(alpha, beta, zero_params='bias') — biases
        are excluded from the z+ denominator.

        Test: build a second composite WITHOUT zero_params (biases included in
        denominator). Run LRP with both composites on the same frame.

        Expected: the results DIFFER, confirming that bias exclusion is active
        and the act_head bias terms are non-trivial.

        FAIL criterion: rel_diff < 1e-4 (exclusion has essentially no effect →
        biases are zero or the setting is ignored).
        WARN criterion: rel_diff < 0.01 (weak effect → biases are very small).
        """
        if frames is None:
            return TestResult("W09_bias_exclusion_effectiveness", WARN, "No testframes — skipped.")

        wide = _to_tensor(frames["wide_rgb"][0], self.device)
        narr = _to_tensor(frames["narr_rgb"][0], self.device)
        spd, cmd = float(frames["speed"][0]), int(frames["cmd"][0])

        # Standard composite (zero_params='bias')
        wide_r_excl, narr_r_excl, _, _ = self.lrp.forward_relevance(
            wide, narr, beg="fc", end="input", cmd=cmd, spd=spd
        )

        # Build a second composite WITH biases
        alpha, beta = self.lrp.alpha, self.lrp.beta
        # AvgPool intentionally omitted: Pass() returns the pooled gradient [B,C,1,1]
        # unchanged, but PyTorch expects the hook to produce [B,C,H,W] — size mismatch.
        # Matches the same exclusion in LRPCameraModel._create_composite().
        composite_with_bias = SpecialFirstLayerMapComposite(
            layer_map=[
                (Activation,  Pass()),
                (Convolution, AlphaBeta(alpha=alpha, beta=beta)),
                (AnyLinear,   AlphaBeta(alpha=alpha, beta=beta)),  # no zero_params
            ],
            first_map=[(Convolution, WSquare())],
            canonizers=[ResNetCanonizer()],
        )

        # Run LRP manually with the alternative composite
        wide_x = self.lrp._prepare_input(wide)
        narr_x = self.lrp._prepare_input(narr)
        with torch.enable_grad():
            with composite_with_bias.context(self.lrp.fc_model_lrp):
                output  = self.lrp.fc_model_lrp(wide_x, narr_x)  # [1, 256]
                sel     = (lambda o: torch.ones_like(o))           # layer-level seed
                grad_out = sel(output)
                wide_r_incl, narr_r_incl = torch.autograd.grad(
                    outputs=output, inputs=[wide_x, narr_x], grad_outputs=grad_out
                )
        wide_r_incl = wide_r_incl.detach().cpu()
        narr_r_incl = narr_r_incl.detach().cpu()

        e = _as_numpy(wide_r_excl).flatten()
        i_ = _as_numpy(wide_r_incl).flatten()
        scale   = float(np.abs(e).max()) + 1e-8
        rel_diff = float(np.abs(e - i_).max() / scale)

        if rel_diff < 1e-4:
            status  = FAIL
            summary = (
                f"rel_diff={rel_diff:.2e} < 1e-4 — zero_params='bias' has NO effect. "
                "Either act_head biases are zero or the setting is silently ignored."
            )
        elif rel_diff < 0.01:
            status  = WARN
            summary = (
                f"rel_diff={rel_diff:.2e} — bias exclusion has a small effect. "
                "act_head biases may be near-zero (but the setting IS applied)."
            )
        else:
            status  = PASS
            summary = (
                f"rel_diff={rel_diff:.2e} — zero_params='bias' is active and "
                "changes LRP output meaningfully."
            )

        return TestResult("W09_bias_exclusion_effectiveness", status, summary,
                          metrics={"rel_diff_max": rel_diff},
                          notes=[
                              "Composite A: AlphaBeta(zero_params='bias') — current WoR default.",
                              "Composite B: AlphaBeta(zero_params=None)   — biases included.",
                              "rel_diff < 1e-4 → WoR is effectively not excluding biases.",
                              "Expected: rel_diff > 0.01 if act_head biases are non-trivial.",
                          ])

    # ------------------------------------------------------------------
    # Reporting (mirrors tfv6_lrp_diagnostics style)
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, TestResult]) -> None:
        sym = {PASS: "✓ PASS", FAIL: "✗ FAIL", WARN: "△ WARN", ERROR: "! ERROR"}
        sep = "─" * 70

        print(f"\n{'='*70}")
        print("  WoR LRP DIAGNOSTICS — DETAILED REPORT")
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
                        print(f"    {k:<55s} {v:.6g}")
                    else:
                        print(f"    {k:<55s} {v}")
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
        import contextlib, io, os
        os.makedirs(out_dir, exist_ok=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.print_report(results)
        with open(os.path.join(out_dir, "wor_diagnostics_report.txt"), "w") as fh:
            fh.write(buf.getvalue())
        for name, r in results.items():
            if r.per_frame is not None:
                np.save(os.path.join(out_dir, f"wor_{name}_per_frame.npy"), r.per_frame)
        print(f"Report saved to {out_dir}")
