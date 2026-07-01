"""
tfv6_lrp_diagnostics.py
=======================
Deep diagnostic tests for the TFV6 AttnLRP implementation.

Unlike tfv6_test_suite.py (which tests outputs for sanity), this module
tests the INTERNAL mathematical properties of LRP at each stage:

  D01  LRPSoftmax backward formula   — unit test, no model needed
  D02  LRPMatMul backward formula    — unit test, no model needed
  D03  _make_speed_seed correctness  — unit test, no model needed
  D04  LiDAR gradient isolation      — no grad path through lidar
  D05  LRP1 conservation             — z+ through target_speed_decoder
  D06  Backbone amplification budget — pixel_sum vs node_sum (signed AND abs-value ratio)
  D07  Two-step decomposition        — output→input == separate LRP1+LRP2
  D08  Forced-seed node_rel distinct — forced_brake vs forced_drive are different
  D09  is_brake independence         — is_brake = model argmax, not forced flag
  D10  Per-node pixel-map cosine     — pairwise cosine sim matrix, up to 4 frames (own node selection each)
  D11  Bias fraction in decoder      — how much z+ denominator comes from bias
  D12  LRP output determinism        — same frame → identical maps on repeat call
  D13  LRPResidualAdd formula        — Ratio-Based Relevance Splitting (Otsuki et al. 2024)

Design principles
-----------------
* Every test has a single, clearly stated FAIL criterion with a numerical threshold.
* PASS thresholds are derived from the mathematical constraints of the LRP rules,
  not from empirical "looks reasonable" observation.
* Unit tests (D01–D03) are self-contained and run without testframes or a model.
* Integration tests call internal methods (_attribute_to_fc, _attribute_backbone,
  _attribute_true_output_to_input, _make_speed_seed) via the lrp instance.

Usage
-----
    from ATOMs_Analysis.utils.tfv6_lrp_diagnostics import TFV6LRPDiagnostics
    diag   = TFV6LRPDiagnostics(lrp_instance)
    report = diag.run_all_tests(testframes)
    diag.print_report(report)

Or standalone, against the real checkpoint + real frames (see the
`if __name__ == "__main__":` block at the end of this file):
    python -m ATOMs_Analysis.utils.tfv6_lrp_diagnostics
    python ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py --n-runs 3 --n-frames 12
"""

from __future__ import annotations

import sys
import time
import traceback
import types
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `python ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py` to work directly:
# running a script (as opposed to `python -m`) only puts the script's own
# directory on sys.path, not the project root, so the absolute
# `ATOMs_Analysis.*` imports below would otherwise fail with
# "ModuleNotFoundError: No module named 'ATOMs_Analysis'". No-op if the
# project root is already importable (e.g. when imported normally from
# another script run from the project root).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ATOMs_Analysis.saliency.lrp_transfuser import (
    LRPSoftmax,
    LRPMatMul,
    LRPResidualAdd,
    LRPTFv6Model,
)
from ATOMs_Analysis.saliency.atoms_carla import _relevance_filter


# ---------------------------------------------------------------------------
# Status constants
# ---------------------------------------------------------------------------

PASS  = "PASS"
FAIL  = "FAIL"
WARN  = "WARN"
ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Data types
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
# Helpers
# ---------------------------------------------------------------------------

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
# TFV6LRPDiagnostics
# ---------------------------------------------------------------------------

class TFV6LRPDiagnostics:
    """
    Parameters
    ----------
    lrp    : LRPTFv6Model — already initialised, backbone in eval mode.
    device : torch.device or str — where tensors live.
    """

    def __init__(self, lrp: LRPTFv6Model, device=None):
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
        print(f"  TFV6 LRP Diagnostics  —  {n} testframes + 3 unit tests")
        print(f"{'='*70}\n")

        tests = [
            ("D01_lrpsoftmax_formula",         self._d01_softmax_formula),
            ("D02_lrpmatmul_formula",           self._d02_matmul_formula),
            ("D03_seed_generation_correctness", self._d03_seed_correctness),
            ("D04_lidar_gradient_isolation",    self._d04_lidar_isolation),
            ("D05_lrp1_conservation",           self._d05_lrp1_conservation),
            ("D06_backbone_amplification",      self._d06_amplification),
            ("D07_two_step_consistency",        self._d07_two_step),
            ("D08_forced_seed_distinctiveness", self._d08_forced_seeds),
            ("D09_is_brake_independence",       self._d09_is_brake),
            ("D10_per_node_cosine_matrix",      self._d10_node_cosines),
            ("D11_decoder_bias_fraction",       self._d11_bias_fraction),
            ("D12_lrp_output_determinism",      self._d12_determinism),
            ("D13_residual_add_formula",        self._d13_residual_add_formula),
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
    # D01 — LRPSoftmax backward formula
    # FAIL criterion: max absolute error > 1e-5 vs the reference formula.
    # ------------------------------------------------------------------

    def _d01_softmax_formula(self, _frames) -> TestResult:
        """
        AttnLRP Proposition 3.1 (Eq. 13):
            R_backward_i = x_i * (R_forward_i - s_i * sum_j(R_forward_j))
        where s = softmax(x).

        Tests three seeds (one-hot, uniform, soft) on two different x vectors.
        A wrong backward (e.g., plain autograd or wrong sign) fails immediately.
        """
        failures = []

        test_cases = [
            # (x, R_forward)
            (torch.tensor([[0.0, 1.0, 2.0]]),  torch.tensor([[1.0, 0.0, 0.0]])),
            (torch.tensor([[0.0, 1.0, 2.0]]),  torch.tensor([[0.333, 0.333, 0.334]])),
            (torch.tensor([[-1.0, 0.0, 1.0, 2.0]]), torch.tensor([[0.0, 1.0, 0.0, 0.0]])),
            (torch.tensor([[5.0, -5.0, 0.0]]), torch.tensor([[0.0, 0.0, 1.0]])),
        ]

        for case_idx, (x, R) in enumerate(test_cases):
            x_in = x.clone().requires_grad_(True)
            s = torch.softmax(x.detach(), dim=-1)
            expected = x.detach() * (R - s * R.sum(dim=-1, keepdim=True))

            y = LRPSoftmax.apply(x_in)
            y.backward(R)

            if x_in.grad is None:
                failures.append(f"case {case_idx}: grad is None after backward")
                continue

            max_err = float((x_in.grad - expected).abs().max().item())
            if max_err > 1e-5:
                failures.append(
                    f"case {case_idx}: max_err={max_err:.2e} > 1e-5  "
                    f"(x={x.tolist()}, R={R.tolist()})"
                )

        status  = FAIL if failures else PASS
        summary = (f"All {len(test_cases)} cases match Prop 3.1 (tol=1e-5)." if not failures
                   else f"{len(failures)} formula violation(s).")
        return TestResult("D01_lrpsoftmax_formula", status, summary,
                          metrics={"n_cases": len(test_cases), "n_failures": len(failures)},
                          notes=failures + [
                              "Formula: R_i^back = x_i * (R_i^fwd - s_i * sum(R^fwd))",
                              "FAIL means the implemented backward is NOT AttnLRP Prop 3.1.",
                          ])

    # ------------------------------------------------------------------
    # D02 — LRPMatMul backward formula
    # FAIL criterion: any grad component off by > 1e-4 OR sum(R_A)+sum(R_B) off
    #   from sum(R_O) by > 1% (approximate conservation check).
    # ------------------------------------------------------------------

    def _d02_matmul_formula(self, _frames) -> TestResult:
        """
        AttnLRP Proposition 3.3 (Eq. 15):
            O = A @ B
            denom = 2*O + eps*sign(O)
            R_A = (R_O / denom) @ B^T * A
            R_B = A^T @ (R_O / denom) * B

        Also verifies approximate conservation:
            sum(R_A) + sum(R_B) ≈ sum(R_O)
        This identity holds up to the epsilon stabilizer; for small eps on
        large outputs the discrepancy is < 1%.
        """
        failures = []
        eps = LRPMatMul.EPS

        test_cases = [
            # (A, B, R_O)
            (torch.randn(2, 3), torch.randn(3, 2), torch.eye(2)),
            (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)),
            (torch.ones(2, 2),  torch.ones(2, 2),  torch.tensor([[1.0, 0.0],[0.0, 1.0]])),
            # near-zero output — tests eps stabilizer
            (torch.tensor([[1e-3, -1e-3]]), torch.tensor([[1e-3], [-1e-3]]),
             torch.tensor([[1.0]])),
        ]

        for ci, (A, B, R_O) in enumerate(test_cases):
            A_in = A.clone().requires_grad_(True)
            B_in = B.clone().requires_grad_(True)

            # Reference
            O_ref = (A @ B).detach()
            sign  = O_ref.sign()
            sign[sign == 0] = 1.0
            denom      = 2.0 * O_ref + eps * sign
            scaled_R   = R_O / denom
            exp_R_A    = (scaled_R @ B.T) * A.detach()
            exp_R_B    = (A.T @ scaled_R) * B.detach()

            # Actual via autograd
            O = LRPMatMul.apply(A_in, B_in)
            O.backward(R_O)

            if A_in.grad is None or B_in.grad is None:
                failures.append(f"case {ci}: grad is None")
                continue

            err_A = float((A_in.grad - exp_R_A).abs().max().item())
            err_B = float((B_in.grad - exp_R_B).abs().max().item())
            if err_A > 1e-4:
                failures.append(f"case {ci}: R_A max_err={err_A:.2e} > 1e-4")
            if err_B > 1e-4:
                failures.append(f"case {ci}: R_B max_err={err_B:.2e} > 1e-4")

            # Approximate conservation: sum(R_A) + sum(R_B) ≈ sum(R_O)
            # Skip when |O_mean| < 100*eps: in the epsilon-dominated regime the
            # stabilizer intentionally absorbs relevance (error ≈ eps/(2|O|+eps)),
            # which can exceed 5% by design and is NOT a bug in LRPMatMul.
            O_mean_abs = float(O_ref.abs().mean().item())
            if O_mean_abs < 100 * eps:
                pass  # epsilon-dominated; conservation intentionally approximate
            else:
                sum_in  = float(A_in.grad.sum().item() + B_in.grad.sum().item())
                sum_out = float(R_O.sum().item())
                if abs(sum_out) > 1e-8:
                    cons_err = abs(sum_in - sum_out) / abs(sum_out)
                    if cons_err > 0.05:
                        failures.append(
                            f"case {ci}: conservation error {cons_err:.2%} > 5% "
                            f"(sum_in={sum_in:.4f}, sum_out={sum_out:.4f}, "
                            f"|O_mean|={O_mean_abs:.2e})"
                        )

        status  = FAIL if failures else PASS
        summary = (f"All {len(test_cases)} cases: formula matches and conservation holds." if not failures
                   else f"{len(failures)} failure(s).")
        return TestResult("D02_lrpmatmul_formula", status, summary,
                          metrics={"n_cases": len(test_cases), "n_failures": len(failures)},
                          notes=failures + [
                              "R_A = (R/denom)@B^T * A,  R_B = A^T@(R/denom) * B",
                              "Conservation: sum(R_A)+sum(R_B) ≈ sum(R_O) (within 5% due to eps).",
                          ])

    # ------------------------------------------------------------------
    # D13 — LRPResidualAdd backward formula (Ratio-Based Relevance Splitting)
    # FAIL criterion: any grad component off by > 1e-5 vs the reference
    #   formula, conservation error > 1% outside the eps-dominated regime,
    #   or the near-cancellation case explodes (max|R| > 10).
    # Numbered out of sequence (added after D01-D12 already existed) —
    # grouped here with the other pure-formula unit tests (D01, D02).
    # ------------------------------------------------------------------

    def _d13_residual_add_formula(self, _frames) -> TestResult:
        """
        Otsuki et al. 2024 (arXiv:2407.09115) Ratio-Based Relevance Splitting,
        applied to residual/skip connections (Eq. 5):
            z = a + b
            R_a = R_z * |a| / (|a| + |b| + eps)
            R_b = R_z * |b| / (|a| + |b| + eps)
        so that R_a + R_b == R_z (exact conservation, up to eps).

        Includes a near-cancellation case (a ~= -b, large magnitude) where a
        SIGNED ratio (e.g. zennit's built-in Sum+Norm rule: a/(a+b+eps)) would
        explode because a+b ~= 0 while |a|+|b| stays large — this is exactly
        why LRPResidualAdd uses |a|, |b| instead of signed a, b (same
        rationale as LRPMatMul.EPS above).
        """
        failures = []
        eps = LRPResidualAdd.EPS

        test_cases = [
            # (a, b, R_z) -- generic mixed-sign, no exactly-zero elements
            (torch.tensor([2.0, -1.0, 0.5]), torch.tensor([1.0, 1.0, -0.5]), torch.tensor([1.0, 1.0, 1.0])),
            (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4, 4)),
            # both branches exactly zero -- fully eps-dominated, must not error/NaN
            (torch.zeros(3), torch.zeros(3), torch.tensor([1.0, -1.0, 0.5])),
            # near-cancellation: a ~= -b, large magnitude -> a+b ~= 0 but |a|+|b| >> 0
            (torch.tensor([1000.0, -500.0]), torch.tensor([-999.999, 500.001]), torch.tensor([1.0, 1.0])),
        ]

        for ci, (a, b, R_z) in enumerate(test_cases):
            a_in = a.clone().requires_grad_(True)
            b_in = b.clone().requires_grad_(True)

            a_abs, b_abs = a.abs(), b.abs()
            denom   = a_abs + b_abs + eps
            exp_R_a = R_z * a_abs / denom
            exp_R_b = R_z * b_abs / denom

            z = LRPResidualAdd.apply(a_in, b_in)

            # Forward must be exactly a+b (relevance-splitting is backward-only).
            fwd_err = float((z.detach() - (a + b)).abs().max().item())
            if fwd_err > 1e-6:
                failures.append(f"case {ci}: forward != a+b, max_err={fwd_err:.2e}")

            z.backward(R_z)

            if a_in.grad is None or b_in.grad is None:
                failures.append(f"case {ci}: grad is None")
                continue

            err_a = float((a_in.grad - exp_R_a).abs().max().item())
            err_b = float((b_in.grad - exp_R_b).abs().max().item())
            if err_a > 1e-5:
                failures.append(f"case {ci}: R_a max_err={err_a:.2e} > 1e-5")
            if err_b > 1e-5:
                failures.append(f"case {ci}: R_b max_err={err_b:.2e} > 1e-5")

            # Conservation: sum(R_a) + sum(R_b) ~= sum(R_z), skipped when any
            # element has |a|+|b| ~= 0 (eps-dominated regime intentionally
            # absorbs relevance there, same convention as D02).
            denom_min = float(denom.min().item())
            if denom_min > 100 * eps:
                sum_in  = float(a_in.grad.sum().item() + b_in.grad.sum().item())
                sum_out = float(R_z.sum().item())
                if abs(sum_out) > 1e-8:
                    cons_err = abs(sum_in - sum_out) / abs(sum_out)
                    if cons_err > 0.01:
                        failures.append(
                            f"case {ci}: conservation error {cons_err:.2%} > 1% "
                            f"(sum_in={sum_in:.6f}, sum_out={sum_out:.6f})"
                        )

            # Robustness check on the near-cancellation case: R_a, R_b must
            # stay bounded (no explosion) even though a+b ~= 0. A signed
            # ratio (a/(a+b+eps)) would blow up to ~1e6 here.
            if ci == 3:
                max_R = float(max(a_in.grad.abs().max().item(), b_in.grad.abs().max().item()))
                if max_R > 10.0:
                    failures.append(
                        f"case {ci}: near-cancellation case exploded, max|R|={max_R:.2e} "
                        "(expected bounded output since |a|+|b| stays large despite a+b~=0)"
                    )

        status  = FAIL if failures else PASS
        summary = (f"All {len(test_cases)} cases match Ratio-Based Splitting (Eq. 5) and conserve."
                   if not failures else f"{len(failures)} failure(s).")
        return TestResult("D13_residual_add_formula", status, summary,
                          metrics={"n_cases": len(test_cases), "n_failures": len(failures)},
                          notes=failures + [
                              "R_a = R_z*|a|/(|a|+|b|+eps),  R_b = R_z*|b|/(|a|+|b|+eps).",
                              "Conservation: sum(R_a)+sum(R_b) ~= sum(R_z) outside eps-dominated regime.",
                              "Near-cancellation (a~=-b) must stay bounded -- this is why |a|,|b| "
                              "(not signed a,b) are used in the denominator.",
                          ])

    # ------------------------------------------------------------------
    # D03 — _make_speed_seed correctness
    # FAIL criterion: any assertion below fails.
    # ------------------------------------------------------------------

    def _d03_seed_correctness(self, _frames) -> TestResult:
        """
        Tests all three seeding modes with hand-crafted logits:
          - Default:       seed = softmax(logits); sums to 1; all >= 0
          - forced_brake:  seed[0] = 1.0, rest = 0
          - forced_drive:  seed[k] = 1.0 where k = argmax(logits[1:]) + 1
          - is_brake = (argmax(logits) == 0), independent of which forced flag is set

        Two scenarios: model predicts driving (bin 5) and model predicts stopping (bin 0).
        """
        failures = []

        # Scenario 1: confident driving — bin 5 is highest
        logits_drive = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.3, 0.9, 0.2, 0.1]])
        expected_drive_bin = int(logits_drive[0, 1:].argmax().item()) + 1  # = 5

        seed_def, is_b_def = LRPTFv6Model._make_speed_seed(logits_drive, False, False)
        seed_brk, is_b_brk = LRPTFv6Model._make_speed_seed(logits_drive, True,  False)
        seed_drv, is_b_drv = LRPTFv6Model._make_speed_seed(logits_drive, False, True)

        # Default: softmax, sums to 1, all >= 0
        if abs(seed_def.sum().item() - 1.0) > 1e-5:
            failures.append(f"[drive/default] seed.sum()={seed_def.sum():.6f} ≠ 1.0")
        if not (seed_def >= 0).all():
            failures.append("[drive/default] seed has negative values (softmax must be >= 0)")
        if (seed_def == seed_def.max()).sum() == 1:
            # Confirm max is at bin 5
            if seed_def.argmax().item() != 5:
                failures.append(f"[drive/default] softmax peak at {seed_def.argmax().item()} != 5")

        # forced_brake: one-hot at bin 0
        if seed_brk[0, 0].item() != 1.0:
            failures.append(f"[drive/forced_brake] seed[0,0]={seed_brk[0,0]:.4f} != 1.0")
        if seed_brk[0, 1:].abs().sum().item() > 1e-6:
            failures.append("[drive/forced_brake] non-zero values outside bin 0")

        # forced_drive: one-hot at expected drive bin
        if seed_drv[0, expected_drive_bin].item() != 1.0:
            failures.append(
                f"[drive/forced_drive] seed[0,{expected_drive_bin}]="
                f"{seed_drv[0, expected_drive_bin]:.4f} != 1.0"
            )
        off_mass = seed_drv.clone(); off_mass[0, expected_drive_bin] = 0.0
        if off_mass.abs().sum().item() > 1e-6:
            failures.append("[drive/forced_drive] non-zero values outside drive bin")

        # is_brake: all three should report False (model predicts driving)
        for flag, is_b in [("default", is_b_def), ("forced_brake", is_b_brk), ("forced_drive", is_b_drv)]:
            if is_b != False:
                failures.append(f"[drive/{flag}] is_brake={is_b} but model predicts bin 5 (driving)")

        # Scenario 2: confident stopping — bin 0 is highest
        logits_stop = torch.tensor([[2.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]])
        expected_drive_bin2 = int(logits_stop[0, 1:].argmax().item()) + 1  # best non-brake

        seed_def2, is_b_def2 = LRPTFv6Model._make_speed_seed(logits_stop, False, False)
        seed_brk2, is_b_brk2 = LRPTFv6Model._make_speed_seed(logits_stop, True,  False)
        seed_drv2, is_b_drv2 = LRPTFv6Model._make_speed_seed(logits_stop, False, True)

        # All three must report is_brake=True (model predicts stopping)
        for flag, is_b in [("default", is_b_def2), ("forced_brake", is_b_brk2), ("forced_drive", is_b_drv2)]:
            if is_b != True:
                failures.append(
                    f"[stop/{flag}] is_brake={is_b} but model predicts bin 0 (stopping); "
                    f"is_brake must reflect model prediction, NOT the forced flag"
                )

        # forced_drive on a stop-predicting model: should pick best non-brake bin
        if seed_drv2[0, expected_drive_bin2].item() != 1.0:
            failures.append(
                f"[stop/forced_drive] seed[0,{expected_drive_bin2}]="
                f"{seed_drv2[0, expected_drive_bin2]:.4f} != 1.0 — "
                f"should be best non-brake bin even when model predicts stopping"
            )

        # Scenario 3: forced_brake seed must always be bin 0 regardless of model prediction
        if seed_brk2[0, 0].item() != 1.0:
            failures.append(f"[stop/forced_brake] seed[0,0]={seed_brk2[0,0]:.4f} != 1.0")

        status  = FAIL if failures else PASS
        summary = ("All seeding scenarios pass." if not failures
                   else f"{len(failures)} seeding violation(s).")
        return TestResult("D03_seed_generation_correctness", status, summary,
                          metrics={"n_failures": len(failures)},
                          notes=failures + [
                              "is_brake MUST reflect argmax(logits)==0, never the forced flag.",
                              "forced_brake always → one-hot at bin 0 regardless of prediction.",
                              "forced_drive always → one-hot at argmax(logits[1:])+1.",
                          ])

    # ------------------------------------------------------------------
    # D04 — LiDAR gradient isolation
    # FAIL criterion: _make_lidar() returns a tensor with requires_grad=True,
    #   OR two calls return different tensors (non-deterministic).
    # ------------------------------------------------------------------

    def _d04_lidar_isolation(self, _frames) -> TestResult:
        """
        TFV6 runs in LTF (LiDAR-as-Template) mode: the lidar 'input' is a
        fixed 2-channel [y, x] positional grid, identical for every frame.
        No gradient must flow back through it.

        Tests:
          1. _make_lidar() returns requires_grad=False (no gradient path)
          2. Two consecutive calls return the SAME tensor values (deterministic)
          3. The tensor has exactly 2 channels, matching the in_chans=2 of LTF mode
        """
        failures = []
        dtype  = torch.float32
        device = self.device

        grid1 = self.lrp.full_model._make_lidar(device, dtype)
        grid2 = self.lrp.full_model._make_lidar(device, dtype)

        # 1. No gradient
        if grid1.requires_grad:
            failures.append(
                "CRITICAL: _make_lidar() returns requires_grad=True. "
                "LRP attributions are NOT vision-only — lidar receives gradients."
            )

        # 2. Determinism
        if not torch.equal(grid1.cpu(), grid2.cpu()):
            max_diff = float((grid1 - grid2).abs().max().item())
            failures.append(
                f"_make_lidar() is non-deterministic: max diff={max_diff:.2e} "
                "between two identical calls."
            )

        # 3. Channel count: must be 2 (LTF mode)
        if grid1.shape[1] != 2:
            failures.append(
                f"LiDAR grid has {grid1.shape[1]} channels, expected 2 (LTF mode: [y_pos, x_pos]). "
                "If LTF=False in config, in_chans=1 and the model expects 1-channel real LiDAR."
            )

        # 4. Content: channels should be spatial position ramps [0, 1]
        ch0, ch1 = grid1[0, 0], grid1[0, 1]
        if not (ch0.min().item() >= -1e-5 and ch0.max().item() <= 1.0 + 1e-5):
            failures.append(f"Channel 0 (y-pos) out of [0,1]: [{ch0.min():.4f}, {ch0.max():.4f}]")
        if not (ch1.min().item() >= -1e-5 and ch1.max().item() <= 1.0 + 1e-5):
            failures.append(f"Channel 1 (x-pos) out of [0,1]: [{ch1.min():.4f}, {ch1.max():.4f}]")

        status  = FAIL if failures else PASS
        summary = (
            f"LiDAR grid: {list(grid1.shape)}, requires_grad=False, deterministic." if not failures
            else failures[0]
        )
        return TestResult("D04_lidar_gradient_isolation", status, summary,
                          metrics={
                              "grid_shape":     list(grid1.shape),
                              "requires_grad":  bool(grid1.requires_grad),
                              "deterministic":  bool(torch.equal(grid1.cpu(), grid2.cpu())),
                          },
                          notes=failures + [
                              "requires_grad=True → LRP can attribute to lidar → maps are NOT vision-only.",
                              "Non-determinism → different attribution maps for the same frame.",
                          ])

    # ------------------------------------------------------------------
    # D05 — LRP1 conservation through target_speed_decoder
    # FAIL criterion: any frame has sum(node_rel) > 1.01 (z+ cannot create relevance)
    #   OR mean sum(node_rel) < 0.05 (near-total absorption = bad attribution quality).
    # ------------------------------------------------------------------

    def _d05_lrp1_conservation(self, frames) -> TestResult:
        """
        z+ rule conservation: Σ R_j^{input} = Σ R_k^{output} at each linear layer.
        For a one-hot seed (sum=1) through the two Linear layers of target_speed_decoder,
        the total at speed_query level must satisfy:

            0 < Σ node_rel ≤ 1.0

        Upper bound 1.0: z+ rule cannot create relevance.
        Values below 1.0 are fine — they reflect speed_query components that are
        negative (clipped to 0 in z+).
        Values near 0 (< 0.05) mean almost all speed_query activations are negative,
        making LRP1 essentially non-informative.
        Values above 1.0 indicate a bug in the z+ backward (relevance created).
        """
        if frames is None:
            return TestResult("D05_lrp1_conservation", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 8)
        totals = []
        failures = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd  = float(frames["speed"][i])
            data = self._get_data(wide, spd, frames, i)
            rgb_x = self.lrp._prepare_input(wide)

            node_rel, _ = self.lrp._attribute_to_fc(rgb_x, data, False, False)
            total = float(node_rel.sum().item())
            totals.append(total)

            if total > 1.01:
                failures.append(
                    f"frame {i}: Σ node_rel = {total:.5f} > 1.01  — "
                    "z+ CREATED relevance (impossible without a bug)"
                )
            if total < 0.0:
                failures.append(
                    f"frame {i}: Σ node_rel = {total:.5f} < 0  — "
                    "negative total suggests sign error in LRP1"
                )

        totals_arr  = np.array(totals)
        mean_total  = float(totals_arr.mean())
        clipped_pct = float((totals_arr < 0.2).mean() * 100)

        # FAIL on any explosion; WARN on near-zero mean
        if failures:
            status  = FAIL
            summary = "; ".join(failures[:2])
        elif mean_total < 0.05:
            status  = FAIL
            summary = (
                f"mean Σ node_rel = {mean_total:.4f} < 0.05 — LRP1 is effectively zero. "
                "speed_query activations may all be negative → z+ denominator = 0."
            )
        elif clipped_pct > 80:
            status  = WARN
            summary = (
                f"mean Σ node_rel = {mean_total:.4f} — {clipped_pct:.0f}% of frames have "
                "Σ < 0.2 (heavy clipping). Consider ε-rule for target_speed_decoder."
            )
        else:
            status  = PASS
            summary = (
                f"Σ node_rel in (0, 1]: mean={mean_total:.4f} across {N} frames. "
                "z+ conservation holds."
            )

        return TestResult("D05_lrp1_conservation", status, summary,
                          metrics={
                              "node_rel_sum_mean":  float(mean_total),
                              "node_rel_sum_min":   float(totals_arr.min()),
                              "node_rel_sum_max":   float(totals_arr.max()),
                              "node_rel_sum_std":   float(totals_arr.std()),
                              "pct_frames_below_0p2": float(clipped_pct),
                              "n_frames": N,
                          },
                          per_frame=totals_arr,
                          notes=failures + [
                              "Σ > 1.01 → z+ backward has a bug (sign flip, wrong denominator).",
                              "Σ < 0.05 → almost all speed_query activations are negative; "
                              "z+ clips them to 0, so no attribution flows. ε-rule would fix this.",
                              "Σ ∈ (0.2, 1.0] is normal — some clipping is expected.",
                          ])

    # ------------------------------------------------------------------
    # D06 — Backbone amplification budget
    # FAIL criterion: |Σ pixel_rel / Σ node_rel| > 50 (extreme amplification)
    # ------------------------------------------------------------------

    def _d06_amplification(self, frames) -> TestResult:
        """
        Measures where relevance is amplified or attenuated between the two LRP stages.

        Stage 1 (target_speed_decoder, LRP1):
            Σ node_rel  (expected: 0 < Σ ≤ 1.0 by z+ rule — tested in D05)

        Stage 2 (backbone, LRP2):
            Σ pixel_rel  (can exceed Σ node_rel due to ε-rule in attention
            and BatchNorm Pass — these are known amplification sources)

        The ratio r = Σ pixel_rel_signed / Σ node_rel tells us the net signed
        amplification through the backbone. A stable ratio across frames indicates
        consistent (even if large) amplification; a wildly varying ratio indicates
        numerical instability.

        Also reports an ABSOLUTE-value ratio, Σ|pixel_rel| / |Σ node_rel|. The
        signed ratio conflates two distinct effects: true magnitude loss (values
        genuinely shrinking through the backbone) and sign cancellation (positive
        and negative regions offsetting in the sum without either actually
        vanishing). This matters concretely when comparing rule choices: e.g.
        AlphaBeta(2,1) (uitb=True) surfaces much more negative relevance than
        pure z+ (pos_frac drops sharply), which shrinks the SIGNED ratio even if
        the underlying magnitude is not more attenuated -- only the absolute
        ratio can tell those two apart. Added 2026-07-02 after exactly this
        ambiguity came up comparing uitb=True against the default composite.

        Additionally: positive fraction of pixel_rel is reported.
        With AttnLRP, pixel maps are signed; positive fraction > 0.45 is expected.
        """
        if frames is None:
            return TestResult("D06_backbone_amplification", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 8)
        ratios, abs_ratios, pos_fracs, node_totals, pixel_totals, pixel_abs_totals = [], [], [], [], [], []
        failures = []

        for i in range(N):
            wide  = _to_tensor(frames["wide_rgb"][i], self.device)
            spd   = float(frames["speed"][i])
            data  = self._get_data(wide, spd, frames, i)
            rgb_x = self.lrp._prepare_input(wide)

            node_rel, _ = self.lrp._attribute_to_fc(rgb_x, data, False, False)
            node_sum    = float(node_rel.sum().item())

            # LRP2: backbone, same node_rel as seed
            rgb_x2 = self.lrp._prepare_input(wide)
            data2  = self._get_data(wide, spd, frames, i)
            nr_cpu = node_rel.detach().cpu()
            selector = lambda q, nr=nr_cpu: nr.unsqueeze(0).to(q.device, dtype=q.dtype)
            pixel_rel = self.lrp._attribute_backbone(rgb_x2, data2, selector)
            pixel_arr = _as_numpy(pixel_rel).flatten()

            pixel_signed = float(pixel_arr.sum())
            pixel_abs    = float(np.abs(pixel_arr).sum())
            pos_frac     = float(pixel_arr[pixel_arr > 0].sum() / (pixel_abs + 1e-12))

            node_totals.append(node_sum)
            pixel_totals.append(pixel_signed)
            pixel_abs_totals.append(pixel_abs)
            pos_fracs.append(pos_frac)

            if abs(node_sum) > 1e-8:
                ratio = pixel_signed / node_sum
                ratios.append(ratio)
                abs_ratios.append(pixel_abs / abs(node_sum))
                if abs(ratio) > 50:
                    failures.append(
                        f"frame {i}: amplification ratio={ratio:.1f} (|pixel_sum/node_sum| > 50). "
                        "Check BatchNorm Pass rule or ε-rule instability."
                    )

        if not ratios:
            return TestResult("D06_backbone_amplification", WARN,
                              "All frames had near-zero node_rel — cannot compute ratio.")

        r_arr   = np.array(ratios)
        abs_arr = np.array(abs_ratios)
        cov     = float(r_arr.std() / (abs(r_arr.mean()) + 1e-12))
        abs_cov = float(abs_arr.std() / (abs_arr.mean() + 1e-12))
        pf_arr  = np.array(pos_fracs)

        status = FAIL if failures else (WARN if cov > 1.0 else PASS)
        summary = (
            f"Backbone amplification: signed ratio mean={r_arr.mean():.3g} (CoV={cov:.3f}), "
            f"abs ratio mean={abs_arr.mean():.3g} (CoV={abs_cov:.3f}), pos_frac={pf_arr.mean():.3f}."
            if not failures else failures[0]
        )

        return TestResult("D06_backbone_amplification", status, summary,
                          metrics={
                              "amplification_ratio_mean": float(r_arr.mean()),
                              "amplification_ratio_std":  float(r_arr.std()),
                              "amplification_ratio_max":  float(np.abs(r_arr).max()),
                              "amplification_cov":        float(cov),
                              "abs_amplification_ratio_mean": float(abs_arr.mean()),
                              "abs_amplification_ratio_std":  float(abs_arr.std()),
                              "abs_amplification_cov":        float(abs_cov),
                              "node_rel_sum_mean":        float(np.mean(node_totals)),
                              "pixel_rel_sum_mean":       float(np.mean(pixel_totals)),
                              "pixel_rel_abs_sum_mean":   float(np.mean(pixel_abs_totals)),
                              "pos_frac_mean":            float(pf_arr.mean()),
                              "n_frames": N,
                          },
                          per_frame=r_arr,
                          notes=failures + [
                              "signed ratio = Σ pixel_rel / Σ node_rel  (can shrink from sign "
                              "cancellation, not just magnitude loss).",
                              "abs ratio = Σ|pixel_rel| / |Σ node_rel|  (magnitude only, not "
                              "confounded by sign cancellation).",
                              "If abs ratio >> signed ratio, most of the apparent 'attenuation' in "
                              "the signed number is internal positive/negative cancellation.",
                              "ratio near 1 → conservative backbone.",
                              "ratio >> 1 → ε-rule / BatchNorm Pass is amplifying.",
                              "High CoV (> 1.0) → numerically unstable across frames.",
                              "BatchNorm with Pass rule is a known amplification source (Issue 7).",
                          ])

    # ------------------------------------------------------------------
    # D07 — Two-step decomposition consistency
    # FAIL criterion: pixel map from _attribute_true_output_to_input differs
    #   from the manually chained LRP1+LRP2 by relative L∞ > 1e-3.
    # ------------------------------------------------------------------

    def _d07_two_step(self, frames) -> TestResult:
        """
        _attribute_true_output_to_input internally does:
            Step 1: node_rel = d(logits)/d(query) * seed   [LRP through decoder]
            Step 2: pixel    = d(query)/d(rgb)    * node_rel [LRP through backbone]

        Both steps share ONE forward pass (retain_graph=True).

        This test replicates the same chain manually with TWO separate calls
        (_attribute_to_fc + _attribute_backbone) and verifies the results are
        numerically identical. Because autograd.grad is linear in grad_outputs
        and the model is deterministic in eval mode, the results MUST be the same.

        A discrepancy indicates the computational graph is inconsistent between
        the two composite contexts (a zennit hook pairing bug or graph mutation).
        """
        if frames is None:
            return TestResult("D07_two_step_consistency", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 4)
        rel_diffs = []
        failures  = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd  = float(frames["speed"][i])
            data = self._get_data(wide, spd, frames, i)

            # --- Method A: internal two-step function ---
            rgb_x_a = self.lrp._prepare_input(wide)
            data_a  = self._get_data(wide, spd, frames, i)
            pixel_a, _ = self.lrp._attribute_true_output_to_input(rgb_x_a, data_a, False, False)

            # --- Method B: explicit LRP1 then LRP2 ---
            rgb_x_b  = self.lrp._prepare_input(wide)
            data_b   = self._get_data(wide, spd, frames, i)
            node_rel, _ = self.lrp._attribute_to_fc(rgb_x_b, data_b, False, False)

            nr_cpu = node_rel.detach().cpu()
            rgb_x_c = self.lrp._prepare_input(wide)
            data_c  = self._get_data(wide, spd, frames, i)
            selector = lambda q, nr=nr_cpu: nr.unsqueeze(0).to(q.device, dtype=q.dtype)
            pixel_b  = self.lrp._attribute_backbone(rgb_x_c, data_c, selector)

            arr_a = _as_numpy(pixel_a).flatten()
            arr_b = _as_numpy(pixel_b).flatten()
            scale = float(np.abs(arr_a).max()) + 1e-8
            rel_diff = float(np.abs(arr_a - arr_b).max()) / scale
            rel_diffs.append(rel_diff)

            if rel_diff > 1e-3:
                failures.append(
                    f"frame {i}: rel_L∞={rel_diff:.2e} > 1e-3  — "
                    "two-step result differs from manual LRP1+LRP2 chain. "
                    "Possible composite context inconsistency or graph mutation."
                )

        arr = np.array(rel_diffs)
        status  = FAIL if failures else PASS
        summary = (f"Two-step ≡ manual LRP1+LRP2 across {N} frames (max rel_L∞={arr.max():.2e})."
                   if not failures else failures[0])
        return TestResult("D07_two_step_consistency", status, summary,
                          metrics={
                              "rel_linf_max":  float(arr.max()),
                              "rel_linf_mean": float(arr.mean()),
                              "n_frames": N,
                          },
                          per_frame=arr,
                          notes=failures + [
                              "By linearity of autograd.grad, these MUST match.",
                              "FAIL → composite hooks applied differently across contexts → bug.",
                          ])

    # ------------------------------------------------------------------
    # D08 — Forced-seed node_rel distinctiveness
    # FAIL criterion: any forced-seed pair has cosine > 0.9999
    #   (practically identical node_rel → forced seed has no effect on LRP1).
    # ------------------------------------------------------------------

    def _d08_forced_seeds(self, frames) -> TestResult:
        """
        forced_brake and forced_drive produce DIFFERENT seeds for LRP1.
        The resulting node_rel vectors must therefore differ.

        If they are identical (cosine ≈ 1.0), the forced flag is not reaching
        _make_speed_seed, and the comparative relevance visualization is broken:
        R_drive - R_brake will be all-zero.

        Tests across multiple frames; a single distinct pair is sufficient to
        confirm routing is alive. All pairs being near-identical is the bug.
        """
        if frames is None:
            return TestResult("D08_forced_seed_distinctiveness", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 6)
        cos_brk_drv_list, cos_def_brk_list, cos_def_drv_list = [], [], []
        failures = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd  = float(frames["speed"][i])
            data = self._get_data(wide, spd, frames, i)

            rgb_def = self.lrp._prepare_input(wide)
            rgb_brk = self.lrp._prepare_input(wide)
            rgb_drv = self.lrp._prepare_input(wide)

            nr_def, _ = self.lrp._attribute_to_fc(rgb_def, self._get_data(wide, spd, frames, i), False, False)
            nr_brk, _ = self.lrp._attribute_to_fc(rgb_brk, self._get_data(wide, spd, frames, i), True,  False)
            nr_drv, _ = self.lrp._attribute_to_fc(rgb_drv, self._get_data(wide, spd, frames, i), False, True)

            a_def = _as_numpy(nr_def)
            a_brk = _as_numpy(nr_brk)
            a_drv = _as_numpy(nr_drv)

            cos_brk_drv = _cosine(a_brk, a_drv)
            cos_def_brk = _cosine(a_def, a_brk)
            cos_def_drv = _cosine(a_def, a_drv)
            cos_brk_drv_list.append(cos_brk_drv)
            cos_def_brk_list.append(cos_def_brk)
            cos_def_drv_list.append(cos_def_drv)

            if cos_brk_drv > 0.9999:
                failures.append(
                    f"frame {i}: brake–drive cosine={cos_brk_drv:.6f} ≈ 1.0 — "
                    "forced seeds produce IDENTICAL node_rel → comparative map will be zero."
                )

        cos_bd_arr = np.array(cos_brk_drv_list)
        status = FAIL if len(failures) == N else (WARN if failures else PASS)
        summary = (
            f"brake–drive cosine: mean={cos_bd_arr.mean():.4f}, max={cos_bd_arr.max():.4f}."
            if not failures else f"{len(failures)}/{N} frames: brake/drive maps identical."
        )
        return TestResult("D08_forced_seed_distinctiveness", status, summary,
                          metrics={
                              "cos_brake_drive_mean":   float(cos_bd_arr.mean()),
                              "cos_brake_drive_max":    float(cos_bd_arr.max()),
                              "cos_default_brake_mean": float(np.mean(cos_def_brk_list)),
                              "cos_default_drive_mean": float(np.mean(cos_def_drv_list)),
                              "n_frames": N,
                          },
                          per_frame=cos_bd_arr,
                          notes=failures + [
                              "cosine ≈ 1.0 → forced_brake and forced_drive produce the same node_rel.",
                              "This means the seed is not changing → comparative map = 0.",
                          ])

    # ------------------------------------------------------------------
    # D09 — is_brake independence from forced flags
    # FAIL criterion: is_brake differs between any two calls on the same frame.
    # ------------------------------------------------------------------

    def _d09_is_brake(self, frames) -> TestResult:
        """
        _make_speed_seed sets is_brake = (argmax(logits) == 0), independently
        of which forced flag is set.  All three calls on the same frame must
        return the same is_brake value.

        A discrepancy would mean the visualisation uses the wrong colormap for
        forced maps vs the default map.
        """
        if frames is None:
            return TestResult("D09_is_brake_independence", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 8)
        failures = []
        brake_votes = []

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd  = float(frames["speed"][i])

            rgb1 = self.lrp._prepare_input(wide); d1 = self._get_data(wide, spd, frames, i)
            rgb2 = self.lrp._prepare_input(wide); d2 = self._get_data(wide, spd, frames, i)
            rgb3 = self.lrp._prepare_input(wide); d3 = self._get_data(wide, spd, frames, i)

            _, b_def = self.lrp._attribute_to_fc(rgb1, d1, False, False)
            _, b_brk = self.lrp._attribute_to_fc(rgb2, d2, True,  False)
            _, b_drv = self.lrp._attribute_to_fc(rgb3, d3, False, True)

            brake_votes.append(int(b_def))

            if b_def != b_brk or b_def != b_drv:
                failures.append(
                    f"frame {i}: is_brake differs — "
                    f"default={b_def}, forced_brake={b_brk}, forced_drive={b_drv}. "
                    "is_brake must be the MODEL's argmax, not the forced flag."
                )

        n_brake  = sum(brake_votes)
        n_drive  = N - n_brake
        status   = FAIL if failures else PASS
        summary  = (
            f"is_brake consistent across all 3 seeds, {N} frames "
            f"({n_brake} brake, {n_drive} drive)."
            if not failures else "; ".join(failures[:2])
        )
        return TestResult("D09_is_brake_independence", status, summary,
                          metrics={
                              "n_failures": len(failures),
                              "n_brake_frames": n_brake,
                              "n_drive_frames": n_drive,
                              "n_frames": N,
                          },
                          notes=failures + [
                              "is_brake = (speed_logits.argmax() == 0) — model's actual prediction.",
                              "forced_brake/drive must NOT change is_brake.",
                          ])

    # ------------------------------------------------------------------
    # D10 — Per-node pixel map pairwise cosine similarity matrix
    # FAIL criterion: any tested frame has min pairwise cosine > 0.9999
    #   (all maps identical in that frame -- node_id not reaching backward)
    # WARN criterion: mean pairwise cosine across all frames > 0.90
    # ------------------------------------------------------------------

    def _d10_node_cosines(self, frames) -> TestResult:
        """
        For each of up to 4 frames, select that frame's own top-K F_c nodes
        (K = min(8, n_selected)) and compute all pairwise cosine similarities
        of their pixel maps. Reports both the pooled cosine distribution
        (all frames' pairs together) and the per-frame mean, so a systematic
        "nodes look alike" effect (consistent across frames -- likely a
        property of the model/rules) can be distinguished from a one-frame
        artifact (high variance across per-frame means).

        Originally single-frame (frame 0 only); extended 2026-07-02 after the
        residual-conservation fix (LRPResidualAdd) raised the question of
        whether skip-path-dominated relevance flow makes node maps more
        self-similar as a systematic property, not a one-off. Node selection
        is re-run per frame (relevant nodes are frame-dependent) rather than
        reusing frame 0's node_ids.

        All-identical maps in any one frame (cosine ≈ 1.0 for every pair)
        means node_id is NOT reaching the fc→input backward seed there --
        the original regression target for Bug #1 (all-ones seed regardless
        of node_id).

        Low mean cosine < 0.3 is also notable: it suggests high per-node
        specialization, which is actually desirable but worth reporting.
        """
        if frames is None:
            return TestResult("D10_per_node_cosine_matrix", WARN, "No testframes — skipped.")

        N = min(len(frames["cmd"]), 4)
        per_frame_mean: List[float] = []
        per_frame_min:  List[float] = []
        per_frame_K:    List[int]   = []
        all_cosines:    List[float] = []
        collapsed_frames: List[int] = []
        skipped_frames:   List[int] = []

        for i in range(N):
            wide  = _to_tensor(frames["wide_rgb"][i], self.device)
            spd   = float(frames["speed"][i])
            data  = self._get_data(wide, spd, frames, i)
            rgb_x = self.lrp._prepare_input(wide)

            r_nodes, _ = self.lrp._attribute_to_fc(rgb_x, data, False, False)
            node_ids   = _relevance_filter(r_nodes, 0.9)

            K = min(8, len(node_ids))
            if K < 2:
                skipped_frames.append(i)
                continue

            probe_ids = node_ids[:K]
            maps = []
            for nid in probe_ids:
                out, _, _, _ = self.lrp.forward_relevance(wide, narr_rgb=None,
                                                           beg="fc", end="input", node_id=nid)
                maps.append(_as_numpy(out).flatten())

            cosines = []
            for a in range(K):
                for b in range(a + 1, K):
                    cosines.append(_cosine(maps[a], maps[b]))

            cos_arr = np.array(cosines)
            per_frame_mean.append(float(cos_arr.mean()))
            per_frame_min.append(float(cos_arr.min()))
            per_frame_K.append(K)
            all_cosines.extend(cosines)
            if float(cos_arr.min()) > 0.9999:
                collapsed_frames.append(i)

        if not per_frame_mean:
            return TestResult("D10_per_node_cosine_matrix", WARN,
                              f"All {N} tested frame(s) had < 2 selected nodes -- "
                              "cannot compute pairwise comparison.",
                              {"n_frames_tested": N, "n_frames_skipped": len(skipped_frames)})

        all_arr        = np.array(all_cosines)
        frame_mean_arr = np.array(per_frame_mean)
        n_frames_ok    = len(per_frame_mean)
        n_pairs_total  = len(all_cosines)
        mean_cos       = float(all_arr.mean())
        min_cos        = float(all_arr.min())
        max_cos        = float(all_arr.max())
        frame_mean_std = float(frame_mean_arr.std())

        if collapsed_frames:
            status  = FAIL
            summary = (
                f"{len(collapsed_frames)}/{n_frames_ok} frame(s) (indices {collapsed_frames}) "
                "have ALL pairs cosine > 0.9999 -- node maps are IDENTICAL in at least one "
                "frame (Bug #1 regression)."
            )
        elif mean_cos > 0.90:
            status  = WARN
            summary = (
                f"Mean pairwise cosine = {mean_cos:.4f} > 0.90 pooled across {n_frames_ok} "
                f"frame(s), {n_pairs_total} pairs. Per-frame means range "
                f"[{frame_mean_arr.min():.4f}, {frame_mean_arr.max():.4f}] "
                f"(std={frame_mean_std:.4f}) -- "
                f"{'consistent across frames (systematic)' if frame_mean_std < 0.02 else 'varies notably by frame'}."
            )
        else:
            status  = PASS
            summary = (
                f"{n_frames_ok} frame(s), {n_pairs_total} pairs: pooled mean cos={mean_cos:.4f}, "
                f"min={min_cos:.4f}."
            )

        return TestResult("D10_per_node_cosine_matrix", status, summary,
                          metrics={
                              "n_frames_tested":     n_frames_ok,
                              "n_frames_skipped":    len(skipped_frames),
                              "n_frames_collapsed":  len(collapsed_frames),
                              "n_pairs_total":       n_pairs_total,
                              "cosine_mean_pooled":  mean_cos,
                              "cosine_min_pooled":   min_cos,
                              "cosine_max_pooled":   max_cos,
                              "per_frame_mean_avg":  float(frame_mean_arr.mean()),
                              "per_frame_mean_std":  frame_mean_std,
                              "per_frame_K_mean":    float(np.mean(per_frame_K)),
                          },
                          per_frame=frame_mean_arr,
                          notes=[
                              "cosine=1.0 for ALL pairs in a frame -> node_id not routed to backward (Bug #1).",
                              "Low per_frame_mean_std (< 0.02) -> node self-similarity is systematic "
                              "across frames, not a one-frame artifact.",
                              "High per_frame_mean_std -> similarity varies by frame content -- worth "
                              "inspecting low-similarity frames (more node-specific evidence available) "
                              "separately from high-similarity ones (likely skip-path dominated).",
                          ])

    # ------------------------------------------------------------------
    # D11 — Bias fraction in target_speed_decoder
    # FAIL criterion: mean bias fraction > 0.50 in any Linear layer
    #   (biases supply more than half the z+ denominator → inputs barely matter)
    # ------------------------------------------------------------------

    def _d11_bias_fraction(self, frames) -> TestResult:
        """
        For the z+ rule on Linear(in→out), the denominator per output neuron k is:
            Z_k = Σ_j max(a_j, 0) * max(w_{jk}, 0) + max(b_k, 0)

        The bias fraction is max(b_k, 0) / Z_k.  When this is large, the bias
        term dominates: the relevance of input features is diluted by a constant
        offset that does not reflect any frame-specific information.

        TFV6 uses AlphaBeta(1, 0) WITHOUT zero_params='bias' — biases are included.
        WoR uses AlphaBeta with zero_params='bias' — biases explicitly excluded.

        This test measures whether TFV6's bias inclusion causes significant
        absorption.  A fraction > 0.50 means biases absorb more than input
        features for that layer on average.
        """
        if frames is None:
            return TestResult("D11_decoder_bias_fraction", WARN, "No testframes — skipped.")

        # Locate the two Linear layers in target_speed_decoder
        tsd = self.lrp.full_model.target_speed_decoder
        lin_layers = [(i, m) for i, m in enumerate(tsd) if isinstance(m, nn.Linear)]
        if len(lin_layers) < 2:
            return TestResult("D11_decoder_bias_fraction", WARN,
                              f"Found only {len(lin_layers)} Linear layers in target_speed_decoder "
                              "(expected 2). Cannot compute bias fraction.")

        N = min(len(frames["cmd"]), 6)
        layer_fracs: Dict[str, List[float]] = {}
        failures = []

        # layer_pos = sequential index of this Linear within the sequential module
        # (0 = first Linear = input is speed_query; 1 = second Linear = input is post-ReLU)
        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            spd  = float(frames["speed"][i])
            data = self._get_data(wide, spd, frames, i)

            with torch.no_grad():
                sq = self.lrp.full_model(wide.float().to(self.device), data).squeeze(0)  # [256]

                activations_at_layer = [sq]
                for m in tsd:
                    activations_at_layer.append(m(activations_at_layer[-1].unsqueeze(0)).squeeze(0))

                for layer_pos, (idx, lin) in enumerate(lin_layers):
                    a_in = activations_at_layer[layer_pos].clamp(min=0)
                    W    = lin.weight.detach()
                    b    = lin.bias.detach() if lin.bias is not None else None

                    Z_input = lin.weight.detach().clamp(min=0) @ a_in  # [out]
                    if b is not None:
                        b_pos   = b.clamp(min=0)
                        Z_total = Z_input + b_pos
                        frac    = (b_pos / (Z_total + 1e-12)).mean().item()
                    else:
                        frac = 0.0

                    key = f"layer{idx}"
                    if key not in layer_fracs:
                        layer_fracs[key] = []
                    layer_fracs[key].append(frac)

        metrics = {"n_frames": N}
        for key, vals in layer_fracs.items():
            mean_f = float(np.mean(vals))
            metrics[f"bias_frac_{key}_mean"] = mean_f
            metrics[f"bias_frac_{key}_max"]  = float(np.max(vals))
            if mean_f > 0.50:
                failures.append(
                    f"{key}: mean bias fraction={mean_f:.3f} > 0.50 — "
                    "biases dominate the z+ denominator. Input relevance is heavily diluted."
                )
            elif mean_f > 0.30:
                failures.append(
                    f"WARN {key}: mean bias fraction={mean_f:.3f} > 0.30. "
                    "Significant bias absorption. Consider zero_params='bias' like WoR."
                )

        hard_fails = [f for f in failures if not f.startswith("WARN")]
        status = FAIL if hard_fails else (WARN if failures else PASS)
        summary = (
            "; ".join(f"{k}: {v:.3f}" for k, v in metrics.items()
                      if k.startswith("bias_frac") and k.endswith("mean"))
            if not failures else failures[0]
        )

        return TestResult("D11_decoder_bias_fraction", status, summary,
                          metrics=metrics,
                          notes=failures + [
                              "bias_frac = max(b,0) / (Σ a^+ * w^+ + max(b,0))  per output neuron.",
                              "> 0.50 → biases absorb more than inputs in the z+ denominator.",
                              "TFV6 uses AlphaBeta without zero_params='bias' (unlike WoR).",
                              "To suppress bias absorption: use zero_params='bias' in TFV6 composite.",
                          ])

    # ------------------------------------------------------------------
    # D12 — LRP output determinism
    # FAIL criterion: any difference between two identical calls > 1e-6.
    # ------------------------------------------------------------------

    def _d12_determinism(self, frames) -> TestResult:
        """
        Two identical calls to forward_relevance with the same input must return
        bit-for-bit identical results.

        In eval() mode with deterministic lidar, there must be no stochastic
        element.  A failure indicates:
          - Dropout accidentally active (model not in eval mode)
          - Non-deterministic GPU operations (use torch.use_deterministic_algorithms)
          - Mutable state in the composite's hooks (zennit context re-use bug)
        """
        if frames is None:
            return TestResult("D12_lrp_output_determinism", WARN, "No testframes — skipped.")

        failures = []
        N = min(len(frames["cmd"]), 4)
        modes = [
            ("output→input", dict(beg="output", end="input")),
            ("output→fc",    dict(beg="output", end="fc")),
            ("fc→input[0]",  dict(beg="fc",     end="input", node_id=0)),
        ]

        for i in range(N):
            wide = _to_tensor(frames["wide_rgb"][i], self.device)
            for mode_name, kwargs in modes:
                r1, _, _, _ = self.lrp.forward_relevance(wide, narr_rgb=None, **kwargs)
                r2, _, _, _ = self.lrp.forward_relevance(wide, narr_rgb=None, **kwargs)
                arr1, arr2 = _as_numpy(r1), _as_numpy(r2)
                max_diff = float(np.abs(arr1 - arr2).max())
                if max_diff > 1e-6:
                    failures.append(
                        f"frame {i} mode {mode_name}: max_diff={max_diff:.2e} > 1e-6 — "
                        "LRP output is non-deterministic."
                    )

        status  = FAIL if failures else PASS
        summary = (f"All {N} frames × {len(modes)} modes: identical on repeat calls."
                   if not failures else "; ".join(failures[:2]))
        return TestResult("D12_lrp_output_determinism", status, summary,
                          metrics={"n_frames": N, "n_modes": len(modes),
                                   "n_failures": len(failures)},
                          notes=failures + [
                              "Non-determinism → dropout still active (model not in eval mode), "
                              "or GPU non-determinism (torch.use_deterministic_algorithms(True)).",
                          ])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_data(self, wide: torch.Tensor, spd: float, frames, idx: int) -> dict:
        """
        Set the lrp data cache for this frame and return the resulting dict.
        testframes may carry a 'data' list of per-frame dicts (TFV6 option B);
        if absent, _make_minimal_data is used (zero command vector).
        """
        frame_data = None
        if frames is not None and "data" in frames:
            frame_data = frames["data"][idx]
        self.lrp.update_context(wide, None, spd, data=frame_data)
        return self.lrp._get_data(spd)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_report(self, results: Dict[str, TestResult]) -> None:
        sym = {PASS: "✓ PASS", FAIL: "✗ FAIL", WARN: "△ WARN", ERROR: "! ERROR"}
        sep = "─" * 70

        print(f"\n{'='*70}")
        print("  TFV6 LRP DIAGNOSTICS — DETAILED REPORT")
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
        with open(os.path.join(out_dir, "tfv6_diagnostics_report.txt"), "w") as fh:
            fh.write(buf.getvalue())
        for name, r in results.items():
            if r.per_frame is not None:
                np.save(os.path.join(out_dir, f"diag_{name}_per_frame.npy"), r.per_frame)
        print(f"Diagnostics report saved to {out_dir}")


# ---------------------------------------------------------------------------
# Standalone CLI entry point
# ---------------------------------------------------------------------------
#
# Loads the real TFV6 checkpoint + a handful of real frames and runs the full
# D01-D13 suite. Mirrors the TFV6 model-loading block in run_analysis.py
# Step 1. Deferred (function-local) imports here are intentional: they pull in
# `lead.*` (needs pcla_agents/transfuserv6 on sys.path) and CARLA-adjacent
# config, which normal importers of TFV6LRPDiagnostics (e.g. an already-loaded
# LRPTFv6Model instance) do not need.
#
# Usage (from the project root, pcla venv active):
#   python -m ATOMs_Analysis.utils.tfv6_lrp_diagnostics
#   python ATOMs_Analysis/utils/tfv6_lrp_diagnostics.py --n-runs 3 --n-frames 12
if __name__ == "__main__":
    import argparse
    import json

    sys.path.insert(0, str(_PROJECT_ROOT / "pcla_agents" / "transfuserv6"))

    from lead.tfv6.tfv6 import TFv6
    from lead.training.config_training import TrainingConfig

    from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
    from ATOMs_Analysis.detection.baseline_dataset import BaselineDataLoader

    parser = argparse.ArgumentParser(
        description="Run the TFV6 LRP diagnostic suite (D01-D13) against real checkpoint + frames."
    )
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="Directory of run_*.npz frame files "
                             "(default: conf.BASELINE_DATA_DIR/frames).")
    parser.add_argument("--n-runs",    type=int, default=2, help="Number of run files to load.")
    parser.add_argument("--n-frames",  type=int, default=8, help="Number of frames to sample.")
    parser.add_argument("--out-dir",   type=str, default="tfv6_diagnostics_out",
                        help="Directory to save the report + per-frame arrays.")
    parser.add_argument("--uitb", action="store_true",
                        help="Use AlphaBeta(2,1) instead of (1,0) for Conv/FFN "
                             "(retains some negative-activation relevance instead "
                             "of discarding it -- candidate lever against relevance "
                             "attenuation; see design_decisions.md).")
    parser.add_argument("--zero-bias", action="store_true",
                        help="Exclude bias terms from the AlphaBeta denominator for "
                             "Linear layers (zennit zero_params='bias'), matching "
                             "WoR/LBC's composite. No effect on Conv layers.")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"uitb (AlphaBeta 2,1): {args.uitb}")
    print(f"zero_bias:            {args.zero_bias}")

    model_dir = _PROJECT_ROOT / "pcla_agents" / "transfuserv6_pretrained" / "visiononly_resnet34"
    with open(model_dir / "config.json") as f:
        training_config = TrainingConfig(json.load(f))
    model = TFv6(device, training_config)

    ckpt_files = sorted(model_dir.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth checkpoint found in {model_dir}")
    print(f"Loading checkpoint: {ckpt_files[0]}")
    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current_state = model.state_dict()
    for k, v in list(state_dict.items()):
        if k in current_state and current_state[k].shape != v.shape:
            print(f"  Dropping mismatched weight: {k}")
            state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    lrp = LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder,
                       uitb=args.uitb, zero_bias=args.zero_bias, device=device)

    frames_dir = Path(args.frames_dir) if args.frames_dir else Path(conf.BASELINE_DATA_DIR) / "frames"
    print(f"Loading {args.n_runs} run(s) from {frames_dir} ...")
    raw = BaselineDataLoader.load_all_runs(frames_dir, max_runs=args.n_runs)

    n_available = len(raw["cmd"])
    n = min(args.n_frames, n_available)
    idx = np.linspace(0, n_available - 1, n).astype(int)
    testframes = {
        "wide_rgb": raw["wide_rgb"][idx],
        "cmd":      raw["cmd"][idx],
        "speed":    raw["speed"][idx],
    }
    print(f"Using {n}/{n_available} loaded frames.")

    diag   = TFV6LRPDiagnostics(lrp, device=device)
    report = diag.run_all_tests(testframes)
    diag.print_report(report)
    diag.save_report(report, args.out_dir)
