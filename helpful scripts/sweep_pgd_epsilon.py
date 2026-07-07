#!/usr/bin/env python3
"""
sweep_pgd_epsilon.py
--------------------
Find the ℓ∞ budget ε at which the brake-target PGD attack forces the *deployed*
TFV6 controller to brake on at least a target fraction of frames.

Success criterion (deployment-faithful)
    The open-loop controller (open_loop_inference.py) applies a full brake when
    P(speed bin 0) > brake_threshold, with brake_threshold = 0.9
    (config_open_loop.py).  So a PGD frame "succeeds" when, after the attack,
        softmax(speed_logits)[0] > --brake-threshold        (default 0.9).
    We report the smallest ε whose success rate reaches --success-target
    (default 0.99).

    For reference we also report the strict p_brake ≥ 0.999 rate — this is the
    number stored as `__pgd_success__` in summary.json (run_analysis.py:908),
    which is far harsher than what actually stops the car.

Faithfulness to production
    The attack is crafted with the SAME code path as the HPC test pipeline
    (hpc/compute_test_chunk.py → PerturbationManager.pgd_attack_tfv6), including
    real target-point conditioning and the zero-radar tensor.  Success is scored
    with lrp.get_speed_logits (zero-TP conditioning), exactly matching how
    test_speed_logits_2.npy — and hence __pgd_success__ — is produced.  The ε
    picked here therefore transfers directly to the pipeline.

    Keep --n-steps equal to conf.PGD_N_STEPS; step_size = 2.5·ε/n_steps scales
    with ε (Madry heuristic, set inside pgd_attack_tfv6), so ε is the only
    variable across the sweep.

Usage (Viper GPU node — recommended)
    python sweep_pgd_epsilon.py \
        --model-dir pcla_agents/transfuserv6_pretrained/visiononly_resnet34 \
        --frames-dir data/TFV6/test_data_alt/frames \
        --n-frames 200 --n-steps 5 \
        --epsilons 1 2 4 6 8 12 16 24 \
        --out data/TFV6/results_alt/pgd_epsilon_sweep

Outputs
    <out>.json   full per-ε statistics + the chosen ε
    <out>.csv    one row per ε (easy to paste into the thesis)
    <out>.png    success-rate-vs-ε curves (if matplotlib is available)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# sys.path setup — mirrors hpc/compute_mdx_features.py so this runs on the
# cluster (no CARLA install, incompatible beartype) exactly like the HPC jobs.
# Order matters (each insert(0,...) prepends, so the LAST insert wins):
#   1. hpc/stubs               → stub carla/beartype shadow the missing/broken
#                                cluster installs (must be highest priority).
#   2. project root            → pcla_agents, ATOMs_Analysis importable.
#   3. pcla_agents/transfuserv6 → the agent's internal `lead` package resolves
#                                 (it uses absolute `import lead.…` internally).
# Must precede the TFV6 imports done inside load_model().
# Find the project root by walking up from this file until we hit the directory
# that contains pcla_agents/ and hpc/ — robust to the script living in a
# subfolder (e.g. "helpful scripts/") rather than at the repo root.
_here = Path(__file__).resolve()
_ROOT = next(
    (p for p in _here.parents if (p / "pcla_agents").is_dir() and (p / "hpc").is_dir()),
    _here.parent,
)
sys.path.insert(0, str(_ROOT / "pcla_agents" / "transfuserv6"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "hpc" / "stubs"))

import numpy as np
import torch


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep PGD epsilon to find the smallest budget that makes "
                    ">= success-target of brake-target attacks force the deployed "
                    "TFV6 controller to brake (p_brake > brake-threshold). "
                    "See module docstring for details.")
    p.add_argument("--model-dir", type=Path,
                   default=Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34"),
                   help="TFV6 pretrained model directory (config.json + model*.pth).")
    p.add_argument("--frames-dir", type=Path,
                   default=Path("data/TFV6/test_data_alt/frames"),
                   help="Directory of clean run_*.npz frame files to attack.")
    p.add_argument("--n-frames", type=int, default=200,
                   help="Number of frames to sample for the sweep (evenly spaced).")
    p.add_argument("--epsilons", type=float, nargs="+",
                   default=[1.0, 2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0],
                   help="L-inf epsilon grid (0-255 pixel scale).")
    p.add_argument("--n-steps", type=int, default=5,
                   help="PGD iterations; keep equal to conf.PGD_N_STEPS.")
    p.add_argument("--target", default="brake",
                   choices=["brake", "max_speed", "steer_left", "steer_right"],
                   help="PGD objective (sweep is designed for 'brake').")
    p.add_argument("--brake-threshold", type=float, default=0.9,
                   help="p_brake above which the deployed agent brakes "
                        "(config_open_loop.brake_threshold).")
    p.add_argument("--success-target", type=float, default=0.99,
                   help="Fraction of frames that must succeed to accept an epsilon.")
    p.add_argument("--measure-with-tp", action="store_true",
                   help="Score success under real-TP conditioning instead of the "
                        "zero-TP get_speed_logits path that __pgd_success__ uses. "
                        "Off by default to match the production metric.")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available else cpu).")
    p.add_argument("--seed", type=int, default=17,
                   help="Torch seed for reproducible PGD random starts.")
    p.add_argument("--out", type=Path,
                   default=Path("data/TFV6/results_alt/pgd_epsilon_sweep"),
                   help="Output path stem (.json/.csv/.png appended).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model + data loading (mirrors hpc/compute_test_chunk.build_tfv6_lrp)
# ---------------------------------------------------------------------------

def load_model(model_dir: Path, device: torch.device):
    # Import via the `lead.` namespace (NOT pcla_agents.transfuserv6.lead...) so
    # the class identities match those the TFV6 code and its beartype hints use
    # — the two import paths resolve to distinct class objects otherwise, and
    # beartype rejects the mismatch. Requires the sys.path.insert at module top.
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.tfv6 import TFv6
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model

    with open(model_dir / "config.json") as f:
        training_config = TrainingConfig(json.load(f))

    model = TFv6(device, training_config)
    ckpt_files = sorted(model_dir.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth checkpoint found in {model_dir}")
    print(f"  checkpoint: {ckpt_files[0].name}")

    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current = model.state_dict()
    for k in [k for k, v in state_dict.items()
              if k in current and current[k].shape != v.shape]:
        print(f"  dropping mismatched weight: {k}")
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    lrp = LRPTFv6Model(backbone_eval=model.backbone,
                       planning_decoder=model.planning_decoder,
                       device=device)
    return lrp, model


def load_frames(frames_dir: Path, n_frames: int) -> dict:
    """Concatenate all run_*.npz frames and evenly subsample n_frames."""
    files = sorted(frames_dir.glob("run_*.npz"))
    if not files:
        raise FileNotFoundError(f"No run_*.npz found in {frames_dir}")
    print(f"  frame files: {len(files)}")

    keys = ["wide_rgb", "cmd", "speed",
            "target_point", "target_point_previous", "target_point_next"]
    parts: dict[str, list] = {k: [] for k in keys}
    for fp in files:
        d = np.load(fp, allow_pickle=False)
        m = d["wide_rgb"].shape[0]
        for k in keys:
            if k in d:
                parts[k].append(d[k])
            elif k in ("target_point", "target_point_previous", "target_point_next"):
                parts[k].append(np.zeros((m, 2), dtype=np.float32))
            else:
                raise KeyError(f"{fp.name} missing required key '{k}'")

    frames = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    total = frames["wide_rgb"].shape[0]

    if n_frames < total:
        idx = np.linspace(0, total - 1, n_frames).round().astype(int)
        frames = {k: v[idx] for k, v in frames.items()}
    print(f"  frames: using {frames['wide_rgb'].shape[0]}/{total}")
    return frames


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    print(f"[sweep] device={device}  target={args.target}  n_steps={args.n_steps}")
    print("[sweep] loading model...")
    lrp, model = load_model(args.model_dir, device)

    print("[sweep] loading frames...")
    frames = load_frames(args.frames_dir, args.n_frames)
    n = frames["wide_rgb"].shape[0]

    from ATOMs_Analysis.perturbation_manager import PerturbationManager
    from ATOMs_Analysis.saliency.lrp_transfuser import _make_minimal_data
    from ATOMs_Analysis.saliency.atoms_carla import extract_target_points

    pm = PerturbationManager(verbose=False)
    pm.attack_interval = 1   # craft a fresh δ for every frame (matches compute_test_chunk)

    has_radar = hasattr(model, "radar_detector")
    if has_radar:
        n_pts = model.config.num_radar_sensors * model.config.num_radar_points_per_sensor

    def p_brake_after_attack(wide_u8: np.ndarray, cmd: int, spd: float, tps) -> float:
        wide = torch.from_numpy(wide_u8[None]).float().to(device)   # [1,3,H,W]
        pgd_data = {**_make_minimal_data(spd, device, cmd=cmd, target_points=tps),
                    "rgb": wide}
        if has_radar:
            pgd_data["radar"] = torch.zeros(1, n_pts, 5, dtype=torch.float32, device=device)
        adv = pm.pgd_attack_tfv6(nets=[model], data=pgd_data,
                                 target=args.target, epsilon=eps, n_steps=args.n_steps)
        adv = adv.detach()
        if args.measure_with_tp:
            # Score under the same TP conditioning the attack used.
            data_eval = {**_make_minimal_data(spd, device, cmd=cmd, target_points=tps),
                         "rgb": adv}
            if has_radar:
                data_eval["radar"] = torch.zeros(1, n_pts, 5, dtype=torch.float32, device=device)
            with torch.no_grad():
                logits = model.forward(data_eval).pred_target_speed_distribution
            logits = logits.squeeze(0).float().cpu().numpy()
        else:
            # Zero-TP path — identical to how test_speed_logits_2.npy (and thus
            # __pgd_success__) is produced. This is the default.
            logits = lrp.get_speed_logits(adv.cpu().float(), cmd=cmd, spd=spd)
        e = np.exp(logits - logits.max())
        return float((e / e.sum())[0])

    rows = []
    for eps in args.epsilons:
        t0 = time.time()
        pbrakes = np.empty(n, dtype=np.float64)
        for i in range(n):
            tps = extract_target_points(frames, i)
            pbrakes[i] = p_brake_after_attack(
                frames["wide_rgb"][i], int(frames["cmd"][i]),
                float(frames["speed"][i]), tps)
        succ_deploy = float((pbrakes > args.brake_threshold).mean())
        succ_strict = float((pbrakes >= 0.999).mean())
        row = {
            "epsilon":          float(eps),
            "success_deploy":   succ_deploy,     # p_brake > brake_threshold
            "success_strict":   succ_strict,     # p_brake >= 0.999 (__pgd_success__)
            "p_brake_mean":     float(pbrakes.mean()),
            "p_brake_median":   float(np.median(pbrakes)),
            "n_frames":         n,
        }
        rows.append(row)
        print(f"[sweep] eps={eps:6.1f}  success@{args.brake_threshold:.2f}="
              f"{succ_deploy*100:5.1f}%  success@0.999={succ_strict*100:5.1f}%  "
              f"p_brake med={row['p_brake_median']:.3f}  ({time.time()-t0:.0f}s)")

    # Smallest ε meeting the deployment-threshold success target.
    hits = [r for r in rows if r["success_deploy"] >= args.success_target]
    chosen = min(hits, key=lambda r: r["epsilon"]) if hits else None

    result = {
        "config": {
            "target":           args.target,
            "n_steps":          args.n_steps,
            "n_frames":         n,
            "brake_threshold":  args.brake_threshold,
            "success_target":   args.success_target,
            "measure_with_tp":  args.measure_with_tp,
            "epsilons":         list(args.epsilons),
            "frames_dir":       str(args.frames_dir),
            "model_dir":        str(args.model_dir),
        },
        "rows":     rows,
        "chosen_epsilon": chosen["epsilon"] if chosen else None,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out.with_suffix(".json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(args.out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    if chosen:
        print(f"[sweep] SMALLEST eps with >={args.success_target*100:.0f}% success "
              f"@ p_brake>{args.brake_threshold}: eps = {chosen['epsilon']}")
    else:
        top = max(rows, key=lambda r: r["success_deploy"])
        print(f"[sweep] NO eps in the grid reached {args.success_target*100:.0f}% "
              f"success. Best: eps={top['epsilon']} at {top['success_deploy']*100:.1f}%. "
              f"Extend --epsilons upward.")
    print(f"[sweep] wrote {args.out.with_suffix('.json')} and .csv")

    # Optional plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        eps_x = [r["epsilon"] for r in rows]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(eps_x, [r["success_deploy"]*100 for r in rows],
                "o-", label=f"p_brake > {args.brake_threshold} (deployment)")
        ax.plot(eps_x, [r["success_strict"]*100 for r in rows],
                "s--", color="gray", label="p_brake ≥ 0.999 (__pgd_success__)")
        ax.axhline(args.success_target*100, color="red", lw=1, ls=":",
                   label=f"{args.success_target*100:.0f}% target")
        if chosen:
            ax.axvline(chosen["epsilon"], color="green", lw=1, ls=":",
                       label=f"chosen ε = {chosen['epsilon']}")
        ax.set_xlabel("PGD ε (ℓ∞, 0–255 pixel units)")
        ax.set_ylabel("attack success rate (%)")
        ax.set_title(f"Brake-target PGD success vs ε  (n={n}, {args.n_steps} steps)")
        ax.set_ylim(0, 101)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out.with_suffix(".png"), dpi=150)
        print(f"[sweep] wrote {args.out.with_suffix('.png')}")
    except Exception as e:
        print(f"[sweep] plot skipped ({e})")


if __name__ == "__main__":
    main()
