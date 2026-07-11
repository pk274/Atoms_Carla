"""
plot_saliency_examples.py
--------------------------
Pick a handful of random frames and save their mode-2 LRP saliency maps
(FC->input layer-level relevance, heatmap + overlay on the input image) as
example figures for the thesis.

Two modes:
  - Clean frames  (default): random draws from baseline/test/val frames/.
  - Perturbed frames (--perturbed): draws from the labeled test/val set
    (test_labeled.npz / val_labeled.npz), one batch of --n-samples per
    perturbation type present (or just --perturbation-type if given).
    TFV6 "pgd" frames store clean pixels in the labeled npz (the attack is
    normally deferred to the HPC array job) — this script crafts the PGD
    attack locally so the saved saliency map is genuinely adversarial.

Usage
-----
python plot_saliency_examples.py --n-samples 5 --dataset baseline
python plot_saliency_examples.py --n-samples 3 --dataset test --perturbed
python plot_saliency_examples.py --n-samples 3 --dataset test --perturbed --perturbation-type gaussian_noise
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before any pyplot import

import numpy as np
import torch
import yaml
import os

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (the repo root — this script lives in
# "helpful scripts/", not at the repo root)
parent_dir = os.path.dirname(current_dir)

# Insert the parent directory at the front of the path
sys.path.insert(0, parent_dir)

# Add transfuserv6 to sys.path so its internal `lead` package resolves correctly
# without modifying the agent's own import statements. Must be based on
# parent_dir (repo root), not this script's own directory.
sys.path.insert(0, str(Path(parent_dir) / "pcla_agents" / "transfuserv6"))

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla, extract_target_points
from ATOMs_Analysis.detection.baseline_dataset import BaselineDataLoader
from ATOMs_Analysis.detection.dataset import LabeledTestLoader
from ATOMs_Analysis.utils.visualization_carla import (
    visualize_relevance, visualize_relevance_comparison, CARLA_CLASSES, TFV6_CLASSES,
)

if conf.AGENT == "TFV6":
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.tfv6 import TFv6
else:  # WOR
    from pcla_agents.wor.rails.models.main_model import CameraModel
    from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel


def parse_args():
    p = argparse.ArgumentParser(description="Save example LRP saliency maps (mode 2).")
    p.add_argument("--n-samples", type=int, default=5,
                   help="Number of random frames to plot. In --perturbed mode, this many "
                        "frames are drawn PER perturbation type.")
    p.add_argument("--dataset", choices=["baseline", "test", "val"], default="baseline")
    p.add_argument("--perturbed", action="store_true",
                   help="Draw from the labeled test/val set (test_labeled.npz / "
                        "val_labeled.npz) instead of clean frames/. Requires "
                        "--dataset test or val.")
    p.add_argument("--perturbation-type", type=str, default="all",
                   help="Restrict --perturbed sampling to one perturbation type "
                        "(e.g. gaussian_noise, brightness_scale, camera_loss, pgd). "
                        "Default 'all' samples --n-samples frames from every type present.")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for sample selection; omit for a fresh random draw each run.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Override output directory (default: <RESULTS_DIR>/atoms_analysis/saliency_examples).")
    args = p.parse_args()
    if args.perturbed and args.dataset == "baseline":
        p.error("--perturbed requires --dataset test or val (baseline has no perturbations).")
    return args


def load_lrp_model():
    """
    Mirrors the Step-1 model loading in run_analysis.py.

    Returns (lrp_model, raw_model). raw_model is the un-wrapped TFv6 net for
    TFV6 (needed to craft PGD attacks locally, mirroring hpc/compute_test_chunk.py),
    or None for WOR (WOR PGD frames already carry adversarial pixels in
    test_labeled.npz — no local crafting needed).
    """
    if conf.AGENT == "TFV6":
        model_dir = parent_dir / Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34")
        with open(model_dir / "config.json") as f:
            training_config = TrainingConfig(json.load(f))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = TFv6(device, training_config)
        ckpt_files = sorted(model_dir.glob("model*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No model*.pth checkpoint found in {model_dir}")
        state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
        current_state = model.state_dict()
        drop_keys = [k for k, v in state_dict.items()
                     if k in current_state and current_state[k].shape != v.shape]
        for k in drop_keys:
            state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        lrp = LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder, device=device)
        return lrp, model
    else:  # WOR
        weights_dir = Path("pcla_agents/wor_pretrained/leaderboard_weights")
        with open(weights_dir / "config_leaderboard.yaml") as f:
            config = yaml.safe_load(f)
        model = CameraModel(config)
        model.load_state_dict(torch.load(weights_dir / "main_model_10.th", map_location="cpu"))
        model.eval()
        return LRPCameraModel(model_eval=model, uitb=False), None


def craft_pgd_frame(pm, raw_model, wide, cmd, spd, target_points, epsilon, device):
    """
    Craft a TFV6 PGD attack locally (mirrors the pgd_enabled branch in
    hpc/compute_test_chunk.py). Only used because test_labeled.npz stores
    CLEAN pixels for TFV6 "pgd" frames — the attack is normally deferred to
    the HPC array job, but for a handful of example figures it's cheap
    enough to craft it here.
    """
    from ATOMs_Analysis.saliency.lrp_transfuser import _make_minimal_data

    pgd_data = {**_make_minimal_data(spd, device, cmd=cmd, target_points=target_points),
                "rgb": wide.to(device)}
    if hasattr(raw_model, "radar_detector"):
        n_pts = raw_model.config.num_radar_sensors * raw_model.config.num_radar_points_per_sensor
        pgd_data["radar"] = torch.zeros(1, n_pts, 5, dtype=torch.float32, device=device)
    adv_rgb = pm.pgd_attack_tfv6(
        nets    = [raw_model],
        data    = pgd_data,
        target  = conf.PGD_TARGET,
        epsilon = epsilon,
        n_steps = conf.PGD_N_STEPS,
    )
    return adv_rgb.detach().cpu().float()


def _save_saliency(atoms, wide, out_dir, stem, i, n_total, run_id, frame_idx, cmd, spd):
    """Runs ATOMs on a single already-conditioned frame and saves its saliency map(s)."""
    rgb_wide = wide[0].permute(1, 2, 0).cpu().detach().numpy()
    save_path = parent_dir / out_dir / f"{stem}.png"
    visualize_relevance(
        atoms.saliency_data_wide_default,
        rgb_image = rgb_wide,
        save_path = save_path,
        is_brake  = atoms._last_is_brake,
    )
    if conf.ADD_BRAKE_SEEDS and atoms.saliency_data_wide_brake is not None:
        visualize_relevance(
            atoms.saliency_data_wide_brake,
            rgb_image = rgb_wide,
            save_path = out_dir / f"{stem}_brake.png",
            is_brake  = True,
        )
    if conf.ADD_WAYPOINT_SEEDS and atoms.saliency_data_wide_wp is not None:
        visualize_relevance(
            atoms.saliency_data_wide_wp,
            rgb_image = rgb_wide,
            save_path = out_dir / f"{stem}_wp.png",
            is_brake  = False,
        )
    print(f"  [{i + 1}/{n_total}] run={run_id} frame={frame_idx} cmd={cmd} speed={spd:.1f} -> {save_path}")


def _save_saliency_comparison(pert_maps, wide_pert, atoms, wide_clean, out_dir, stem,
                               i, n_total, run_id, frame_idx, cmd, spd):
    """
    Like _save_saliency, but for perturbed frames with a clean counterpart:
    saves a two-row figure (perturbed on top, clean below) per saliency map so
    the attention shift caused by the perturbation is visible directly.

    pert_maps: dict captured from the perturbed pass BEFORE `atoms` was reset
               and re-run on the clean frame (atoms' own tensors get overwritten
               by that second process_frame call). Keys: "default", "is_brake",
               and optionally "brake" / "wp".
    atoms:     the ATOMsCarla instance, holding the CLEAN frame's just-computed
               results at call time.
    """
    rgb_pert  = wide_pert[0].permute(1, 2, 0).cpu().detach().numpy()
    rgb_clean = wide_clean[0].permute(1, 2, 0).cpu().detach().numpy()
    save_path = parent_dir / out_dir / f"{stem}.png"

    visualize_relevance_comparison(
        pert_maps["default"], rgb_pert,
        atoms.saliency_data_wide_default, rgb_clean,
        save_path       = save_path,
        is_brake_top    = pert_maps["is_brake"],
        is_brake_bottom = atoms._last_is_brake,
    )
    if conf.ADD_BRAKE_SEEDS and pert_maps.get("brake") is not None and atoms.saliency_data_wide_brake is not None:
        visualize_relevance_comparison(
            pert_maps["brake"], rgb_pert,
            atoms.saliency_data_wide_brake, rgb_clean,
            save_path       = parent_dir / out_dir / f"{stem}_brake.png",
            is_brake_top    = True,
            is_brake_bottom = True,
        )
    if conf.ADD_WAYPOINT_SEEDS and pert_maps.get("wp") is not None and atoms.saliency_data_wide_wp is not None:
        visualize_relevance_comparison(
            pert_maps["wp"], rgb_pert,
            atoms.saliency_data_wide_wp, rgb_clean,
            save_path       = parent_dir / out_dir / f"{stem}_wp.png",
            is_brake_top    = False,
            is_brake_bottom = False,
        )
    print(f"  [{i + 1}/{n_total}] run={run_id} frame={frame_idx} cmd={cmd} speed={spd:.1f} -> {save_path}")


def _build_clean_frame_lookup(data_dir):
    """
    Load all raw clean frames for a test/val dataset and index them by
    (run_id, frame_idx), so a perturbed frame from test_labeled.npz /
    val_labeled.npz can be matched back to its pre-perturbation pixels.

    Matches by construction: PerturbationApplier assigns run_id/frame_idx from
    the exact same sorted `frames/*.npz` glob that BaselineDataLoader.load_all_runs
    uses here, so every (run_id, frame_idx) key in the labeled set has a
    corresponding entry in this lookup.
    """
    frames_dir = Path(data_dir) / "frames"
    raw = BaselineDataLoader.load_all_runs(frames_dir)
    lookup = {
        (int(rid), int(fidx)): row
        for row, (rid, fidx) in enumerate(zip(raw["run_id"], raw["frame_idx"]))
    }
    return raw, lookup


def _lookup_clean_frame(clean_raw, clean_lookup, run_id, frame_idx, has_narr):
    """Return (wide_clean, narr_clean) for a (run_id, frame_idx) key, or (None, None) if missing."""
    row = clean_lookup.get((run_id, frame_idx))
    if row is None:
        return None, None
    wide_clean = torch.from_numpy(clean_raw["wide_rgb"][row:row + 1]).float()
    narr_clean = None
    if has_narr and clean_raw.get("narr_rgb") is not None:
        narr_clean = torch.from_numpy(clean_raw["narr_rgb"][row:row + 1]).float()
    return wide_clean, narr_clean


def plot_clean_examples(atoms, args, out_dir):
    data_dir = {
        "baseline": conf.BASELINE_DATA_DIR,
        "test":     conf.TEST_DATA_DIR,
        "val":      conf.VAL_DATA_DIR,
    }[args.dataset]
    frames_dir = Path(data_dir) / "frames"

    runs = BaselineDataLoader.load_all_runs(frames_dir)
    n_total = runs["wide_rgb"].shape[0]

    rng = np.random.default_rng(args.seed)
    n_pick = min(args.n_samples, n_total)
    indices = rng.choice(n_total, size=n_pick, replace=False)

    has_narr = runs.get("narr_rgb") is not None
    has_seg_narr = runs.get("seg_red_narr") is not None

    print(f"Sampling {n_pick} frame(s) from {frames_dir} ({n_total} total frames).\n")

    for rank, idx in enumerate(indices):
        idx = int(idx)
        wide = torch.from_numpy(runs["wide_rgb"][idx:idx + 1]).float()
        narr = torch.from_numpy(runs["narr_rgb"][idx:idx + 1]).float() if has_narr else None
        seg_wide = runs["seg_red_wide"][idx]
        seg_narr = runs["seg_red_narr"][idx] if has_seg_narr else None
        cmd = int(runs["cmd"][idx])
        spd = float(runs["speed"][idx])
        run_id = int(runs["run_id"][idx])
        frame_idx = int(runs["frame_idx"][idx])

        atoms.reset()
        atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd,
                            target_points=extract_target_points(runs, idx))

        stem = f"saliency_{rank}_run{run_id}_frame{frame_idx}"
        _save_saliency(atoms, wide, out_dir, stem, rank, n_pick, run_id, frame_idx, cmd, spd)

    print(f"\nSaved {n_pick} saliency map(s) to {out_dir}")


def plot_perturbed_examples(atoms, args, out_dir, raw_model):
    loader = LabeledTestLoader.load_val if args.dataset == "val" else LabeledTestLoader.load
    data = loader()
    print(LabeledTestLoader.summary(data))
    print()

    labels = np.asarray(data["label"])
    pert_names = np.asarray(data["perturbation"])
    perturbed_mask = labels == 1
    available_types = sorted(set(pert_names[perturbed_mask].tolist()))
    if not available_types:
        raise RuntimeError(f"No perturbed frames found in the labeled {args.dataset} set.")

    if args.perturbation_type == "all":
        types_to_sample = available_types
    else:
        if args.perturbation_type not in available_types:
            raise ValueError(
                f"Perturbation type '{args.perturbation_type}' not present in this dataset. "
                f"Available: {available_types}"
            )
        types_to_sample = [args.perturbation_type]

    rng = np.random.default_rng(args.seed)
    selections = []   # list of (perturbation_type, frame_index)
    for ptype in types_to_sample:
        type_idxs = np.where(pert_names == ptype)[0]
        n_pick = min(args.n_samples, len(type_idxs))
        picked = rng.choice(type_idxs, size=n_pick, replace=False)
        selections.extend((ptype, int(i)) for i in picked)

    has_narr = data.get("narr_rgb") is not None
    has_seg_narr = data.get("seg_red_narr") is not None

    # TFV6 "pgd" frames carry CLEAN pixels in the labeled npz (attack deferred
    # to the HPC array job) — craft the attack locally so the saliency map is
    # actually adversarial. WOR pgd pixels are already baked into the npz.
    needs_pgd_crafting = (
        conf.AGENT == "TFV6" and not has_narr
        and any(ptype == "pgd" for ptype, _ in selections)
    )
    pm = None
    if needs_pgd_crafting:
        if raw_model is None:
            raise RuntimeError("PGD frames selected but no raw TFV6 model was loaded.")
        from ATOMs_Analysis.perturbation_manager import PerturbationManager
        pm = PerturbationManager(verbose=False)
        pm.attack_interval = 1   # craft a fresh delta for every attacked frame

    # Clean (pre-perturbation) counterparts are looked up by (run_id, frame_idx)
    # from the raw frames/ directory, so the saved figure can show how the
    # saliency map shifts under the perturbation.
    data_dir = conf.VAL_DATA_DIR if args.dataset == "val" else conf.TEST_DATA_DIR
    print(f"Loading clean frames from {Path(data_dir) / 'frames'} for before/after comparison...")
    clean_raw, clean_lookup = _build_clean_frame_lookup(data_dir)
    print()

    print(f"Sampling {len(selections)} frame(s) across {len(types_to_sample)} "
          f"perturbation type(s): {types_to_sample}\n")

    n_missing_clean = 0
    n_total = len(selections)
    for rank, (ptype, idx) in enumerate(selections):
        wide = torch.from_numpy(data["wide_rgb"][idx:idx + 1]).float()
        narr = torch.from_numpy(data["narr_rgb"][idx:idx + 1]).float() if has_narr else None
        seg_wide = data["seg_red_wide"][idx]
        seg_narr = data["seg_red_narr"][idx] if has_seg_narr else None
        cmd = int(data["cmd"][idx])
        spd = float(data["speed"][idx])
        run_id = int(data["run_id"][idx]) if "run_id" in data else -1
        frame_idx = int(data["frame_idx"][idx])
        target_points = extract_target_points(data, idx)

        is_local_pgd = (ptype == "pgd" and pm is not None)
        if is_local_pgd:
            # test_labeled.npz still holds CLEAN pixels here (attack deferred to
            # HPC) -- capture that as the clean reference before crafting locally.
            wide_clean, narr_clean = wide.clone(), (narr.clone() if narr is not None else None)
            eps = float(data["intensity"][idx]) if "intensity" in data else 0.0
            if eps <= 0.0:
                eps = conf.PGD_EPSILON
            wide = craft_pgd_frame(pm, raw_model, wide, cmd, spd, target_points,
                                    epsilon=eps, device=raw_model.device)
        else:
            wide_clean, narr_clean = _lookup_clean_frame(clean_raw, clean_lookup, run_id, frame_idx, has_narr)

        atoms.reset()
        atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd,
                            target_points=target_points)

        stem = f"saliency_{ptype}_{rank}_run{run_id}_frame{frame_idx}"

        if wide_clean is None:
            n_missing_clean += 1
            print(f"  [warn] no clean counterpart found for run={run_id} frame={frame_idx}; "
                  "saving perturbed-only.")
            _save_saliency(atoms, wide, out_dir, stem, rank, n_total, run_id, frame_idx, cmd, spd)
            continue

        pert_maps = {
            "default":  atoms.saliency_data_wide_default.clone(),
            "is_brake": atoms._last_is_brake,
        }
        if conf.ADD_BRAKE_SEEDS and atoms.saliency_data_wide_brake is not None:
            pert_maps["brake"] = atoms.saliency_data_wide_brake.clone()
        if conf.ADD_WAYPOINT_SEEDS and atoms.saliency_data_wide_wp is not None:
            pert_maps["wp"] = atoms.saliency_data_wide_wp.clone()

        atoms.reset()
        atoms.process_frame(wide_clean, narr_clean, seg_wide, seg_narr, cmd=cmd, spd=spd,
                            target_points=target_points)

        _save_saliency_comparison(pert_maps, wide, atoms, wide_clean, out_dir, stem,
                                   rank, n_total, run_id, frame_idx, cmd, spd)

    print(f"\nSaved {n_total} saliency map(s) to {out_dir}"
          + (f"  ({n_missing_clean} without a clean counterpart)" if n_missing_clean else ""))


def main():
    args = parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        Path(conf.RESULTS_DIR) / "atoms_analysis" / "saliency_examples"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "perturbed" if args.perturbed else "clean"
    print(f"[plot_saliency_examples] Agent={conf.AGENT}  Dataset={args.dataset}  Mode={mode}  Out={out_dir}")

    lrp, raw_model = load_lrp_model()

    class_map = TFV6_CLASSES if conf.AGENT == "TFV6" else CARLA_CLASSES
    atoms = ATOMsCarla(
        lrp_model     = lrp,
        p_relevance   = conf.FC_RELEVANCE_FILTER,
        default_cmd   = conf.DEFAULT_CMD,
        mode_analysis = 2,          # layer-level FC -> input, as requested
        use_reduced   = False,
        class_map     = class_map,
    )

    if args.perturbed:
        plot_perturbed_examples(atoms, args, out_dir, raw_model)
    else:
        plot_clean_examples(atoms, args, out_dir)


if __name__ == "__main__":
    main()
