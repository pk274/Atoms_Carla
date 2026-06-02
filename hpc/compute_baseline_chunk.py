#!/usr/bin/env python3
"""
compute_baseline_chunk.py
-------------------------
HPC worker: process one run .npz through LRP + ATOMs and write a partial
attention series (.npz with keys: series, class_ids, class_names).

Called by array_task.sh — one SLURM array task per run file.

Usage (standalone test):
    python hpc/compute_baseline_chunk.py \
        --run-file  data/TFV6/baseline_data/frames/run_001.npz \
        --output    /ptmp/$USER/atoms_baseline/partials/partial_0.npz \
        --model-dir pcla_agents/transfuserv6_pretrained/visiononly_resnet34
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-file",  required=True, type=Path,
                   help="Path to a single run_*.npz baseline frame file.")
    p.add_argument("--output",    required=True, type=Path,
                   help="Output path for the partial .npz (series + class metadata).")
    p.add_argument("--model-dir", required=True, type=Path,
                   help="Directory containing config.json and model*.pth for TFV6.")
    p.add_argument("--agent",     default="TFV6", choices=["TFV6", "WOR"],
                   help="Agent architecture: TFV6 or WOR.")
    return p.parse_args()


def load_run_npz(filepath: Path) -> dict:
    """Load one run .npz; mirrors BaselineDataLoader.load_run."""
    data = np.load(filepath, allow_pickle=False)
    return {
        "wide_rgb":     data["wide_rgb"],
        "narr_rgb":     data["narr_rgb"]     if "narr_rgb"     in data else None,
        "seg_red_wide": data["seg_red_wide"],
        "seg_red_narr": data["seg_red_narr"] if "seg_red_narr" in data else None,
        "cmd":          data["cmd"],
        "speed":        data["speed"],
        "frame_idx":    data["frame_idx"],
    }


def build_wor_lrp(model_dir: Path, device: torch.device):
    import yaml
    from pcla_agents.wor.rails.models import CameraModel
    from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel

    config_path  = model_dir / "config_leaderboard.yaml"
    weights_path = next(model_dir.glob("main_model*.th"), None)
    if weights_path is None:
        raise FileNotFoundError(f"No main_model*.th found in {model_dir}")
    print(f"  Loading WOR config  : {config_path.name}")
    print(f"  Loading WOR weights : {weights_path.name}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = CameraModel(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()

    lrp = LRPCameraModel(model_eval=model, device=device)
    return lrp


def build_tfv6_lrp(model_dir: Path, device: torch.device):
    from pcla_agents.transfuserv6.lead.training.config_training import TrainingConfig
    from pcla_agents.transfuserv6.lead.tfv6.tfv6 import TFv6
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model

    with open(model_dir / "config.json") as f:
        training_config = TrainingConfig(json.load(f))

    model = TFv6(device, training_config)

    ckpt_files = sorted(model_dir.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth checkpoint found in {model_dir}")
    print(f"  Loading checkpoint: {ckpt_files[0].name}")

    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current_state = model.state_dict()
    drop_keys = [k for k, v in state_dict.items()
                 if k in current_state and current_state[k].shape != v.shape]
    for k in drop_keys:
        print(f"  Dropping mismatched weight: {k}")
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    lrp = LRPTFv6Model(
        backbone_eval    = model.backbone,
        planning_decoder = model.planning_decoder,
        device           = device,
    )
    return lrp


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chunk] device={device}  agent={args.agent}  run={args.run_file.name}")

    # --- LRP model ---
    print("[chunk] Loading model...")
    if args.agent == "WOR":
        lrp = build_wor_lrp(args.model_dir, device)
    else:
        lrp = build_tfv6_lrp(args.model_dir, device)

    # --- ATOMs ---
    from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla
    from ATOMs_Analysis.utils.visualization_carla import TFV6_CLASSES, CARLA_CLASSES

    class_map = CARLA_CLASSES if args.agent == "WOR" else TFV6_CLASSES
    atoms = ATOMsCarla(
        lrp_model     = lrp,
        p_relevance   = 0.25,   # FC_RELEVANCE_FILTER
        default_cmd   = 2,      # DEFAULT_CMD (FOLLOW_LANE)
        mode_analysis = 1,      # MODE_ANALYSIS (paper default)
        use_reduced   = False,
        class_map     = class_map,
    )
    print(f"[chunk] ATOMs tracking {atoms.num_classes} classes: "
          f"{', '.join(atoms.class_names[:4])}, ...")

    # --- Load frame data ---
    data = load_run_npz(args.run_file)
    n_frames = data["wide_rgb"].shape[0]
    print(f"[chunk] {n_frames} frames loaded.")

    has_narr     = data["narr_rgb"]     is not None
    has_seg_narr = data["seg_red_narr"] is not None

    # --- Process frames ---
    attention_series    = []
    backbone_feat_list  = []   # [N, D] — for MDX fitting (D=512 TFV6, D=576 WOR)
    mdx_actions_list    = []   # [N, 3] — [steer, throt, brake] proxy
    t0 = time.time()

    for i in range(n_frames):
        wide     = torch.from_numpy(data["wide_rgb"][i:i+1]).float()
        narr     = torch.from_numpy(data["narr_rgb"][i:i+1]).float() if has_narr     else None
        seg_wide = data["seg_red_wide"][i]
        seg_narr = data["seg_red_narr"][i]                           if has_seg_narr else None
        cmd      = int(data["cmd"][i])
        spd      = float(data["speed"][i])

        frame_att = atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)
        attention_series.append(frame_att)

        if args.agent == "WOR":
            # WOR: 576-dim backbone (512 wide + 64 narr bottleneck)
            with torch.no_grad():
                feat = lrp._model_eval.get_features(wide.float().to(device),
                                                    narr.float().to(device))
                backbone_feat_list.append(feat.squeeze(0).cpu().numpy())
            # MDX actions from 28-dim joint π(a|s): true steer/throt/brake distribution
            joint_logits = lrp.get_action_logits(wide.float(), narr.float(), cmd, spd)
            probs = np.exp(joint_logits - joint_logits.max())
            probs /= probs.sum()
            brake_prob   = float(probs[-1])
            drive_probs  = probs[:27].reshape(3, 9)          # [throt=3, steer=9]
            drive_total  = drive_probs.sum() + 1e-9
            steer_marg   = drive_probs.sum(axis=0) / drive_total  # [9]
            throt_marg   = drive_probs.sum(axis=1) / drive_total  # [3]
            steer_val    = float(np.linspace(-1.0, 1.0, 9) @ steer_marg)
            throt_val    = float(np.linspace(0.0,  1.0, 3) @ throt_marg)
            mdx_actions_list.append([steer_val, throt_val, brake_prob])
        else:
            # TFV6: 512-dim globally-pooled backbone features
            backbone_feat_list.append(lrp.get_backbone_features(wide.float()))
            brake_proxy    = 1.0 if spd < 0.5 else 0.0
            throttle_proxy = min(spd / 25.0, 1.0)
            mdx_actions_list.append([0.0, throttle_proxy, brake_proxy])

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            fps     = (i + 1) / elapsed
            eta     = (n_frames - i - 1) / max(fps, 1e-6)
            print(f"  {i+1}/{n_frames}  ({fps:.2f} fr/s, ETA {eta:.0f}s)")

    atoms.reset()

    series           = np.stack(attention_series, axis=0)              # [N, num_classes]
    backbone_features = np.stack(backbone_feat_list, axis=0)           # [N, 512]
    mdx_actions       = np.array(mdx_actions_list, dtype=np.float32)   # [N, 3]
    print(f"[chunk] series shape: {series.shape}")
    print(f"[chunk] backbone_features shape: {backbone_features.shape}")

    # --- Save partial result ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        series             = series.astype(np.float32),
        backbone_features  = backbone_features.astype(np.float32),
        mdx_actions        = mdx_actions,
        class_ids          = np.array(atoms.class_ids,   dtype=np.int32),
        class_names        = np.array(atoms.class_names, dtype=object),
    )

    elapsed = time.time() - t0
    print(f"[chunk] Done. {n_frames} frames in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
