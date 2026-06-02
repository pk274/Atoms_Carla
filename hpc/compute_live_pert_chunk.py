#!/usr/bin/env python3
"""
compute_live_pert_chunk.py
--------------------------
HPC worker: process a frame-index slice of live_pert_concat.npz through
LRP + ATOMs and write a partial profile file.

Called by array_live_pert_task.sh — one SLURM array task per chunk of frames.
The live-pert data is pre-collected in CARLA (already perturbed); no
perturbation application is needed here.

Usage (standalone test):
    python hpc/compute_live_pert_chunk.py \
        --concat-file  /ptmp/$USER/atoms_live_pert/live_pert_concat.npz \
        --chunk-start  0 \
        --chunk-end    20 \
        --output       /ptmp/$USER/atoms_live_pert/partials/partial_live_pert_0.npz \
        --model-dir    /u/$USER/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34
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
    p.add_argument("--concat-file",  required=True, type=Path,
                   help="Path to live_pert_concat.npz produced by prep_live_pert.py.")
    p.add_argument("--chunk-start",  required=True, type=int,
                   help="First frame index (inclusive) in this chunk.")
    p.add_argument("--chunk-end",    required=True, type=int,
                   help="Last frame index (exclusive) in this chunk.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output path for the partial profile .npz.")
    p.add_argument("--model-dir",    required=True, type=Path,
                   help="Directory containing model weights and config.")
    p.add_argument("--agent",        default="TFV6", choices=["TFV6", "WOR"],
                   help="Agent architecture: TFV6 or WOR.")
    return p.parse_args()


def build_wor_lrp(model_dir: Path, device: torch.device):
    import yaml
    from pcla_agents.wor.rails.models import CameraModel
    from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel

    config_path  = model_dir / "config_leaderboard.yaml"
    weights_path = next(model_dir.glob("main_model*.th"), None)
    if weights_path is None:
        raise FileNotFoundError(f"No main_model*.th found in {model_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model = CameraModel(config).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.eval()
    return LRPCameraModel(model_eval=model, device=device)


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

    data    = np.load(args.concat_file, allow_pickle=True)
    n_total = data["wide_rgb"].shape[0]

    chunk_start = args.chunk_start
    chunk_end   = min(args.chunk_end, n_total)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if chunk_start >= n_total:
        print(f"[chunk] chunk_start={chunk_start} >= n_total={n_total}, skipping.")
        np.savez_compressed(
            args.output,
            profiles    = np.empty((0, 0), dtype=np.float32),
            chunk_start = np.array([chunk_start], dtype=np.int32),
            chunk_end   = np.array([chunk_end],   dtype=np.int32),
            class_ids   = np.array([], dtype=np.int32),
            class_names = np.array([], dtype=object),
        )
        return

    n_chunk = chunk_end - chunk_start
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chunk] device={device}  frames={chunk_start}:{chunk_end}  ({n_chunk} frames)")

    print(f"[chunk] Loading model (agent={args.agent})...")
    if args.agent == "WOR":
        lrp = build_wor_lrp(args.model_dir, device)
    else:
        lrp = build_tfv6_lrp(args.model_dir, device)

    from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla
    from ATOMs_Analysis.utils.visualization_carla import TFV6_CLASSES, CARLA_CLASSES

    class_map = CARLA_CLASSES if args.agent == "WOR" else TFV6_CLASSES
    atoms = ATOMsCarla(
        lrp_model     = lrp,
        p_relevance   = 0.25,
        default_cmd   = 2,
        mode_analysis = 1,
        use_reduced   = False,
        class_map     = class_map,
    )
    print(f"[chunk] ATOMs: {atoms.num_classes} classes: "
          f"{', '.join(atoms.class_names[:4])}, ...")

    has_narr     = "narr_rgb"     in data and data["narr_rgb"]     is not None
    has_seg_narr = "seg_red_narr" in data and data["seg_red_narr"] is not None

    profiles    = []
    peoc_logits = []
    t0 = time.time()

    for i in range(chunk_start, chunk_end):
        wide     = torch.from_numpy(data["wide_rgb"][i:i+1]).float()
        narr     = torch.from_numpy(data["narr_rgb"][i:i+1]).float() if has_narr     else None
        seg_wide = data["seg_red_wide"][i]
        seg_narr = data["seg_red_narr"][i]                           if has_seg_narr else None
        cmd      = int(data["cmd"][i])
        spd      = float(data["speed"][i])

        profile = atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)
        profiles.append(profile)

        if args.agent == "WOR":
            peoc_logits.append(lrp.get_action_logits(wide.float(), narr.float(), cmd, spd))
        else:
            peoc_logits.append(lrp.get_speed_logits(wide.float(), cmd=cmd, spd=spd))

        local_i = i - chunk_start + 1
        if local_i % 10 == 0:
            elapsed = time.time() - t0
            fps     = local_i / elapsed
            eta     = (n_chunk - local_i) / max(fps, 1e-6)
            print(f"  {local_i}/{n_chunk}  ({fps:.2f} fr/s, ETA {eta:.0f}s)")

    atoms.reset()

    profiles_arr    = np.stack(profiles,    axis=0).astype(np.float32)
    peoc_logits_arr = np.stack(peoc_logits, axis=0).astype(np.float32)
    logit_key       = "action_logits" if args.agent == "WOR" else "speed_logits"
    print(f"[chunk] profiles shape : {profiles_arr.shape}")
    print(f"[chunk] {logit_key} shape: {peoc_logits_arr.shape}")

    np.savez_compressed(
        args.output,
        profiles    = profiles_arr,
        chunk_start = np.array([chunk_start], dtype=np.int32),
        chunk_end   = np.array([chunk_end],   dtype=np.int32),
        class_ids   = np.array(atoms.class_ids,   dtype=np.int32),
        class_names = np.array(atoms.class_names, dtype=object),
        **{logit_key: peoc_logits_arr},
    )

    elapsed = time.time() - t0
    print(f"[chunk] Done. {n_chunk} frames in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
