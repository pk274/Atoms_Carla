#!/usr/bin/env python3
"""
compute_test_chunk.py
---------------------
HPC worker: process a frame-index slice of test_labeled.npz through
LRP + ATOMs and write a partial profile file.

Called by array_test_task.sh — one SLURM array task per chunk of frames.

Usage (standalone test):
    python hpc/compute_test_chunk.py \
        --labeled-file /ptmp/$USER/atoms_test/test_labeled.npz \
        --chunk-start  0 \
        --chunk-end    20 \
        --output       /ptmp/$USER/atoms_test/partials/partial_test_0.npz \
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
    p.add_argument("--labeled-file", required=True, type=Path,
                   help="Path to test_labeled.npz produced by prep_test.py.")
    p.add_argument("--chunk-start",  required=True, type=int,
                   help="First frame index (inclusive) in this chunk.")
    p.add_argument("--chunk-end",    required=True, type=int,
                   help="Last frame index (exclusive) in this chunk.")
    p.add_argument("--output",       required=True, type=Path,
                   help="Output path for the partial profile .npz.")
    p.add_argument("--model-dir",    required=True, type=Path,
                   help="Directory containing config.json and model*.pth for TFV6.")
    return p.parse_args()


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

    data    = np.load(args.labeled_file, allow_pickle=True)
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

    print("[chunk] Loading model...")
    lrp = build_tfv6_lrp(args.model_dir, device)

    from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla
    from ATOMs_Analysis.utils.visualization_carla import TFV6_CLASSES

    atoms = ATOMsCarla(
        lrp_model     = lrp,
        p_relevance   = 0.25,
        default_cmd   = 2,
        mode_analysis = 1,
        use_reduced   = False,
        class_map     = TFV6_CLASSES,
    )
    print(f"[chunk] ATOMs: {atoms.num_classes} classes: "
          f"{', '.join(atoms.class_names[:4])}, ...")

    profiles      = []
    speed_logits  = []   # [N_chunk, 8] — for PEOC scoring (no fitting needed)
    t0 = time.time()

    for i in range(chunk_start, chunk_end):
        wide     = torch.from_numpy(data["wide_rgb"][i:i+1]).float()
        seg_wide = data["seg_red_wide"][i]
        cmd      = int(data["cmd"][i])
        spd      = float(data["speed"][i])

        profile = atoms.process_frame(wide, None, seg_wide, None, cmd=cmd, spd=spd)
        profiles.append(profile)

        # Speed logits for PEOC (cheap: no LRP backward, model already loaded)
        speed_logits.append(lrp.get_speed_logits(wide.float(), cmd=cmd, spd=spd))

        local_i = i - chunk_start + 1
        if local_i % 10 == 0:
            elapsed = time.time() - t0
            fps     = local_i / elapsed
            eta     = (n_chunk - local_i) / max(fps, 1e-6)
            print(f"  {local_i}/{n_chunk}  ({fps:.2f} fr/s, ETA {eta:.0f}s)")

    atoms.reset()

    profiles_arr     = np.stack(profiles,     axis=0).astype(np.float32)  # [N_chunk, C]
    speed_logits_arr = np.stack(speed_logits, axis=0).astype(np.float32)  # [N_chunk, 8]
    print(f"[chunk] profiles shape    : {profiles_arr.shape}")
    print(f"[chunk] speed_logits shape: {speed_logits_arr.shape}")

    np.savez_compressed(
        args.output,
        profiles     = profiles_arr,
        speed_logits = speed_logits_arr,
        chunk_start  = np.array([chunk_start], dtype=np.int32),
        chunk_end    = np.array([chunk_end],   dtype=np.int32),
        class_ids    = np.array(atoms.class_ids,   dtype=np.int32),
        class_names  = np.array(atoms.class_names, dtype=object),
    )

    elapsed = time.time() - t0
    print(f"[chunk] Done. {n_chunk} frames in {elapsed:.1f}s → {args.output}")


if __name__ == "__main__":
    main()
