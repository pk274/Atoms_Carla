#!/usr/bin/env python3
"""
compute_mdx_features.py
-----------------------
Standalone script: extract 512-dim backbone features + speed-derived action
proxy from all baseline run_*.npz files and write mdx_features.npz.

Run this when mdx_features.npz is missing from a prior gather (e.g. because
the partials were produced by an older version of compute_baseline_chunk.py
that did not yet include backbone feature extraction).

Runs in a few minutes on CPU — no LRP backward pass, just forward passes
through the ResNet34 backbone.

Usage (on Viper, after activating the venv):
    python hpc/compute_mdx_features.py \
        --frames-dir /ptmp/paulkull/atoms_baseline/frames \
        --output     /ptmp/paulkull/atoms_baseline/partials/mdx_features.npz \
        --model-dir  /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34

Or as a quick SLURM job:
    sbatch --ntasks=1 --cpus-per-task=4 --mem=16000MB --time=00:30:00 \
        --wrap="python hpc/compute_mdx_features.py \
            --frames-dir /ptmp/paulkull/atoms_baseline/frames \
            --output     /ptmp/paulkull/atoms_baseline/partials/mdx_features.npz \
            --model-dir  /u/paulkull/pcla/pcla_agents/transfuserv6_pretrained/visiononly_resnet34"
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", required=True, type=Path,
                   help="Directory containing run_*.npz baseline frame files.")
    p.add_argument("--output",     required=True, type=Path,
                   help="Output path for mdx_features.npz.")
    p.add_argument("--model-dir",  required=True, type=Path,
                   help="Directory containing config.json and model*.pth for TFV6.")
    return p.parse_args()


def build_lrp(model_dir: Path, device: torch.device):
    from pcla_agents.transfuserv6.lead.training.config_training import TrainingConfig
    from pcla_agents.transfuserv6.lead.tfv6.tfv6 import TFv6
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model

    with open(model_dir / "config.json") as f:
        training_config = TrainingConfig(json.load(f))
    model = TFv6(device, training_config)

    ckpt_files = sorted(model_dir.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth found in {model_dir}")
    print(f"  Loading checkpoint: {ckpt_files[0].name}")

    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current    = model.state_dict()
    drop_keys  = [k for k, v in state_dict.items()
                  if k in current and current[k].shape != v.shape]
    for k in drop_keys:
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return LRPTFv6Model(
        backbone_eval    = model.backbone,
        planning_decoder = model.planning_decoder,
        device           = device,
    )


def main() -> None:
    args = parse_args()

    run_files = sorted(args.frames_dir.glob("run_*.npz"))
    if not run_files:
        raise FileNotFoundError(f"No run_*.npz files found in {args.frames_dir}")
    print(f"Found {len(run_files)} run files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading model...")
    lrp = build_lrp(args.model_dir, device)

    features_list = []
    actions_list  = []
    t0 = time.time()
    total = 0

    for run_file in run_files:
        data     = np.load(run_file, allow_pickle=False)
        n_frames = data["wide_rgb"].shape[0]
        print(f"  {run_file.name}: {n_frames} frames")

        for i in range(n_frames):
            wide_t = torch.from_numpy(data["wide_rgb"][i:i+1])
            features_list.append(lrp.get_backbone_features(wide_t))

            spd = float(data["speed"][i])
            actions_list.append([0.0, min(spd / 25.0, 1.0), 1.0 if spd < 0.5 else 0.0])

            total += 1
            if total % 50 == 0:
                elapsed = time.time() - t0
                print(f"    {total} frames  ({total / elapsed:.2f} fr/s)")

    features = np.array(features_list, dtype=np.float32)  # [N, 512]
    actions  = np.array(actions_list,  dtype=np.float32)  # [N, 3]
    print(f"\nTotal: {total} frames  features: {features.shape}  actions: {actions.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, features=features, actions=actions)
    print(f"Saved → {args.output}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")
    print(f"\nNext — copy into the repo and push (run on Viper):")
    print(f"  cp {args.output} /u/$USER/pcla/data/TFV6/baseline_data/mdx_features.npz")
    print(f"  cd /u/$USER/pcla")
    print(f"  git add -f data/TFV6/baseline_data/mdx_features.npz")
    print(f"  git commit -m 'add TFV6 mdx_features.npz from HPC'")
    print(f"  git push")


if __name__ == "__main__":
    main()
