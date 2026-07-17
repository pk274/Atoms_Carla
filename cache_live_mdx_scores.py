#!/usr/bin/env python
"""
cache_live_mdx_scores.py — precompute MDX scores for the live-perturbation runs.

run_online_analysis.py computes MDX scores on the fly (one backbone forward
per frame), so unlike the ATOMs profiles and speed logits they are never
cached.  make_thesis_figures.py needs them for the live-score comparison
figures without dragging torch/timm into the figure script — this utility
computes them once and stores them next to the other cached live arrays:

    <TEST_DATA_DIR>/attention/live_pert/<pert>/
        live_pert_mdx_scores_<variant>.npy          # perturbed run
        live_pert_mdx_scores_<variant>_clean.npy    # clean-RGB counterpart

(No MODE_ANALYSIS suffix — MDX operates on backbone features and is
independent of the ATOMs mode.)

Feature extraction and scoring mirror run_online_analysis.py exactly:
LRPTFv6Model.get_backbone_features per frame, MDXDetector loaded from
<BASELINE_DATA_DIR>/mdx_parameters.

Run with the conda `PCLA` env (needs torch + timm):

    PYTHONUTF8=1 <PCLA python> cache_live_mdx_scores.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / "pcla_agents" / "transfuserv6"))

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.detection.detectors import MDXDetector
from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model
from lead.training.config_training import TrainingConfig
from lead.tfv6.tfv6 import TFv6

# (perturbation, variant) pairs to cache — the thesis live-score figures.
TARGETS = [
    ("brightness_scale", "20260623_3front_000"),
    ("gaussian_noise",   "20260622_224036_000"),
    ("pgd",              "20260630_weak_000"),
]


def load_model_and_lrp():
    model_dir = Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34")
    with open(model_dir / "config.json") as f:
        training_config = TrainingConfig(json.load(f))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TFv6(device, training_config)
    ckpt = sorted(model_dir.glob("model*.pth"))[0]
    state_dict = torch.load(ckpt, map_location=device, weights_only=True)
    current = model.state_dict()
    for k in [k for k, v in state_dict.items()
              if k in current and current[k].shape != v.shape]:
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return LRPTFv6Model(backbone_eval=model.backbone,
                        planning_decoder=model.planning_decoder, device=device)


def score_frames(lrp, mdx, wide_rgb: np.ndarray) -> np.ndarray:
    scores = np.empty(len(wide_rgb))
    for i in range(len(wide_rgb)):
        wide_t = torch.from_numpy(wide_rgb[i]).unsqueeze(0)
        scores[i] = mdx.score(lrp.get_backbone_features(wide_t))
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(wide_rgb)}")
    return scores


def main() -> None:
    lrp = load_model_and_lrp()
    mdx = MDXDetector()
    mdx.load(Path(conf.BASELINE_DATA_DIR) / "mdx_parameters")

    frames_dir = Path(conf.TEST_DATA_DIR) / "live_pert_frames"
    att_dir = Path(conf.TEST_DATA_DIR) / "attention" / "live_pert"

    for pert, variant in TARGETS:
        out_dir = att_dir / pert
        out_pert = out_dir / f"live_pert_mdx_scores_{variant}.npy"
        out_clean = out_dir / f"live_pert_mdx_scores_{variant}_clean.npy"
        if out_pert.exists() and out_clean.exists():
            print(f"[{pert}/{variant}] cached — skipping")
            continue

        run_npz = frames_dir / f"run_{pert}_live_pert_{variant}.npz"
        clean_npz = frames_dir / f"run_{pert}_live_pert_{variant}_clean_rgb.npz"
        test_data = np.load(run_npz)
        clean_data = np.load(clean_npz)

        print(f"[{pert}/{variant}] perturbed frames ...")
        np.save(out_pert, score_frames(lrp, mdx, test_data["wide_rgb"]))
        print(f"[{pert}/{variant}] clean frames ...")
        np.save(out_clean, score_frames(lrp, mdx, clean_data["wide_rgb"]))
        print(f"  -> {out_pert.name}, {out_clean.name}")

    print("Done.")


if __name__ == "__main__":
    main()
