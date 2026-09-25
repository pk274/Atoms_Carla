"""
export_relevance_example.py
---------------------------
Compute the mode-2 relevance map of one or more reference frames and save,
per frame, everything the thesis figure `relevance_segmentation` needs: the
camera image, its segmentation mask, the relevance map the h-profile is
projected from, and the resulting profile.

Mirrors the reference pipeline exactly (run_analysis.py Step 1 model loading,
BaselineComputer's per-frame call of ATOMsCarla.process_frame), so the saved
profile must match the frame's row in baseline_2.npz. The script prints that
comparison; a mismatch means the local computation is not the one the thesis
reports, and the figure must not be made from it.

Frames are addressed by their row in baseline_2.npz ("series"), i.e. the
position in the name-sorted concatenation of frames/run_*.npz, which is the
order BaselineDataLoader.load_all_runs uses on every platform.

Output: <BASELINE_DATA_DIR>/relevance_examples/relevance_example_row<N>.npz
  rgb            uint8   [H, W, 3]   camera image (6 cameras, 360 degrees)
  seg            uint8   [H, W]      LEAD grouped segmentation (TFV6_CLASSES ids)
  relevance      float32 [H, W]      channel-summed relevance, the map that
                                     _give_element_selectivity projects
  profile        float64 [C]         h-profile returned by process_frame
  profile_stored float64 [C]         the same frame's row of baseline_2.npz
  class_ids, class_names, run_file, frame_in_run, row, cmd, speed, checkpoint

Needs torch + timm (the cluster env, or locally the packages listed in
PCLA's environment.yml). Runs on CPU in about a minute per frame.

Usage
-----
python "helpful scripts/export_relevance_example.py"                 # default row 4808
python "helpful scripts/export_relevance_example.py" --rows 4808 4365 1476
"""

import argparse
import json
import os
import sys
import typing
from pathlib import Path

import numpy as np
import torch

# torch >= 2.x no longer re-exports typing.Union, and beartype evaluates the
# stringified hint 'torch.Union[Tensor, None]' in lead's center_net_decoder.
# Restoring the alias is a no-op on the cluster's older torch.
if not hasattr(torch, "Union"):
    torch.Union = typing.Union

# TFv6 builds its timm ResNet with pretrained=True, which fetches ImageNet
# weights from the Hugging Face hub before the checkpoint overwrites them.
# Build it empty instead; load_lrp_model fails if the checkpoint leaves any
# model weight uncovered, so nothing can silently stay random.
import timm
_create_model = timm.create_model
def _create_model_offline(*a, **kw):
    kw["pretrained"] = False
    return _create_model(*a, **kw)
timm.create_model = _create_model_offline

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pcla_agents" / "transfuserv6"))
os.chdir(REPO)   # conf paths and the checkpoint path are repo-relative

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.saliency.atoms_carla import (
    ATOMsCarla, TFV6_CLASSES, extract_target_points,
)
from ATOMs_Analysis.detection.baseline_dataset import BaselineDataLoader
from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model
from lead.training.config_training import TrainingConfig
from lead.tfv6.tfv6 import TFv6

DEFAULT_ROW = 4808   # Town15, clear day: several vehicles, a traffic light, lane lines


def load_lrp_model():
    """run_analysis.py Step 1, TFV6 branch."""
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
    result = model.load_state_dict(state_dict, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"checkpoint leaves {len(result.missing_keys)} weights "
                           f"uncovered, e.g. {result.missing_keys[:3]}")
    model.eval()
    lrp = LRPTFv6Model(backbone_eval=model.backbone,
                       planning_decoder=model.planning_decoder, device=device)
    return lrp, ckpt.name


def locate(row: int, files: list) -> tuple:
    """Map a baseline_2.npz row to (run file, frame index inside it)."""
    counts = np.array([np.load(f)["frame_idx"].shape[0] for f in files])
    bounds = np.concatenate([[0], np.cumsum(counts)])
    fi = int(np.searchsorted(bounds, row, side="right") - 1)
    return files[fi], row - int(bounds[fi])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--rows", type=int, nargs="+", default=[DEFAULT_ROW],
                    help="rows of baseline_2.npz to export")
    args = ap.parse_args()

    base_dir = Path(conf.BASELINE_DATA_DIR)
    files = sorted((base_dir / "frames").glob("run_*.npz"), key=lambda p: p.name)
    stored = np.load(base_dir / f"baseline_{conf.MODE_ANALYSIS}.npz",
                     allow_pickle=True)["series"]
    if stored.shape[0] != sum(np.load(f)["frame_idx"].shape[0] for f in files):
        raise RuntimeError("baseline series and frames/ differ in length; "
                           "the row mapping would be wrong")

    lrp, ckpt_name = load_lrp_model()
    atoms = ATOMsCarla(
        lrp_model=lrp,
        p_relevance=conf.FC_RELEVANCE_FILTER,
        default_cmd=conf.DEFAULT_CMD,
        mode_analysis=conf.MODE_ANALYSIS,
        use_reduced=False,
        class_map=TFV6_CLASSES,
    )
    out_dir = base_dir / "relevance_examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    for row in args.rows:
        run_file, j = locate(row, files)
        data = BaselineDataLoader.load_run(run_file)
        wide = torch.from_numpy(data["wide_rgb"][j:j + 1]).float()
        seg = data["seg_red_wide"][j]
        cmd, spd = int(data["cmd"][j]), float(data["speed"][j])

        atoms.reset()
        profile = atoms.process_frame(
            wide, None, seg, None, cmd=cmd, spd=spd,
            target_points=extract_target_points(data, j),
        )
        rel = atoms.saliency_data_wide_default
        if rel.dim() == 4:
            rel = rel.squeeze(0)
        rel = rel.sum(dim=0).detach().cpu().numpy().astype(np.float32)

        diff = float(np.abs(profile - stored[row]).max())
        print(f"row {row}: {run_file.name} frame {j}, cmd {cmd}, speed {spd:.1f}")
        print("  profile (local) ", np.round(profile, 4))
        print("  profile (stored)", np.round(stored[row], 4))
        print(f"  max |difference| {diff:.2e}  "
              f"{'OK' if diff < 1e-3 else 'MISMATCH, do not use for the thesis'}")

        out = out_dir / f"relevance_example_row{row}.npz"
        np.savez_compressed(
            out,
            rgb=np.transpose(data["wide_rgb"][j], (1, 2, 0)),
            seg=seg,
            relevance=rel,
            profile=np.asarray(profile, dtype=np.float64),
            profile_stored=np.asarray(stored[row], dtype=np.float64),
            class_ids=np.array(atoms.class_ids),
            class_names=np.array(atoms.class_names),
            run_file=run_file.name,
            frame_in_run=j,
            row=row,
            cmd=cmd,
            speed=spd,
            checkpoint=ckpt_name,
        )
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
