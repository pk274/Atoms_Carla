#!/usr/bin/env python
"""
sweep_mdx_variants.py
---------------------
Compare four MDX ablation variants on the current labeled test set.

Two design choices are crossed independently:

  Feature space:  backbone  — 512-d globally-pooled ResNet34 output (fast)
                  F_c       — 256-d speed_query from PlanningDecoder (slower)

  Action bins:    equal-width — same as MDX v1 / Zhang et al. 2024 default
                  quantile    — bins cover equal mass; better for skewed actions

This yields:
  A) backbone + equal-width  → identical to the existing MDX v1
  B) F_c      + equal-width
  C) backbone + quantile
  D) F_c      + quantile     → MDX v2 as described in atoms_config comments

Caching
-------
Feature arrays are extracted once from the baseline and test sets and saved,
so they can be reused across --n-pca runs without any model inference:

  <BASELINE_DATA_DIR>/mdx_features.npz             backbone features + speed actions (may exist)
  <BASELINE_DATA_DIR>/mdx_fc_features.npz          F_c features + real actions
  <TEST_DATA_DIR>/attention/mdx_test_backbone.npz  test backbone features
  <TEST_DATA_DIR>/attention/mdx_test_fc.npz        test F_c features

Results
-------
Saved to: <RESULTS_DIR>/mdx_ablation/mdx_ablation_pca{N}.json

Usage
-----
    python sweep_mdx_variants.py                       # n_pca=50 (default)
    python sweep_mdx_variants.py --n-pca 30
    python sweep_mdx_variants.py --n-pca 50 --recompute-features
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# This script lives in "helpful scripts/", not at the repo root, so anchor
# on the parent directory rather than this file's own directory.
_ROOT = Path(__file__).resolve().parent.parent
# Repo root, so ATOMs_Analysis is importable.
sys.path.insert(0, str(_ROOT))
# transfuserv6 to sys.path so its internal `lead` package resolves.
sys.path.insert(0, str(_ROOT / "pcla_agents" / "transfuserv6"))

import numpy as np
import torch

from ATOMs_Analysis.atoms_config import ExperimentConfig as conf
from ATOMs_Analysis.detection.baseline_dataset import BaselineDataLoader
from ATOMs_Analysis.detection.dataset import LabeledTestLoader
from ATOMs_Analysis.detection.detectors import MDXDetector, DetectorEvaluator


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--n-pca", type=int, default=50, metavar="K",
        help="Number of PCA components used inside each MDX detector (default: 50).",
    )
    ap.add_argument(
        "--recompute-features", action="store_true",
        help="Re-extract all feature arrays even if cached files exist.",
    )
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Model loading (TFV6 only)
# ---------------------------------------------------------------------------

def _load_model(device: torch.device):
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.tfv6 import TFv6
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model

    TFV6_MODEL_DIR = _ROOT / "pcla_agents/transfuserv6_pretrained/visiononly_resnet34"
    with open(TFV6_MODEL_DIR / "config.json") as fh:
        training_config = TrainingConfig(json.load(fh))

    model = TFv6(device, training_config)
    ckpt_files = sorted(TFV6_MODEL_DIR.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth found in {TFV6_MODEL_DIR}")
    print(f"  Loading checkpoint: {ckpt_files[0]}")
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
        backbone_eval=model.backbone,
        planning_decoder=model.planning_decoder,
        device=device,
    )
    return lrp


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_baseline_backbone(lrp, baseline_cache: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load or extract 512-d backbone features + speed-derived actions for the baseline."""
    legacy_path = Path(conf.BASELINE_DATA_DIR) / "mdx_features.npz"

    if legacy_path.exists():
        print(f"  [backbone] Reusing existing {legacy_path.name}")
        d = np.load(legacy_path)
        return d["features"].astype(np.float64), d["actions"].astype(np.float64)

    print("  [backbone] Extracting baseline backbone features (slow)...")
    runs = BaselineDataLoader.load_all_runs(Path(conf.BASELINE_DATA_DIR) / "frames")
    n = len(runs["frame_idx"])
    feats, acts = [], []
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0:
            print(f"    {i}/{n}  ({i / max(time.time() - t0, 1):.1f} fr/s)")
        wide_t = torch.from_numpy(runs["wide_rgb"][i]).unsqueeze(0)
        feats.append(lrp.get_backbone_features(wide_t))
        spd = float(runs["speed"][i])
        acts.append([0.0, min(spd / 25.0, 1.0), 1.0 if spd < 0.5 else 0.0])

    feats_arr = np.array(feats, dtype=np.float64)
    acts_arr  = np.array(acts,  dtype=np.float64)
    np.savez_compressed(baseline_cache, features=feats_arr.astype(np.float32),
                        actions=acts_arr.astype(np.float32))
    print(f"  Saved -> {baseline_cache}")
    return feats_arr, acts_arr


def _extract_baseline_fc(lrp, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load or extract 256-d F_c features + real steering/throttle/brake for the baseline."""
    if cache_path.exists():
        print(f"  [F_c] Reusing existing {cache_path.name}")
        d = np.load(cache_path)
        return d["features"].astype(np.float64), d["actions"].astype(np.float64)

    print("  [F_c] Extracting baseline F_c features (slow — full PlanningDecoder per frame)...")
    runs = BaselineDataLoader.load_all_runs(Path(conf.BASELINE_DATA_DIR) / "frames")
    n = len(runs["frame_idx"])
    feats, acts = [], []
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0:
            print(f"    {i}/{n}  ({i / max(time.time() - t0, 1):.1f} fr/s)")
        wide_t = torch.from_numpy(runs["wide_rgb"][i]).unsqueeze(0)
        feat, st, th, br = lrp.get_planning_action_and_features(
            wide_t, cmd=int(runs["cmd"][i]), spd=float(runs["speed"][i])
        )
        feats.append(feat)
        acts.append([st, th, br])

    feats_arr = np.array(feats, dtype=np.float64)
    acts_arr  = np.array(acts,  dtype=np.float64)
    np.savez_compressed(cache_path, features=feats_arr.astype(np.float32),
                        actions=acts_arr.astype(np.float32))
    steer_std = acts_arr[:, 0].std()
    print(f"  Saved -> {cache_path}  (steer std={steer_std:.4f})")
    return feats_arr, acts_arr


def _extract_test_backbone(lrp, test_data: dict, cache_path: Path) -> np.ndarray:
    """Load or extract 512-d backbone features for each test frame."""
    if cache_path.exists():
        print(f"  [test backbone] Reusing existing {cache_path.name}")
        return np.load(cache_path)["features"].astype(np.float64)

    print("  [test backbone] Extracting test backbone features...")
    n = len(test_data["frame_idx"])
    feats = []
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0:
            print(f"    {i}/{n}  ({i / max(time.time() - t0, 1):.1f} fr/s)")
        wide_t = torch.from_numpy(test_data["wide_rgb"][i]).unsqueeze(0)
        feats.append(lrp.get_backbone_features(wide_t))

    feats_arr = np.array(feats, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feats_arr.astype(np.float32))
    print(f"  Saved -> {cache_path}")
    return feats_arr


def _extract_test_fc(lrp, test_data: dict, cache_path: Path) -> np.ndarray:
    """Load or extract 256-d F_c features for each test frame."""
    if cache_path.exists():
        print(f"  [test F_c] Reusing existing {cache_path.name}")
        return np.load(cache_path)["features"].astype(np.float64)

    print("  [test F_c] Extracting test F_c features (slow)...")
    n = len(test_data["frame_idx"])
    feats = []
    t0 = time.time()
    for i in range(n):
        if i % 200 == 0:
            print(f"    {i}/{n}  ({i / max(time.time() - t0, 1):.1f} fr/s)")
        wide_t = torch.from_numpy(test_data["wide_rgb"][i]).unsqueeze(0)
        feats.append(lrp.get_fc_features(
            wide_t,
            cmd=int(test_data["cmd"][i]),
            spd=float(test_data["speed"][i]),
        ))

    feats_arr = np.array(feats, dtype=np.float64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=feats_arr.astype(np.float32))
    print(f"  Saved -> {cache_path}")
    return feats_arr


# ---------------------------------------------------------------------------
# Per-perturbation breakdown
# ---------------------------------------------------------------------------

def _per_pert_auc(scores: np.ndarray, test_data: dict, labels: np.ndarray) -> dict:
    """Compute per-perturbation AUC using clean frames as negatives."""
    from sklearn.metrics import roc_auc_score

    perturbations = np.asarray(test_data["perturbation"])
    unique_perts  = [p for p in np.unique(perturbations) if p != "None" and p != ""]
    results = {}
    clean_scores = scores[labels == 0]

    for pert in unique_perts:
        pert_mask = perturbations == pert
        pert_scores = scores[pert_mask]
        combined_scores = np.concatenate([clean_scores, pert_scores])
        combined_labels = np.concatenate([
            np.zeros(len(clean_scores), dtype=int),
            np.ones(len(pert_scores),  dtype=int),
        ])
        try:
            auc = float(roc_auc_score(combined_labels, combined_scores))
        except ValueError:
            auc = float("nan")
        results[pert] = round(auc, 4)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    n_pca = args.n_pca

    if conf.AGENT != "TFV6":
        raise SystemExit("sweep_mdx_variants.py only supports AGENT='TFV6'.")

    print(f"\n{'='*60}")
    print(f"MDX Variant Ablation  —  n_pca={n_pca}")
    print(f"  Baseline : {conf.BASELINE_DATA_DIR}")
    print(f"  Test     : {conf.TEST_DATA_DIR}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    baseline_dir = Path(conf.BASELINE_DATA_DIR)
    test_att_dir = Path(conf.TEST_DATA_DIR) / "attention"
    out_dir      = Path(conf.RESULTS_DIR) / "mdx_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    cache_backbone_baseline = baseline_dir / "mdx_features.npz"           # reuse existing
    cache_fc_baseline       = baseline_dir / "mdx_fc_features.npz"        # new
    cache_backbone_test     = test_att_dir / "mdx_test_backbone.npz"
    cache_fc_test           = test_att_dir / "mdx_test_fc.npz"

    if args.recompute_features:
        for p in [cache_fc_baseline, cache_backbone_test, cache_fc_test]:
            if p.exists():
                p.unlink()
                print(f"  Removed cached {p.name} (--recompute-features)")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print("[Step 1] Loading TFV6 model...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lrp = _load_model(device)
    print()

    # ------------------------------------------------------------------
    # Baseline features
    # ------------------------------------------------------------------
    print("[Step 2] Loading / extracting baseline features...")
    bb_feats, bb_acts = _extract_baseline_backbone(lrp, cache_backbone_baseline)
    fc_feats, fc_acts = _extract_baseline_fc(lrp, cache_fc_baseline)
    print(f"  Backbone: {bb_feats.shape}  F_c: {fc_feats.shape}\n")

    # ------------------------------------------------------------------
    # Test data + features
    # ------------------------------------------------------------------
    print("[Step 3] Loading test set and extracting test features...")
    test_data   = LabeledTestLoader.load()
    test_labels = test_data["label"].astype(np.int32)
    n_test      = len(test_labels)
    print(f"  Test frames: {n_test}  ({int(test_labels.sum())} perturbed, "
          f"{int((test_labels==0).sum())} clean)")

    bb_test = _extract_test_backbone(lrp, test_data, cache_backbone_test)
    fc_test = _extract_test_fc(lrp, test_data, cache_fc_test)
    print()

    # ------------------------------------------------------------------
    # Define the 4 variants
    # ------------------------------------------------------------------
    VARIANTS = [
        {
            "name":        "A_backbone_equal-width",
            "label":       "standard (backbone + equal-width)",
            "feat_train":  bb_feats,
            "act_train":   bb_acts,
            "feat_test":   bb_test,
            "bin_strategy":"equal-width",
        },
        {
            "name":        "B_fc_equal-width",
            "label":       "fc only   (F_c     + equal-width)",
            "feat_train":  fc_feats,
            "act_train":   fc_acts,
            "feat_test":   fc_test,
            "bin_strategy":"equal-width",
        },
        {
            "name":        "C_backbone_quantile",
            "label":       "quantile  (backbone + quantile)",
            "feat_train":  bb_feats,
            "act_train":   bb_acts,
            "feat_test":   bb_test,
            "bin_strategy":"quantile",
        },
        {
            "name":        "D_fc_quantile",
            "label":       "full v2   (F_c     + quantile)  [MDX v2]",
            "feat_train":  fc_feats,
            "act_train":   fc_acts,
            "feat_test":   fc_test,
            "bin_strategy":"quantile",
        },
    ]

    # ------------------------------------------------------------------
    # Fit, score, evaluate
    # ------------------------------------------------------------------
    print(f"[Step 4] Fitting and evaluating all 4 variants  (n_pca={n_pca})...\n")
    evaluator = DetectorEvaluator()
    all_results = []

    for v in VARIANTS:
        print(f"  --- {v['label']} ---")

        det = MDXDetector(n_pca_components=n_pca, bin_strategy=v["bin_strategy"])
        det.fit(v["feat_train"], v["act_train"])

        scores = det.score_batch(v["feat_test"])

        res = evaluator.evaluate(scores, test_labels, detector_name=v["label"])
        res["variant_name"]  = v["name"]
        res["n_pca"]         = n_pca
        res["bin_strategy"]  = v["bin_strategy"]
        res["feat_dim"]      = int(v["feat_train"].shape[1])
        res["per_pert_auc"]  = _per_pert_auc(scores, test_data, test_labels)

        clean_mean = scores[test_labels == 0].mean()
        pert_mean  = scores[test_labels == 1].mean()
        print(f"    AUC={res['auc']:.4f}  Youden-J={res['youden_j']:.4f}  "
              f"clean_mean={clean_mean:.3f}  pert_mean={pert_mean:.3f}\n")

        all_results.append(res)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("=" * 78)
    print(f"  MDX VARIANT COMPARISON   n_pca={n_pca}")
    print("=" * 78)
    header = f"  {'Variant':<40} {'AUC':>6}  {'Youden-J':>8}  {'TPR':>5}  {'FPR':>5}"
    print(header)
    print("  " + "-" * 74)
    for r in sorted(all_results, key=lambda x: -x["auc"]):
        print(
            f"  {r['detector_name']:<40} "
            f"{r['auc']:>6.4f}  "
            f"{r['youden_j']:>8.4f}  "
            f"{r['tpr_at_threshold']:>5.3f}  "
            f"{r['fpr_at_threshold']:>5.3f}"
        )
    print()

    # Per-perturbation breakdown
    all_perts = sorted({p for r in all_results for p in r["per_pert_auc"]})
    if all_perts:
        print(f"  Per-perturbation AUC  (n_pca={n_pca})")
        print("  " + "-" * 74)
        pert_header = f"  {'Variant':<40}" + "".join(f"  {p[:12]:>12}" for p in all_perts)
        print(pert_header)
        for r in sorted(all_results, key=lambda x: -x["auc"]):
            row = f"  {r['detector_name']:<40}"
            for p in all_perts:
                auc_p = r["per_pert_auc"].get(p, float("nan"))
                row += f"  {auc_p:>12.4f}"
            print(row)
        print()

    # ------------------------------------------------------------------
    # Save JSON
    # ------------------------------------------------------------------
    out_path = out_dir / f"mdx_ablation_pca{n_pca}.json"
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
