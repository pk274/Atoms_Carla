"""
run_analysis.py
---------------
Full ATOMs-based anomaly detection analysis pipeline.

Assumes:
  - A clean baseline dataset has already been collected
    (conf.BASELINE_DATA_DIR/frames/run_*.npz)
  - A clean test dataset has already been collected
    (conf.TEST_DATA_DIR/frames/run_*.npz)
  - The pretrained CameraModel weights are available at conf.MODEL_PATH
  - conf is globally importable and exposes the constants referenced below

Steps
-----
  1.  Load model + initialize LRP and ATOMs
  2.  Compute ATOMs attention profiles on the baseline set
  3.  Fit single-Gaussian Mahalanobis detector on baseline profiles
  4.  Select optimal GMM component count (BIC/AIC sweep)
  5.  Fit GMM on baseline profiles
  6.  Visualize baseline: attention bar chart, PCA coloured by run, PCA coloured by cluster
  7.  Apply perturbation mix to clean test set  →  labeled test dataset
  8.  Compute ATOMs attention profiles on the labeled test set
  9.  Score all test profiles with both detectors + action entropy baseline
  10. Evaluate each detector (ROC, AUC, Youden threshold)
  11. Visualize detection results: score distributions, ROC curves, PCA OOD overlay
  12. Save all results as JSON + figures

Adjustable parameters are marked with  # <<< ADJUST
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — must be set before any pyplot import

import json
import time
from pathlib import Path

import numpy as np
import torch

import yaml

from collections import defaultdict

import sys
# Add transfuserv6 to sys.path so its internal `lead` package resolves correctly
# without modifying the agent's own import statements.
sys.path.insert(0, str(Path(__file__).parent / "pcla_agents" / "transfuserv6"))


# ---------------------------------------------------------------------------
# Project imports — adjust paths to match your project layout
# ---------------------------------------------------------------------------
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf   # global config

from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla

if conf.AGENT == "TFV6":
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model
    from lead.training.config_training import TrainingConfig
    from lead.tfv6.tfv6 import TFv6
else:  # WoR
    from pcla_agents.wor.rails.models.main_model import CameraModel
    from pcla_agents.wor.image_agent import ImageAgent
    from ATOMs_Analysis.saliency.lrp_analysis import LRPCameraModel

from ATOMs_Analysis.detection.baseline_dataset import BaselineDataLoader, BaselineComputer
from ATOMs_Analysis.detection.dataset import (
    LabeledTestLoader, PerturbationApplier, PerturbationSpec, PerturbationEntry,
)
from ATOMs_Analysis.detection.detectors import (
    MahalanobisDetector, ActionEntropyDetector, DetectorEvaluator,
)
from ATOMs_Analysis.detection.clustering import GMMClustering

from ATOMs_Analysis.utils.visualization_carla import (plot_bic_aic,
    save_figure, plot_distance_over_time, visualize_comparative_relevance,
    CARLA_CLASSES,
)
from ATOMs_Analysis.utils.distance_computer import DistanceComputer
from ATOMs_Analysis.detection.detectors import MDXDetector


# ---------------------------------------------------------------------------
# Output directory — all figures and result JSONs go here
# ---------------------------------------------------------------------------

OUT_DIR = Path(conf.RESULTS_DIR) / "atoms_analysis"  # <<< ADJUST if needed
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Name of the live perturbation being analysed — drives all output paths below.
LIVE_PERT_NAME = conf.PERTURBATION   # e.g. "phantom_obstacle"

ATT_DIR = Path(conf.TEST_DATA_DIR) / "attention" / "live_pert" / LIVE_PERT_NAME
ATT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"ATOMs Analysis Pipeline")
print(f"Results → {OUT_DIR}")
print(f"{'='*60}\n")



# ===========================================================================
# STEP 1 — Load model and initialize LRP / ATOMs
# ===========================================================================
print("[Step 1] Loading model and initializing LRP / ATOMs...")

if conf.AGENT == "TFV6":
    TFV6_MODEL_DIR = Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34")
    with open(TFV6_MODEL_DIR / "config.json") as f:
        training_config = TrainingConfig(json.load(f))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TFv6(device, training_config)
    ckpt_files = sorted(TFV6_MODEL_DIR.glob("model*.pth"))
    state_dict = torch.load(ckpt_files[0], map_location=device, weights_only=True)
    current_state = model.state_dict()
    drop_keys = [k for k, v in state_dict.items()
                 if k in current_state and current_state[k].shape != v.shape]
    for k in drop_keys:
        state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    lrp = LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder, device=device)
else:  # WoR
    WOR_WEIGHTS_DIR = Path("pcla_agents/wor_pretrained/leaderboard_weights")
    with open(WOR_WEIGHTS_DIR / "config_leaderboard.yaml") as f:
        config = yaml.safe_load(f)
    model = CameraModel(config)
    model.load_state_dict(torch.load(WOR_WEIGHTS_DIR / "main_model_10.th", map_location="cpu"))
    model.eval()
    lrp = LRPCameraModel(model_eval=model, uitb=False)

# WoR has discrete steer×throt×brake action logits → PEOC via get_action_logits().
# TFV6 uses speed logits (get_speed_logits) handled separately in run_analysis.py.
action_logits_available = (conf.AGENT != "TFV6")

# Initialize ATOMs.
#
# use_reduced=True tracks only 7 driving-relevant classes instead of all 23.
# Good for quick experiments; set False for the full analysis.
atoms = ATOMsCarla(
    lrp_model     = lrp,
    p_relevance   = conf.FC_RELEVANCE_FILTER,   # <<< typically 0.9 (90% mass filter)
    default_cmd   = conf.DEFAULT_CMD,   # <<< 3 = FOLLOW_LANE
    mode_analysis = conf.MODE_ANALYSIS,                  # <<< ADJUST: 1 is paper default
    use_reduced   = False,              # <<< ADJUST
)

print(f"  Classes tracked : {atoms.num_classes}  ({', '.join(atoms.class_names[:5])}, ...)")
print(f"  Mode            : {atoms.mode_analysis}")
print()


# ===========================================================================
# STEP 2 — Compute ATOMs profiles on baseline set
# ===========================================================================
# BaselineComputer loads all run files, processes each frame through ATOMsCarla,
# and saves the per-frame series + mean + covariance to baseline.npz.
# If baseline.npz already exists and you just want to reload it, skip this
# block and go straight to loading.
# ---------------------------------------------------------------------------
print("[Step 2] Computing ATOMs on baseline dataset...")

RECOMPUTE_BASELINE = conf.RECOMPUTE_BASELINE  # <<< set False to load cached baseline.npz

baseline_npz = Path(conf.BASELINE_DATA_DIR) / "baseline.npz"

if RECOMPUTE_BASELINE or not baseline_npz.exists():
    computer = BaselineComputer(lrp, atoms)
    computer.compute_and_save(
        cmd_filter = None,          # <<< set to an int to build a command-specific baseline
        max_runs   = None,          # <<< set to e.g. 5 for a quick smoke test
    )
else:
    print(f"  Skipping recompute — loading cached {baseline_npz}")

# Load the computed baseline
baseline_data    = np.load(baseline_npz, allow_pickle=True)
baseline_series  = baseline_data["series"].astype(np.float64)   # [N, C]
baseline_mean    = baseline_data["mean"].astype(np.float64)      # [C]
baseline_cov     = baseline_data["cov"].astype(np.float64)       # [C, C]


print(f"  Baseline: {baseline_series.shape[0]} frames, {baseline_series.shape[1]} classes")
print()

# ===========================================================================
# STEP 2.5 — Compute MDX baseline
# ===========================================================================
mdx = None
if conf.RECOMPUTE_MDX_BASELINE:
    loader = BaselineDataLoader()
    runs_dict = loader.load_all_runs(conf.BASELINE_DATA_DIR / "frames")
    features_list, actions_list = [], []

    if conf.AGENT == "WOR":
        configPath = "C:/Users/paulk/Desktop/Unistuff/Masterarbeit/Code/PCLA/pcla_agents/wor_pretrained/leaderboard_weights/config_leaderboard.yaml"
        agent = ImageAgent(configPath)

    for i in range(len(runs_dict["frame_idx"])):
        if i % 20 == 0:
            print("Extracting MDX information from frame", i)

        if conf.AGENT == "TFV6":
            wide_t = torch.from_numpy(runs_dict["wide_rgb"][i]).unsqueeze(0)
            features_list.append(lrp.get_backbone_features(wide_t))
            spd = float(runs_dict["speed"][i])
            actions_list.append([0.0, min(spd / 25.0, 1.0), 1.0 if spd < 0.5 else 0.0])
        else:  # WOR
            steer_l, throt_l, brake_l = model.policy(torch.from_numpy(runs_dict["wide_rgb"][i]),
                                                      torch.from_numpy(runs_dict["narr_rgb"][i]),
                                                      runs_dict["cmd"][i])
            features = model.get_features(torch.from_numpy(runs_dict["wide_rgb"][i]),
                                          torch.from_numpy(runs_dict["narr_rgb"][i]),
                                          runs_dict["speed"][i])
            features_list.append(features[0].cpu().detach().numpy())
            steer_logit = agent._lerp(steer_l, runs_dict["speed"][i])
            throt_logit = agent._lerp(throt_l, runs_dict["speed"][i])
            brake_logit = agent._lerp(brake_l, runs_dict["speed"][i])
            action_prob = agent.action_prob(steer_logit, throt_logit, brake_logit)
            brake_prob  = float(action_prob[-1])
            steer       = float(agent.steers @ torch.softmax(steer_logit, dim=0))
            throt       = float(agent.throts @ torch.softmax(throt_logit, dim=0))
            actions_list.append([steer, throt, brake_prob])

    mdx = MDXDetector(n_pca_components=50)
    print("\nFitting MDX Detector!\n")
    features_list_np = np.array(features_list)
    actions_list_np = np.array(actions_list)
    mdx.fit(features_list_np, actions_list_np)
    mdx.save(conf.BASELINE_DATA_DIR / "mdx_parameters")
else:
    _pkl_path = Path(conf.BASELINE_DATA_DIR) / "mdx_parameters.pkl"
    _npz_path = Path(conf.BASELINE_DATA_DIR) / "mdx_features.npz"
    if _pkl_path.exists():
        mdx = MDXDetector()
        mdx.load(conf.BASELINE_DATA_DIR / "mdx_parameters")
    elif _npz_path.exists():
        print(f"  mdx_parameters.pkl not found — fitting from {_npz_path}")
        _mdx_data    = np.load(_npz_path)
        features_arr = _mdx_data["features"].astype(np.float64)
        actions_arr  = _mdx_data["actions"].astype(np.float64)
        print(f"  features: {features_arr.shape}  actions: {actions_arr.shape}")
        mdx = MDXDetector(n_pca_components=50)
        mdx.fit(features_arr, actions_arr)
        mdx.save(conf.BASELINE_DATA_DIR / "mdx_parameters")
        print("  Saved fitted MDX → mdx_parameters.pkl")
    else:
        raise FileNotFoundError(
            "Neither mdx_parameters.pkl nor mdx_features.npz found in "
            f"{conf.BASELINE_DATA_DIR}. Set RECOMPUTE_MDX_BASELINE=True to recompute."
        )


# ===========================================================================
# STEP 3 — Fit single-Gaussian Mahalanobis detector
# ===========================================================================
# This is the primary detector: Mahalanobis distance from the baseline mean
# in the full attention-profile space.
# ---------------------------------------------------------------------------
print("[Step 3] Fitting single-Gaussian Mahalanobis detector...")

mahal_detector = MahalanobisDetector(
    ridge = conf.MAHAL_RIDGE,   # <<< 1e-6 default; increase to 1e-4 if unstable
)
mahal_detector.fit(baseline_series)

# Fit the in-distribution threshold at the 99th percentile of baseline scores.
# This means ~1% of clean frames will be false-positived at runtime.
# Lower percentile = more sensitive, more false positives.
mahal_threshold = mahal_detector.fit_threshold(
    baseline_data  = baseline_series,
    percentile     = 99.0,          # <<< ADJUST: 95 more sensitive, 99.5 more specific
)
mahal_detector.save(OUT_DIR / "mahal_detector.npz")
print(f"  Threshold (p=99): {mahal_threshold:.4f}")
print()


# ===========================================================================
# STEP 4 — GMM model selection: sweep K, pick best by BIC
# ===========================================================================
# We check K=1..MAX_K and pick the K that minimises BIC.
# AIC tends to favour more components; BIC is more conservative.
# Both are plotted so you can make an informed choice.
# ---------------------------------------------------------------------------
print("[Step 4] GMM model selection (BIC/AIC sweep)...")

MAX_K = conf.GMM_MAX_K   # <<< e.g. 8 — increase if you expect many driving modes

best_k_bic, scores_bic = GMMClustering.select_n_components(
    data            = baseline_series,
    max_components  = MAX_K,
    criterion       = "bic",
    covariance_type = conf.GMM_COV_TYPE,   # <<< "full" or "diag"
)
_, scores_aic = GMMClustering.select_n_components(
    data            = baseline_series,
    max_components  = MAX_K,
    criterion       = "aic",
    covariance_type = conf.GMM_COV_TYPE,
)

fig_bic = plot_bic_aic(scores_bic, scores_aic)
save_figure(fig_bic, OUT_DIR / "gmm_model_selection.png")

# You can override the auto-selected K here if the sweep result looks wrong.
N_COMPONENTS = best_k_bic   # <<< ADJUST: override if needed, e.g. N_COMPONENTS = 4
#N_COMPONENTS = 3
print(f"  Selected K = {N_COMPONENTS}")
print()


# ===========================================================================
# STEP 5 — Fit GMM
# ===========================================================================
print(f"[Step 5] Fitting GMM with K={N_COMPONENTS}...")

gmm = GMMClustering(
    n_components    = N_COMPONENTS,
    covariance_type = conf.GMM_COV_TYPE,   # <<< "full" recommended if N >> C^2
    random_state    = conf.RANDOM_SEED,    # <<< for reproducibility
    ridge           = conf.MAHAL_RIDGE,
)
gmm.fit(baseline_series)
gmm.save(OUT_DIR / "gmm.npz")

# Assign each baseline frame to its most probable cluster
baseline_cluster_labels = gmm.predict_batch(baseline_series)

# Log cluster sizes — very unequal sizes may indicate that K is too large
# or that one cluster dominates (e.g. mostly straight driving)
unique, counts = np.unique(baseline_cluster_labels, return_counts=True)
print("  Cluster sizes:")
for k, cnt in zip(unique, counts):
    print(f"    Cluster {k}: {cnt} frames ({cnt/len(baseline_cluster_labels)*100:.1f}%)")
print()



# ===========================================================================
# STEP 7 — Load live perturbation test data
# ===========================================================================
# ---------------------------------------------------------------------------

# Load and summarise
test_data = LabeledTestLoader.load_live_pert(LIVE_PERT_NAME)


# ===========================================================================
# STEP 8 — Compute ATOMs profiles on test set
# ===========================================================================
# We process each test frame through ATOMsCarla to get attention profiles,
# then collect them alongside their ground-truth labels for evaluation.
# ---------------------------------------------------------------------------
from ATOMs_Analysis.utils.visualization_carla import visualize_relevance, visualize_comparative_relevance
REL_DIR = Path(conf.TEST_DATA_DIR) / "relevance_live_pert" / LIVE_PERT_NAME
REL_DIR.mkdir(parents=True, exist_ok=True)

has_narr_test     = test_data.get("narr_rgb")     is not None
has_seg_narr_test = test_data.get("seg_red_narr") is not None

if conf.RECOMPUTE_TEST_ATOMS:
    print("[Step 8] Computing ATOMs on test set...")
    atoms.reset()
    n_test          = test_data["wide_rgb"].shape[0]
    test_profiles   = np.zeros((n_test, atoms.num_classes), dtype=np.float64)
    test_logits_all = [] if action_logits_available else None
    t0 = time.time()
    for i in range(n_test):
        wide     = torch.from_numpy(test_data["wide_rgb"][i:i+1]).float()
        narr     = torch.from_numpy(test_data["narr_rgb"][i:i+1]).float() if has_narr_test else None
        seg_wide = test_data["seg_red_wide"][i]
        seg_narr = test_data["seg_red_narr"][i] if has_seg_narr_test else None
        cmd      = int(test_data["cmd"][i])
        spd      = float(test_data["speed"][i])

        profile = atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd, spd=spd)
        test_profiles[i] = profile

        savepath_rel_w = REL_DIR / f"relevance_wide_{i}"
        rgb_wide = wide[0].permute(1, 2, 0).cpu().detach().numpy()

        # Normal map (default seed — what feeds _hierarchical)
        default_wide = atoms.saliency_data_wide_default
        if default_wide is None:
            default_wide = atoms.saliency_data_wide_brake if atoms._last_is_brake else atoms.saliency_data_wide_drive
        visualize_relevance(default_wide, rgb_image=rgb_wide,
                            save_path=savepath_rel_w, is_brake=atoms._last_is_brake)

        if conf.PLOT_COMPARATIVE_REL:
            # Brake map (one-hot bin 0 seed)
            if atoms.saliency_data_wide_brake is not None:
                visualize_relevance(atoms.saliency_data_wide_brake, rgb_image=rgb_wide,
                                    save_path=f"{savepath_rel_w}_brake", is_brake=True)
            # Drive map (one-hot best non-brake bin seed)
            if atoms.saliency_data_wide_drive is not None:
                visualize_relevance(atoms.saliency_data_wide_drive, rgb_image=rgb_wide,
                                    save_path=f"{savepath_rel_w}_drive", is_brake=False)
            # Difference map: drive − brake (diverging colormap)
            if atoms.saliency_data_wide_brake is not None and atoms.saliency_data_wide_drive is not None:
                comp_map_wide = atoms.saliency_data_wide_drive - atoms.saliency_data_wide_brake
                vmax_w = comp_map_wide.abs().max().item() + 1e-12
                # For dual-camera agents normalize both maps to a shared scale
                if has_narr_test and narr is not None and atoms.saliency_data_narr_brake is not None:
                    comp_map_narr = atoms.saliency_data_narr_drive - atoms.saliency_data_narr_brake
                    vmax_w = max(vmax_w, comp_map_narr.abs().max().item()) + 1e-12
                visualize_comparative_relevance(comp_map_wide / vmax_w, rgb_image=rgb_wide,
                                                save_path=f"{savepath_rel_w}_comparative",
                                                is_brake=atoms._last_is_brake)
            # Narr maps — dual-camera agents (WoR/LBC) only
            if has_narr_test and narr is not None:
                rgb_narr = narr[0].permute(1, 2, 0).cpu().detach().numpy()
                savepath_rel_n = REL_DIR / f"relevance_narr_{i}"
                if atoms.saliency_data_narr_brake is not None:
                    visualize_relevance(atoms.saliency_data_narr_brake, rgb_image=rgb_narr,
                                        save_path=f"{savepath_rel_n}_brake", is_brake=True)
                if atoms.saliency_data_narr_drive is not None:
                    visualize_relevance(atoms.saliency_data_narr_drive, rgb_image=rgb_narr,
                                        save_path=f"{savepath_rel_n}_drive", is_brake=False)
                if atoms.saliency_data_narr_brake is not None and atoms.saliency_data_narr_drive is not None:
                    visualize_comparative_relevance(comp_map_narr / vmax_w, rgb_image=rgb_narr,
                                                    save_path=f"{savepath_rel_n}_comparative",
                                                    is_brake=atoms._last_is_brake)


        # PEOC: 28-dim joint action logits at the interpolated speed (true π(a|s))
        if action_logits_available:
            test_logits_all.append(lrp.get_action_logits(wide, narr, cmd=cmd, spd=spd))

        if (i + 1) % 100 == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n_test}  ({fps:.1f} fr/s)")

    atoms.reset()
    np.save(ATT_DIR / "live_pert_profiles.npy", test_profiles)
    if action_logits_available:
        test_logits_all = np.array(test_logits_all, dtype=np.float32)
        np.save(ATT_DIR / "live_pert_speed_logits.npy", test_logits_all)
    print(f"  Done. {n_test} frames processed.\n")

else:
    test_data     = LabeledTestLoader.load_live_pert(LIVE_PERT_NAME)
    test_profiles = np.load(ATT_DIR / "live_pert_profiles.npy")
    test_logits_all = (
        np.load(ATT_DIR / "live_pert_speed_logits.npy") if action_logits_available else None
    )
    print(f"  Loaded {len(test_profiles)} test profiles.")


# ===========================================================================
# STEP 9 — Score all test profiles with every detector
# ===========================================================================
print("[Step 9] Scoring test profiles...")

# --- 9a: Single-Gaussian Mahalanobis (ATOMs) ---
# compute_mahalanobis() takes (mu_ref, cov_ref, mu_target) — called once per frame.
scores_mahal_single = np.array([
    DistanceComputer.compute_mahalanobis(
        mu_ref         = baseline_mean,
        cov_ref        = baseline_cov,
        mu_target      = test_profiles[i],
        regularization = conf.MAHAL_RIDGE,
    )
    for i in range(len(test_profiles))
])

scores_euclid_single = np.array([
    DistanceComputer.compute_euclidean(
        mu_ref         = baseline_mean,
        mu_target      = test_profiles[i]
    )
    for i in range(len(test_profiles))
])

scores_knn_single = np.array([
    DistanceComputer.compute_knn_distance(
        reference_samples=baseline_series,
        target_point   = test_profiles[i],
        k              = 10,
        normalize      = True,
    )
    for i in range(len(test_profiles))
])

scores_jsd_single = np.array([
    DistanceComputer.compute_jsd(
        p           = baseline_mean,
        q           = test_profiles[i],
    )
    for i in range(len(test_profiles))
])

# --- 9b: GMM Mahalanobis (nearest cluster) ---
# compute_gmm_distance() takes the full GMM parameters + one target point.
# Returns a DistanceResult; we extract .distance for the score.
# mode="nearest"   → distance to closest cluster centre      [default, recommended]
# mode="weighted"  → probability-weighted average distance   [alternative]
gmm_results = [
    DistanceComputer.compute_gmm_distance(
        means          = gmm.means_,
        covariances    = gmm.covariances_,
        weights        = gmm.weights_,
        mu_target      = test_profiles[i],
        mode           = "nearest",         # <<< ADJUST: "nearest" or "weighted"
        regularization = conf.MAHAL_RIDGE,
    )
    for i in range(len(test_profiles))
]
scores_mahal_gmm    = np.array([r.distance          for r in gmm_results])
nearest_clusters    = np.array([r.nearest_component for r in gmm_results])  # useful for analysis

# --- 9c: Action entropy (WoR only) ---
scores_entropy = None
if test_logits_all is not None:
    entropy_detector = ActionEntropyDetector(from_logits=True, cmd=None)
    scores_entropy   = entropy_detector.score_batch(test_logits_all)

# --- 9d: MDX detection ---
scores_mdx = None
if mdx is not None:
    scores_list = []
    for i in range(len(test_data["frame_idx"])):
        if conf.AGENT == "TFV6":
            wide_t   = torch.from_numpy(test_data["wide_rgb"][i]).unsqueeze(0)
            feat_vec = lrp.get_backbone_features(wide_t)
        else:  # WOR
            features = model.get_features(torch.from_numpy(test_data["wide_rgb"][i]),
                                          torch.from_numpy(test_data["narr_rgb"][i]),
                                          test_data["speed"][i])
            feat_vec = features[0].cpu().detach().numpy()
        scores_list.append(mdx.score(feat_vec))
    scores_mdx = np.array(scores_list)


print()

### ================= PLOT ALL OF THIS =================================================
live_pert_dir = Path(conf.RESULTS_DIR) / "live_perturbation" / LIVE_PERT_NAME
live_pert_dir.mkdir(parents=True, exist_ok=True)

# Determine injection frame for the vertical marker.
# New recordings carry is_perturbed; old files fall back to tick-based estimate.
if "is_perturbed" in test_data and test_data["is_perturbed"].any():
    _injection_frame = int(np.argmax(test_data["is_perturbed"]))
    print(f"  Injection frame from is_perturbed flag: {_injection_frame}")
else:
    _CARLA_HZ = 20
    _injection_frame = int(np.searchsorted(
        test_data["frame_idx"], conf.INJECTION_TIME * _CARLA_HZ
    ))
    print(f"  Injection frame estimated from INJECTION_TIME={conf.INJECTION_TIME}s "
          f"@ {_CARLA_HZ} Hz: {_injection_frame}  (no is_perturbed key in data)")

plot_distance_over_time(scores_mahal_single,  LIVE_PERT_NAME, "mahalanobis_single", live_pert_dir, _injection_frame)
plot_distance_over_time(scores_mahal_gmm,     LIVE_PERT_NAME, "mahalanobis_gmm",    live_pert_dir, _injection_frame)
plot_distance_over_time(scores_euclid_single, LIVE_PERT_NAME, "euclidean",          live_pert_dir, _injection_frame)
plot_distance_over_time(scores_jsd_single,    LIVE_PERT_NAME, "jsd",                live_pert_dir, _injection_frame)
plot_distance_over_time(scores_knn_single,    LIVE_PERT_NAME, "knn",                live_pert_dir, _injection_frame)
if scores_entropy is not None:
    plot_distance_over_time(scores_entropy,   LIVE_PERT_NAME, "PEOC",               live_pert_dir, _injection_frame)
if scores_mdx is not None:
    plot_distance_over_time(scores_mdx,       LIVE_PERT_NAME, "mdx",                live_pert_dir, _injection_frame)


