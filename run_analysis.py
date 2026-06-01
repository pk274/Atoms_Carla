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



# ---------------------------------------------------------------------------
# Project imports — adjust paths to match your project layout
# ---------------------------------------------------------------------------
from ATOMs_Analysis.atoms_config import ExperimentConfig as conf   # global config
from ATOMs_Analysis.saliency.atoms_carla import ATOMsCarla

# Agent-specific LRP imports (loaded conditionally in Step 1 to avoid hard deps)
if conf.AGENT == "TFV6":
    from ATOMs_Analysis.saliency.lrp_transfuser import LRPTFv6Model
    from pcla_agents.transfuserv6.lead.training.config_training import TrainingConfig
    from pcla_agents.transfuserv6.lead.tfv6.tfv6 import TFv6
else:  # WOR (default)
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

from ATOMs_Analysis.utils.visualization_carla import (
    plot_pca_baseline, plot_pca_clusters, plot_pca_ood,
    plot_tsne_baseline, plot_tsne_clusters, plot_tsne_ood,
    fit_pca, fit_tsne, fit_tsne_joint,
    get_cluster_colors, make_output_dirs,
    plot_attention_bar, plot_attention_comparison, plot_attention_bars_separate,
    plot_cluster_representative,
    plot_roc, plot_mahal_distribution, plot_bic_aic,
    plot_knn_sensitivity,
    save_figure,
    compute_perturbation_displacement_stats,
    format_displacement_stats_text,
    plot_pca_perturbation_trajectories,
    plot_displacement_coherence_bar,
    plot_displacement_magnitude_boxplot,
    CARLA_CLASSES,
    TFV6_CLASSES,
)
from ATOMs_Analysis.utils.distance_computer import DistanceComputer
from ATOMs_Analysis.detection.detectors import MDXDetector


# ---------------------------------------------------------------------------
# Output directory — all figures and result JSONs go here
# ---------------------------------------------------------------------------

OUT_DIR = Path(conf.RESULTS_DIR) / "atoms_analysis"  # <<< ADJUST if needed
OUT_DIR.mkdir(parents=True, exist_ok=True)
dirs = make_output_dirs(OUT_DIR)

ATT_DIR = Path(conf.TEST_DATA_DIR) / "attention"
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
    # -----------------------------------------------------------------------
    # TFV6: visiononly_resnet34, LTF mode
    # -----------------------------------------------------------------------
    TFV6_MODEL_DIR = Path("pcla_agents/transfuserv6_pretrained/visiononly_resnet34")
    with open(TFV6_MODEL_DIR / "config.json") as f:
        training_config = TrainingConfig(json.load(f))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = TFv6(device, training_config)

    ckpt_files = sorted(TFV6_MODEL_DIR.glob("model*.pth"))
    if not ckpt_files:
        raise FileNotFoundError(f"No model*.pth checkpoint found in {TFV6_MODEL_DIR}")
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

    lrp = LRPTFv6Model(backbone_eval=model.backbone, planning_decoder=model.planning_decoder, device=device)
    # TFV6 has no WoR-compatible discrete action head; WoR-style MDX is disabled.
    # speed_logits_available enables PEOC (Sedlmeier et al. 2020) via the 8-bin
    # speed distribution from target_speed_decoder, and TFV6 MDX via backbone features.
    action_logits_available = False
    speed_logits_available  = True

    # ------------------------------------------------------------------
    # ANSWERED (2026-05-28): SequentialMergeBatchNorm covers all 72 BN
    # layers in TFv6FullModelForLRP.  Canonizer is now part of the
    # composite (see lrp_transfuser.py:_create_composite).  Probe removed.
    # ------------------------------------------------------------------

else:
    # -----------------------------------------------------------------------
    # WoR: CameraModel (default)
    # -----------------------------------------------------------------------
    WOR_WEIGHTS_DIR = Path("pcla_agents/wor_pretrained/leaderboard_weights")
    with open(WOR_WEIGHTS_DIR / "config_leaderboard.yaml", 'r') as f:
        config = yaml.safe_load(f)
    model = CameraModel(config)
    model.load_state_dict(torch.load(
        WOR_WEIGHTS_DIR / "main_model_10.th",
        map_location=torch.device('cpu'),
    ))
    model.eval()

    lrp = LRPCameraModel(
        model_eval = model,
        uitb       = False,    # <<< set True if using UITB-style model
    )
    action_logits_available = True
    speed_logits_available  = False

# Select segmentation class map based on the active agent.
# TFV6/LEAD data uses the 10-class TransFuser grouped scheme (save_grouped_semantic=True).
# WoR uses the raw 23-class CARLA scheme.
_seg_class_map = TFV6_CLASSES if conf.AGENT == "TFV6" else CARLA_CLASSES

# Initialize ATOMs.
#
# use_reduced=True tracks only 7 driving-relevant classes instead of all 23.
# Good for quick experiments; set False for the full analysis.
atoms = ATOMsCarla(
    lrp_model     = lrp,
    p_relevance   = conf.FC_RELEVANCE_FILTER,   # <<< typically 0.9 (90% mass filter)
    default_cmd   = conf.DEFAULT_CMD,   # <<< 3 = FOLLOW_LANE
    mode_analysis = conf.MODE_ANALYSIS, # <<< ADJUST: 1 is paper default
    use_reduced   = False,              # <<< ADJUST
    class_map     = _seg_class_map,
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
# STEP 3 — Compute MDX baseline
# ===========================================================================
# WoR  : uses model.get_features() (penultimate layer) + policy() for action
#         classes, exactly as in Zhang et al. 2024.
# TFV6 : backbone 512-dim features from lrp._attribute_to_backbone().
#         Action proxy: normalised speed → throttle, speed<0.5 m/s → brake,
#         steer=0 (no offline steering signal).  This adapts MDX to the
#         single-stream backbone without requiring a live model forward pass
#         through the full planning decoder.
# ===========================================================================
mdx = None
if action_logits_available:
    # -----------------------------------------------------------------------
    # WoR MDX: penultimate-layer features + policy logits
    # -----------------------------------------------------------------------
    if conf.RECOMPUTE_MDX_BASELINE:
        runs_dict = BaselineDataLoader.load_all_runs(
            Path(conf.BASELINE_DATA_DIR) / "frames"
        )
        WOR_WEIGHTS_DIR = Path("pcla_agents/wor_pretrained/leaderboard_weights")
        agent = ImageAgent(str(WOR_WEIGHTS_DIR / "config_leaderboard.yaml"))
        features_list, actions_list = [], []

        for i in range(len(runs_dict["frame_idx"])):
            if i % 20 == 0:
                print("Extracting MDX information from frame", i)
            wide = torch.from_numpy(runs_dict["wide_rgb"][i])
            narr = torch.from_numpy(runs_dict["narr_rgb"][i])
            steer_l, throt_l, brake_l = model.policy(wide, narr, runs_dict["cmd"][i])
            features = model.get_features(wide, narr, runs_dict["speed"][i])
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
        mdx.fit(np.array(features_list), np.array(actions_list))
        mdx.save(conf.BASELINE_DATA_DIR / "mdx_parameters")
    else:
        mdx = MDXDetector()
        mdx.load(conf.BASELINE_DATA_DIR / "mdx_parameters")

elif conf.AGENT == "TFV6":
    # -----------------------------------------------------------------------
    # TFV6 MDX: backbone 512-dim features + speed-derived action proxy
    # -----------------------------------------------------------------------
    if conf.RECOMPUTE_MDX_BASELINE:
        # Fast path: backbone features were pre-computed on HPC by
        # compute_baseline_chunk.py and gathered into mdx_features.npz.
        # If the file exists, skip the slow per-frame extraction loop.
        mdx_features_path = Path(conf.BASELINE_DATA_DIR) / "mdx_features.npz"
        if mdx_features_path.exists():
            print(f"  Loading pre-computed MDX features from {mdx_features_path}")
            _mdx_data     = np.load(mdx_features_path)
            features_arr  = _mdx_data["features"].astype(np.float64)  # [N, 512]
            actions_arr   = _mdx_data["actions"].astype(np.float64)   # [N, 3]
            print(f"  features: {features_arr.shape}  actions: {actions_arr.shape}")
        else:
            # Slow path: extract backbone features locally frame by frame.
            print("  mdx_features.npz not found — extracting locally (slow).")
            print("  Tip: run compute_baseline_chunk.py + gather_baseline.py on HPC "
                  "to generate mdx_features.npz and avoid this step.")
            runs_dict = BaselineDataLoader.load_all_runs(
                Path(conf.BASELINE_DATA_DIR) / "frames"
            )
            features_list, actions_list = [], []
            n_mdx = len(runs_dict["frame_idx"])

            for i in range(n_mdx):
                if i % 20 == 0:
                    print(f"  Extracting TFV6 MDX backbone features from frame {i}/{n_mdx}")
                wide_t = torch.from_numpy(runs_dict["wide_rgb"][i]).unsqueeze(0)
                features_list.append(lrp.get_backbone_features(wide_t))

                spd = float(runs_dict["speed"][i])
                actions_list.append([0.0, min(spd / 25.0, 1.0), 1.0 if spd < 0.5 else 0.0])

            features_arr = np.array(features_list)
            actions_arr  = np.array(actions_list)

        mdx = MDXDetector(n_pca_components=50)
        print("\nFitting TFV6 MDX Detector!\n")
        mdx.fit(features_arr, actions_arr)
        mdx.save(conf.BASELINE_DATA_DIR / "mdx_parameters")
    else:
        mdx = MDXDetector()
        mdx.load(conf.BASELINE_DATA_DIR / "mdx_parameters")


# ===========================================================================
# STEP 4 — Fit single-Gaussian Mahalanobis detector
# ===========================================================================
# This is the primary detector: Mahalanobis distance from the baseline mean
# in the full attention-profile space.
# ---------------------------------------------------------------------------
print("[Step 4] Fitting single-Gaussian Mahalanobis detector...")

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
# STEP 5 — GMM model selection: sweep K, pick best by BIC
# ===========================================================================
# We check K=1..MAX_K and pick the K that minimises BIC.
# AIC tends to favour more components; BIC is more conservative.
# Both are plotted so you can make an informed choice.
# ---------------------------------------------------------------------------
print("[Step 5] GMM model selection (BIC/AIC sweep)...")

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
save_figure(fig_bic, dirs["clustering"] / "gmm_model_selection.png")

# You can override the auto-selected K here if the sweep result looks wrong.
N_COMPONENTS = best_k_bic   # <<< ADJUST: override if needed, e.g. N_COMPONENTS = 4
N_COMPONENTS = 5
print(f"  Selected K = {N_COMPONENTS}")
print()


# ===========================================================================
# STEP 6 — Fit GMM
# ===========================================================================
print(f"[Step 6] Fitting GMM with K={N_COMPONENTS}...")

gmm = GMMClustering(
    n_components    = N_COMPONENTS,
    covariance_type = conf.GMM_COV_TYPE,   # <<< "full" recommended if N >> C^2
    random_state    = conf.RANDOM_SEED,    # <<< for reproducibility
    ridge           = conf.MAHAL_RIDGE,
)
gmm.fit(baseline_series)
gmm.save(OUT_DIR / "gmm.npz")
cluster_colors = get_cluster_colors(N_COMPONENTS)

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
# STEP 7 — Visualize baseline
# ===========================================================================
print("[Step 7] Visualizing baseline attention profiles...")

# --- 6a: Mean attention bar chart ---
# Shows which semantic classes the agent attends to on average.
# Large bars for Vehicle, RoadLine, Road are expected for a healthy agent.
fig_bar = plot_attention_bar(
    attention   = baseline_mean,
    class_names = atoms.class_names,
    title       = "Baseline Mean Attention (all classes)",
    error       = baseline_series.std(axis=0),   # std as error bars
    top_k       = 15,    # <<< show top 15 classes; set None for all 23
)
save_figure(fig_bar, dirs["attention"] / "baseline_attention_bar.png")

# --- 6a.5: Mean attention per GMM cluster ---
# Compares what each cluster "looks at" on average.
# Distinct profiles confirm the clusters represent genuinely different
# driving situations. Similar profiles suggest K may be too large.
cluster_attention = {}
for k in range(N_COMPONENTS):
    mask = baseline_cluster_labels == k
    if mask.sum() == 0:
        continue
    cluster_mean = baseline_series[mask].mean(axis=0)   # [C]
    cluster_attention[f"Cluster {k}  (n={mask.sum()})"] = cluster_mean

fig_cluster_bar = plot_attention_comparison(
    attention_dict = cluster_attention,
    class_names    = atoms.class_names,
    top_k          = 10,       # <<< show top-10 classes by max attention across clusters
    title          = f"Mean Attention per GMM Cluster (K={N_COMPONENTS})",
    colors         = cluster_colors,   # link bar colors to PCA/t-SNE cluster scatter
)
save_figure(fig_cluster_bar, dirs["attention"] / "attention_by_cluster.png")

# --- 6a.6: Separate attention bar chart per cluster (for collage) ---
# One identically-styled figure per cluster so they can be tiled next to
# the PCA scatter without any post-processing rescaling.
figs_per_cluster = plot_attention_bars_separate(
    attention_dict = cluster_attention,
    class_names    = atoms.class_names,
    top_k          = 10,
    colors         = cluster_colors,
)
for label, fig in figs_per_cluster.items():
    # label is e.g. "Cluster 0  (n=412)" — sanitise for filename
    safe_label = label.split("(")[0].strip().lower().replace(" ", "_")
    save_figure(fig, dirs["attention"] / f"{safe_label}_attention_bar.png")

# --- 6b: Mean attention per navigation command ---
# Re-load the raw run files to get cmd metadata, then group by cmd.
raw_baseline = BaselineDataLoader.load_all_runs(
    Path(conf.BASELINE_DATA_DIR) / "frames"
)
cmd_attention = {}
for cmd_id in np.unique(raw_baseline["cmd"]):
    mask = raw_baseline["cmd"] == cmd_id
    if mask.sum() > 0:
        # We already have the per-frame series in baseline_series but need
        # to align with the raw data order — use the same cmd filter as
        # BaselineComputer would have used (no filter = all cmds)
        cmd_label = f"cmd={cmd_id}"
        cmd_attention[cmd_label] = baseline_series[
            raw_baseline["cmd"][:len(baseline_series)] == cmd_id
        ].mean(axis=0)

if len(cmd_attention) > 1:
    fig_cmd = plot_attention_comparison(
        attention_dict = cmd_attention,
        class_names    = atoms.class_names,
        top_k          = 10,
        title          = "Mean Attention by Navigation Command",
    )
    save_figure(fig_cmd, dirs["attention"] / "attention_by_command.png")

# --- 6b.5: Representative image per GMM cluster (closest to cluster mean) ---
# For each cluster find the frame whose attention vector is nearest (L2) to
# the cluster mean.  raw_baseline["wide_rgb"] is aligned with baseline_series
# (same sorted-file order, no filtering applied to either), so a flat index
# into baseline_series maps directly into raw_baseline.
_n = len(baseline_series)   # guard against any trailing frames in raw files
for k in range(N_COMPONENTS):
    _mask = baseline_cluster_labels == k
    if _mask.sum() == 0:
        continue
    _cluster_mean = baseline_series[_mask].mean(axis=0)
    _dists        = np.linalg.norm(baseline_series[_mask] - _cluster_mean, axis=1)
    _global_idx   = np.where(_mask)[0][np.argmin(_dists)]
    _img_chw      = raw_baseline["wide_rgb"][min(_global_idx, _n - 1)]   # [3,H,W] uint8
    _label        = f"Cluster {k}  (n={_mask.sum()})"
    fig_rep = plot_cluster_representative(_img_chw, title=_label)
    save_figure(fig_rep, dirs["attention"] / f"cluster_{k}_representative.png")

# --- 6c: PCA of baseline coloured by run ---
# Checks whether different collection runs produce similar attention distributions.
# If runs form separate clusters, you may want per-run normalisation.
run_ids = raw_baseline["run_id"][:len(baseline_series)]
fig_pca_run = plot_pca_baseline(
    baseline_series = baseline_series,
    class_names     = atoms.class_names,
    color_by        = run_ids,
    color_label     = "Run ID",
    title           = "Baseline ATOMs — PCA (coloured by run)",
)
save_figure(fig_pca_run, dirs["pca"] / "pca_baseline_by_run.png")

# --- 6d: PCA coloured by GMM cluster ---
# Checks whether clusters are spatially coherent in attention space.
# Well-separated clusters support the multimodal hypothesis.
fig_pca_clust = plot_pca_clusters(
    baseline_series = baseline_series,
    cluster_labels  = baseline_cluster_labels,
    gmm_means       = gmm.means_,
    title           = f"Baseline ATOMs — GMM Clusters (K={N_COMPONENTS}, PCA)",
)
save_figure(fig_pca_clust, dirs["pca"] / "pca_baseline_clusters.png")

# Fit PCA once on baseline and reuse it for all subsequent OOD plots
# so that test points are projected into the same space.
pca_obj, _ = fit_pca(baseline_series, n_components=2)

# --- 6e: t-SNE of baseline ---
# Fit once on baseline and reuse the same embedding for both run-color and
# cluster-color plots so the spatial layout is identical.
print("  Computing baseline t-SNE (may take a moment)...")
tsne_baseline = fit_tsne(baseline_series)

fig_tsne_run = plot_tsne_baseline(
    baseline_series = baseline_series,
    class_names     = atoms.class_names,
    color_by        = run_ids,
    color_label     = "Run ID",
    title           = "Baseline ATOMs — t-SNE (coloured by run)",
    tsne_embedding  = tsne_baseline,
)
save_figure(fig_tsne_run, dirs["tsne"] / "tsne_baseline_by_run.png")

fig_tsne_clust = plot_tsne_clusters(
    baseline_series = baseline_series,
    cluster_labels  = baseline_cluster_labels,
    title           = f"Baseline ATOMs — GMM Clusters (K={N_COMPONENTS}, t-SNE)",
    tsne_embedding  = tsne_baseline,
)
save_figure(fig_tsne_clust, dirs["tsne"] / "tsne_baseline_clusters.png")

print("  Figures saved.\n")


# ===========================================================================
# STEP 8 — Apply perturbations to test set
# ===========================================================================
# The perturbation mix is defined here.  Fractions must sum to 1.0.
# Add, remove, or reweight entries to match your experimental design.
#
# Available perturbation types depend on your PerturbationManager.
# Common options (adjust names to match your pm interface):
#   "gaussian_noise", "brightness", "dropout", "fgsm", "right_camera_loss"
# ---------------------------------------------------------------------------
if conf.REAPPLY_PERTURBATIONS:
    print("[Step 8] Applying perturbations to test set...")

    # Import your PerturbationManager
    from ATOMs_Analysis.perturbation_manager import PerturbationManager
    pm = PerturbationManager()

    # Define the perturbation mix.
    # <<< ADJUST fractions and perturbation types to your experiment
    if conf.AGENT == "TFV6":
        # PGD requires model.policy() which is WoR-specific and needs route
        # context unavailable offline.  Replace with an extra gaussian_noise
        # entry at a higher intensity so all four slots remain populated.
        spec = PerturbationSpec([
            PerturbationEntry(fraction=0.25, perturbation=None),
            PerturbationEntry(fraction=0.25, perturbation="gaussian_noise",  intensity=conf.NOISE_INTENSITY),
            PerturbationEntry(fraction=0.25, perturbation="brightness_scale", intensity=conf.BRIGHTNESS_INTENSITY),
            PerturbationEntry(fraction=0.25, perturbation="camera_loss",     intensity=0),
        ])
    else:
        spec = PerturbationSpec([
            PerturbationEntry(fraction=0.2, perturbation=None),
            PerturbationEntry(fraction=0.20, perturbation="gaussian_noise",   intensity=conf.NOISE_INTENSITY),
            PerturbationEntry(fraction=0.20, perturbation="brightness_scale", intensity=conf.BRIGHTNESS_INTENSITY),
            PerturbationEntry(fraction=0.2,  perturbation="camera_loss",      intensity=0),
            PerturbationEntry(fraction=0.2,  perturbation="pgd",              intensity=conf.EPSILON, fgsm_target="max_steer"),
        ])

    applier = PerturbationApplier(pm, model)
    labeled_path = applier.apply(
        spec        = spec,
        seed        = conf.RANDOM_SEED,    # <<< fix seed for reproducibility
        output_name = "test_labeled",
    )

# Load and summarise
test_data = LabeledTestLoader.load()
print("\n  Labeled test set summary:")
print("  " + LabeledTestLoader.summary(test_data).replace("\n", "\n  "))
print()


# ===========================================================================
# STEP 9 — Compute ATOMs profiles on labeled test set
# ===========================================================================
# We process each test frame through ATOMsCarla to get attention profiles,
# then collect them alongside their ground-truth labels for evaluation.
# ---------------------------------------------------------------------------
if conf.RECOMPUTE_TEST_ATOMS:
    print("[Step 9] Computing ATOMs on test set...")

    atoms.reset()   # clear any accumulated state from baseline computation

    n_test                 = test_data["wide_rgb"].shape[0]
    test_profiles          = np.zeros((n_test, atoms.num_classes), dtype=np.float64)
    test_logits_all        = [] if action_logits_available  else None
    test_speed_logits_list = [] if speed_logits_available   else None

    has_narr_test     = test_data["narr_rgb"]     is not None
    has_seg_narr_test = test_data["seg_red_narr"] is not None

    t0 = time.time()
    for i in range(n_test):
        wide     = torch.from_numpy(test_data["wide_rgb"][i:i+1]).float()
        narrow   = torch.from_numpy(test_data["narr_rgb"][i:i+1]).float()   if has_narr_test     else None
        seg_wide = test_data["seg_red_wide"][i]
        seg_narr = test_data["seg_red_narr"][i]                             if has_seg_narr_test else None
        cmd      = int(test_data["cmd"][i])

        profile = atoms.process_frame(wide, narrow, seg_wide, seg_narr, cmd=cmd)
        test_profiles[i] = profile

        # Action logits for entropy detector (WoR only)
        if action_logits_available:
            with torch.no_grad():
                steer, throt, brake = model.policy(wide, narrow, cmd=cmd)
                flat_logits = torch.cat([
                    steer.flatten(),
                    throt.flatten(),
                    brake.unsqueeze(0).flatten(),
                ]).cpu().numpy()
            test_logits_all.append(flat_logits)

        # Speed logits for PEOC (TFV6 only)
        if speed_logits_available:
            spd_logits = lrp.get_speed_logits(
                wide.float(),
                cmd=cmd,
                spd=float(test_data["speed"][i]),
            )
            test_speed_logits_list.append(spd_logits)

        if (i + 1) % 100 == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n_test}  ({fps:.1f} fr/s)")

    atoms.reset()
    if action_logits_available:
        test_logits_all = np.array(test_logits_all, dtype=np.float32)   # [N, num_logits]
    if speed_logits_available:
        test_speed_logits = np.array(test_speed_logits_list, dtype=np.float32)  # [N, 8]
    else:
        test_speed_logits = None
    test_labels = test_data["label"].astype(np.int32)
    print(f"  Done. {n_test} frames processed.\n")

    np.save(ATT_DIR / "test_profiles.npy", test_profiles)
    if action_logits_available:
        np.save(ATT_DIR / "test_logits.npy", test_logits_all)
    if speed_logits_available:
        np.save(ATT_DIR / "test_speed_logits.npy", test_speed_logits)
    print("  Test profiles saved.\n")

else:
    # Reload test data and pre-computed profiles
    test_data         = LabeledTestLoader.load()
    test_profiles     = np.load(ATT_DIR / "test_profiles.npy")
    test_logits_all   = (
        np.load(ATT_DIR / "test_logits.npy") if action_logits_available else None
    )
    test_speed_logits = (
        np.load(ATT_DIR / "test_speed_logits.npy") if speed_logits_available else None
    )
    test_labels       = test_data["label"].astype(np.int32)

    print(f"  Loaded {len(test_profiles)} test profiles, "
          f"{int(test_labels.sum())} perturbed.\n")


# ===========================================================================
# STEP 10 — Perturbation trajectory analysis in ATOMs attention space
# ===========================================================================
# Compares how each perturbation type moves samples in attention space by:
#   1. Computing ATOMs profiles for the clean test frames (mirroring Step 9).
#   2. Loading pre-computed ATOMs profiles for the perturbed test frames
#      (test_profiles.npy, produced in Step 8).
#   3. Matching clean ↔ perturbed pairs via the composite key (run_id, frame_idx).
#   4. Grouping pairs by perturbation type.
#   5. Computing displacement statistics (mean direction, magnitude, cosine sim).
#   6. Producing three figures:
#        trajectory_pca.png          — arrows in standard PCA space
#        trajectory_whitened_pca.png — arrows in Mahalanobis-whitened PCA space
#        displacement_similarity.png — cosine sim heatmap between pert. directions
#        displacement_magnitude.png  — boxplot of displacement magnitudes
#   7. Writing a text summary to displacement_stats.txt.
#
# Prerequisites (must exist in scope from earlier steps):
#   atoms          — ATOMsCarla instance (will be reset before and after use)
#   baseline_data  — dict from BaselineDataLoader (used to fit the PCA axes and
#                    to supply the covariance for whitened PCA)
#   baseline_profiles — np.ndarray [N_b, C]  (from Step 7)
#   test_data      — dict from LabeledTestLoader.load() (labeled test set)
#   test_profiles  — np.ndarray [N_t, C]     (perturbed profiles, from Step 9)
#   test_labels    — np.ndarray [N_t] int    (0=clean, 1=perturbed)
#   OUT_DIR        — pathlib.Path  output directory
#   conf           — ExperimentConfig
#   CARLA_CLASSES  — dict {int: str}  from visualization.py
# ---------------------------------------------------------------------------


TRAJ_OUT_DIR = OUT_DIR / "trajectory_analysis"
TRAJ_OUT_DIR.mkdir(parents=True, exist_ok=True)

# Class names in index order for annotation (visualization optional)
_max_class   = max(_seg_class_map.keys()) + 1
class_names  = [_seg_class_map.get(i, f"cls_{i}") for i in range(_max_class)]


# ---------------------------------------------------------------------------
# 10.a — Compute ATOMs profiles for the CLEAN test frames
# ---------------------------------------------------------------------------
# The clean frames live in conf.TEST_DATA_DIR/frames/ and are NOT stored in
# test_labeled.npz (which only holds copies of whichever frames were
# selected, some perturbed and some label-0 "pseudo-clean" copies that went
# through PerturbationApplier).  We load the raw frames directly so we have
# the original clean version of every frame.
# ---------------------------------------------------------------------------

CLEAN_PROFILES_PATH = TRAJ_OUT_DIR / "clean_test_profiles.npy"

if CLEAN_PROFILES_PATH.exists():
    print("[Step 10a] Loading cached clean test profiles...")
    clean_raw         = BaselineDataLoader.load_all_runs(conf.TEST_DATA_DIR / "frames")
    clean_profiles    = np.load(CLEAN_PROFILES_PATH)
    print(f"  Loaded {len(clean_profiles)} clean profiles.\n")
else:
    print("[Step 10a] Computing ATOMs profiles on clean test frames...")
    clean_raw      = BaselineDataLoader.load_all_runs(conf.TEST_DATA_DIR / "frames")
    n_clean        = clean_raw["wide_rgb"].shape[0]
    clean_profiles = np.zeros((n_clean, atoms.num_classes), dtype=np.float64)

    has_narr_clean     = clean_raw["narr_rgb"]     is not None
    has_seg_narr_clean = clean_raw["seg_red_narr"] is not None

    atoms.reset()
    t0 = time.time()
    for i in range(n_clean):
        wide     = torch.from_numpy(clean_raw["wide_rgb"][i:i+1]).float()
        narr     = torch.from_numpy(clean_raw["narr_rgb"][i:i+1]).float()   if has_narr_clean     else None
        seg_wide = clean_raw["seg_red_wide"][i]
        seg_narr = clean_raw["seg_red_narr"][i]                             if has_seg_narr_clean else None
        cmd      = int(clean_raw["cmd"][i])
        profile  = atoms.process_frame(wide, narr, seg_wide, seg_narr, cmd=cmd)
        clean_profiles[i] = profile
        if (i + 1) % 100 == 0:
            fps = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n_clean}  ({fps:.1f} fr/s)")

    atoms.reset()
    np.save(CLEAN_PROFILES_PATH, clean_profiles)
    print(f"  Done. {n_clean} frames processed. Saved to {CLEAN_PROFILES_PATH}\n")


# ---------------------------------------------------------------------------
# 10.b — Build lookup: (run_id, frame_idx) → clean profile index
# ---------------------------------------------------------------------------
# Both clean_raw and test_labeled use the same (run_id, frame_idx) keys
# inherited from _load_all_runs / PerturbationApplier, so this composite
# key is the reliable unique identifier.

clean_key_to_idx: dict = {
    (int(clean_raw["run_id"][i]), int(clean_raw["frame_idx"][i])): i
    for i in range(len(clean_profiles))
}

print(f"[Step 10b] Clean index built: {len(clean_key_to_idx)} unique (run_id, frame_idx) pairs.")


# ---------------------------------------------------------------------------
# 10.c — Build per-perturbation-type paired arrays
# ---------------------------------------------------------------------------
# Only use perturbed frames (label == 1).  For each one, look up the
# matching clean profile by composite key.  Group by perturbation type.
# ---------------------------------------------------------------------------

# paired_dict: {pert_name: {"clean": list[np.ndarray[C]], "perturbed": list}}
#paired_dict: dict = defaultdict(lambda: {"clean": [], "perturbed": []})
#
#n_matched   = 0
#n_unmatched = 0
#
#perturbed_mask = test_labels == 1
#
#for i in np.where(perturbed_mask)[0]:
#    run_id    = int(test_data["run_id"][i])
#    frame_idx = int(test_data["frame_idx"][i])
#    key       = (run_id, frame_idx)
#
#    if key not in clean_key_to_idx:
#        n_unmatched += 1
#        continue
#
#    clean_idx   = clean_key_to_idx[key]
#    pert_name   = str(test_data["perturbation"][i])
#
#    paired_dict[pert_name]["clean"].append(clean_profiles[clean_idx])
#    paired_dict[pert_name]["perturbed"].append(test_profiles[i])
#    n_matched += 1
#
## Convert lists → arrays
#paired_dict = {
#    name: {
#        "clean":     np.stack(pair["clean"]),      # [N, C]
#        "perturbed": np.stack(pair["perturbed"]),  # [N, C]
#    }
#    for name, pair in paired_dict.items()
#}
#
#print(f"[Step 10c] Matched {n_matched} perturbed frames to clean originals.")
#if n_unmatched:
#    print(f"  WARNING: {n_unmatched} frames could not be matched (run_id/frame_idx mismatch).")
#for name, pair in paired_dict.items():
#    print(f"  [{name}]  {len(pair['clean'])} pairs")
#print()


# ---------------------------------------------------------------------------
# 10.d — Displacement statistics
# ---------------------------------------------------------------------------

#print("[Step 10d] Computing displacement statistics...")
#stats = compute_perturbation_displacement_stats(paired_dict)
#
#stats_text = format_displacement_stats_text(stats)
#print(stats_text)
#
#stats_path = TRAJ_OUT_DIR / "displacement_stats.txt"
#stats_path.write_text(stats_text)
#print(f"\n  Stats saved → {stats_path}\n")



# ---------------------------------------------------------------------------
# 10.f — Figures
# ---------------------------------------------------------------------------

print("[Step 10f] Generating figures...")

# --- Standard PCA trajectories ---
#fig_traj, proj_info = plot_pca_perturbation_trajectories(
#    baseline_profiles = baseline_series,
#    paired_dict       = paired_dict,
#    cov               = None,           # standard PCA
#    subsample         = 60,
#    arrow_alpha       = 0.3,
#    title             = "ATOMs Attention Trajectories Under Perturbation (PCA)",
#    class_names       = class_names,
#)
#save_figure(fig_traj, TRAJ_OUT_DIR / "trajectory_pca.png")
#
## --- Whitened PCA trajectories (Mahalanobis geometry) ---
#fig_wtraj, _ = plot_pca_perturbation_trajectories(
#    baseline_profiles = baseline_series,
#    paired_dict       = paired_dict,
#    cov               = baseline_cov,            # activates whitened PCA
#    subsample         = 60,
#    arrow_alpha       = 0.3,
#    title             = "ATOMs Attention Trajectories Under Perturbation (Whitened PCA)",
#    class_names       = class_names,
#)
#save_figure(fig_wtraj, TRAJ_OUT_DIR / "trajectory_whitened_pca.png")
#
## --- Cosine similarity heatmap ---
#fig_coh = plot_displacement_coherence_bar(stats)
#save_figure(fig_coh, TRAJ_OUT_DIR / "displacement_coherence.png")
#
## --- Magnitude boxplot ---
#fig_mag = plot_displacement_magnitude_boxplot(stats)
#save_figure(fig_mag, TRAJ_OUT_DIR / "displacement_magnitude.png")
#
#print(f"\n[Step 10] Done. All outputs in {TRAJ_OUT_DIR}\n")


# ===========================================================================
# STEP 11 — Score all test profiles with every detector
# ===========================================================================
print("[Step 11] Scoring test profiles...")

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

KNN_K_VALUES = [1, 5, 10, 25, 50, 100, 250]
scores_knn_by_k: dict = {}
for _k in KNN_K_VALUES:
    scores_knn_by_k[_k] = np.array([
        DistanceComputer.compute_knn_distance(
            reference_samples = baseline_series,
            target_point      = test_profiles[i],
            k                 = _k,
            normalize         = True,
        )
        for i in range(len(test_profiles))
    ])
    print(f"  k-NN k={_k}: done")

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
if action_logits_available and test_logits_all is not None:
    entropy_detector = ActionEntropyDetector(from_logits=True, cmd=None)
    scores_entropy   = entropy_detector.score_batch(test_logits_all)

# ----- 9d: MDX detection ----
scores_mdx = None
if mdx is not None:
    scores_list = []
    n_mdx_test = len(test_data["frame_idx"])
    for i in range(n_mdx_test):
        if i % 100 == 0:
            print(f"  MDX scoring frame {i}/{n_mdx_test}")
        if action_logits_available:
            # WoR: penultimate-layer features via model.get_features()
            features = model.get_features(
                torch.from_numpy(test_data["wide_rgb"][i]),
                torch.from_numpy(test_data["narr_rgb"][i]),
                test_data["speed"][i],
            )
            feat_vec = features[0].cpu().detach().numpy()
        else:
            # TFV6: 512-dim globally-pooled backbone features
            wide_t   = torch.from_numpy(test_data["wide_rgb"][i]).unsqueeze(0)
            feat_vec = lrp.get_backbone_features(wide_t)
        scores_list.append(mdx.score(feat_vec))
    scores_mdx = np.array(scores_list)
    print(f"  MDX: scored {len(scores_mdx)} frames")

# ----- 9e: PEOC — Policy Entropy OOD Classifier (Sedlmeier et al., 2020) ----
# H(π) of the 8-bin speed distribution. No fitting required.
scores_peoc = None
if speed_logits_available and test_speed_logits is not None:
    peoc_detector = ActionEntropyDetector(from_logits=True, cmd=None)
    scores_peoc   = peoc_detector.score_batch(test_speed_logits)
    print(f"  PEOC: scored {len(scores_peoc)} frames")

# --- Quick sanity check ---
_knn_sanity = [(f"k-NN (k={k})", scores_knn_by_k[k]) for k in KNN_K_VALUES]
_optional = [
    ("Action entropy",  scores_entropy  if action_logits_available else None),
    ("MDX Detection",   scores_mdx),
    ("PEOC",            scores_peoc),
]
for name, scores in [
    ("Mahalanobis (single)", scores_mahal_single),
    ("Mahalanobis (GMM)",    scores_mahal_gmm),
    ("Euclidean",            scores_euclid_single),
    ("Jensen-Shannon",       scores_jsd_single),
] + [(n, s) for n, s in _optional if s is not None] + _knn_sanity:
    clean_mean = scores[test_labels == 0].mean()
    pert_mean  = scores[test_labels == 1].mean()
    sep        = pert_mean - clean_mean   # positive = detector pointing the right way
    print(f"  {name:<30}  clean: {clean_mean:.3f}  perturbed: {pert_mean:.3f}  "
          f"separation: {sep:+.3f}")
print()


# ===========================================================================
# STEP 12 — Evaluate each detector (ROC / AUC / Youden threshold)
# ===========================================================================
print("[Step 12] Evaluating detectors...")

evaluator = DetectorEvaluator()

results_single = evaluator.evaluate(
    scores        = scores_mahal_single,
    labels        = test_labels,
    detector_name = "ATOMs-Mahalanobis (single Gaussian)",
)
results_gmm = evaluator.evaluate(
    scores        = scores_mahal_gmm,
    labels        = test_labels,
    detector_name = f"ATOMs-Mahalanobis (GMM K={N_COMPONENTS})",
)
results_entropy = evaluator.evaluate(
    scores        = scores_entropy,
    labels        = test_labels,
    detector_name = "Action entropy",
) if action_logits_available else None

results_jsd = evaluator.evaluate(
    scores        = scores_jsd_single,
    labels        = test_labels,
    detector_name = "ATOMs-JSD",
)

results_euclidean = evaluator.evaluate(
    scores        = scores_euclid_single,
    labels        = test_labels,
    detector_name = "ATOMs-Euclidean",
)

# Evaluate all k-NN variants for the sensitivity analysis
results_knn_by_k: dict = {}
for k_val, knn_scores in scores_knn_by_k.items():
    results_knn_by_k[k_val] = evaluator.evaluate(
        scores        = knn_scores,
        labels        = test_labels,
        detector_name = f"ATOMs-k-NN (k={k_val})",
    )

# Select best k by AUC — only this variant competes in the combined ROC plot
best_k = max(results_knn_by_k, key=lambda k: results_knn_by_k[k]["auc"])
results_knn = results_knn_by_k[best_k]
results_knn["detector_name"] = f"ATOMs-k-NN (k={best_k}, best)"
scores_knn_best = scores_knn_by_k[best_k]
print(f"  Best k-NN: k={best_k}  AUC={results_knn['auc']:.4f}")

results_mdx = evaluator.evaluate(
    scores        = scores_mdx,
    labels        = test_labels,
    detector_name = "MDX Detection",
) if scores_mdx is not None else None

results_peoc = evaluator.evaluate(
    scores        = scores_peoc,
    labels        = test_labels,
    detector_name = "PEOC (speed entropy)",
) if scores_peoc is not None else None

all_results = [
    r for r in [
        results_single, results_gmm, results_entropy,
        results_euclidean, results_jsd, results_knn, results_mdx, results_peoc,
    ] if r is not None
]
evaluator.compare(all_results)

# Save each result as JSON
for res in all_results:
    safe_name = res["detector_name"].replace(" ", "_").replace("(", "").replace(")", "")
    evaluator.save_results(res, OUT_DIR / f"results_{safe_name}.json")

# Also save a combined summary
summary = {r["detector_name"]: {"auc": r["auc"], "youden_j": r["youden_j"]} for r in all_results}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print()


# ===========================================================================
# STEP 13 — Per-perturbation breakdown
# ===========================================================================
# Evaluate each detector on each perturbation type independently.
# This reveals which perturbations are detectable and which are not,
# which is important for your thesis narrative.
# ---------------------------------------------------------------------------
print("[Step 13] Per-perturbation breakdown...")

split_data  = LabeledTestLoader.split_by_perturbation(test_data)
perturb_results: dict = {}

for pert_name, subset in split_data.items():
    if pert_name == "clean":
        continue  # skip the clean subset — it has no positive labels

    # Indices of this subset in the full test array
    mask = test_data["perturbation"] == pert_name

    # Build labels: clean frames from the full set vs this perturbation type.
    # Evaluation is: can the detector separate clean from *this specific* perturbation?
    clean_mask  = test_data["perturbation"] == "clean"
    eval_mask   = clean_mask | mask
    eval_labels = test_labels[eval_mask]

    r_single = evaluator.evaluate(
        scores_mahal_single[eval_mask], eval_labels,
        detector_name=f"Mahalanobis-single | {pert_name}",
    )
    r_gmm = evaluator.evaluate(
        scores_mahal_gmm[eval_mask], eval_labels,
        detector_name=f"Mahalanobis-GMM | {pert_name}",
    )
    r_jsd = evaluator.evaluate(
        scores_jsd_single[eval_mask], eval_labels,
        detector_name=f"JSD-single | {pert_name}",
    )

    r_euclidean = evaluator.evaluate(
        scores_euclid_single[eval_mask], eval_labels,
        detector_name=f"Euclidean-single | {pert_name}",
    )

    r_knn = evaluator.evaluate(
        scores_knn_best[eval_mask], eval_labels,
        detector_name=f"kNN (k={best_k}) | {pert_name}",
    )

    r_mdx = evaluator.evaluate(
        scores_mdx[eval_mask], eval_labels,
        detector_name=f"MDX Detection | {pert_name}",
    ) if scores_mdx is not None else None

    r_entropy = evaluator.evaluate(
        scores_entropy[eval_mask], eval_labels,
        detector_name=f"Entropy | {pert_name}",
    ) if action_logits_available else None

    r_peoc = evaluator.evaluate(
        scores_peoc[eval_mask], eval_labels,
        detector_name=f"PEOC | {pert_name}",
    ) if scores_peoc is not None else None

    perturb_results[pert_name] = [
        r for r in [r_single, r_gmm, r_entropy, r_jsd, r_euclidean, r_mdx, r_knn, r_peoc]
        if r is not None
    ]
    print(f"\n  Perturbation: {pert_name}")
    evaluator.compare([r for r in [r_single, r_gmm, r_peoc] if r is not None])

with open(OUT_DIR / "results_per_perturbation.json", "w") as f:
    json.dump(
        {k: [{r["detector_name"]: r["auc"]} for r in v] for k, v in perturb_results.items()},
        f, indent=2
    )
print()


# ===========================================================================
# STEP 14 — Visualize detection results
# ===========================================================================
print("[Step 14] Saving detection figures...")

# --- 12a: ROC curves — all detectors on full test set ---
fig_roc = plot_roc(
    results_list = all_results,
    title        = "ROC — Full test set (all perturbations)",
)
save_figure(fig_roc, dirs["roc"] / "roc_all_detectors.png")

# --- 12b: Score distributions for best detector ---
# Split by clean / perturbed to show separation quality.
fig_dist_mahal = plot_mahal_distribution(
    in_scores  = scores_mahal_single[test_labels == 0],
    out_scores = scores_mahal_single[test_labels == 1],
    threshold  = results_single["optimal_threshold"],
    title      = "Mahalanobis score distribution (single Gaussian)",
)
save_figure(fig_dist_mahal, dirs["scores"] / "score_dist_mahal_single.png")

fig_dist_gmm = plot_mahal_distribution(
    in_scores  = scores_mahal_gmm[test_labels == 0],
    out_scores = scores_mahal_gmm[test_labels == 1],
    threshold  = results_gmm["optimal_threshold"],
    title      = f"Mahalanobis score distribution (GMM K={N_COMPONENTS})",
)
save_figure(fig_dist_gmm, dirs["scores"] / "score_dist_mahal_gmm.png")

# --- 12c: One ROC plot per perturbation type, comparing all detectors ---
for pert_name, res_list in perturb_results.items():
    fig_roc_p = plot_roc(
        results_list = res_list,
        title        = f"ROC — {pert_name}",
    )
    safe = pert_name.replace(" ", "_")
    save_figure(fig_roc_p, dirs["roc"] / f"roc_{safe}.png")

# --- 12d.0: k-NN sensitivity — AUC vs k ---
knn_k_list   = list(results_knn_by_k.keys())
knn_auc_list = [results_knn_by_k[k]["auc"] for k in knn_k_list]
fig_knn_sens = plot_knn_sensitivity(knn_k_list, knn_auc_list, best_k)
save_figure(fig_knn_sens, dirs["roc"] / "knn_k_sensitivity.png")

# Save per-k results as JSON for later reference
for k_val, res in results_knn_by_k.items():
    evaluator.save_results(res, OUT_DIR / f"results_knn_k{k_val}.json")

# --- 12d: PCA OOD overlay — test samples projected into baseline PCA space ---
# Clean test points should sit within the baseline cloud; perturbed should scatter away.
fig_pca_ood = plot_pca_ood(
    baseline_series = baseline_series,
    test_series     = test_profiles,
    test_labels     = test_labels,
    pca             = pca_obj,       # reuse baseline PCA — do not refit on test data
    title           = "PCA — Baseline vs Test (clean / perturbed)",
)
save_figure(fig_pca_ood, dirs["pca"] / "pca_ood_overlay.png")

# --- 12e: One OOD PCA per perturbation type ---
for pert_name in split_data:
    if pert_name == "clean":
        continue
    mask        = test_data["perturbation"] == pert_name
    sub_labels  = np.zeros(len(test_profiles), dtype=np.int32)
    sub_labels[mask] = 1
    fig_p = plot_pca_ood(
        baseline_series = baseline_series,
        test_series     = test_profiles,
        test_labels     = sub_labels,
        pca             = pca_obj,
        title           = f"PCA OOD — {pert_name} vs clean",
    )
    save_figure(fig_p, dirs["pca"] / f"pca_ood_{pert_name.replace(' ', '_')}.png")

# --- 12f: t-SNE OOD overlay ---
# Fit once on baseline + all test combined, then reuse for per-perturbation plots.
print("  Computing joint t-SNE for OOD overlay (may take a moment)...")
tsne_joint = fit_tsne_joint(baseline_series, test_profiles)

fig_tsne_ood = plot_tsne_ood(
    baseline_series = baseline_series,
    test_series     = test_profiles,
    test_labels     = test_labels,
    tsne_embedding  = tsne_joint,
    title           = "t-SNE — Baseline vs Test (clean / perturbed)",
)
save_figure(fig_tsne_ood, dirs["tsne"] / "tsne_ood_overlay.png")

# --- 12g: One OOD t-SNE per perturbation type (reuses tsne_joint) ---
for pert_name in split_data:
    if pert_name == "clean":
        continue
    mask       = test_data["perturbation"] == pert_name
    sub_labels = np.zeros(len(test_profiles), dtype=np.int32)
    sub_labels[mask] = 1
    fig_tsne_p = plot_tsne_ood(
        baseline_series = baseline_series,
        test_series     = test_profiles,
        test_labels     = sub_labels,
        tsne_embedding  = tsne_joint,
        title           = f"t-SNE OOD — {pert_name} vs clean",
    )
    save_figure(fig_tsne_p, dirs["tsne"] / f"tsne_ood_{pert_name.replace(' ', '_')}.png")

print(f"\n{'='*60}")
print(f"Analysis complete.  All outputs saved to {OUT_DIR}")
print(f"{'='*60}")
