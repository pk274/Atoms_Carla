"""
viz_config.py
-------------
Shared visualization configuration for the master's thesis on OOD detection in RL.

This file is intended to be IDENTICAL in both project directories (CARLA and Atari).
When you change something here, copy the file to the other project to keep them
in sync.  Both visualization.py files import from this module, so any plot showing
the same conceptual thing (baseline cloud, gaussian_noise perturbation, ...) uses
the same colors, figure sizes, marker styles, and DPI in both domains.

Things that exist in only one domain (CARLA segmentation, Atari frame stacks, etc.)
can use these constants where applicable but are not forced to mirror anything.
"""

from __future__ import annotations
from typing import Dict, Tuple

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Semantic colors -- same conceptual role uses the same color in both projects
# ---------------------------------------------------------------------------

# The in-distribution reference cloud shown as background context in every
# OOD/PCA plot.  Muted grey so it does not compete with perturbation colors.
BASELINE_COLOR     = "#9aa0a6"

# Clean test samples drawn from the same distribution as baseline but held out
# for evaluation (CARLA's plot_pca_ood).
CLEAN_TEST_COLOR   = "tab:blue"

# Generic "perturbed / OOD" color when the individual perturbation type does
# not matter (e.g. in-vs-out histograms).
PERTURBED_COLOR    = "tab:red"

# Decision thresholds, chance lines, reference lines.
THRESHOLD_COLOR    = "black"
CHANCE_LINE_COLOR  = "black"


# ---------------------------------------------------------------------------
# Per-perturbation colors -- semantic mapping shared across both projects.
#
# Same concept = same color, even if the names differ across the two codebases
# (salt_pepper_noise vs salt_and_pepper -- two keys, one color).
#
# If a perturbation is not in this dict, get_perturbation_color() falls back
# to a stable tab10 index so the plot still works.
# ---------------------------------------------------------------------------

PERTURBATION_COLORS: Dict[str, str] = {
    # Shared concepts (both projects)
    "gaussian_noise":      "tab:red",
    "blur":                "tab:purple",

    # Salt-and-pepper noise: same concept, two names across projects.
    "salt_pepper_noise":   "tab:orange",   # Atari naming
    "salt_and_pepper":     "tab:orange",   # CARLA naming -- intentionally identical

    # Brightness family -- distinct but in the same warm/earth family.
    "decrease_brightness": "tab:green",    # Atari
    "increase_brightness": "tab:olive",    # Atari
    "brightness_scale":    "tab:brown",    # CARLA (signed scale, covers both directions)

    # Atari-only
    "random_occlusion":    "tab:gray",

    # CARLA-only
    "phantom_obstacle":    "tab:pink",     # NOTE: spelling preserved from CARLA codebase
    "pgd_attack":          "tab:cyan",
    "fgsm_attack":         "#0e7c93",      # darker cyan -- groups visually with pgd_attack
    "camera_loss":         "#525252",      # dark grey, distinct from tab:gray
}

TAB10_FALLBACK = [f"C{i}" for i in range(10)]


def get_perturbation_color(name: str, fallback_index: int = 0) -> str:
    """Return the configured color for a perturbation, or a stable tab10 fallback."""
    if name in PERTURBATION_COLORS:
        return PERTURBATION_COLORS[name]
    return TAB10_FALLBACK[fallback_index % len(TAB10_FALLBACK)]


# ---------------------------------------------------------------------------
# Cluster styling -- baseline cluster visualization (CARLA plot_pca_clusters)
# ---------------------------------------------------------------------------

CLUSTER_CMAP        = "tab10"      # use for up to 10 clusters
CLUSTER_CMAP_LARGE  = "tab20"      # use for more

CENTROID_MARKER     = "*"
CENTROID_SIZE       = 250
CENTROID_EDGECOLOR  = "black"
CENTROID_LINEWIDTH  = 0.8


# ---------------------------------------------------------------------------
# Saliency / attention heatmap colormaps (mostly CARLA)
# ---------------------------------------------------------------------------

SALIENCY_CMAP_POSITIVE   = "hot"           # one-sided relevance
SALIENCY_CMAP_ALT        = "gist_earth"    # CARLA: brake action
SALIENCY_CMAP_DIVERGING  = "seismic"       # signed relevance

SALIENCY_OVERLAY_ALPHA   = 0.5


# ---------------------------------------------------------------------------
# Distance-type colors (CARLA-only -- centralized here for organization)
# ---------------------------------------------------------------------------

DISTANCE_TYPE_COLORS = {
    "knn":              "tab:green",
    "gmm_knn":          "#006400",     # dark green -- GMM variant
    "jsd":              "tab:purple",
    "gmm_jsd":          "#4B0082",     # indigo -- GMM variant
    "wasserstein":      "tab:brown",
    "gmm_wasserstein":  "#8B4513",     # saddlebrown -- GMM variant
    "mahalanobis":      "tab:red",
    "gmm_mahalanobis":  "#810000",     # dark red -- GMM variant
    "mdx":              "#cc7979",     # IndianRed -- class-conditional Mahalanobis
    "euclidean":        "tab:blue",
    "gmm_euclidean":    "tab:cyan",
    "peoc":             "tab:olive",   # entropy-based, not a distance
}

DISTANCE_TYPE_YLABELS = {
    "knn":              "k-NN Distance",
    "gmm_knn":          "k-NN Distance (GMM)",
    "jsd":              "Jensen-Shannon Divergence",
    "gmm_jsd":          "Jensen-Shannon Divergence (GMM)",
    "wasserstein":      "Wasserstein Distance",
    "gmm_wasserstein":  "Wasserstein Distance (GMM)",
    "mahalanobis":      "Mahalanobis Distance",
    "gmm_mahalanobis":  "Mahalanobis Distance (GMM)",
    "mdx":              "MDX Distance",
    "euclidean":        "Euclidean Distance",
    "gmm_euclidean":    "Euclidean Distance (GMM)",
    "peoc":             "PEOC (entropy)",
}


# ---------------------------------------------------------------------------
# Figure sizes -- one canonical size per plot type
# ---------------------------------------------------------------------------

FIGSIZE_PCA                = (8, 6)     # baseline / OOD / cluster PCA scatter
FIGSIZE_TRAJECTORY         = (8, 6)     # trajectory plots -- same as PCA for coherence
FIGSIZE_ROC                = (5, 5)     # ROC -- square
FIGSIZE_HISTOGRAM          = (6, 4)     # score distributions
FIGSIZE_BIC_AIC            = (5, 3.5)   # model selection
FIGSIZE_BAR                = (7, 4)     # bar charts with fixed number of bars
FIGSIZE_ATTENTION_OVERLAY  = (15, 4)    # CARLA saliency input/heatmap/overlay triptych
FIGSIZE_DISTANCE_OVER_TIME = (7, 5)     # CARLA distance vs frame index


def figsize_bar_scaled(n_items: int,
                       per_item: float = 0.5,
                       min_w: float = 4.0,
                       height: float = 4.0) -> Tuple[float, float]:
    """Figure size for a bar chart whose width should scale with the number of bars."""
    return (max(min_w, n_items * per_item), height)


def figsize_attention_bar(n_classes: int,
                          per_class: float = 0.35,
                          width: float = 7.0,
                          min_h: float = 3.0) -> Tuple[float, float]:
    """Figure size for a horizontal bar of attention per class."""
    return (width, max(min_h, n_classes * per_class))


# ---------------------------------------------------------------------------
# Marker sizes, alphas, line widths -- per semantic role
# ---------------------------------------------------------------------------

# Baseline cloud (background context, visible but not loud)
BASELINE_MARKER_SIZE      = 14
BASELINE_MARKER_ALPHA     = 0.40

# Clean test points (CARLA plot_pca_ood)
CLEAN_MARKER_SIZE         = 18
CLEAN_MARKER_ALPHA        = 0.65

# Perturbed test points
PERTURBED_MARKER_SIZE     = 18
PERTURBED_MARKER_ALPHA    = 0.65

# Trajectory line/points (Atari per-frame trajectories)
TRAJ_POINT_SIZE           = 30
TRAJ_LINE_WIDTH           = 1.5
TRAJ_LINE_ALPHA           = 0.70
TRAJ_ARROW_LINEWIDTH      = 2.0

# Start / end markers on trajectories (Atari) or clean / perturbed centroids (CARLA)
TRAJ_START_MARKER         = "o"
TRAJ_END_MARKER           = "s"
TRAJ_ENDPOINT_SIZE        = 80
TRAJ_ENDPOINT_EDGECOLOR   = "black"
TRAJ_ENDPOINT_LINEWIDTH   = 1.5

# Per-sample / mean displacement arrows (CARLA plot_pca_perturbation_trajectories)
ARROW_ALPHA_INDIVIDUAL    = 0.25
ARROW_LINEWIDTH_INDIV     = 0.8
ARROW_MUTATION_INDIV      = 8

ARROW_LINEWIDTH_MEAN      = 2.5
ARROW_MUTATION_MEAN       = 18

MEAN_MARKER_SIZE_CLEAN     = 80
MEAN_MARKER_CLEAN          = "o"
MEAN_MARKER_SIZE_PERTURBED = 60
MEAN_MARKER_PERTURBED      = "D"


# ---------------------------------------------------------------------------
# Fonts (most are applied via rcParams in apply_default_style)
# ---------------------------------------------------------------------------

FONTSIZE_TITLE      = 13
FONTSIZE_SUBTITLE   = 10
FONTSIZE_AXIS_LABEL = 11
FONTSIZE_TICK       = 9
FONTSIZE_LEGEND     = 9

TITLE_WEIGHT        = "bold"


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

SAVE_DPI            = 200
SAVE_BBOX_INCHES    = "tight"
SAVE_FORMAT_DEFAULT = "png"


# ---------------------------------------------------------------------------
# Grid / legend
# ---------------------------------------------------------------------------

GRID_LINESTYLE      = "--"
GRID_ALPHA          = 0.3
LEGEND_FRAMEALPHA   = 0.85


# ---------------------------------------------------------------------------
# Apply default style globally
# ---------------------------------------------------------------------------

def apply_default_style() -> None:
    """
    Set matplotlib rcParams to the thesis-wide defaults.

    Call this once at the entry point of your script (or in the visualizer's
    constructor) and every figure made afterwards picks up the fonts, tick
    sizes, save DPI, etc., without you having to specify them per call.
    """
    plt.rcParams.update({
        "figure.dpi":        100,                  # screen preview
        "savefig.dpi":       SAVE_DPI,
        "savefig.bbox":      SAVE_BBOX_INCHES,
        "axes.titlesize":    FONTSIZE_TITLE,
        "axes.titleweight":  TITLE_WEIGHT,
        "axes.labelsize":    FONTSIZE_AXIS_LABEL,
        "xtick.labelsize":   FONTSIZE_TICK,
        "ytick.labelsize":   FONTSIZE_TICK,
        "legend.fontsize":   FONTSIZE_LEGEND,
        "legend.framealpha": LEGEND_FRAMEALPHA,
        "axes.grid":         False,                # set per-plot when wanted
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })


__all__ = [
    # semantic colors
    "BASELINE_COLOR", "CLEAN_TEST_COLOR", "PERTURBED_COLOR",
    "THRESHOLD_COLOR", "CHANCE_LINE_COLOR",
    # perturbation colors
    "PERTURBATION_COLORS", "TAB10_FALLBACK", "get_perturbation_color",
    # clusters
    "CLUSTER_CMAP", "CLUSTER_CMAP_LARGE",
    "CENTROID_MARKER", "CENTROID_SIZE", "CENTROID_EDGECOLOR", "CENTROID_LINEWIDTH",
    # saliency
    "SALIENCY_CMAP_POSITIVE", "SALIENCY_CMAP_ALT", "SALIENCY_CMAP_DIVERGING",
    "SALIENCY_OVERLAY_ALPHA",
    # distance types (CARLA)
    "DISTANCE_TYPE_COLORS", "DISTANCE_TYPE_YLABELS",
    # figure sizes
    "FIGSIZE_PCA", "FIGSIZE_TRAJECTORY", "FIGSIZE_ROC", "FIGSIZE_HISTOGRAM",
    "FIGSIZE_BIC_AIC", "FIGSIZE_BAR", "FIGSIZE_ATTENTION_OVERLAY",
    "FIGSIZE_DISTANCE_OVER_TIME",
    "figsize_bar_scaled", "figsize_attention_bar",
    # markers
    "BASELINE_MARKER_SIZE", "BASELINE_MARKER_ALPHA",
    "CLEAN_MARKER_SIZE", "CLEAN_MARKER_ALPHA",
    "PERTURBED_MARKER_SIZE", "PERTURBED_MARKER_ALPHA",
    "TRAJ_POINT_SIZE", "TRAJ_LINE_WIDTH", "TRAJ_LINE_ALPHA", "TRAJ_ARROW_LINEWIDTH",
    "TRAJ_START_MARKER", "TRAJ_END_MARKER", "TRAJ_ENDPOINT_SIZE",
    "TRAJ_ENDPOINT_EDGECOLOR", "TRAJ_ENDPOINT_LINEWIDTH",
    "ARROW_ALPHA_INDIVIDUAL", "ARROW_LINEWIDTH_INDIV", "ARROW_MUTATION_INDIV",
    "ARROW_LINEWIDTH_MEAN", "ARROW_MUTATION_MEAN",
    "MEAN_MARKER_SIZE_CLEAN", "MEAN_MARKER_CLEAN",
    "MEAN_MARKER_SIZE_PERTURBED", "MEAN_MARKER_PERTURBED",
    # fonts
    "FONTSIZE_TITLE", "FONTSIZE_SUBTITLE", "FONTSIZE_AXIS_LABEL",
    "FONTSIZE_TICK", "FONTSIZE_LEGEND", "TITLE_WEIGHT",
    # saving
    "SAVE_DPI", "SAVE_BBOX_INCHES", "SAVE_FORMAT_DEFAULT",
    # grid
    "GRID_LINESTYLE", "GRID_ALPHA", "LEGEND_FRAMEALPHA",
    # helper
    "apply_default_style",
]
