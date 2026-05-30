"""
detection/visualization.py
--------------------------
Visualization utilities for ATOMs attention profiles and OOD detection results.

Style and color choices are read from ``viz_config.py`` so that this file and
its counterpart in the Atari project produce visually coherent plots when shown
side-by-side in the thesis.

Functions
---------
fit_pca / fit_tsne / fit_tsne_joint   Dimensionality-reduction helpers
make_output_dirs                      Create standard output subdirectory tree
get_cluster_colors                    K colors matching the cluster scatter colormap

plot_pca_baseline        PCA of baseline attention profiles, coloured by run or cmd
plot_tsne_baseline       t-SNE counterpart of plot_pca_baseline
plot_pca_clusters        PCA coloured by GMM cluster assignment
plot_tsne_clusters       t-SNE counterpart of plot_pca_clusters
plot_pca_ood             PCA with test points overlaid (clean vs perturbed)
plot_tsne_ood            t-SNE counterpart of plot_pca_ood
plot_attention_bar       Mean normalized attention per semantic class
plot_attention_comparison Grouped bar chart; accepts cluster colors for linked styling
plot_roc                 ROC curve for one or multiple detectors
plot_bic_aic             BIC/AIC vs K for GMM model selection
plot_mahal_distribution  Distribution of Mahalanobis scores for in vs out distribution

All functions return matplotlib Figure objects so they can be saved or
shown by the caller -- no plt.show() calls inside.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

# Shared style/colors/sizes -- keep this file's viz_config.py identical to
# the one in the Atari project.  We add the directory of this file to sys.path
# so the import works regardless of how this module is loaded.
import sys as _sys
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
import viz_config as vc

# Apply thesis-wide style (fonts, save DPI, spine visibility, etc.) on import.
vc.apply_default_style()


# Raw CARLA semantic class IDs (CARLA 0.9.x tags 0-28).
# Used for WoR data where the segmentation camera stores these IDs directly.
CARLA_CLASSES: Dict[int, str] = {
    0:  "Unlabeled",
    1:  "Roads",
    2:  "SideWalks",
    3:  "Building",
    4:  "Wall",
    5:  "Fence",
    6:  "Pole",
    7:  "TrafficLight",
    8:  "TrafficSign",
    9:  "Vegetation",
    10: "Terrain",
    11: "Sky",
    12: "Pedestrian",
    13: "Rider",
    14: "Car",
    15: "Truck",
    16: "Bus",
    17: "Train",
    18: "Motorcycle",
    19: "Bycicle",
    20: "Static",
    21: "Dynamic",
    22: "Other",
    23: "Water",
    24: "RoadLine",
    25: "Ground",
    26: "Bridge",
    27: "RailTrack",
    28: "GuardRail"
}

# TransFuser / LEAD grouped semantic class IDs.
# Used for TFV6 data: the LEAD dataset applies save_grouped_semantic=True,
# collapsing the 32 raw CARLA classes into 10 classes via
# constants.SEMANTIC_SEGMENTATION_CONVERTER before saving the PNG.
TFV6_CLASSES: Dict[int, str] = {
    0: "Unlabeled",       # background: sky, buildings, vegetation, sidewalks, …
    1: "Vehicle",         # Car, Truck, Bus, Motorcycle
    2: "Road",            # drivable surface
    3: "TrafficLight",
    4: "Pedestrian",
    5: "RoadLine",        # lane markings
    6: "Obstacle",        # cones, traffic warnings
    7: "SpecialVehicle",
    8: "StopSign",
    9: "Biker",           # Rider, Bicycle
}


def visualize_relevance(relevance, rgb_image=None, alpha=None, save_path: Optional[str] = None, is_brake: bool = None):
    """
    relevance:  [1, 3, H, W] or [3, H, W] tensor (output of attribute_action)
    rgb_image:  [H, W, 3] numpy array in [0,255] for overlay (optional)
    alpha:      overlay transparency (defaults to viz_config.SALIENCY_OVERLAY_ALPHA)
    """
    if alpha is None:
        alpha = vc.SALIENCY_OVERLAY_ALPHA

    rel = relevance.squeeze(0)          # [3, H, W]
    heatmap = rel.sum(dim=0).numpy()    # [H, W]
    heatmap = np.maximum(heatmap, 0)    # keep only positive relevance
    heatmap /= heatmap.max() + 1e-8     # normalize to [0, 1]

    if is_brake is None or is_brake is False:
        cmap = vc.SALIENCY_CMAP_POSITIVE
    else:
        cmap = vc.SALIENCY_CMAP_ALT

    if rgb_image is None:
        plt.imshow(heatmap, cmap=cmap)
        plt.colorbar(); plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
            print(f"[Visualizer] Saved image to {save_path}")
        #plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=vc.FIGSIZE_ATTENTION_OVERLAY)
        axes[0].imshow(heatmap, cmap=cmap);  axes[0].set_title('Relevance'); axes[0].axis('off')
        cmap = plt.get_cmap(cmap)
        colored = (cmap(heatmap)[..., :3] * 255).astype(np.uint8)
        overlay = ((1 - alpha) * rgb_image + alpha * colored).astype(np.uint8)
        axes[1].imshow(overlay)
        axes[1].set_title('Overlay'); axes[1].axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
        else:
            plt.show()
    plt.close()


def visualize_comparative_relevance(relevance, rgb_image=None, alpha=None, save_path: Optional[str] = None, is_brake: bool = None):
    """
    Visualize a signed (drive − brake) relevance map with a diverging colormap.

    relevance:  [1, 3, H, W] or [3, H, W] tensor — should be pre-normalized to [-1, 1]
    rgb_image:  [H, W, 3] numpy array in [0,255] — shown as first panel for reference
    alpha:      unused (per-pixel alpha is derived from signal magnitude)

    Layout (with rgb_image): Input | Drive − Brake | Overlay
    """
    rel = relevance.squeeze(0)          # [3, H, W]
    heatmap = rel.sum(dim=0).numpy()    # [H, W]
    cmap = vc.SALIENCY_CMAP_DIVERGING
    vmax = np.abs(heatmap).max() + 1e-12

    if rgb_image is None:
        plt.imshow(heatmap, cmap=cmap, vmin=-vmax, vmax=vmax)
        plt.colorbar(); plt.axis('off')
        if save_path:
            plt.savefig(save_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
            print(f"[Visualizer] Saved image to {save_path}")
    else:
        fig, axes = plt.subplots(1, 2, figsize=vc.FIGSIZE_ATTENTION_OVERLAY)
        axes[0].imshow(heatmap, cmap=cmap, vmin=-vmax, vmax=vmax)
        axes[0].set_title('Drive − Brake'); axes[0].axis('off')

        heatmap_norm = heatmap / vmax   # [-1, 1], sign preserved
        cmap_fn = plt.get_cmap(cmap)
        colored = (cmap_fn((heatmap_norm + 1) / 2)[..., :3] * 255).astype(np.uint8)
        # Opacity proportional to |signal|: strong features opaque, near-zero transparent
        pixel_alpha = (np.abs(heatmap_norm) ** 0.99)[..., np.newaxis]
        overlay = (pixel_alpha * colored + (1 - pixel_alpha) * rgb_image).astype(np.uint8)
        axes[1].imshow(overlay)
        axes[1].set_title('Overlay'); axes[1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
        else:
            plt.show()
    plt.close()


def visualize_segmentation(
    seg_red:   np.ndarray,
    title:     str = "Segmentation",
    save_path: Optional[str] = None,
    class_map: Optional[Dict[int, str]] = None,
) -> None:
    """
    Visualise a CARLA semantic segmentation map.

    Parameters
    ----------
    seg_red : np.ndarray [H, W] uint8  -- class ID per pixel (red channel of
              CARLA seg image in BGRA layout, i.e. seg_array[:, :, 2])
    class_map : optional dict mapping class ID -> name; defaults to CARLA_CLASSES.
                Pass TFV6_CLASSES when visualising LEAD/TFV6 data.
    """
    import matplotlib.patches as mpatches
    import torch

    _class_map = class_map if class_map is not None else CARLA_CLASSES

    # Ensure numpy
    if isinstance(seg_red, torch.Tensor):
        seg_red = seg_red.cpu().numpy()
    seg_red = np.asarray(seg_red, dtype=np.int32)

    # Build a discrete colormap covering only the classes present
    present_ids = sorted(np.unique(seg_red).tolist())
    n = len(present_ids)
    base_cmap = plt.cm.get_cmap("tab20", max(n, 2))
    # Map original class IDs -> contiguous indices for the colormap
    id_to_idx = {cid: i for i, cid in enumerate(present_ids)}
    remapped  = np.vectorize(id_to_idx.__getitem__)(seg_red)

    # Segmentation is a wide road-image; keep its non-square figsize -- it has
    # no analog in the Atari project so it does not need to match anything.
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(remapped, cmap=base_cmap, vmin=0, vmax=n - 1, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    # Legend showing only classes that appear in the frame
    patches = [
        mpatches.Patch(
            color=base_cmap(id_to_idx[cid]),
            label=f"{cid}: {_class_map.get(cid, 'Unknown')}"
        )
        for cid in present_ids
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.01, 1),
              loc="upper left", fontsize=vc.FONTSIZE_LEGEND,
              framealpha=vc.LEGEND_FRAMEALPHA)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
        print(f"[Visualizer] Saved image to {save_path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# PCA helper
# ---------------------------------------------------------------------------

def fit_pca(data: np.ndarray, n_components: int = 2):
    """Fit a PCA on data and return (pca, transformed_data)."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, random_state=42)
    projected = pca.fit_transform(data)
    return pca, projected


def fit_tsne(data: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    """
    Fit t-SNE on data and return the [N, n_components] embedding.

    kwargs are forwarded to sklearn.manifold.TSNE (e.g. perplexity, n_iter).
    Note: t-SNE is non-parametric — new points cannot be projected onto an
    existing embedding.  Use fit_tsne_joint() when comparing baseline and test
    points in the same space.
    """
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=n_components, random_state=42, **kwargs)
    return tsne.fit_transform(data)


def fit_tsne_joint(
    baseline: np.ndarray,
    test:     np.ndarray,
    **kwargs,
) -> np.ndarray:
    """
    Fit t-SNE on baseline + test concatenated; return [N_b + N_t, 2] embedding.

    The first N_b rows are baseline points, the remaining N_t are test points.
    Reuse this single embedding for the full OOD plot and all per-perturbation
    plots to avoid multiple expensive t-SNE fits.
    """
    combined = np.concatenate([baseline, test], axis=0)
    return fit_tsne(combined, **kwargs)


# ---------------------------------------------------------------------------
# Baseline exploration
# ---------------------------------------------------------------------------

def plot_pca_baseline(
    baseline_series: np.ndarray,
    class_names:     List[str],
    color_by:        Optional[np.ndarray] = None,
    color_label:     str = "run_id",
    title:           str = "Baseline ATOMs — PCA",
):
    """
    PCA scatter of baseline attention profiles.

    Parameters
    ----------
    baseline_series : np.ndarray [N, C]  row-normalized attention profiles.
    class_names     : List[str] [C]
    color_by        : np.ndarray [N] int/float  -- colour-coding variable
                      (e.g. run_id, cmd).  If None, all points same colour.
    color_label     : label for the colorbar / legend.
    title           : figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pca, proj = fit_pca(baseline_series, n_components=2)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)

    if color_by is None:
        ax.scatter(proj[:, 0], proj[:, 1],
                   s=vc.BASELINE_MARKER_SIZE,
                   alpha=vc.BASELINE_MARKER_ALPHA,
                   color=vc.BASELINE_COLOR)
    else:
        sc = ax.scatter(
            proj[:, 0], proj[:, 1],
            c=color_by,
            s=vc.BASELINE_MARKER_SIZE,
            alpha=vc.BASELINE_MARKER_ALPHA,
            cmap=vc.CLUSTER_CMAP,
            vmin=color_by.min(), vmax=color_by.max(),
        )
        fig.colorbar(sc, ax=ax, label=color_label)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_tsne_baseline(
    baseline_series: np.ndarray,
    class_names:     List[str],
    color_by:        Optional[np.ndarray] = None,
    color_label:     str = "run_id",
    title:           str = "Baseline ATOMs — t-SNE",
    tsne_embedding:  Optional[np.ndarray] = None,
    **tsne_kwargs,
) -> "plt.Figure":
    """
    t-SNE scatter of baseline attention profiles.  Mirrors plot_pca_baseline.

    Parameters
    ----------
    tsne_embedding : np.ndarray [N, 2]  pre-computed t-SNE coords.  If None,
                     t-SNE is fitted here (slow for large N — prefer passing the
                     result of fit_tsne(baseline_series) so baseline and cluster
                     plots share the same geometry).
    """
    if tsne_embedding is None:
        tsne_embedding = fit_tsne(baseline_series, **tsne_kwargs)

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)
    if color_by is None:
        ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1],
                   s=vc.BASELINE_MARKER_SIZE, alpha=vc.BASELINE_MARKER_ALPHA,
                   color=vc.BASELINE_COLOR)
    else:
        sc = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=color_by,
                        s=vc.BASELINE_MARKER_SIZE, alpha=vc.BASELINE_MARKER_ALPHA,
                        cmap=vc.CLUSTER_CMAP, vmin=color_by.min(), vmax=color_by.max())
        fig.colorbar(sc, ax=ax, label=color_label)

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pca_clusters(
    baseline_series: np.ndarray,
    cluster_labels:  np.ndarray,
    gmm_means:       Optional[np.ndarray] = None,
    title:           str = "ATOMs — GMM Clusters (PCA)",
):
    """
    PCA scatter coloured by GMM cluster assignment, with optional cluster
    centroids projected into PCA space.

    Parameters
    ----------
    baseline_series : np.ndarray [N, C]
    cluster_labels  : np.ndarray [N] int  -- output of GMMClustering.predict_batch()
    gmm_means       : np.ndarray [K, C]   -- cluster means (optional, plotted as stars)
    title           : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    pca, proj = fit_pca(baseline_series, n_components=2)
    var = pca.explained_variance_ratio_
    k   = len(np.unique(cluster_labels))

    cmap_name = vc.CLUSTER_CMAP if k <= 10 else vc.CLUSTER_CMAP_LARGE

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)
    sc = ax.scatter(
        proj[:, 0], proj[:, 1],
        c=cluster_labels,
        s=vc.BASELINE_MARKER_SIZE,
        alpha=vc.BASELINE_MARKER_ALPHA + 0.2,   # slightly more saturated than plain baseline
        cmap=cmap_name,
        vmin=0, vmax=max(k - 1, 1),
    )
    fig.colorbar(sc, ax=ax, label="Cluster")

    if gmm_means is not None:
        means_proj = pca.transform(gmm_means)
        ax.scatter(
            means_proj[:, 0], means_proj[:, 1],
            marker=vc.CENTROID_MARKER,
            s=vc.CENTROID_SIZE,
            c=range(len(gmm_means)),
            cmap=cmap_name,
            edgecolors=vc.CENTROID_EDGECOLOR,
            linewidths=vc.CENTROID_LINEWIDTH,
            zorder=5,
            label="Cluster centroid",
        )
        ax.legend(fontsize=vc.FONTSIZE_LEGEND)

    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def get_cluster_colors(k: int) -> List:
    """
    Return K colors matching the colormap used in plot_pca_clusters / plot_tsne_clusters.

    The i-th color corresponds to cluster i.  Pass the result as the ``colors``
    argument to plot_attention_comparison() so the attention bars use the same
    colors as the PCA / t-SNE scatter.
    """
    cmap_name = vc.CLUSTER_CMAP if k <= 10 else vc.CLUSTER_CMAP_LARGE
    cmap = plt.get_cmap(cmap_name)
    norm = max(k - 1, 1)
    return [cmap(i / norm) for i in range(k)]


def plot_tsne_clusters(
    baseline_series: np.ndarray,
    cluster_labels:  np.ndarray,
    title:           str = "ATOMs — GMM Clusters (t-SNE)",
    tsne_embedding:  Optional[np.ndarray] = None,
    **tsne_kwargs,
) -> "plt.Figure":
    """
    t-SNE scatter coloured by GMM cluster assignment.  Mirrors plot_pca_clusters.

    Parameters
    ----------
    tsne_embedding : np.ndarray [N, 2]  pre-computed coords.  Pass the same
                     embedding as plot_tsne_baseline so both plots share the
                     same spatial layout.
    """
    if tsne_embedding is None:
        tsne_embedding = fit_tsne(baseline_series, **tsne_kwargs)

    k         = len(np.unique(cluster_labels))
    cmap_name = vc.CLUSTER_CMAP if k <= 10 else vc.CLUSTER_CMAP_LARGE

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)
    sc = ax.scatter(tsne_embedding[:, 0], tsne_embedding[:, 1], c=cluster_labels,
                    s=vc.BASELINE_MARKER_SIZE, alpha=vc.BASELINE_MARKER_ALPHA + 0.2,
                    cmap=cmap_name, vmin=0, vmax=max(k - 1, 1))
    fig.colorbar(sc, ax=ax, label="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pca_ood(
    baseline_series: np.ndarray,
    test_series:     np.ndarray,
    test_labels:     np.ndarray,
    pca=None,
    title:           str = "ATOMs — OOD Detection (PCA)",
):
    """
    PCA plot with baseline samples (grey), clean test samples (blue),
    and perturbed test samples (red).

    Parameters
    ----------
    baseline_series : np.ndarray [N_b, C]
    test_series     : np.ndarray [N_t, C]
    test_labels     : np.ndarray [N_t] int  0 = clean, 1 = perturbed
    pca             : pre-fitted PCA object (optional).  If None, fitted on
                      baseline_series.
    title           : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    if pca is None:
        pca, base_proj = fit_pca(baseline_series, n_components=2)
    else:
        base_proj = pca.transform(baseline_series)

    test_proj = pca.transform(test_series)
    var = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)
    ax.scatter(
        base_proj[:, 0], base_proj[:, 1],
        s=vc.BASELINE_MARKER_SIZE,
        alpha=vc.BASELINE_MARKER_ALPHA,
        color=vc.BASELINE_COLOR,
        label="Baseline", zorder=1,
    )
    clean_mask = test_labels == 0
    ax.scatter(
        test_proj[clean_mask, 0], test_proj[clean_mask, 1],
        s=vc.CLEAN_MARKER_SIZE,
        alpha=vc.CLEAN_MARKER_ALPHA,
        color=vc.CLEAN_TEST_COLOR,
        label="Test clean", zorder=2,
    )
    ax.scatter(
        test_proj[~clean_mask, 0], test_proj[~clean_mask, 1],
        s=vc.PERTURBED_MARKER_SIZE,
        alpha=vc.PERTURBED_MARKER_ALPHA,
        color=vc.PERTURBED_COLOR,
        label="Test perturbed", zorder=3,
    )
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND, markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_tsne_ood(
    baseline_series: np.ndarray,
    test_series:     np.ndarray,
    test_labels:     np.ndarray,
    tsne_embedding:  Optional[np.ndarray] = None,
    title:           str = "ATOMs — OOD Detection (t-SNE)",
    **tsne_kwargs,
) -> "plt.Figure":
    """
    t-SNE plot with baseline (grey), clean test (blue), perturbed (red).
    Mirrors plot_pca_ood.

    Parameters
    ----------
    tsne_embedding : np.ndarray [N_b + N_t, 2]  joint embedding from
                     fit_tsne_joint().  If None, fitted here (slow).
                     Reuse one embedding across all per-perturbation plots
                     so the geometry stays consistent.
    """
    if tsne_embedding is None:
        tsne_embedding = fit_tsne_joint(baseline_series, test_series, **tsne_kwargs)

    n_base    = len(baseline_series)
    base_proj = tsne_embedding[:n_base]
    test_proj = tsne_embedding[n_base:]

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_PCA)
    ax.scatter(base_proj[:, 0], base_proj[:, 1],
               s=vc.BASELINE_MARKER_SIZE, alpha=vc.BASELINE_MARKER_ALPHA,
               color=vc.BASELINE_COLOR, label="Baseline", zorder=1)
    clean_mask = test_labels == 0
    ax.scatter(test_proj[clean_mask, 0], test_proj[clean_mask, 1],
               s=vc.CLEAN_MARKER_SIZE, alpha=vc.CLEAN_MARKER_ALPHA,
               color=vc.CLEAN_TEST_COLOR, label="Test clean", zorder=2)
    ax.scatter(test_proj[~clean_mask, 0], test_proj[~clean_mask, 1],
               s=vc.PERTURBED_MARKER_SIZE, alpha=vc.PERTURBED_MARKER_ALPHA,
               color=vc.PERTURBED_COLOR, label="Test perturbed", zorder=3)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND, markerscale=1.5)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Attention bar charts
# ---------------------------------------------------------------------------

def plot_attention_bar(
    attention:   np.ndarray,
    class_names: List[str],
    title:       str = "Mean Hierarchical Attention",
    error:       Optional[np.ndarray] = None,
    top_k:       Optional[int] = None,
    color:       Optional[str] = None,
):
    """
    Horizontal bar chart of normalized attention per semantic class.

    Parameters
    ----------
    attention   : np.ndarray [C]  -- mean attention (normalized to sum 1).
    class_names : List[str] [C]
    error       : np.ndarray [C]  -- std or CI for error bars (optional).
    top_k       : if set, show only the top-k classes by attention.
    color       : bar colour (defaults to viz_config.CLEAN_TEST_COLOR).

    Returns
    -------
    matplotlib.figure.Figure
    """
    if color is None:
        color = vc.CLEAN_TEST_COLOR

    assert len(attention) == len(class_names)
    idx = np.argsort(attention)
    if top_k is not None:
        idx = idx[-top_k:]

    vals   = attention[idx]
    names  = [class_names[i] for i in idx]
    errors = error[idx] if error is not None else None

    fig, ax = plt.subplots(figsize=vc.figsize_attention_bar(len(idx)))
    y_pos   = np.arange(len(idx))
    ax.barh(y_pos, vals, xerr=errors, color=color, alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=vc.FONTSIZE_TICK)
    ax.set_xlabel("Normalized attention")
    ax.set_title(title)
    ax.set_xlim(0, None)
    fig.tight_layout()
    return fig


def plot_attention_comparison(
    attention_dict: Dict[str, np.ndarray],
    class_names:    List[str],
    top_k:          int = 10,
    title:          str = "ATOMs Attention Comparison",
    colors:         Optional[List] = None,
):
    """
    Grouped bar chart comparing attention profiles across conditions.

    Parameters
    ----------
    attention_dict : dict mapping condition label (str) -> attention [C].
    class_names    : List[str] [C]
    top_k          : show only top-k classes by max attention across conditions.
    colors         : optional list of colors, one per condition.  When provided
                     (e.g. the output of get_cluster_colors(K)), these override
                     the default perturbation-name colour lookup so the bars
                     match the cluster scatter plots exactly.

    Returns
    -------
    matplotlib.figure.Figure
    """
    labels    = list(attention_dict.keys())
    profiles  = np.stack(list(attention_dict.values()), axis=0)  # [n_cond, C]
    max_att   = profiles.max(axis=0)
    top_idx   = np.argsort(max_att)[-top_k:][::-1]

    x     = np.arange(len(top_idx))
    width = 0.8 / len(labels)
    cmap  = plt.get_cmap(vc.CLUSTER_CMAP)

    fig, ax = plt.subplots(figsize=vc.figsize_bar_scaled(top_k, per_item=0.8, min_w=8.0, height=4.0))
    for i, (label, prof) in enumerate(zip(labels, profiles)):
        offset = (i - len(labels) / 2 + 0.5) * width
        if colors is not None:
            bar_color = colors[i % len(colors)]
        else:
            bar_color = vc.get_perturbation_color(label, fallback_index=i)
        ax.bar(x + offset, prof[top_idx], width=width * 0.9,
               label=label, color=bar_color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([class_names[i] for i in top_idx], rotation=30, ha="right", fontsize=vc.FONTSIZE_TICK)
    ax.set_ylabel("Normalized attention")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Detector evaluation
# ---------------------------------------------------------------------------

def plot_roc(
    results_list: List[Dict],
    title:        str = "ROC Curves",
):
    """
    ROC curves for one or multiple detectors.

    Curve colours are derived from each detector's name via string matching
    (e.g. "ATOMs-Mahalanobis (GMM K=5)" -> gmm_mahalanobis colour from
    viz_config.DISTANCE_TYPE_COLORS).  Detectors that don't match any known
    distance type fall back to matplotlib's default cycle.
    """
    fig, ax = plt.subplots(figsize=vc.FIGSIZE_ROC)
    ax.plot([0, 1], [0, 1], color=vc.CHANCE_LINE_COLOR,
            linestyle="--", lw=0.8, label="Chance")

    fallback_idx = 0
    for r in results_list:
        label = f"{r['detector_name']}  (AUC={r['auc']:.3f})"

        dt = _distance_type_from_detector_name(r["detector_name"])
        if dt is not None:
            color = vc.DISTANCE_TYPE_COLORS[dt]
        else:
            color = f"C{fallback_idx % 10}"
            fallback_idx += 1

        ax.plot(r["roc_fpr"], r["roc_tpr"], lw=1.8, label=label, color=color)
        ax.scatter(
            [r["fpr_at_threshold"]], [r["tpr_at_threshold"]],
            s=60, zorder=5, color=color,
        )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND, loc="lower right")
    fig.tight_layout()
    return fig


def plot_mahal_distribution(
    in_scores:  np.ndarray,
    out_scores: np.ndarray,
    threshold:  Optional[float] = None,
    title:      str = "Mahalanobis Score Distribution",
    bins:       int = 50,
):
    """
    Overlapping histograms of in-distribution vs out-of-distribution scores.

    Parameters
    ----------
    in_scores  : np.ndarray [N]  -- scores for clean (in-distribution) samples.
    out_scores : np.ndarray [M]  -- scores for anomalous samples.
    threshold  : float            -- decision threshold (drawn as vertical line).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=vc.FIGSIZE_HISTOGRAM)
    ax.hist(in_scores,  bins=bins, alpha=0.6, color=vc.CLEAN_TEST_COLOR, label="In-distribution", density=True)
    ax.hist(out_scores, bins=bins, alpha=0.6, color=vc.PERTURBED_COLOR,  label="Perturbed",       density=True)
    if threshold is not None:
        ax.axvline(threshold, color=vc.THRESHOLD_COLOR, lw=1.5, linestyle="--",
                   label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND)
    fig.tight_layout()
    return fig


def plot_bic_aic(
    scores_bic: np.ndarray,
    scores_aic: np.ndarray,
    title:      str = "GMM Model Selection",
):
    """
    BIC and AIC vs number of components.

    Parameters
    ----------
    scores_bic : np.ndarray [max_K]  -- output of GMMClustering.select_n_components (bic)
    scores_aic : np.ndarray [max_K]  -- same for aic

    Returns
    -------
    matplotlib.figure.Figure
    """
    ks = np.arange(1, len(scores_bic) + 1)
    fig, ax = plt.subplots(figsize=vc.FIGSIZE_BIC_AIC)
    ax.plot(ks, scores_bic, "o-", label="BIC")
    ax.plot(ks, scores_aic, "s-", label="AIC")
    ax.axvline(ks[np.argmin(scores_bic)], color=vc.CLEAN_TEST_COLOR, linestyle="--", lw=1,
               label=f"Best BIC K={ks[np.argmin(scores_bic)]}")
    ax.axvline(ks[np.argmin(scores_aic)], color=vc.PERTURBED_COLOR, linestyle="--", lw=1,
               label=f"Best AIC K={ks[np.argmin(scores_aic)]}")
    ax.set_xlabel("Number of components K")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND)
    fig.tight_layout()
    return fig


def plot_knn_sensitivity(
    k_values: List[int],
    aucs:     List[float],
    best_k:   int,
    title:    str = "k-NN Sensitivity Analysis — AUC vs k",
) -> "plt.Figure":
    """
    Bar chart of ROC-AUC as a function of k for the k-NN detector.

    The bar for `best_k` is highlighted in the k-NN color; all others are
    shown in the muted baseline grey so the winner stands out immediately.
    Each bar is annotated with its AUC value.

    Parameters
    ----------
    k_values : List[int]   — k values evaluated (x-axis).
    aucs     : List[float] — corresponding AUC scores (y-axis).
    best_k   : int         — k that achieved the highest AUC (highlighted).
    title    : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors = [
        vc.DISTANCE_TYPE_COLORS["knn"] if k == best_k else vc.BASELINE_COLOR
        for k in k_values
    ]
    labels = [str(k) for k in k_values]

    fig, ax = plt.subplots(
        figsize=vc.figsize_bar_scaled(len(k_values), per_item=1.2, min_w=5.0, height=4.0)
    )
    bars = ax.bar(labels, aucs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)

    for bar, auc in zip(bars, aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{auc:.3f}",
            ha="center", va="bottom",
            fontsize=vc.FONTSIZE_TICK,
        )

    y_lo = max(0.0, min(aucs) - 0.05)
    y_hi = min(1.0, max(aucs) + 0.07)
    ax.set_ylim(y_lo, y_hi)
    ax.axhline(
        max(aucs),
        color=vc.DISTANCE_TYPE_COLORS["knn"], lw=0.8, linestyle="--", alpha=0.5,
        label=f"Best AUC = {max(aucs):.3f}  (k={best_k})",
    )
    ax.set_xlabel("k  (number of neighbours)")
    ax.set_ylabel("AUC (ROC)")
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig, path: str, dpi: int = None) -> None:
    """Save a matplotlib figure to disk, creating parent dirs as needed."""
    if dpi is None:
        dpi = vc.SAVE_DPI
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches=vc.SAVE_BBOX_INCHES)
    plt.close(fig)
    print(f"[visualization] Saved → {p}")


def make_output_dirs(base_dir) -> dict:
    """
    Create the standard output subdirectory tree under base_dir.

    Returns a dict mapping category name to Path so callers can write
    ``save_figure(fig, dirs["pca"] / "my_plot.png")`` instead of constructing
    paths manually.

    Subdirectories
    --------------
    pca/        PCA scatter plots (baseline, clusters, OOD overlays)
    tsne/       t-SNE counterparts of each PCA plot
    roc/        ROC curves (full test set and per perturbation type)
    attention/  Attention bar charts (mean, per cluster, per command)
    scores/     Score distribution histograms (Mahalanobis, etc.)
    clustering/ GMM model selection (BIC / AIC)
    """
    base_dir = Path(base_dir)
    cats = ["pca", "tsne", "roc", "attention", "scores", "clustering"]
    subdirs = {c: base_dir / c for c in cats}
    for d in subdirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return subdirs


# ---------------------------------------------------------------------------
# Whitened PCA (geometrically correct for Mahalanobis-based detectors)
# ---------------------------------------------------------------------------

def fit_whitened_pca(
    data:         np.ndarray,
    cov:          np.ndarray,
    n_components: int = 2,
) -> Tuple:
    """
    PCA whitening using the provided covariance matrix.

    Projects onto the top n_components principal directions of cov and scales
    by lambda^{-1/2}, so that Euclidean distance in the projected space equals
    Mahalanobis distance along those components.  This is the geometrically
    correct representation for Mahalanobis-based detectors.

    Returns
    -------
    mean              : np.ndarray [D]
    V_top             : np.ndarray [D, n_components]  top eigenvectors of cov.
    lam_top           : np.ndarray [n_components]     corresponding eigenvalues.
    explained_var_ratio : np.ndarray [n_components]
    projected         : np.ndarray [N, n_components]  whitened baseline coords.
    """
    mean = data.mean(axis=0)
    eigenvalues, V = np.linalg.eigh(cov)
    eigenvalues    = np.maximum(eigenvalues, 1e-12)

    idx     = np.argsort(eigenvalues)[::-1][:n_components]
    V_top   = V[:, idx]
    lam_top = eigenvalues[idx]

    explained_var_ratio = lam_top / eigenvalues.sum()

    projected = (data - mean) @ V_top / np.sqrt(lam_top)
    return mean, V_top, lam_top, explained_var_ratio, projected


def apply_whitened_pca(
    data:    np.ndarray,
    mean:    np.ndarray,
    V_top:   np.ndarray,
    lam_top: np.ndarray,
) -> np.ndarray:
    """Apply a previously fitted whitened PCA to new data."""
    return (data - mean) @ V_top / np.sqrt(lam_top)


# ---------------------------------------------------------------------------
# Perturbation displacement statistics
# ---------------------------------------------------------------------------

def compute_perturbation_displacement_stats(
    paired_dict: Dict[str, Dict[str, np.ndarray]],
) -> Dict:
    """
    Compute displacement statistics for each perturbation type in attention space.

    For each perturbation type k:
      - displacement vectors  d_i = perturbed_i − clean_i          [N_k, C]
      - mean displacement     mu_k = mean(d_i)                      [C]
      - per-sample magnitudes ||d_i||                               [N_k]
      - within-type coherence R_k = ||mean(d_i / ||d_i||)||         scalar
          R=0: displacements are isotropic (no consistent direction)
          R=1: every sample moves in exactly the same direction
    """
    types      = list(paired_dict.keys())
    mean_disps = {}
    mags       = {}
    coherence  = {}

    for name, pair in paired_dict.items():
        clean     = pair["clean"]
        perturbed = pair["perturbed"]
        disp      = perturbed - clean

        mean_disps[name] = disp.mean(axis=0)
        mags[name]       = np.linalg.norm(disp, axis=1)

        norms          = mags[name][:, None].clip(1e-12)
        unit_disps     = disp / norms
        coherence[name] = float(np.linalg.norm(unit_disps.mean(axis=0)))

    return {
        "types":            types,
        "mean_disp":        mean_disps,
        "magnitudes":       mags,
        "mean_magnitude":   {n: float(v.mean()) for n, v in mags.items()},
        "std_magnitude":    {n: float(v.std())  for n, v in mags.items()},
        "within_coherence": coherence,
    }


def format_displacement_stats_text(stats: Dict) -> str:
    """Format displacement statistics as a human-readable text block."""
    lines = ["Perturbation Displacement Statistics (attention space)", "=" * 55]
    for name in stats["types"]:
        n = len(stats["magnitudes"][name])
        lines.append(f"\n[{name}]  n={n}")
        lines.append(f"  Mean displacement magnitude  : {stats['mean_magnitude'][name]:.6f}")
        lines.append(f"  Std  displacement magnitude  : {stats['std_magnitude'][name]:.6f}")
        lines.append(f"  Within-type coherence (R)    : {stats['within_coherence'][name]:.4f}"
                     "  (0=isotropic, 1=perfectly aligned)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trajectory visualizations
# ---------------------------------------------------------------------------

def plot_pca_perturbation_trajectories(
    baseline_profiles: np.ndarray,
    paired_dict:       Dict[str, Dict[str, np.ndarray]],
    pca                = None,
    cov:               Optional[np.ndarray] = None,
    subsample:         int = 60,
    arrow_alpha:       float = None,
    mean_arrow_scale:  float = 1.0,
    title:             str = "Attention Trajectories Under Perturbation (PCA)",
    class_names:       Optional[List[str]] = None,
) -> Tuple:
    """
    Visualise how perturbations move samples in ATOMs attention space.

    For each perturbation type, draws:
      - Thin arrows from each (subsampled) clean point to its perturbed image
      - A bold arrow from the clean centroid to the perturbed centroid,
        representing the mean population shift

    If `cov` is supplied the plot uses **whitened PCA**, so that Euclidean
    distance in the plot equals Mahalanobis distance in feature space -- the
    correct geometry for Mahalanobis-based detectors.
    """
    if arrow_alpha is None:
        arrow_alpha = vc.ARROW_ALPHA_INDIVIDUAL

    whitened  = cov is not None
    proj_info = {}

    # ---- Fit projection on baseline ----------------------------------------
    if whitened:
        mean_b, V_top, lam_top, var, base_proj = fit_whitened_pca(
            baseline_profiles, cov, n_components=2
        )
        proj_info = {"mean": mean_b, "V_top": V_top, "lam_top": lam_top}
        _project  = lambda x: apply_whitened_pca(x, mean_b, V_top, lam_top)
        axis_label = "Whitened PC"
    else:
        if pca is None:
            pca_fit, base_proj = fit_pca(baseline_profiles, n_components=2)
        else:
            pca_fit   = pca
            base_proj = pca_fit.transform(baseline_profiles)
        proj_info = {"pca": pca_fit}
        _project  = lambda x: pca_fit.transform(x)
        var = pca_fit.explained_variance_ratio_
        axis_label = "PC"

    # ---- Per-perturbation colors (semantic, shared with Atari project) -----
    types  = list(paired_dict.keys())
    colors = {name: vc.get_perturbation_color(name, fallback_index=i)
              for i, name in enumerate(types)}

    fig, ax = plt.subplots(figsize=vc.FIGSIZE_TRAJECTORY)

    # Baseline cloud
    ax.scatter(
        base_proj[:, 0], base_proj[:, 1],
        s=vc.BASELINE_MARKER_SIZE,
        alpha=vc.BASELINE_MARKER_ALPHA,
        color=vc.BASELINE_COLOR,
        label="Baseline", zorder=1,
    )

    rng = np.random.default_rng(0)

    for name, pair in paired_dict.items():
        clean_proj     = _project(pair["clean"])
        perturbed_proj = _project(pair["perturbed"])
        color          = colors[name]
        N              = len(clean_proj)

        # Subsample indices for individual arrows
        idx = rng.choice(N, size=min(subsample, N), replace=False)

        # Individual arrows
        for i in idx:
            dx = perturbed_proj[i, 0] - clean_proj[i, 0]
            dy = perturbed_proj[i, 1] - clean_proj[i, 1]
            ax.annotate(
                "", xy=(clean_proj[i, 0] + dx, clean_proj[i, 1] + dy),
                xytext=(clean_proj[i, 0], clean_proj[i, 1]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    alpha=arrow_alpha,
                    lw=vc.ARROW_LINEWIDTH_INDIV,
                    mutation_scale=vc.ARROW_MUTATION_INDIV,
                ),
                zorder=2,
            )

        # Clean centroid (hollow marker)
        c_mean = clean_proj.mean(axis=0)
        p_mean = perturbed_proj.mean(axis=0)

        ax.scatter(*c_mean,
                   s=vc.MEAN_MARKER_SIZE_CLEAN, color=color,
                   edgecolors=vc.TRAJ_ENDPOINT_EDGECOLOR,
                   linewidths=vc.TRAJ_ENDPOINT_LINEWIDTH,
                   zorder=4, marker=vc.MEAN_MARKER_CLEAN)

        # Mean displacement arrow (bold)
        dx_m = (p_mean[0] - c_mean[0]) * mean_arrow_scale
        dy_m = (p_mean[1] - c_mean[1]) * mean_arrow_scale
        ax.annotate(
            "", xy=(c_mean[0] + dx_m, c_mean[1] + dy_m),
            xytext=(c_mean[0], c_mean[1]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=vc.ARROW_LINEWIDTH_MEAN,
                mutation_scale=vc.ARROW_MUTATION_MEAN,
            ),
            zorder=5,
        )

        # Perturbed centroid (filled marker)
        ax.scatter(*p_mean,
                   s=vc.MEAN_MARKER_SIZE_PERTURBED, color=color,
                   edgecolors=vc.TRAJ_ENDPOINT_EDGECOLOR,
                   linewidths=vc.TRAJ_ENDPOINT_LINEWIDTH,
                   zorder=4, marker=vc.MEAN_MARKER_PERTURBED)

        # Invisible scatter for legend entry
        ax.scatter([], [], color=color, label=name, s=40, marker="o")

    # ---- Labels & decoration -----------------------------------------------
    xlabel = f"{axis_label}1 ({var[0]*100:.1f}%)"
    ylabel = f"{axis_label}2 ({var[1]*100:.1f}%)"
    if whitened:
        xlabel += "  [Mahal.-whitened]"
        ylabel += "  [Mahal.-whitened]"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND, markerscale=1.4, loc="best")

    # Optional subtitle: top contributing attention classes in PC1
    if class_names is not None and not whitened:
        top3       = np.argsort(np.abs(pca_fit.components_[0]))[-3:][::-1]
        names_top3 = [class_names[i] for i in top3]
        ax.set_title(title + f"\nPC1 top drivers: {', '.join(names_top3)}",
                     fontsize=vc.FONTSIZE_SUBTITLE)
    elif class_names is not None and whitened:
        top3       = np.argsort(np.abs(V_top[:, 0]))[-3:][::-1]
        names_top3 = [class_names[i] for i in top3]
        ax.set_title(title + f"\nPC1 top drivers: {', '.join(names_top3)}",
                     fontsize=vc.FONTSIZE_SUBTITLE)

    fig.tight_layout()
    return fig, proj_info


def plot_displacement_coherence_bar(
    stats: Dict,
    title: str = "Within-Type Displacement Coherence",
) -> "plt.Figure":
    """
    Bar chart of within-type displacement coherence (mean resultant length R)
    for each perturbation type.

    R = 0  ->  displacements scatter isotropically (no consistent direction).
    R = 1  ->  every displaced sample moves in exactly the same direction.
    """
    types     = stats["types"]
    coherence = [stats["within_coherence"][t] for t in types]
    bar_colors = [vc.get_perturbation_color(t, fallback_index=i)
                  for i, t in enumerate(types)]

    fig, ax = plt.subplots(
        figsize=vc.figsize_bar_scaled(len(types), per_item=0.9, min_w=4.0, height=3.5)
    )
    bars = ax.bar(types, coherence, color=bar_colors,
                  alpha=0.85, edgecolor="black", linewidth=0.6)

    for bar, val in zip(bars, coherence):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=vc.FONTSIZE_TICK)

    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="grey", lw=0.8, linestyle="--")
    ax.set_ylabel("Mean resultant length R  (0 = isotropic, 1 = aligned)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_displacement_magnitude_boxplot(
    stats: Dict,
    title: str = "Displacement Magnitude per Perturbation Type",
) -> "plt.Figure":
    """
    Box plot of per-sample displacement magnitudes (L2 in attention space)
    for each perturbation type.  Gives a sense of how consistently and how
    strongly each perturbation moves samples.
    """
    types = stats["types"]
    data  = [stats["magnitudes"][t] for t in types]

    fig, ax = plt.subplots(
        figsize=vc.figsize_bar_scaled(len(types), per_item=1.1, min_w=5.0, height=4.0)
    )
    bp = ax.boxplot(data, labels=types, patch_artist=True, notch=False)

    for i, (patch, t) in enumerate(zip(bp["boxes"], types)):
        patch.set_facecolor(vc.get_perturbation_color(t, fallback_index=i))
        patch.set_alpha(0.7)

    ax.set_ylabel("L2 displacement (attention space)")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_distance_over_time(
        dist: np.ndarray,
        perturbation: str,
        distance_type: str,
        results_dir: Path = None,
    ) -> None:
    """
    Plot mean distance vs frame index for a single perturbation type.

    Parameters
    ----------
    dist          : np.ndarray [T]  -- mean distance per frame.
    perturbation  : str            -- perturbation name (used in title & filename).
    distance_type : str            -- one of: knn, jsd, wasserstein, gmm_*, mahalanobis, euclidean.
    results_dir   : Path           -- directory to save the figure.  REQUIRED.

    Notes
    -----
    `results_dir` used to default to a hardcoded Windows path; that has been
    removed.  Callers must now pass a results directory explicitly.
    """
    if results_dir is None:
        raise ValueError(
            "plot_distance_over_time: results_dir is required.  "
            "Pass a Path explicitly (the previous hardcoded default has been removed)."
        )

    color, ylabel = _get_plot_style(distance_type)
    frame_idx = np.arange(len(dist))
    fig, ax = plt.subplots(figsize=vc.FIGSIZE_DISTANCE_OVER_TIME)
    ax.plot(frame_idx, dist, "o-", color=color, linewidth=2, label="Mean across agents")

    ax.set_title(f"{ylabel} from Baseline\n — {perturbation}")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=vc.FONTSIZE_LEGEND)
    ax.grid(True, linestyle=vc.GRID_LINESTYLE, alpha=vc.GRID_ALPHA)
    fig.tight_layout()

    fig_path = Path(results_dir) / f"{distance_type}_{perturbation}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=vc.SAVE_DPI, bbox_inches=vc.SAVE_BBOX_INCHES)
    plt.close(fig)
    print(f"[INFO] Saved plot → {fig_path}")
    return None

def _distance_type_from_detector_name(name: str) -> Optional[str]:
    """
    Best-effort: extract a DISTANCE_TYPE_COLORS key from a detector display
    name (e.g. "ATOMs-Mahalanobis (GMM K=5)" -> "gmm_mahalanobis").

    Returns None for names that aren't distance-based (e.g. "Action entropy",
    "MDX Detection") so the caller can fall back to a default colour cycle.
    """
    n = name.lower()
    gmm = "gmm" in n
    if "knn" in n or "k-nn" in n:
        return "knn"
    if "jsd" in n or "jensen" in n:
        return "jsd"
    if "wasserstein" in n:
        return "wasserstein"
    if "peoc" in n or "entropy" in n or "PEOC" in n:
        return "peoc"
    if "mdx" in n:
        return "mdx"
    if "mahalanobis" in n or "mahal" in n:
        return "gmm_mahalanobis" if gmm else "mahalanobis"
    if "euclidean" in n or "euclid" in n:
        return "gmm_euclidean" if gmm else "euclidean"

    return None


def _get_plot_style(distance_type: str) -> Tuple[str, str]:
    """Return (color, ylabel) for a given distance type string.

    Looks up DISTANCE_TYPE_COLORS / DISTANCE_TYPE_YLABELS in viz_config, using
    substring matching on `distance_type` so that callers can pass strings
    like "mahalanobis_full" or "gmm_mahalanobis_run3" and still get a hit.
    """
    dt = distance_type.lower()

    # Most-specific keys first so "gmm_mahalanobis" matches before "mahalanobis".
    if "knn" in dt:
        key = "knn"
    elif "jsd" in dt:
        key = "jsd"
    elif "wasserstein" in dt:
        key = "wasserstein"
    elif "peoc" in dt or "entropy" in dt or "PEOC" in dt:
        key = "peoc"
    elif "mdx" in dt:
        key = "mdx"
    elif "gmm" in dt and "mahalanobis" in dt:
        key = "gmm_mahalanobis"
    elif "gmm" in dt:
        key = "gmm_euclidean"
    elif "mahalanobis" in dt:
        key = "mahalanobis"
    else:
        key = "euclidean"

    return vc.DISTANCE_TYPE_COLORS[key], vc.DISTANCE_TYPE_YLABELS[key]
