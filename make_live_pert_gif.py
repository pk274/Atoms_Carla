#!/usr/bin/env python3
"""
make_live_pert_gif.py
---------------------
Create a presentation-ready animated GIF from a live-perturbation run.

Per-frame layout
----------------
  ┌────────────────────────────────────────────────────────┐
  │   RGB camera image  [green border=clean / red=perturbed]│
  │   [optional: + segmentation panel / + diff panel]       │
  ├────────────────────────┬───────────────────────────────┤
  │  ATOMs attention bars  │  Mahalanobis distance trace   │
  │  (current vs baseline) │  + threshold + OOD indicator  │
  └────────────────────────┴───────────────────────────────┘

Quick start
-----------
  # PGD nocrash run (auto-picks the run, mode 2)
  python make_live_pert_gif.py --pert pgd

  # Brightness scale heavy-crash with segmentation panel
  python make_live_pert_gif.py --pert brightness_scale --run heavycrash_170131_000 --show-seg

  # PGD nocrash with amplified perturbation diff
  python make_live_pert_gif.py --pert pgd --run nocrash_155706_000 --show-diff

  # Phantom obstacle, slower playback
  python make_live_pert_gif.py --pert phantom_obstacle --run noreaction_150958_000 --fps 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import inv

try:
    from PIL import Image
except ImportError:
    print("PIL not found — install Pillow:  pip install Pillow")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TFV6_CLASSES = {
    0: "Unlabeled",
    1: "Vehicle",
    2: "Road",
    3: "TrafficLight",
    4: "Pedestrian",
    5: "RoadLine",
    6: "Obstacle",
    7: "SpecialVehicle",
    8: "StopSign",
    9: "Biker",
}

# Per-class color palette for segmentation overlay (RGB 0-255)
SEG_PALETTE: dict[int, tuple] = {
    0: (100, 100, 100),   # Unlabeled  – gray
    1: (  0, 114, 189),   # Vehicle    – blue
    2: ( 60,  60,  60),   # Road       – dark gray
    3: (237, 176,  33),   # TrafficLight – amber
    4: (216,  82,  24),   # Pedestrian – red-orange
    5: (240, 240, 240),   # RoadLine   – white
    6: (255, 140,   0),   # Obstacle   – orange
    7: (  0, 200, 200),   # SpecialVehicle – cyan
    8: (148,   0, 211),   # StopSign   – purple
    9: (  0, 200,  80),   # Biker      – green
}

DATA_ROOT   = Path("data/TFV6")
RESULTS_DIR = DATA_ROOT / "results_alt"   # always use the alternative (same-distribution) split

# Colors matching viz_config.py style
C_CLEAN     = "#43a047"   # green
C_PERTURBED = "#e53935"   # red
C_THRESHOLD = "black"
C_CURSOR    = "#ffa000"   # amber

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a presentation GIF from a live-perturbation run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pert",      default="pgd",
                   choices=["pgd", "phantom_obstacle", "brightness_scale", "gaussian_noise"],
                   help="Perturbation type (default: pgd)")
    p.add_argument("--run",       default=None,
                   help="Run label, e.g. nocrash_155706_000 (auto if omitted)")
    p.add_argument("--mode",      default=2, type=int, choices=[1, 2],
                   help="ATOMs analysis mode (default: 2)")
    p.add_argument("--fps",       default=8, type=int,
                   help="GIF frame rate in frames/second (default: 8)")
    p.add_argument("--output",    default=None, type=Path,
                   help="Output GIF path (default: gifs/<auto>.gif)")
    p.add_argument("--show-seg",  action="store_true",
                   help="Add semantic segmentation panel next to the RGB image")
    p.add_argument("--show-diff", action="store_true",
                   help="Add amplified perturbation diff panel (PGD only, needs clean_rgb)")
    p.add_argument("--show-pca",  action="store_true",
                   help="Add PCA panel showing current frame in baseline attention space")
    p.add_argument("--dpi",       default=85, type=int,
                   help="Render DPI (lower = smaller file, default: 85)")
    p.add_argument("--list-runs", action="store_true",
                   help="List available runs and exit")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def list_available_runs(pert: str, mode: int) -> list[tuple[str, bool]]:
    """Return [(run_label, has_profiles), ...] for all runs of this pert type."""
    frames_dir   = DATA_ROOT / "test_data_alt" / "live_pert_frames"
    profiles_dir = DATA_ROOT / "test_data_alt" / "attention" / "live_pert" / pert
    runs = []
    for npz in sorted(frames_dir.glob(f"run_{pert}_live_pert_*.npz")):
        if "_clean_rgb" in npz.name:
            continue
        stem  = npz.stem.removeprefix(f"run_{pert}_live_pert_")
        prof  = profiles_dir / f"live_pert_profiles_{stem}_{mode}.npy"
        clean = frames_dir / f"run_{pert}_live_pert_{stem}_clean_rgb.npz"
        runs.append((stem, prof.exists(), clean.exists()))
    return runs


def find_run(pert: str, mode: int) -> str:
    runs = list_available_runs(pert, mode)
    for stem, has_prof, _ in runs:
        if has_prof:
            return stem
    raise FileNotFoundError(
        f"No run with profiles found for pert={pert} mode={mode}.\n"
        f"Available frames: {[r[0] for r in runs]}"
    )


def load_frames(pert: str, run: str) -> dict:
    path = DATA_ROOT / "test_data_alt" / "live_pert_frames" / f"run_{pert}_live_pert_{run}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Frame file not found: {path}")
    return np.load(path, allow_pickle=True)


def load_profiles(pert: str, run: str, mode: int) -> np.ndarray:
    path = (DATA_ROOT / "test_data_alt" / "attention" / "live_pert"
            / pert / f"live_pert_profiles_{run}_{mode}.npy")
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    return np.load(path)


def load_clean_rgb(pert: str, run: str) -> np.ndarray | None:
    path = (DATA_ROOT / "test_data_alt" / "live_pert_frames"
            / f"run_{pert}_live_pert_{run}_clean_rgb.npz")
    if path.exists():
        return np.load(path, allow_pickle=True)["wide_rgb"]
    return None


def load_logits(pert: str, run: str, mode: int) -> np.ndarray | None:
    path = (DATA_ROOT / "test_data_alt" / "attention" / "live_pert"
            / pert / f"live_pert_speed_logits_{run}_{mode}.npy")
    return np.load(path) if path.exists() else None


def load_detector(mode: int) -> tuple[np.ndarray, np.ndarray, float]:
    path = RESULTS_DIR / f"atoms_analysis_mode_{mode}" / "mahal_detector.npz"
    if not path.exists():
        raise FileNotFoundError(f"Detector not found: {path}")
    det = np.load(path, allow_pickle=True)
    mean  = det["mean"].astype(np.float64)
    cov   = det["cov"].astype(np.float64)
    ridge = float(det["ridge"])
    threshold = float(det["threshold"])
    reg_cov = cov + ridge * np.eye(len(mean))
    return mean, inv(reg_cov), threshold


def load_gmm(mode: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (means [K,D], inv_covs [K,D,D]) from the fitted GMM."""
    path = RESULTS_DIR / f"atoms_analysis_mode_{mode}" / "gmm.npz"
    if not path.exists():
        raise FileNotFoundError(f"GMM not found: {path}")
    gmm   = np.load(path, allow_pickle=True)
    means = gmm["means"].astype(np.float64)           # (K, D)
    covs  = gmm["covariances"].astype(np.float64)     # (K, D, D)
    ridge = float(gmm["ridge"])
    D     = means.shape[1]
    inv_covs = np.stack([inv(c + ridge * np.eye(D)) for c in covs])  # (K, D, D)
    return means, inv_covs


def load_baseline_series(mode: int) -> np.ndarray:
    """Load the alt-split baseline ATOMs profile series (N, D)."""
    path = DATA_ROOT / "baseline_data_alt" / f"baseline_{mode}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Baseline series not found: {path}")
    return np.load(path, allow_pickle=True)["series"].astype(np.float64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_distances(profiles: np.ndarray,
                      mean: np.ndarray,
                      inv_cov: np.ndarray) -> np.ndarray:
    diff = profiles.astype(np.float64) - mean
    return np.sqrt(np.einsum("ni,ij,nj->n", diff, inv_cov, diff))


def assign_clusters(profiles: np.ndarray,
                    gmm_means: np.ndarray,
                    gmm_inv_covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign each profile to its nearest GMM component (by Mahalanobis distance).

    Returns
    -------
    assignments : (N,) int   — cluster index per profile
    cluster_means : (N, D)   — nearest cluster mean per profile
    """
    K = len(gmm_means)
    assignments = np.empty(len(profiles), dtype=np.intp)
    for i, x in enumerate(profiles.astype(np.float64)):
        dists = np.array([
            (x - gmm_means[k]) @ gmm_inv_covs[k] @ (x - gmm_means[k])
            for k in range(K)
        ])
        assignments[i] = np.argmin(dists)
    return assignments, gmm_means[assignments]


# ---------------------------------------------------------------------------
# PCA context (built once in main, reused across frames)
# ---------------------------------------------------------------------------

class PCAContext:
    """Holds a fitted PCA and precomputed baseline projections for the PCA panel."""

    def __init__(self,
                 pca,
                 baseline_proj:      np.ndarray,   # (N, 2)
                 baseline_clusters:  np.ndarray,   # (N,) int
                 gmm_means_proj:     np.ndarray,   # (K, 2)
                 cluster_colors:     list):
        self.pca               = pca
        self.baseline_proj     = baseline_proj
        self.baseline_clusters = baseline_clusters
        self.gmm_means_proj    = gmm_means_proj
        self.cluster_colors    = cluster_colors
        self.K                 = len(gmm_means_proj)


class _NumpyPCA:
    """Minimal PCA (n_components=2) backed by numpy SVD — avoids sklearn/scipy.stats import."""

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self._mean = X.mean(axis=0)
        Xc = X - self._mean
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self._components = Vt[:2]          # (2, D)
        total_var = (s ** 2).sum()
        self.explained_variance_ratio_ = (s[:2] ** 2) / total_var if total_var > 0 else np.zeros(2)
        return Xc @ self._components.T     # (N, 2)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) @ self._components.T


def build_pca_context(baseline_series: np.ndarray,
                      gmm_means: np.ndarray,
                      gmm_inv_covs: np.ndarray) -> PCAContext:
    pca  = _NumpyPCA()
    proj = pca.fit_transform(baseline_series)          # (N, 2)

    K = len(gmm_means)
    # Assign each baseline point to its nearest cluster for background coloring
    assignments, _ = assign_clusters(baseline_series, gmm_means, gmm_inv_covs)

    # Stable color palette: tab10 for K≤10, tab20 for K≤20, fallback cycler after
    cmap = plt.cm.get_cmap("tab10" if K <= 10 else "tab20")
    colors = [cmap(k % cmap.N) for k in range(K)]

    means_proj = pca.transform(gmm_means)              # (K, 2)

    return PCAContext(pca, proj, assignments, means_proj, colors)


def _draw_pca_panel(ax,
                    pca_ctx:          PCAContext,
                    live_profiles:    np.ndarray,   # all live-pert profiles (N, D)
                    live_assignments: np.ndarray,   # (N,) cluster ids for live frames
                    frame_idx:        int,
                    trail_len:        int = 12) -> None:
    """Render the PCA panel for frame `frame_idx`."""
    K = pca_ctx.K

    # --- baseline cloud (one scatter per cluster, low alpha) ---
    for k in range(K):
        mask = pca_ctx.baseline_clusters == k
        if mask.any():
            ax.scatter(
                pca_ctx.baseline_proj[mask, 0],
                pca_ctx.baseline_proj[mask, 1],
                c=[pca_ctx.cluster_colors[k]], alpha=0.12, s=5,
                linewidths=0, rasterized=True,
            )

    # --- cluster centroids (stars) ---
    for k in range(K):
        ax.scatter(
            pca_ctx.gmm_means_proj[k, 0],
            pca_ctx.gmm_means_proj[k, 1],
            marker="*", s=180, c=[pca_ctx.cluster_colors[k]],
            edgecolors="black", linewidths=0.6, zorder=6,
        )

    # --- trail of recent live frames (fading dots) ---
    t_start = max(0, frame_idx - trail_len)
    if t_start < frame_idx:
        trail_profiles = live_profiles[t_start:frame_idx].astype(np.float64)
        trail_proj     = pca_ctx.pca.transform(trail_profiles)    # (trail, 2)
        n_trail        = len(trail_proj)
        for j in range(n_trail):
            alpha = 0.15 + 0.55 * (j / max(n_trail - 1, 1))
            size  = 8   + 30  * (j / max(n_trail - 1, 1))
            k_j   = int(live_assignments[t_start + j])
            ax.scatter(
                trail_proj[j, 0], trail_proj[j, 1],
                c=[pca_ctx.cluster_colors[k_j]], alpha=alpha, s=size,
                edgecolors="none", zorder=7,
            )

    # --- current frame (large, cluster-colored, white edge) ---
    cur_proj = pca_ctx.pca.transform(
        live_profiles[frame_idx].astype(np.float64).reshape(1, -1)
    )[0]
    k_cur = int(live_assignments[frame_idx])
    ax.scatter(
        cur_proj[0], cur_proj[1],
        c=[pca_ctx.cluster_colors[k_cur]], s=140, marker="o",
        edgecolors="white", linewidths=1.8, zorder=10,
    )

    var_exp = pca_ctx.pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=8)
    ax.set_title("Attention Profile in PCA Space", fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def speed_from_logits(logits_row: np.ndarray) -> float:
    """Decode predicted speed (m/s) from 8-bin logits via softmax + weighted sum."""
    bins = np.array([0.0, 4.0, 8.0, 10.0, 13.89, 16.0, 17.78, 20.0])
    probs = np.exp(logits_row - logits_row.max())
    probs /= probs.sum()
    return float(probs @ bins)


def seg_to_rgb(seg_map: np.ndarray) -> np.ndarray:
    """(H, W) class-ID map → (H, W, 3) RGB."""
    rgb = np.zeros((*seg_map.shape, 3), dtype=np.uint8)
    for cls, color in SEG_PALETTE.items():
        rgb[seg_map == cls] = color
    return rgb


def chw_to_hwc(img: np.ndarray) -> np.ndarray:
    return np.transpose(img, (1, 2, 0))


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = canvas.get_width_height()
    return Image.frombuffer("RGBA", (w, h), buf, "raw", "RGBA", 0, 1).convert("RGB")

# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frame(
    i:               int,
    frame_data,
    profiles:        np.ndarray,
    distances:       np.ndarray,
    logits:          np.ndarray | None,
    cluster_mean:    np.ndarray,         # nearest GMM cluster mean for this frame
    live_assignments: np.ndarray,        # (N,) cluster index per live frame
    threshold:       float,
    onset_idx:       int,
    run_label:       str,
    pert_label:      str,
    clean_rgb,
    show_seg:        bool,
    show_diff:       bool,
    pca_ctx:         "PCAContext | None",
    dpi:             int,
) -> plt.Figure:

    rgb_hwc  = chw_to_hwc(frame_data["wide_rgb"][i])
    seg_map  = frame_data["seg_red_wide"][i]
    is_pert  = bool(frame_data["is_perturbed"][i])
    speed_ms = float(frame_data["speed"][i])
    profile  = profiles[i]
    n_frames = len(distances)

    # Predicted speed from logits (if available)
    pred_speed_str = ""
    if logits is not None:
        pred_ms = speed_from_logits(logits[i])
        pred_speed_str = f"  |  pred {pred_ms:.1f} m/s"

    # Colour scheme
    border_col = C_PERTURBED if is_pert else C_CLEAN
    bar_col    = C_PERTURBED if is_pert else "#1565c0"

    # --- top panel count ---
    show_pca_panel = pca_ctx is not None
    n_top = 1 + int(show_seg) + int(show_diff and clean_rgb is not None) + int(show_pca_panel)

    # --- figure ---
    fig = plt.figure(figsize=(15, 8), dpi=dpi)
    fig.patch.set_facecolor("#f5f5f5")

    gs_outer = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[2.6, 1],
        hspace=0.10,
        top=0.93, bottom=0.07, left=0.05, right=0.98,
    )
    gs_top = gridspec.GridSpecFromSubplotSpec(1, n_top,  subplot_spec=gs_outer[0], wspace=0.03)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 2,      subplot_spec=gs_outer[1], wspace=0.28)

    # ---- RGB panel ----
    ax_img = fig.add_subplot(gs_top[0, 0])
    ax_img.imshow(rgb_hwc)
    ax_img.axis("off")
    for spine in ax_img.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(border_col)
        spine.set_linewidth(6)

    status_txt = "PERTURBED" if is_pert else "CLEAN"
    ax_img.text(
        0.01, 0.98,
        f"{status_txt}  |  frame {i+1}/{n_frames}  |  {speed_ms:.1f} m/s{pred_speed_str}",
        transform=ax_img.transAxes,
        color=border_col, fontweight="bold", fontsize=10,
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
    )

    top_panel_idx = 1

    # ---- optional: segmentation panel ----
    if show_seg:
        ax_seg = fig.add_subplot(gs_top[0, top_panel_idx])
        ax_seg.imshow(seg_to_rgb(seg_map))
        ax_seg.axis("off")
        ax_seg.set_title("Semantic Segmentation", fontsize=9, pad=2)
        _add_seg_legend(ax_seg)
        top_panel_idx += 1

    # ---- optional: PCA panel ----
    if show_pca_panel:
        ax_pca = fig.add_subplot(gs_top[0, top_panel_idx])
        ax_pca.set_facecolor("white")
        _draw_pca_panel(ax_pca, pca_ctx, profiles, live_assignments, i)
        top_panel_idx += 1

    # ---- optional: amplified perturbation diff ----
    if show_diff and clean_rgb is not None:
        clean_hwc = chw_to_hwc(clean_rgb[i])
        diff = rgb_hwc.astype(np.int16) - clean_hwc.astype(np.int16)
        diff_vis = np.clip(diff * 5 + 128, 0, 255).astype(np.uint8)
        ax_diff = fig.add_subplot(gs_top[0, top_panel_idx])
        ax_diff.imshow(diff_vis)
        ax_diff.axis("off")
        ax_diff.set_title("Perturbation (×5, centred at 128)", fontsize=9, pad=2)

    # ---- figure title ----
    # mode is available via closure from render_frame's caller; pass it explicitly
    fig.suptitle(
        f"Live-perturbation run  —  {pert_label}  |  {run_label}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    # ================================================================
    # Bottom-left: ATOMs attention bar chart
    # ================================================================
    ax_bar = fig.add_subplot(gs_bot[0, 0])
    ax_bar.set_facecolor("white")

    class_names = list(TFV6_CLASSES.values())
    n_cls = len(class_names)
    y_pos = np.arange(n_cls)

    ax_bar.barh(y_pos, profile, color=bar_col, alpha=0.80, height=0.6, label="Current frame")
    ax_bar.scatter(
        cluster_mean, y_pos,
        marker="|", s=150, linewidths=2.0, color="black", zorder=5,
        label="Nearest cluster mean",
    )

    # Annotate the highest-attention class
    top_cls = int(np.argmax(profile))
    ax_bar.text(
        profile[top_cls] + 0.01, top_cls,
        f"{profile[top_cls]:.2f}", va="center", fontsize=7.5, color=bar_col,
    )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(class_names, fontsize=8.5)
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Attention weight", fontsize=9)
    ax_bar.set_title("ATOMs Profile vs. Nearest Cluster", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=7.5, loc="lower right")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)

    # ================================================================
    # Bottom-right: Mahalanobis distance trace
    # ================================================================
    ax_dist = fig.add_subplot(gs_bot[0, 1])
    ax_dist.set_facecolor("white")

    x = np.arange(n_frames)

    # Two-colour line: green (clean) → red (perturbed)
    ax_dist.plot(x[:onset_idx + 1], distances[:onset_idx + 1],
                 color=C_CLEAN,     linewidth=1.8, label="Clean frames")
    if onset_idx < n_frames:
        ax_dist.plot(x[onset_idx:], distances[onset_idx:],
                     color=C_PERTURBED, linewidth=1.8, label="Perturbed frames")

    # Threshold line
    ax_dist.axhline(threshold, color=C_THRESHOLD, linestyle="--", linewidth=1.4,
                    label=f"Threshold ({threshold:.2f})", zorder=3)

    # Perturbation onset
    if onset_idx < n_frames:
        ax_dist.axvline(onset_idx, color="gray", linestyle=":", linewidth=1.0, alpha=0.65,
                        label="Pert. onset")

    # Playback cursor
    ax_dist.axvline(i, color=C_CURSOR, linestyle="-", linewidth=2.2, alpha=0.9, zorder=4)
    ax_dist.scatter([i], [distances[i]], color=border_col, s=55, zorder=6, edgecolors="white",
                    linewidths=0.8)

    # OOD label badge
    is_ood = distances[i] > threshold
    ood_txt = "OOD DETECTED" if is_ood else "in-distribution"
    ood_col = C_PERTURBED    if is_ood else C_CLEAN
    ax_dist.text(
        0.98, 0.97, ood_txt,
        transform=ax_dist.transAxes,
        color=ood_col, fontweight="bold", fontsize=9,
        va="top", ha="right",
        bbox=dict(facecolor="white", alpha=0.85, edgecolor=ood_col, pad=3, linewidth=1.3),
    )

    ax_dist.set_xlabel("Frame index", fontsize=9)
    ax_dist.set_ylabel("Mahalanobis distance", fontsize=9)
    ax_dist.set_title("OOD Score (Mahalanobis)", fontsize=10, fontweight="bold")
    ax_dist.legend(fontsize=7.5, loc="upper left")
    ax_dist.spines["top"].set_visible(False)
    ax_dist.spines["right"].set_visible(False)
    ax_dist.grid(linestyle="--", alpha=0.25)
    ax_dist.set_xlim(0, n_frames - 1)

    return fig


def _add_seg_legend(ax) -> None:
    """Tiny class-colour legend inside the segmentation panel."""
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color=np.array(c) / 255, label=TFV6_CLASSES[k])
        for k, c in SEG_PALETTE.items()
    ]
    ax.legend(
        handles=patches, fontsize=5, ncol=2,
        loc="lower right",
        framealpha=0.8, handlelength=1.0, handleheight=0.8,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.list_runs:
        for pert in ["pgd", "phantom_obstacle", "brightness_scale"]:
            runs = list_available_runs(pert, args.mode)
            if runs:
                print(f"\n[{pert}]")
                for stem, has_prof, has_clean in runs:
                    prof_tag  = "✓ profiles" if has_prof  else "✗ no profiles"
                    clean_tag = "  clean_rgb" if has_clean else ""
                    print(f"  {stem:40s}  {prof_tag}{clean_tag}")
        return

    # ---- resolve run ----
    if args.run is None:
        args.run = find_run(args.pert, args.mode)
        print(f"[gif] Auto-selected run: {args.run}")

    # ---- load data ----
    print(f"[gif] Loading frames   : pert={args.pert}  run={args.run}")
    frame_data = load_frames(args.pert, args.run)
    profiles   = load_profiles(args.pert, args.run, args.mode)
    logits     = load_logits(args.pert, args.run, args.mode)
    clean_rgb  = load_clean_rgb(args.pert, args.run) if args.show_diff else None

    n_frames = profiles.shape[0]
    print(f"[gif] Frames: {n_frames}")
    if logits is not None:
        print(f"[gif] Speed logits loaded ({logits.shape})")
    if clean_rgb is not None:
        print(f"[gif] Clean RGB loaded (for diff panel)")
    elif args.show_diff:
        print("[gif] Warning: --show-diff requested but no clean_rgb file found; skipping diff panel.")

    # ---- detector + GMM ----
    print(f"[gif] Loading Mahalanobis detector + GMM (mode={args.mode}, alt split)")
    mean, inv_cov, threshold = load_detector(args.mode)
    gmm_means, gmm_inv_covs  = load_gmm(args.mode)
    print(f"[gif] GMM: {len(gmm_means)} clusters")

    # ---- distances ----
    distances = compute_distances(profiles, mean, inv_cov)
    print(f"[gif] Distance range: [{distances.min():.3f}, {distances.max():.3f}]  threshold={threshold:.3f}")

    # ---- nearest cluster assignment per frame ----
    print(f"[gif] Computing nearest cluster assignment for {n_frames} frames...")
    live_assignments, c_means = assign_clusters(profiles, gmm_means, gmm_inv_covs)

    # ---- PCA context (optional, built once) ----
    pca_ctx = None
    if args.show_pca:
        print(f"[gif] Building PCA context from alt baseline...")
        baseline_series = load_baseline_series(args.mode)
        pca_ctx = build_pca_context(baseline_series, gmm_means, gmm_inv_covs)
        var = pca_ctx.pca.explained_variance_ratio_
        print(f"[gif] PCA: PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  (cumulative {sum(var)*100:.1f}%)")

    # Perturbation onset
    is_pert_arr = frame_data["is_perturbed"].astype(bool)
    onset_idx   = int(np.argmax(is_pert_arr)) if is_pert_arr.any() else n_frames
    print(f"[gif] Perturbation onset at frame {onset_idx} / {n_frames}")

    # ---- render frames ----
    print(f"[gif] Rendering {n_frames} frames @ {args.fps} fps, DPI={args.dpi}...")
    pil_frames: list[Image.Image] = []

    for i in range(n_frames):
        fig = render_frame(
            i                = i,
            frame_data       = frame_data,
            profiles         = profiles,
            distances        = distances,
            logits           = logits,
            cluster_mean     = c_means[i],
            live_assignments = live_assignments,
            threshold        = threshold,
            onset_idx        = onset_idx,
            run_label        = args.run,
            pert_label       = args.pert,
            clean_rgb        = clean_rgb,
            show_seg         = args.show_seg,
            show_diff        = args.show_diff,
            pca_ctx          = pca_ctx,
            dpi              = args.dpi,
        )
        pil_frames.append(fig_to_pil(fig))
        plt.close(fig)

        if (i + 1) % 10 == 0 or i == n_frames - 1:
            print(f"  rendered {i+1}/{n_frames}")

    # ---- output path ----
    if args.output is None:
        out_dir = Path("gifs")
        out_dir.mkdir(exist_ok=True)
        suffix = ("_seg" if args.show_seg else "") + ("_diff" if args.show_diff else "") + ("_pca" if args.show_pca else "")
        args.output = (
            out_dir / f"live_pert_{args.pert}_{args.run}_mode{args.mode}{suffix}.gif"
        )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)

    # ---- quantize to palette (reduces GIF size significantly) ----
    print("[gif] Quantizing frames to 256-colour palette...")
    # Build a global palette from the first frame and apply to all others so
    # PIL doesn't generate a per-frame palette (which inflates the file).
    palette_frame = pil_frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0)
    quant_frames  = [palette_frame] + [
        f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=0, palette=palette_frame)
        for f in pil_frames[1:]
    ]

    # ---- save GIF ----
    duration_ms = max(40, int(1000 / args.fps))   # minimum 40 ms (25 fps cap)
    print(f"[gif] Saving GIF → {args.output}  ({duration_ms} ms/frame)")

    quant_frames[0].save(
        args.output,
        save_all      = True,
        append_images = quant_frames[1:],
        duration      = duration_ms,
        loop          = 0,
        optimize      = True,
    )

    size_mb = args.output.stat().st_size / 1e6
    print(f"[gif] Done!  {n_frames} frames  |  {size_mb:.1f} MB  →  {args.output}")


if __name__ == "__main__":
    main()
