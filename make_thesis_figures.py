#!/usr/bin/env python
"""
make_thesis_figures.py — CARLA-chapter thesis figures in the shared thesis style.

Recreates the key result figures of the alternative-split TFV6 experiment
(`EXPERIMENT_VARIANT = "alternative"`, mode 2, val-selected K=10) following
`documents/14_thesis_figure_style.md` / `thesis_style.py`, so the CARLA chapter
matches the ATOMs (Atari) chapter visually.

Every figure also writes a companion `<name>.txt` beside it (see
figure_notes.py) holding the exact numbers behind every visual feature the
caption or the prose would describe, so the chapter can be drafted off one
file instead of read off a plot.

Figures written to `thesis_figures/` (each as .pdf + .png + .txt):

  1. gmm_auc_vs_K
       Test AUROC of every GMM detector vs cluster count K, plus their mean;
       the val-selected K=10 is marked.  (Thesis version of
       results_summary_alt/curve_meanGMM_vs_K_TFV6.png.)
  2. pca_baseline_run_vs_gmm
       1x2: the same baseline PCA, left coloured by collection run, right by
       GMM component (K=10) with component means.  (Combines
       pca/pca_baseline_by_run.png and pca/pca_baseline_clusters.png.)
  3. score_dist_per_perturbation
       2x2: Mahalanobis-GMM score distribution on the labelled test set, the
       same clean set overlaid with each perturbation in its own panel, each
       marked with its TPR@5%FPR (AUROC is the subject of the per-perturbation
       AUROC figure, so it is kept out of this figure and lives in the sidecar).
       Replaces the earlier single stacked panel.  Scores are recomputed from
       the saved detector parameters; AUCs are checked against the stored
       results JSON.  (The single-kNN check is kept in the sidecar.)
  4. auroc_per_perturbation_gmm
       Grouped bars: test AUROC per perturbation for the four GMM detectors
       plus the two non-attention baselines (MDX, PEOC).  Bars are anchored
       at chance (0.5).  Static single-Gaussian variants are omitted (they
       are dominated by their GMM twins — see figure 5).
  5. auroc_gmm_vs_single
       Parity scatter: single-Gaussian AUROC (x) vs GMM AUROC (y) per
       detector family and perturbation; points above the diagonal mean the
       GMM baseline wins.
  6. attention_per_cluster
       4x2 grid: mean ATOMs attention profile per GMM cluster (horizontal
       bars, min-max whiskers), shared class order and axes.
  7. attention_per_cluster_frames
       Alternative version of 6: each cluster panel is paired with its
       representative frame (the baseline frame closest to the cluster mean).
  8. attention_by_cluster
       All clusters in one grouped bar chart (mean attention per class,
       8 bars per class group).
  9-11. live_scores_<perturbation>
       1x3 live-run score traces (Mahalanobis-GMM | MDX | PEOC) for one live
       variant each of brightness_scale (main text), gaussian_noise and pgd
       (appendix): perturbed run vs clean-RGB counterpart, with the
       perturbation-onset frame marked.  MDX scores must be cached first via
       cache_live_mdx_scores.py (needs torch/timm); everything else loads
       from the cached live_pert arrays.
  12. auroc_val_test_vs_K
       K-selection view: mean GMM-detector AUROC on the validation set (the
       selection criterion) as a solid line, test-set counterpart dotted;
       BOTH exclude gaussian-noise frames so the comparison is
       like-for-like; the selected K=10 marked.
  13. knn_k_selection
       Validation AUROC vs neighbour count k (full val set, as used by the
       pipeline's k selection) for single kNN and kNN-GMM; the selected k of
       each variant is circled.

Run with any env that has numpy / matplotlib / sklearn — both conda `PCLA`
(numpy 1.x) and `atoms3` (numpy 2.x) work; a shim below handles the
numpy-2-pickled object arrays in the alt-split npz files:

    python make_thesis_figures.py                  # writes thesis_figures/
    python make_thesis_figures.py --out-dir DIR    # anywhere else
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# numpy 1.x compatibility: the object arrays inside the alt-split npz files
# (e.g. baseline_2.npz "profile_names") were pickled by numpy >= 2 on the HPC
# and reference the "numpy._core" module path, which does not exist in 1.x.
# Alias it to numpy.core so both conda envs (PCLA: numpy 1.x, atoms3: 2.x)
# can run this script.  No-op on numpy >= 2.
if not hasattr(np, "_core"):
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.multiarray"] = _np_core.multiarray
    sys.modules["numpy._core.umath"] = _np_core.umath
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_notes import FigureNotes, spearman
from thesis_style import (
    METRIC_AXIS,
    METRIC_COLORS,
    METRIC_LEGEND,
    MUTED,
    TEXT_WIDTH_IN,
    apply_thesis_style,
    save_figure,
)

# --------------------------------------------------------------------------- #
# Paths / constants
# --------------------------------------------------------------------------- #
ROOT         = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "data" / "TFV6" / "results_alt"
SELECTED_K   = 10           # val-selected winner (max __val_auc_gmm_avg__)
RUN_DIR      = RESULTS_ROOT / f"{SELECTED_K} clusters" / "atoms_analysis_mode_2"
BASELINE_NPZ = ROOT / "data" / "TFV6" / "baseline_data_alt" / "baseline_2.npz"
FRAMES_DIR   = ROOT / "data" / "TFV6" / "baseline_data_alt" / "frames"
TEST_DIR     = ROOT / "data" / "TFV6" / "test_data_alt"
OUT_DIR      = ROOT / "thesis_figures"

# GMM detector families (canonical thesis_style keys) -> matcher on the raw
# summary.json / results_per_perturbation.json detector names.
GMM_FAMILIES = ["mahalanobis", "euclidean", "knn", "jsd"]

# Lighter tint of the kNN green for the single (non-GMM) kNN variant — same
# hue as METRIC_COLORS["knn"], lightness-only contrast (survives CVD/print).
KNN_SINGLE_COLOR = "#66bb6a"

# Cluster palette: 8 categorical colors that avoid the metric hue families
# (red, blue, green, purple, rose, gold) so cluster figures are never read
# as detector figures.  Four free hue families (orange, cyan, magenta/pink,
# brown) in dark+light steps, plus gray; the largest clusters (1, 3, 5, 7)
# get the strongest hues.  Shared by the PCA scatter and all attention bars.
CLUSTER_COLORS = ["#e6851f", "#00acc1", "#f0a3d3", "#6d4c41",
                  "#f7b565", "#d63fa6", "#7fd6e4", "#8d8d8d",
                  # added when K moved from 8 to 10; same orange/cyan/magenta/
                  # brown/gray families, separated from their neighbours by
                  # lightness so the set stays CVD-safe.
                  "#9c3d00", "#4a4a4a"]


def cluster_color(k: int) -> str:
    return CLUSTER_COLORS[k % len(CLUSTER_COLORS)]

# Perturbation shades, used only where the four perturbations subdivide one
# population (the stacked score histogram).  Deliberately NOT a fourth
# categorical family: the metric and cluster palettes have taken every free hue,
# and the four segments here are subdivisions of "perturbed", which is already
# drawn in the Mahalanobis red.  A dark-to-light ramp within that red keeps the
# stack readable in colour and in grayscale, which a categorical set would not.
# The shade order is PERT_ORDER, which coincides with the AUROC order.
PERT_COLORS = {
    "brightness_scale": "#8c1515",
    "camera_loss":      "#c0392b",
    "gaussian_noise":   "#e08e86",
    "pgd":              "#f6d3cf",
}


def run_colors(n: int) -> list[tuple[float, float, float]]:
    """`n` genuinely distinct colours, one per collection run.

    A categorical palette cannot do this at n = 190, and the earlier figure used
    tab20 modulo 20, so ten runs shared each colour.  Golden-angle hue rotation
    spreads the hues as evenly as possible for any n, and cycling saturation and
    value separates neighbouring indices further.  Uniqueness is asserted at
    8-bit, which is what actually lands in the PNG.

    The colours are unique, not individually identifiable: at 190 levels no
    reader can look up a run.  The panel's claim is that runs interleave rather
    than separate, and the per-component route counts in the sidecar are what
    carry it quantitatively.
    """
    import colorsys
    golden = 0.6180339887498949
    cols = []
    for i in range(n):
        h = (i * golden) % 1.0
        s = 0.55 + 0.15 * (i % 3)          # 0.55, 0.70, 0.85
        v = 0.95 - 0.20 * (i % 2)          # 0.95, 0.75
        cols.append(colorsys.hsv_to_rgb(h, s, v))
    keys = {tuple(round(c * 255) for c in rgb) for rgb in cols}
    if len(keys) != n:
        raise RuntimeError(f"run_colors({n}) produced {len(keys)} distinct 8-bit "
                           "colours; adjust the saturation/value cycle")
    return cols

PERT_LABELS = {                       # display names for perturbation types
    "brightness_scale": "Brightness",
    "camera_loss":      "Camera loss",
    "gaussian_noise":   "Gaussian noise",
    "pgd":              "PGD",
}
PERT_ORDER = ["brightness_scale", "camera_loss", "gaussian_noise", "pgd"]

# TFv6 target-speed bins in m/s (config_training.TrainingConfig.target_speed_classes).
# The predicted target speed is the two-hot decoding of the softmaxed logits,
# which is simply the expectation sum(p_i * v_i) (planning_decoder.decode_two_hot).
SPEED_BINS = np.array([0.0, 4.0, 8.0, 10.0, 13.88888888, 16.0, 17.77777777, 20.0])

# --------------------------------------------------------------------------- #
# Sidecar helpers (figure_notes.py)
# --------------------------------------------------------------------------- #
# Three different aggregations appear across these figures and they are NOT
# interchangeable.  Every sidecar states which one it is showing, because the
# same word ("overall", "mean AUROC") is used for all three in the sources.
AGG_POOLED_TEST = (
    "pooled AUROC over the whole mixed test set (1000 frames: 200 clean and 200 "
    "each of brightness / camera loss / Gaussian noise / PGD). Gaussian noise IS "
    "included.")
AGG_POOLED_EXGN = (
    "pooled AUROC over clean + brightness + camera loss + PGD only. "
    "Gaussian-noise frames are dropped from the pool entirely, not counted as "
    "negatives (run_analysis.py: they are labelled OOD but the ATOMs signal is "
    "expected to be clean-like). This is the K-selection criterion.")
AGG_MEAN_OVER_DETECTORS = (
    "the plotted mean is the unweighted mean over the FOUR GMM detectors of "
    "their pooled AUROC. Averaging over detectors is an extra step on top of "
    "the pooling.")

# Verified identity, checked to machine precision for all ten detector variants
# on both splits.  Both splits hold exactly 200 clean and 200 of each of the
# four perturbations, and every perturbation is scored against the same clean
# frames, so a pooled AUROC decomposes exactly:
#     AUROC(pooled) = P(perturbed > clean)
#                   = mean over the equal-sized perturbation groups of their own
#                     AUROC.
# So "pooled AUROC on the mixed set" and "unweighted mean of the per-perturbation
# AUROCs" are the SAME NUMBER here, not two competing aggregations.  What still
# differs between the figures is WHICH perturbations are in the pool.
AGG_IDENTITY = (
    "Because the split holds exactly 200 frames of each perturbation and all of "
    "them are scored against the same 200 clean frames, a pooled AUROC is "
    "identically the unweighted mean of the per-perturbation AUROCs in the pool. "
    "Verified to machine precision for every detector. The two are not competing "
    "quantities; only the set of perturbations in the pool distinguishes the "
    "figures.")


# Plain-text detector names. METRIC_LEGEND / METRIC_AXIS carry matplotlib
# mathtext ("$k^\\mathrm{th}$-NN"), which is right on an axis and unreadable in
# a text file.
PLAIN_NAME = {
    "mahalanobis": "Mahalanobis (MD)",
    "euclidean":   "Euclidean (ED)",
    "knn":         "k-th-NN",
    "knn_single":  "k-th-NN (single, no clustering)",
    "jsd":         "Jensen-Shannon (JSD)",
    "mdx":         "MDX",
    "peoc":        "PEOC",
    "mean":        "Mean over the four detectors",
}

PLAIN_AXIS = {
    "mahalanobis": "Mahalanobis distance",
    "euclidean":   "Euclidean distance",
    "knn":         "k-th-NN distance",
    "jsd":         "Jensen-Shannon divergence",
    "mdx":         "MDX distance",
    "peoc":        "PEOC entropy",
}


def _note_listing(notes: FigureNotes, label: str, xs, ys,
                  x_fmt: str = "{:g}", y_fmt: str = "{:.4f}",
                  per_line: int = 7) -> None:
    """An explicit x=y listing under `label`, wrapped over several lines.
    Sweeps are short enough that the reader wants every value, not a summary."""
    notes.line(f"    {label}:")
    items = [f"{x_fmt.format(x)}={y_fmt.format(y)}" for x, y in zip(xs, ys)]
    for i in range(0, len(items), per_line):
        notes.line("        " + ", ".join(items[i:i + per_line]))


def _quantiles(a: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, float)
    qs = np.percentile(a, [5, 25, 50, 75, 95, 99])
    return {"min": float(a.min()), "q05": float(qs[0]), "q25": float(qs[1]),
            "median": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4]),
            "q99": float(qs[5]), "max": float(a.max()), "mean": float(a.mean())}


def _note_distribution(notes: FigureNotes, label: str, a: np.ndarray) -> None:
    q = _quantiles(a)
    notes.line(f"    {label} (n = {len(a)}):")
    notes.line("        min {min:.4g}, q05 {q05:.4g}, q25 {q25:.4g}, median "
               "{median:.4g}, q75 {q75:.4g}, q95 {q95:.4g}, q99 {q99:.4g}, "
               "max {max:.4g}".format(**q))
    notes.line(f"        mean {q['mean']:.4g}")


def tpr_at_fpr(clean: np.ndarray, pert: np.ndarray, fpr: float) -> tuple[float, float]:
    """(TPR, threshold) at a target FPR, thresholding at the clean scores'
    (1 - fpr) quantile.  Higher score = more OOD throughout this script.

    The reason this belongs in every score sidecar: an AUROC of 0.6 says
    nothing about whether a deployable threshold exists, and the chapter's
    claim is that the separation lives in the right tail."""
    thr = float(np.quantile(clean, 1.0 - fpr))
    return float((pert > thr).mean()), thr


def _note_tail(notes: FigureNotes, clean: np.ndarray, pert: np.ndarray,
               prefix: str = "") -> None:
    """The operating points a reader can judge: TPR at 5 % and 1 % FPR."""
    for fpr in (0.05, 0.01):
        tpr, thr = tpr_at_fpr(clean, pert, fpr)
        notes.value(f"{prefix}TPR at {fpr:.0%} FPR", tpr,
                    note=f"threshold {thr:.4g} = clean q{100 * (1 - fpr):.0f}")


# --------------------------------------------------------------------------- #
# Data access helpers
# --------------------------------------------------------------------------- #
def _family_of(raw_name: str) -> str | None:
    """Map a raw detector name to a thesis_style metric key (or None)."""
    n = raw_name.lower()
    if "mdx" in n:
        return "mdx"
    if "peoc" in n or "entropy" in n:
        return "peoc"
    if "k-nn" in n or "knn" in n:
        return "knn"
    if "mahalanobis" in n:
        return "mahalanobis"
    if "euclidean" in n:
        return "euclidean"
    if "jsd" in n:
        return "jsd"
    return None


def load_sweep() -> dict[str, dict[int, float]]:
    """{family: {K: test AUC}} for the four GMM detectors, from the named
    '<K> clusters' snapshot folders (mode 2, alternative split)."""
    sweep: dict[str, dict[int, float]] = {f: {} for f in GMM_FAMILIES}
    for kdir in RESULTS_ROOT.glob("* clusters"):
        summ = kdir / "atoms_analysis_mode_2" / "summary.json"
        if not summ.exists():
            continue
        K = int(kdir.name.split()[0])
        raw = json.loads(summ.read_text())
        for key, val in raw.items():
            if key.startswith("__") or not isinstance(val, dict):
                continue
            if "gmm" not in key.lower():
                continue
            fam = _family_of(key)
            if fam in sweep:
                sweep[fam][K] = val["auc"]
    return sweep


def load_per_perturbation() -> dict[str, dict[str, float]]:
    """{perturbation: {'<family>_single'|'<family>_gmm'|'mdx'|'peoc': auc}}."""
    raw = json.loads((RUN_DIR / "results_per_perturbation.json").read_text())
    out: dict[str, dict[str, float]] = {}
    for pert, entries in raw.items():
        d: dict[str, float] = {}
        for entry in entries:
            for key, v in entry.items():
                name = key.split("|")[0].strip()
                fam = _family_of(name)
                if fam is None:
                    continue
                if fam in ("mdx", "peoc"):
                    d[fam] = v
                else:
                    kind = "gmm" if "gmm" in name.lower() else "single"
                    d[f"{fam}_{kind}"] = v
        out[pert] = d
    return out


def load_overall() -> dict[str, float]:
    """Same key scheme as load_per_perturbation(), from summary.json (mixed
    test set: 200 clean + 800 perturbed)."""
    raw = json.loads((RUN_DIR / "summary.json").read_text())
    d: dict[str, float] = {}
    for key, val in raw.items():
        if key.startswith("__") or not isinstance(val, dict):
            continue
        fam = _family_of(key)
        if fam is None:
            continue
        if fam in ("mdx", "peoc"):
            d[fam] = val["auc"]
        else:
            kind = "gmm" if "gmm" in key.lower() else "single"
            d[f"{fam}_{kind}"] = val["auc"]
    return d


def load_baseline_series() -> np.ndarray:
    return np.load(BASELINE_NPZ, allow_pickle=True)["series"].astype(np.float64)


def load_run_ids(n_frames: int) -> np.ndarray:
    """Per-frame run index, reproducing BaselineDataLoader.load_all_runs:
    0-based file index in sorted(run_*.npz) order.  Only the small frame_idx
    member is decompressed per file."""
    ids = []
    for run_id, f in enumerate(sorted(FRAMES_DIR.glob("run_*.npz"), key=lambda p: p.name)):
        n = np.load(f)["frame_idx"].shape[0]
        ids.append(np.full(n, run_id, dtype=np.int32))
    ids = np.concatenate(ids)
    if len(ids) < n_frames:
        raise RuntimeError(
            f"frames dir yields {len(ids)} frames < baseline series {n_frames}"
        )
    return ids[:n_frames]


# --------------------------------------------------------------------------- #
# Detector math (replicates DistanceComputer / GMMClustering semantics)
# --------------------------------------------------------------------------- #
def mahalanobis_batch(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Mahalanobis distance of each row of X, matching
    DistanceComputer.compute_mahalanobis (ridge 1e-6, pseudo-inverse, sqrt)."""
    cov_inv = np.linalg.pinv(cov + 1e-6 * np.eye(cov.shape[0]))
    diff = X - mean
    m2 = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)
    return np.sqrt(np.maximum(m2, 0.0))


def gmm_min_mahalanobis(X: np.ndarray, means: np.ndarray, covs: np.ndarray) -> np.ndarray:
    """GMMClustering.score_batch: distance to the nearest component centre."""
    d = np.stack([mahalanobis_batch(X, means[k], covs[k]) for k in range(len(means))])
    return d.min(axis=0)


def load_gmm():
    """(means, covariances, weights, K) from the run's gmm.npz.  The stored
    covariances carry the fit-time shrinkage; since the uniform-shrinkage fix
    (2026-07-16) the pipeline predicts AND scores with exactly these."""
    g = np.load(RUN_DIR / "gmm.npz", allow_pickle=True)
    return (g["means"].astype(np.float64),
            g["covariances"].astype(np.float64),
            g["weights"].astype(np.float64),
            int(g["n_components"][0]))


def gmm_predict(X: np.ndarray, means: np.ndarray, covs: np.ndarray,
                weights: np.ndarray) -> np.ndarray:
    """Most probable component per row (sklearn GaussianMixture.predict on
    the saved, shrinkage-regularised parameters — matches GMMClustering)."""
    N, D = X.shape
    log_prob = np.empty((N, len(weights)))
    for k in range(len(weights)):
        L = np.linalg.cholesky(covs[k])
        sol = np.linalg.solve(L, (X - means[k]).T)          # [D, N]
        maha = (sol ** 2).sum(axis=0)
        logdet = 2.0 * np.log(np.diag(L)).sum()
        log_prob[:, k] = -0.5 * (maha + logdet + D * np.log(2 * np.pi)) \
                         + np.log(weights[k])
    return log_prob.argmax(axis=1)


def mean_pixel_intensity() -> np.ndarray:
    """Mean pixel value of each labelled test frame, cached beside the data.

    The reference point for the obvious objection to the chapter's one positive
    result: if a perturbation is a global photometric change, a statistic that
    needs no model at all might detect it just as well.  Reading it out costs a
    2.6 GB decompression of `wide_rgb`, so the 1000 means are cached and the
    raw array is touched once."""
    cache = TEST_DIR / "mean_pixel_intensity_2.npy"
    if cache.exists():
        return np.load(cache)
    d = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)
    rgb = d["wide_rgb"]
    mi = rgb.reshape(len(rgb), -1).mean(axis=1)
    np.save(cache, mi)
    print(f"  [info] cached mean pixel intensity -> {cache.name}")
    return mi


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


# --------------------------------------------------------------------------- #
# Shared legend helpers
# --------------------------------------------------------------------------- #
def _fig_legend(fig: plt.Figure, handles: list, ncol: int) -> None:
    """The collision-safe below-figure legend (spec section 6)."""
    fig.legend(handles=handles, loc="upper center", ncol=ncol,
               bbox_to_anchor=(0.5, -0.002), bbox_transform=fig.transFigure,
               columnspacing=1.4, handlelength=1.9)


def _stats_box(ax: plt.Axes, text: str) -> None:
    """Top-right annotation box (spec section 5)."""
    ax.annotate(text, xy=(0.97, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                          alpha=0.85, edgecolor="0.8", linewidth=0.5))


# --------------------------------------------------------------------------- #
# Figure 1 — GMM detector AUROC vs cluster count K
# --------------------------------------------------------------------------- #
def fig_gmm_auc_vs_k() -> None:
    sweep = load_sweep()
    Ks = sorted(set().union(*[set(v) for v in sweep.values()]))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    ax.axvline(SELECTED_K, color="0.78", linewidth=0.8,
               linestyle=(0, (2, 2)), zorder=1)

    for fam in GMM_FAMILIES:
        ys = [sweep[fam][K] for K in Ks]
        ax.plot(Ks, ys, color=METRIC_COLORS[fam], linewidth=1.25, alpha=0.95,
                marker="o", markersize=2.6, markeredgewidth=0, zorder=2)

    mean = [float(np.mean([sweep[f][K] for f in GMM_FAMILIES])) for K in Ks]
    ax.plot(Ks, mean, color="0.15", linewidth=1.9,
            marker="o", markersize=3.4, markeredgewidth=0, zorder=3)

    ax.text(SELECTED_K + 0.25, 0.5205, f"$K={SELECTED_K}$ (selected on val.)",
            fontsize=7.5, color=MUTED, ha="left", va="bottom")

    ax.set_xlabel("GMM components $K$")
    ax.set_ylabel("Test AUROC")
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlim(1.4, 20.6)
    ax.set_ylim(0.515, 0.645)
    ax.set_yticks([0.52, 0.55, 0.58, 0.61, 0.64])

    handles = [Line2D([], [], color=METRIC_COLORS[f], linewidth=1.6,
                      label=METRIC_LEGEND[f]) for f in GMM_FAMILIES]
    handles.append(Line2D([], [], color="0.15", linewidth=1.9, label="Mean"))
    _fig_legend(fig, handles, ncol=5)

    save_figure(fig, OUT_DIR, "gmm_auc_vs_K")

    notes = FigureNotes(
        "gmm_auc_vs_K",
        title="Test AUROC of the four GMM attention detectors against the "
              "component count K",
        source=str(RESULTS_ROOT) + "/<K> clusters/atoms_analysis_mode_2/summary.json")
    notes.line(f"    Sweep: {len(Ks)} values of K, {min(Ks)} to {max(Ks)} "
               f"({', '.join(str(k) for k in Ks)}).")
    notes.line(f"    AUROC is {AGG_POOLED_TEST}")
    notes.line(f"    {AGG_MEAN_OVER_DETECTORS}")
    notes.line("    NOTE the criterion K was actually selected on is a different "
               "quantity: the validation split, Gaussian noise dropped from the "
               "pool, then averaged over detectors. See auroc_val_test_vs_K.txt. "
               "Name the perturbation set before comparing a value from there "
               "with one from here.")
    notes.line(f"    The same 1000 test frames are reused at every K, so the "
               f"{len(Ks)} points are not independent comparisons.")
    notes.line("    No rise/plateau/shape descriptors: those are defined for a "
               "smooth sweep over a severity knob, which K is not. The per-K "
               "listing carries the shape.")

    curves = [(fam, [sweep[fam][K] for K in Ks]) for fam in GMM_FAMILIES]
    curves.append(("mean", mean))
    for fam, ys in curves:
        notes.section(PLAIN_NAME[fam])
        notes.curve("test AUROC vs K", Ks, ys, rise=False, plateau=False, shape=False)
        best = int(np.argmax(ys))
        notes.value("best K", f"{Ks[best]}", note=f"AUROC {ys[best]:.4f}")
        notes.value(f"at the selected K = {SELECTED_K}", ys[Ks.index(SELECTED_K)])
        notes.value("spread over the sweep", max(ys) - min(ys),
                    note="max minus min across K")
        _note_listing(notes, "per K", Ks, ys)
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 2 — baseline PCA: coloured by run vs by GMM component
# --------------------------------------------------------------------------- #
def fig_pca_run_vs_gmm() -> None:
    from sklearn.decomposition import PCA

    series = load_baseline_series()
    run_ids = load_run_ids(len(series))

    means, covs, weights, _ = load_gmm()
    labels = gmm_predict(series, means, covs, weights)

    pca = PCA(n_components=2, random_state=42)
    proj = pca.fit_transform(series)
    cent = pca.transform(means)
    var = pca.explained_variance_ratio_ * 100

    fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.9),
                            sharex=True, sharey=True)

    n_runs = int(run_ids.max()) + 1
    palette = run_colors(n_runs)
    axs[0].scatter(proj[:, 0], proj[:, 1],
                   c=[palette[r] for r in run_ids], s=6, alpha=0.5, linewidths=0,
                   rasterized=True)

    K = len(weights)
    axs[1].scatter(proj[:, 0], proj[:, 1],
                   c=[cluster_color(l) for l in labels], s=6, alpha=0.5,
                   linewidths=0, rasterized=True)
    for k in range(K):
        axs[1].scatter(*cent[k], marker="*", s=120, facecolor=cluster_color(k),
                       edgecolor="black", linewidths=0.6, zorder=3)

    fig.supxlabel(f"PC1 ({var[0]:.1f} %)")
    fig.supylabel(f"PC2 ({var[1]:.1f} %)")

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=cluster_color(k), markeredgewidth=0,
                      label=str(k)) for k in range(K)]
    handles.append(Line2D([], [], marker="*", linestyle="none", markersize=9,
                          markerfacecolor="0.85", markeredgecolor="0.15",
                          markeredgewidth=0.6, label="component mean"))
    _fig_legend(fig, handles, ncol=K + 1)

    save_figure(fig, OUT_DIR, "pca_baseline_run_vs_gmm")

    full = PCA(random_state=42).fit(series)
    evr = full.explained_variance_ratio_ * 100

    notes = FigureNotes(
        "pca_baseline_run_vs_gmm",
        title="Baseline attention cloud in its first two principal components, "
              "coloured by collection run (left) and by GMM component (right)",
        source=f"{BASELINE_NPZ} + {RUN_DIR}/gmm.npz")
    notes.line(f"    {len(series)} baseline profiles, {series.shape[1]} classes, "
               f"from {n_runs} collection runs (routes). This is the TRAINING "
               f"split only.")
    notes.line(f"    Left-panel palette: {n_runs} genuinely distinct colours, one "
               f"per run, from a golden-angle hue rotation with cycling "
               f"saturation and value; uniqueness is asserted at 8-bit. "
               f"(The earlier version used tab20 modulo 20, so ten runs shared "
               f"each colour.) They are unique but NOT individually "
               f"identifiable at 190 levels: the panel shows that runs "
               f"interleave rather than separate, and the per-component route "
               f"counts below are what carry that quantitatively.")
    notes.line("    Both panels show the same projection; only the colouring differs.")

    notes.section("Explained variance")
    for i in range(min(6, len(evr))):
        notes.value(f"PC{i + 1}", evr[i], unit="%",
                    note=f"cumulative {evr[:i + 1].sum():.2f} %")
    notes.value("components reaching 99 % of the variance",
                int(np.searchsorted(np.cumsum(evr), 99.0) + 1),
                note="the cloud is effectively this low-dimensional")
    notes.value("variance left in PC5 and above", evr[4:].sum(), unit="%")

    notes.section("Components (right panel)")
    sizes = np.bincount(labels, minlength=K)
    notes.value("K", K)
    notes.counts("component sizes (index = n)",
                 {str(k): int(sizes[k]) for k in range(K)})
    notes.value("sizes ascending", ", ".join(str(int(s)) for s in np.sort(sizes)))
    notes.value("sizes sum", int(sizes.sum()),
                note="equals the training split, so the baseline is fitted on "
                     "training frames only")
    notes.value("largest / smallest", f"{sizes.max()} / {sizes.min()}",
                note=f"share {sizes.max() / len(series):.1%} / "
                     f"{sizes.min() / len(series):.1%}")
    for k in range(K):
        notes.value(f"component {k} centre (PC1, PC2)",
                    f"{cent[k, 0]:+.3f}, {cent[k, 1]:+.3f}",
                    note=f"n = {sizes[k]} ({sizes[k] / len(series):.1%})")

    notes.section("How components relate to runs")
    notes.line("    Answers the figure's open question: the left and right "
               "panels are only comparable if a component is not simply a run.")
    per_cluster = {str(k): int(len(np.unique(run_ids[labels == k])))
                   for k in range(K)}
    notes.counts("runs contributing to each component", per_cluster)
    cpr = np.array([len(np.unique(labels[run_ids == r])) for r in range(n_runs)])
    notes.value("components visited per run", f"{cpr.min()} to {cpr.max()}",
                note=f"median {np.median(cpr):.0f}, mean {cpr.mean():.2f}")
    notes.counts("runs by number of components visited",
                 {str(c): int(n) for c, n in enumerate(np.bincount(cpr)) if n})
    notes.value("runs confined to a single component", int((cpr == 1).sum()),
                note=f"of {n_runs}")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 3 — score distributions of the two best GMM detectors
# --------------------------------------------------------------------------- #
def knn_gmm_scores(profiles: np.ndarray, series: np.ndarray, means: np.ndarray,
                   covs: np.ndarray, weights: np.ndarray, k: int = 1) -> np.ndarray:
    """kNN-GMM scores, replicating run_analysis.py step 9b.3: baseline frames
    are pooled by their GMM-predicted cluster; each test point is routed to
    the nearest component (by Mahalanobis) and scored by the k-th-NN
    Euclidean distance within that pool, on L2-normalised vectors.  All steps
    use the shrunk covariances (uniform-shrinkage fix)."""
    baseline_labels = gmm_predict(series, means, covs, weights)
    pools = {c: series[baseline_labels == c]
             for c in range(len(weights)) if (baseline_labels == c).sum() > 0}

    comp_dist = np.stack([mahalanobis_batch(profiles, means[c], covs[c])
                          for c in range(len(weights))])       # [K, N]
    nearest = comp_dist.argmin(axis=0)

    def _norm(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

    scores = np.empty(len(profiles))
    for i, x in enumerate(profiles):
        pool = pools.get(int(nearest[i]), series)
        if len(pool) < k:
            pool = series
        d = np.linalg.norm(_norm(pool) - _norm(x), axis=1)
        scores[i] = np.sort(d)[k - 1]
    return scores


def fig_score_distributions() -> None:
    profiles = np.load(TEST_DIR / "attention" / "test_profiles_2.npy").astype(np.float64)
    tl = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)
    labels = tl["label"].astype(int)
    perts = tl["perturbation"].astype(str)

    means, covs, weights, _ = load_gmm()

    # Plain single kNN (full-baseline pool, no clustering) — it dominates
    # kNN-GMM on the test set.  Its neighbour count is selected on the val
    # set and changes between runs: parse it from summary.json (always
    # fresh) and read the per-k result file, which run_analysis rewrites
    # every run.  The "results_ATOMs-k-NN..._best.json" alias files can be
    # stale copies from other K runs (scratch-dir accumulation pre-2026-07-17).
    summ = json.loads((RUN_DIR / "summary.json").read_text())
    knn_key = next(k for k in summ if "k-NN" in k and "GMM" not in k)
    best_k = int(re.search(r"k=(\d+)", knn_key).group(1))
    print(f"  [info] val-selected single-kNN k = {best_k}")

    s_mahal = gmm_min_mahalanobis(profiles, means, covs)
    series_n = load_baseline_series()
    series_n /= (np.linalg.norm(series_n, axis=1, keepdims=True) + 1e-12)
    prof_n = profiles / (np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-12)
    from scipy.spatial.distance import cdist
    s_knn = np.sort(cdist(prof_n, series_n), axis=1)[:, best_k - 1]

    res_mahal = json.loads(
        (RUN_DIR / f"results_ATOMs-Mahalanobis_GMM_K={SELECTED_K}.json").read_text())
    res_knn = json.loads(
        (RUN_DIR / f"results_knn_k{best_k}.json").read_text())

    # sanity: recomputed scores must reproduce the stored AUCs
    for name, scores, res in (("mahal-gmm", s_mahal, res_mahal),
                              ("knn-single", s_knn, res_knn)):
        auc = roc_auc(scores, labels)
        if abs(auc - res["auc"]) > 1e-3:
            raise RuntimeError(
                f"Recomputed {name} AUC {auc:.4f} != stored {res['auc']:.4f} — "
                "detector parameters and test profiles are out of sync.")
        print(f"  [check] {name}: recomputed AUC {auc:.4f} == stored {res['auc']:.4f}")

    # 2x2 grid, one panel per perturbation: clean against that perturbation,
    # each with its own AUROC and detection rate at the 5% FPR threshold.  This
    # replaces the earlier single stacked panel (through 2026-09-16), which
    # pooled the four perturbations into one density and hid each one's shape.
    # Colours are the Atari appendix TABLE tints (compute_detection_summary.py
    # PERT_FILL: noise=red, attack=blue, increase=yellow, decrease=green),
    # nudged ~35% toward a soft same-hue edge so they read on a histogram while
    # staying clearly paler than the saturated attention-distance palette.
    # brightness_scale is the increase analogue, camera_loss the decrease one.
    clean_face, clean_edge = to_rgba("0.5", 0.45), "0.35"
    PANEL = {   # perturbation: (pale table tint, soft same-hue edge)
        "brightness_scale": ("#f5f5de", "#bcb46a"),   # yellow  (increase)
        "camera_loss":      ("#dff1df", "#7bbd88"),   # green   (decrease)
        "gaussian_noise":   ("#f9dfdf", "#d1665e"),   # red     (noise)
        "pgd":              ("#dbebef", "#6fa5bd"),   # blue    (attack)
    }

    def _mix(hex_a, hex_b, t):               # t of b blended into a
        a, b = np.array(to_rgba(hex_a)[:3]), np.array(to_rgba(hex_b)[:3])
        return tuple((1.0 - t) * a + t * b)

    bins = np.linspace(0.0, float(s_mahal.max()) * 1.02, 40)
    width = bins[1] - bins[0]
    clean_scores = s_mahal[labels == 0]
    thr95 = float(np.quantile(clean_scores, 0.95))
    c_counts, _ = np.histogram(clean_scores, bins=bins)
    c_dens = c_counts / (len(clean_scores) * width)
    clean_step_x = np.append(bins, bins[-1])
    clean_step_y = np.append(np.append(0.0, c_dens), 0.0)

    # panels in AUROC order, strongest first
    panel_order = ["brightness_scale", "camera_loss", "gaussian_noise", "pgd"]
    fig, axs = plt.subplots(2, 2, figsize=(TEXT_WIDTH_IN, 4.6),
                            sharex=True, sharey=True)
    for pt, ax in zip(panel_order, axs.ravel()):
        ps = s_mahal[perts == pt]
        t5, _ = tpr_at_fpr(clean_scores, ps, 0.05)
        face, edge = PANEL[pt]
        face = _mix(face, edge, 0.35)        # turn the colour up a touch
        # clean reference (grey), same in every panel
        ax.hist(clean_scores, bins=bins, density=True, histtype="stepfilled",
                facecolor=clean_face, edgecolor=clean_edge, linewidth=0.9,
                zorder=2)
        # this perturbation
        ax.hist(ps, bins=bins, density=True, histtype="stepfilled",
                facecolor=to_rgba(face, 0.9), edgecolor=edge, linewidth=1.3,
                zorder=3)
        # clean outline on top so its shape stays visible under the fill
        ax.step(clean_step_x, clean_step_y, where="pre", color=clean_edge,
                linewidth=0.9, zorder=4)
        ax.axvline(thr95, color="0.25", linestyle=(0, (1, 1.6)), linewidth=0.9,
                   zorder=5)
        ax.set_title(PERT_LABELS[pt], fontsize=9)
        _stats_box(ax, f"TPR@5%FPR = {t5:.0%}")

    axs[1, 0].annotate("5 % FPR", xy=(thr95, axs[1, 0].get_ylim()[1] * 0.88),
                       xytext=(-4, 0), textcoords="offset points",
                       fontsize=7.0, color=MUTED, ha="right", va="center",
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                 alpha=0.85, edgecolor="none"))
    for ax in axs[-1, :]:
        ax.set_xlabel(METRIC_AXIS["mahalanobis"])
    for ax in axs[:, 0]:
        ax.set_ylabel("Density")
    axs[0, 0].set_xlim(left=0.0)

    handles = [Patch(facecolor=clean_face, edgecolor=clean_edge, linewidth=0.9,
                     label="clean")]
    _fig_legend(fig, handles, ncol=1)

    save_figure(fig, OUT_DIR, "score_dist_per_perturbation")

    notes = FigureNotes(
        "score_dist_per_perturbation",
        title="Mahalanobis-GMM score distribution on the labelled test set, "
              "clean against each perturbation in its own panel",
        source=f"{TEST_DIR}/attention/test_profiles_2.npy + {RUN_DIR}")
    notes.line(f"    The pooled AUROC is {AGG_POOLED_TEST}")
    notes.line("    Scores are recomputed from the saved detector parameters and "
               "checked against the stored results JSONs.")
    notes.line("    FOUR panels since 2026-09-16 (replacing the single stacked "
               "panel), Mahalanobis-GMM only. Each panel overlays the same clean "
               "test set (grey) with one perturbation, in AUROC order: brightness "
               "increase, camera loss, Gaussian noise, PGD. Only the TPR@5%FPR is "
               "drawn on each panel; AUROC is reported by the per-perturbation "
               "AUROC figure and kept out of this one, but the per-panel and "
               "pooled AUROCs are recorded below. Colours are the Atari appendix "
               "table tints (PERT_FILL) nudged 35% toward a soft same-hue edge.")
    notes.line("    The dotted vertical rule is the clean scores' 95th "
               "percentile, i.e. the threshold of a detector admitting one false "
               "alarm in twenty. Everything to its right is what that detector "
               "flags. Higher score = more OOD.")
    notes.line("    The threshold rows below are the point of this sidecar: an "
               "AUROC does not say whether a usable operating point exists, and "
               "the chapter's claim is that the separation sits in the right tail.")
    notes.counts("test frames",
                 {"clean": int((labels == 0).sum()),
                  "perturbed": int((labels == 1).sum())})
    notes.counts("perturbed frames by perturbation",
                 {PERT_LABELS.get(pt, pt): int((perts == pt).sum())
                  for pt in PERT_ORDER})

    for panel, scores, res in (("Mahalanobis-GMM (the panel)", s_mahal, res_mahal),
                               (f"single k-NN, k = {best_k} (NOT DRAWN since "
                                "2026-09-03, kept as a check on the stored "
                                "results and because the metric comparison "
                                "cites it)", s_knn, res_knn)):
        notes.section(panel)
        notes.value("AUROC, whole mixed test set", res["auc"])
        clean, pert = scores[labels == 0], scores[labels == 1]
        _note_distribution(notes, "clean", clean)
        _note_distribution(notes, "perturbed", pert)
        notes.value("median shift, perturbed minus clean",
                    float(np.median(pert) - np.median(clean)))
        notes.value("perturbed above the clean median",
                    float((pert > np.median(clean)).mean()),
                    note="0.5 would mean no shift of the bulk")
        _note_tail(notes, clean, pert)
        notes.line("    per perturbation, against the same clean set:")
        for pt in PERT_ORDER:
            sel = perts == pt
            if not sel.any():
                continue
            sub_p = scores[sel]
            auc = roc_auc(np.concatenate([clean, sub_p]),
                          np.concatenate([np.zeros(len(clean)),
                                          np.ones(len(sub_p))]))
            t5, _ = tpr_at_fpr(clean, sub_p, 0.05)
            t1, _ = tpr_at_fpr(clean, sub_p, 0.01)
            notes.value(f"    {PERT_LABELS.get(pt, pt)}",
                        f"AUROC {auc:.4f}, median {np.median(sub_p):.4g}, "
                        f"TPR@5%FPR {t5:.3f}, TPR@1%FPR {t1:.3f}",
                        note=f"n = {int(sel.sum())}")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 4 — test AUROC per perturbation (GMM detectors + MDX / PEOC)
# --------------------------------------------------------------------------- #
def fig_auroc_per_perturbation() -> None:
    per_pert = load_per_perturbation()
    overall = load_overall()

    # The single-component (no-clustering) kNN was dropped from this headline
    # figure on 2026-09-22 (author): it shows every attention distance against
    # the same K=10 reference, and the single-vs-clustered kNN comparison lives
    # in auroc_gmm_vs_single (@clusterAdvantage) and knn_k_selection
    # (@knnSelection). "knn" here is the GMM-pooled variant.
    detectors = ["mahalanobis", "euclidean", "knn", "jsd", "mdx", "peoc"]
    hatches = {"mdx": "///", "peoc": "xxx"}   # bar analogue of the dashed lines

    def value(src: dict[str, float], fam: str) -> float:
        if fam in ("mdx", "peoc", "knn_single"):
            return src[fam] if fam in ("mdx", "peoc") else src["knn_single"]
        return src[f"{fam}_gmm"]

    # The "Overall" group was removed on 2026-09-03 (author): it is exactly the
    # unweighted mean of the four groups beside it (equal 200-frame perturbation
    # sets against a shared clean set, verified to machine precision), and the
    # chapter's position, established in the Atari study and visible again here,
    # is that averaging over perturbations carries little information.  The
    # mixed-set values are still in the sidecar, since the metric ranking reads
    # off them.
    groups = list(PERT_ORDER)
    group_labels = [PERT_LABELS[p] for p in PERT_ORDER]
    sources = [per_pert[p] for p in PERT_ORDER]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))

    n_det = len(detectors)
    width = 0.8 / n_det
    x0 = np.arange(len(groups))
    for j, fam in enumerate(detectors):
        xs = x0 + (j - (n_det - 1) / 2) * width
        ys = np.array([value(src, fam) for src in sources])
        if fam == "knn_single":
            ax.bar(xs, ys - 0.5, bottom=0.5, width=width * 0.92,
                   facecolor=KNN_SINGLE_COLOR, edgecolor="white",
                   linewidth=0.0)
        else:
            ax.bar(xs, ys - 0.5, bottom=0.5, width=width * 0.92,
                   facecolor=METRIC_COLORS[fam], edgecolor="white",
                   linewidth=0.0, hatch=hatches.get(fam, None))

    ax.axhline(0.5, color="0.3", linewidth=0.8, zorder=1)
    ax.text(-0.44, 0.489, "chance", fontsize=7.5, color=MUTED,
            ha="left", va="top")

    ax.set_xticks(x0)
    ax.set_xticklabels(group_labels)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(0.03, 0.88)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.grid(axis="x", visible=False)

    handles = []
    for f in detectors:
        if f == "knn_single":
            handles.append(Patch(facecolor=KNN_SINGLE_COLOR, edgecolor="white",
                                 linewidth=0.0,
                                 label=METRIC_LEGEND["knn"] + " (single)"))
        else:
            handles.append(Patch(facecolor=METRIC_COLORS[f], edgecolor="white",
                                 linewidth=0.0, hatch=hatches.get(f, None),
                                 label=METRIC_LEGEND[f]))
    _fig_legend(fig, handles, ncol=4)

    save_figure(fig, OUT_DIR, "auroc_per_perturbation_gmm")

    det_labels = {f: PLAIN_NAME[f] for f in detectors}

    notes = FigureNotes(
        "auroc_per_perturbation_gmm",
        title="Test AUROC per perturbation for the GMM attention detectors and "
              "the two non-attention baselines",
        source=f"{RUN_DIR}/results_per_perturbation.json + summary.json")
    notes.line(f"    K = {SELECTED_K}. Bars are drawn from the chance line 0.5, "
               "so bar height is AUROC minus 0.5 and the sign is the direction.")
    notes.line("    A per-perturbation column is that perturbation's frames "
               "against the clean frames only.")
    notes.line("    There is no aggregate group in this figure. The 'Overall' "
               "bar was removed on 2026-09-03: it was exactly the unweighted "
               "mean of the four groups drawn here.")
    notes.line(f"    {AGG_IDENTITY}")
    notes.line("    The mixed-set values are kept in the last section below, "
               "marked as not drawn, because the metric ranking reads off them.")
    notes.line("    The K-selection criterion is yet another pool, because it "
               "drops Gaussian noise and then averages over detectors as well. "
               "See auroc_val_test_vs_K.txt.")
    notes.line("    The k-NN entry is the GMM-pooled variant. The single "
               "(no-clustering) k-NN was dropped from this figure on 2026-09-22; "
               "it is compared in auroc_gmm_vs_single.txt and knn_k_selection.txt.")

    for gi, (g, glabel, src_d) in enumerate(zip(groups, group_labels, sources)):
        notes.section(glabel)
        vals = {fam: value(src_d, fam) for fam in detectors}
        for fam in detectors:
            notes.value(det_labels[fam], vals[fam],
                        note=f"{vals[fam] - 0.5:+.4f} from chance")
        best = max(vals, key=vals.get)
        worst = min(vals, key=vals.get)
        notes.value("highest", f"{det_labels[best]} {vals[best]:.4f}")
        notes.value("lowest", f"{det_labels[worst]} {vals[worst]:.4f}")
        att = ["mahalanobis", "euclidean", "knn", "jsd"]
        b_att = max(att, key=lambda f: vals[f])
        notes.value("highest attention distance",
                    f"{det_labels[b_att]} {vals[b_att]:.4f}")
        notes.value("detectors above 0.70", 
                    ", ".join(det_labels[f] for f in detectors if vals[f] > 0.70)
                    or "none")
        notes.value("detectors below 0.50",
                    ", ".join(f"{det_labels[f]} ({vals[f]:.4f})"
                              for f in detectors if vals[f] < 0.50) or "none")

    notes.section("NOT DRAWN: mean pixel intensity, a model-free reference")
    notes.line("    Not a detector we propose and not in the figure. It is the "
               "answer to the obvious objection to the brightness result: that a "
               "trivial input statistic would do as well on a global photometric "
               "change. Higher intensity = more OOD, one-sided like every other "
               "score here.")
    mi = mean_pixel_intensity()
    tl_ = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)
    lab_ = tl_["label"].astype(int)
    pert_ = tl_["perturbation"].astype(str)
    clean_mi = mi[lab_ == 0]
    notes.value("clean frames, median mean-pixel value", float(np.median(clean_mi)))
    for pt in PERT_ORDER:
        sub_mi = mi[pert_ == pt]
        auc = roc_auc(np.concatenate([clean_mi, sub_mi]),
                      np.concatenate([np.zeros(len(clean_mi)),
                                      np.ones(len(sub_mi))]))
        notes.value(PERT_LABELS[pt], auc,
                    note=f"median {np.median(sub_mi):.1f}, "
                         f"{abs(auc - 0.5):.4f} from chance")
    notes.line("    Reading: on the brightness increase it reaches 0.78, which is "
               "BELOW both MDX and the attention MD, so the attention distance is "
               "not merely re-detecting a change any pixel average would see. It "
               "is at chance on Gaussian noise and PGD and inverted on camera "
               "loss, where removing a camera lowers the frame mean.")

    notes.section("NOT DRAWN: the mixed test set, and the ranking on it")
    notes.line("    Removed from the figure on 2026-09-03. Retained here because "
               "it is the only place the detectors are ranked against each other "
               "on one number, and it is exactly the mean of the four groups "
               "above.")
    ov = {fam: value(overall, fam) for fam in detectors}
    for fam in detectors:
        notes.value(det_labels[fam], ov[fam],
                    note=f"{ov[fam] - 0.5:+.4f} from chance")
    notes.line("    ranking:")
    for rank, fam in enumerate(sorted(ov, key=ov.get, reverse=True), start=1):
        notes.value(f"{rank}. {det_labels[fam]}", ov[fam])
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 5 — parity scatter: GMM vs single-Gaussian AUROC
# --------------------------------------------------------------------------- #
def fig_gmm_vs_single_parity() -> None:
    per_pert = load_per_perturbation()
    overall = load_overall()

    cases = [(p, per_pert[p], "o", 26) for p in PERT_ORDER]
    cases.append(("overall", overall, "D", 34))

    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    lo, hi = 0.44, 0.86
    ax.plot([lo, hi], [lo, hi], color="black", linestyle=(0, (4, 2)),
            linewidth=0.8, zorder=2)

    for fam in GMM_FAMILIES:
        for _, src, marker, size in cases:
            ax.scatter(src[f"{fam}_single"], src[f"{fam}_gmm"],
                       marker=marker, s=size, facecolor=METRIC_COLORS[fam],
                       edgecolor="black", linewidths=0.3, zorder=3)

    ax.text(0.04, 0.96, "GMM better", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, ha="left", va="top")
    ax.text(0.96, 0.04, "single Gaussian better", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, ha="right", va="bottom")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Test AUROC — single-Gaussian baseline")
    ax.set_ylabel(f"Test AUROC — GMM baseline ($K={SELECTED_K}$)")

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5.5,
                      markerfacecolor=METRIC_COLORS[f], markeredgecolor="black",
                      markeredgewidth=0.3, label=METRIC_LEGEND[f])
               for f in GMM_FAMILIES]
    handles += [
        Line2D([], [], marker="o", linestyle="none", markersize=5.5,
               markerfacecolor="0.7", markeredgecolor="black",
               markeredgewidth=0.3, label="per perturbation"),
        Line2D([], [], marker="D", linestyle="none", markersize=5.5,
               markerfacecolor="0.7", markeredgecolor="black",
               markeredgewidth=0.3, label="overall (mixed test set)"),
    ]
    _fig_legend(fig, handles, ncol=3)

    save_figure(fig, OUT_DIR, "auroc_gmm_vs_single")

    notes = FigureNotes(
        "auroc_gmm_vs_single",
        title="Test AUROC against a single-Gaussian baseline versus a "
              f"{SELECTED_K}-component GMM baseline",
        source=f"{RUN_DIR}/results_per_perturbation.json + summary.json")
    notes.line("    Each point is one detector family on one evaluation set. "
               "Above the diagonal = the GMM baseline scores higher.")
    notes.line("    Circles are per-perturbation (that perturbation against the "
               f"clean frames). The diamond is {AGG_POOLED_TEST}")
    notes.line(f"    {AGG_IDENTITY} The diamond is therefore the centroid of the "
               "four circles of its own colour.")
    notes.line("    'gain' below is GMM minus single, so a negative gain means "
               "clustering hurt that family.")

    for fam in GMM_FAMILIES:
        notes.section(PLAIN_NAME[fam])
        gains = []
        for gname, srcd, _, _ in cases:
            s, g = srcd[f"{fam}_single"], srcd[f"{fam}_gmm"]
            gains.append(g - s)
            label = PERT_LABELS.get(gname, "overall (mixed test set)")
            notes.value(label, f"single {s:.4f}, GMM {g:.4f}",
                        note=f"gain {g - s:+.4f}")
        per_pert_gains = gains[:len(PERT_ORDER)]
        notes.counts("perturbations where the GMM baseline wins",
                     {"wins": int(sum(x > 0 for x in per_pert_gains)),
                      "losses": int(sum(x < 0 for x in per_pert_gains)),
                      "of": len(per_pert_gains)})
        notes.value("mean gain over the four perturbations",
                    float(np.mean(per_pert_gains)))
        notes.value("gain on the mixed test set", gains[-1])

    notes.section("Summary across families")
    for fam in GMM_FAMILIES:
        g = overall[f"{fam}_gmm"] - overall[f"{fam}_single"]
        pp = [per_pert[pt][f"{fam}_gmm"] - per_pert[pt][f"{fam}_single"]
              for pt in PERT_ORDER]
        notes.value(PLAIN_NAME[fam],
                    f"mixed set {g:+.4f}, per-perturbation "
                    f"{sum(x > 0 for x in pp)} up / {sum(x < 0 for x in pp)} down")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figures 6-8 — per-cluster ATOMs attention profiles
# --------------------------------------------------------------------------- #
def cluster_attention_stats():
    """Baseline series, GMM cluster labels, profile names and the shared
    class display order (descending by max mean attention across clusters —
    same rule as run_analysis.py step 7)."""
    series = load_baseline_series()
    means, covs, weights, K = load_gmm()
    labels = gmm_predict(series, means, covs, weights)
    names = [str(n) for n in
             np.load(BASELINE_NPZ, allow_pickle=True)["profile_names"]]
    cluster_mean = np.stack([series[labels == k].mean(axis=0) for k in range(K)])
    order_desc = np.argsort(cluster_mean.max(axis=0))[::-1]
    return series, labels, names, K, cluster_mean, order_desc


# A class is called "unused" when its mean attention stays below this in every
# component.  A convention, not a measurement, hence [def] wherever it is quoted.
ATTENTION_ZERO_TOL = 0.005


def _note_cluster_attention(notes, series, labels, names, K, cluster_mean,
                            order_desc) -> None:
    """The numbers behind every per-cluster attention figure: the full
    component x class matrix, the per-class range across components, and which
    classes carry no attention at all.  Shared by all three, since they draw the
    same matrix in three layouts and the prose must not disagree between them."""
    sizes = np.bincount(labels, minlength=K)
    N = len(series)

    notes.line(f"    {N} baseline profiles over {len(names)} semantic classes, "
               f"{K} GMM components. Profiles are normalised to sum to 1, so a "
               f"value is a share of the frame's total relevance.")
    notes.line("    Class display order is descending by the largest component "
               "mean, the same rule the figure uses.")

    notes.section("Classes")
    unused = []
    for j in order_desc:
        col = cluster_mean[:, j]
        raw = series[:, j]
        if col.max() < ATTENTION_ZERO_TOL:
            unused.append(names[j])
        notes.value(names[j],
                    f"component means {col.min():.4g} to {col.max():.4g}",
                    note=(f"highest in component {int(np.argmax(col))}, "
                          f"lowest in {int(np.argmin(col))}; " if col.max() > 0
                          else "") +
                         f"over all frames max {raw.max():.4g}, nonzero in "
                         f"{int((raw > 0).sum())} of {N}")
    notes.value("classes with any attention in some component",
                len(names) - len(unused), note=f"of {len(names)}", derived=True)
    notes.value("classes below the unused threshold in every component",
                ", ".join(unused) or "none",
                note=f"threshold {ATTENTION_ZERO_TOL}", derived=True)
    exact_zero = [names[j] for j in range(len(names)) if series[:, j].max() == 0.0]
    notes.value("classes identically zero over every training frame",
                ", ".join(exact_zero) or "none")
    notes.line("        the four unused classes are near-zero rather than "
               "absent: " + ", ".join(
                   f"{names[j]} max {series[:, j].max():.3g} over {N} frames"
                   for j in range(len(names))
                   if cluster_mean[:, j].max() < ATTENTION_ZERO_TOL))

    notes.section("Components")
    for k in range(K):
        rows = series[labels == k]
        notes.value(f"component {k}", f"n = {int(sizes[k])}",
                    note=f"{sizes[k] / N:.1%} of the baseline")
        top = np.argsort(cluster_mean[k])[::-1]
        shown = [j for j in top if cluster_mean[k][j] >= ATTENTION_ZERO_TOL]
        notes.line("        mean: " + ", ".join(
            f"{names[j]} {cluster_mean[k][j]:.4f}" for j in shown))
        notes.line("        min-max whiskers: " + ", ".join(
            f"{names[j]} [{rows[:, j].min():.3f}, {rows[:, j].max():.3f}]"
            for j in shown))
        notes.value("    classes at or above the threshold", len(shown),
                    derived=True)
    notes.value("largest / smallest component",
                f"{int(sizes.max())} / {int(sizes.min())}",
                note=f"{sizes.max() / N:.1%} / {sizes.min() / N:.1%}")
    notes.value("component sizes ascending",
                ", ".join(str(int(s)) for s in np.sort(sizes)))
    notes.value("sizes sum", int(sizes.sum()),
                note="equals the training split, so the baseline is fitted on "
                     "training frames only")


def _cluster_bars(ax, series, labels, k, order_asc, with_whiskers=True):
    """Horizontal mean-attention bars (min-max whiskers) for one cluster."""
    rows = series[labels == k]
    mean, lo, hi = rows.mean(axis=0), rows.min(axis=0), rows.max(axis=0)
    vals = mean[order_asc]
    xerr = np.vstack([np.clip(vals - lo[order_asc], 0, None),
                      np.clip(hi[order_asc] - vals, 0, None)]) if with_whiskers else None
    ax.barh(np.arange(len(order_asc)), vals, height=0.72,
            color=cluster_color(k), linewidth=0,
            xerr=xerr, error_kw=dict(ecolor="0.35", elinewidth=0.8,
                                     capsize=2.0, capthick=0.8))
    ax.text(0.97, 0.05, f"Component {k} (n = {(labels == k).sum()})",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
            ha="right", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=False)


def fig_attention_per_cluster() -> None:
    series, labels, names, K, cluster_mean, order_desc = cluster_attention_stats()
    order_asc = order_desc[::-1]          # barh: largest class ends up on top

    n_rows = -(-K // 2)                       # 2 columns, rows follow K
    fig, axs = plt.subplots(n_rows, 2, figsize=(TEXT_WIDTH_IN, 1.725 * n_rows),
                            sharex=True, sharey=True)
    for k, ax in enumerate(axs.ravel()):
        if k >= K:                            # odd K leaves one cell empty
            ax.set_axis_off()
            continue
        _cluster_bars(ax, series, labels, k, order_asc)
    axs[0, 0].set_yticks(np.arange(len(order_asc)))
    axs[0, 0].set_yticklabels([names[j] for j in order_asc])

    fig.supxlabel("Normalized attention")
    save_figure(fig, OUT_DIR, "attention_per_cluster")

    notes = FigureNotes(
        "attention_per_cluster",
        title="Mean ATOMs attention profile per GMM component, one panel per "
              "component, min-max whiskers",
        source=f"{BASELINE_NPZ} + {RUN_DIR}/gmm.npz")
    notes.line("    Bars are the component mean, whiskers the min and max over "
               "that component's frames. Panels share the class order and axes.")
    _note_cluster_attention(notes, series, labels, names, K, cluster_mean,
                            order_desc)
    notes.write(OUT_DIR)


def representative_frames(series, labels, K):
    """Per cluster: the baseline frame closest (L2) to the cluster mean —
    same rule as run_analysis.py step 6b.5.  Loads only the two npz members
    needed (frame counts, then the one wide_rgb array per hit).

    Returns ``(imgs, provenance)``; provenance names the run file, the frame
    index inside it and the distance to the cluster mean, so the sidecar can
    say which frame a panel shows."""
    files = sorted(FRAMES_DIR.glob("run_*.npz"), key=lambda p: p.name)
    counts = np.array([np.load(f)["frame_idx"].shape[0] for f in files])
    bounds = np.concatenate([[0], np.cumsum(counts)])
    imgs, prov = {}, {}
    for k in range(K):
        mask = labels == k
        mean = series[mask].mean(axis=0)
        dists = np.linalg.norm(series[mask] - mean, axis=1)
        j = int(np.argmin(dists))
        gidx = int(np.where(mask)[0][j])
        fi = int(np.searchsorted(bounds, gidx, side="right") - 1)
        img = np.load(files[fi])["wide_rgb"][gidx - bounds[fi]]   # [3, H, W]
        imgs[k] = np.transpose(img, (1, 2, 0))
        prov[k] = (files[fi].name, gidx - bounds[fi], float(dists[j]),
                   float(np.median(dists)))
    return imgs, prov


def fig_attention_per_cluster_frames() -> None:
    series, labels, names, K, cluster_mean, order_desc = cluster_attention_stats()
    order_asc = order_desc[::-1]
    imgs, prov = representative_frames(series, labels, K)

    # 4 card rows x 2 columns; each card = frame strip (6:1) above its bars.
    n_rows = -(-K // 2)                       # 2 columns, rows follow K
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 2.025 * n_rows))
    gs = fig.add_gridspec(2 * n_rows, 2, height_ratios=[1.0, 2.9] * n_rows)

    bar_axs = []
    for k in range(K):
        r, c = divmod(k, 2)
        ax_img = fig.add_subplot(gs[2 * r, c])
        ax_img.imshow(imgs[k], aspect="auto")
        ax_img.set_axis_off()

        ax_bar = fig.add_subplot(gs[2 * r + 1, c],
                                 sharex=bar_axs[0] if bar_axs else None,
                                 sharey=bar_axs[0] if bar_axs else None)
        _cluster_bars(ax_bar, series, labels, k, order_asc)
        if c == 0:
            ax_bar.set_yticks(np.arange(len(order_asc)))
            ax_bar.set_yticklabels([names[j] for j in order_asc], fontsize=7.5)
        else:
            ax_bar.tick_params(labelleft=False)
        if r < n_rows - 1:
            ax_bar.tick_params(labelbottom=False)
        bar_axs.append(ax_bar)

    fig.supxlabel("Normalized attention")
    save_figure(fig, OUT_DIR, "attention_per_cluster_frames")

    notes = FigureNotes(
        "attention_per_cluster_frames",
        title="Mean ATOMs attention profile per GMM component, each paired with "
              "the baseline frame closest to that component's mean",
        source=f"{BASELINE_NPZ} + {FRAMES_DIR}")
    notes.line("    Same numbers as attention_per_cluster; only the "
               "representative frame strip is added.")
    notes.line("    A representative is the single frame nearest the component "
               "mean in profile space. It is an illustration of the component's "
               "centre, NOT evidence about what the component encodes: the "
               "spread below shows how far the component's frames reach.")
    _note_cluster_attention(notes, series, labels, names, K, cluster_mean,
                            order_desc)

    notes.section("Representative frames")
    for k in range(K):
        fname, fidx, d, dmed = prov[k]
        notes.value(f"component {k}", f"{fname}, frame {fidx}",
                    note=f"distance to the component mean {d:.4f}, median "
                         f"distance in the component {dmed:.4f}")
    notes.write(OUT_DIR)


def fig_attention_by_cluster() -> None:
    series, labels, names, K, cluster_mean, order_desc = cluster_attention_stats()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    x0 = np.arange(len(order_desc))
    width = 0.8 / K
    for k in range(K):
        ax.bar(x0 + (k - (K - 1) / 2) * width, cluster_mean[k][order_desc],
               width=width * 0.92, color=cluster_color(k), linewidth=0)

    ax.set_xticks(x0)
    ax.set_xticklabels([names[j] for j in order_desc], rotation=30, ha="right")
    ax.set_ylabel("Normalized attention")
    ax.grid(axis="x", visible=False)

    handles = [Patch(facecolor=cluster_color(k),
                     label=f"Component {k} (n = {(labels == k).sum()})")
               for k in range(K)]
    _fig_legend(fig, handles, ncol=4)

    save_figure(fig, OUT_DIR, "attention_by_cluster")

    notes = FigureNotes(
        "attention_by_cluster",
        title="Mean ATOMs attention per semantic class, all GMM components in "
              "one grouped bar chart",
        source=f"{BASELINE_NPZ} + {RUN_DIR}/gmm.npz")
    notes.line("    One bar group per class, one bar per component, no whiskers. "
               "The component sizes in the legend are the n values below.")
    _note_cluster_attention(notes, series, labels, names, K, cluster_mean,
                            order_desc)

    notes.section("Per class, every component mean")
    for j in order_desc:
        _note_listing(notes, names[j], range(K), cluster_mean[:, j],
                      x_fmt="c{:d}", per_line=8)
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 12 — K selection: mean GMM AUROC on val (criterion) vs test
# --------------------------------------------------------------------------- #
def fig_val_test_auc_vs_k() -> None:
    """Mean GMM AUROC vs K: validation (the selection criterion) vs test.

    Both curves EXCLUDE gaussian-noise frames: the val criterion
    (__val_auc_gmm_avg__) is defined ex-GN, so the test counterpart is
    recomputed here on the same footing — pooled AUC over clean + the three
    remaining perturbations, per detector, averaged.  kNN-GMM uses each K's
    val-selected k (parsed from summary.json).

    K = 1 is the single-Gaussian, no-clustering baseline, computed here rather
    than read from a snapshot (there is no "1 clusters" directory).  It really
    is the K = 1 member of the same family: one component means one mean and
    one covariance for Mahalanobis / Euclidean / JSD, and a kNN pool that is
    the whole baseline.  Including it is what lets the figure show what the
    clustering buys, which is the question the chapter asks of it."""
    from sklearn.metrics import roc_auc_score
    from ATOMs_Analysis.utils.distance_computer import DistanceComputer as DC

    sweep = load_sweep()
    Ks = sorted(set().union(*[set(v) for v in sweep.values()]))

    profiles = np.load(TEST_DIR / "attention" / "test_profiles_2.npy").astype(np.float64)
    tl = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)
    labels = tl["label"].astype(int)
    mask = tl["perturbation"].astype(str) != "gaussian_noise"
    series = load_baseline_series()

    vprof = np.load(ROOT / "data" / "TFV6" / "val_data_alt" / "attention"
                    / "val_profiles_2.npy").astype(np.float64)
    vl = np.load(ROOT / "data" / "TFV6" / "val_data_alt" / "val_labeled.npz",
                 allow_pickle=True)
    vlabels = vl["label"].astype(int)
    vmask = vl["perturbation"].astype(str) != "gaussian_noise"

    def _single_gaussian_aucs():
        """The four detector AUROCs at K = 1, on both splits, ex-Gaussian-noise.

        Same four families and the same ex-GN pool as every other point on the
        curve, scored against the run's stored single-Gaussian parameters
        (mahal_detector.npz, shrinkage already applied) instead of a mixture.
        kNN uses the whole baseline, which is what a single component's pool
        is, at the val-selected single-kNN k."""
        from scipy.spatial.distance import cdist
        md = np.load(RUN_DIR / "mahal_detector.npz", allow_pickle=True)
        mean = md["mean"].astype(np.float64)
        cov = md["cov"].astype(np.float64)

        summ8 = json.loads((RUN_DIR / "summary.json").read_text())
        knn_key = next(k for k in summ8 if "k-NN" in k and "GMM" not in k)
        k1 = int(re.search(r"k=(\d+)", knn_key).group(1))

        def _norm(a):
            return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        base_n = _norm(series)

        out = {}
        for tag, X, lab, msk in (("val", vprof, vlabels, vmask),
                                 ("test", profiles, labels, mask)):
            s_m = mahalanobis_batch(X, mean, cov)
            s_e = np.linalg.norm(X - mean, axis=1)
            s_j = np.array([DC.compute_jsd(mean, x) for x in X])
            s_k = np.sort(cdist(_norm(X), base_n), axis=1)[:, k1 - 1]
            out[tag] = [float(roc_auc_score(lab[msk], s[msk]))
                        for s in (s_m, s_e, s_j, s_k)]
        # sanity: the incl-GN single-Gaussian mahal AUC must match the stored one
        auc_all = roc_auc_score(labels, mahalanobis_batch(profiles, mean, cov))
        stored = json.loads((RUN_DIR / "results_ATOMs-Mahalanobis_single_Gaussian"
                             ".json").read_text())["auc"]
        if abs(auc_all - stored) > 1e-3:
            raise RuntimeError(
                f"K=1: single-Gaussian mahal {auc_all:.4f} != stored {stored:.4f}")
        print(f"  [check] K=1 single Gaussian: mahal incl-GN {auc_all:.4f} "
              f"== stored {stored:.4f}  (k-NN k={k1})")
        return out, k1

    single, k_single = _single_gaussian_aucs()

    val_avg, test_avg = [], []
    test_per_det: dict[int, list[float]] = {}
    for K in Ks:
        run = RESULTS_ROOT / f"{K} clusters" / "atoms_analysis_mode_2"
        summ = json.loads((run / "summary.json").read_text())
        val_avg.append(summ["__val_auc_gmm_avg__"])

        g = np.load(run / "gmm.npz", allow_pickle=True)
        means = g["means"].astype(np.float64)
        covs = g["covariances"].astype(np.float64)
        weights = g["weights"].astype(np.float64)

        comp_d = np.stack([mahalanobis_batch(profiles, means[c], covs[c])
                           for c in range(len(weights))])
        route = comp_d.argmin(axis=0)
        s_mahal = comp_d.min(axis=0)
        s_euclid = np.linalg.norm(profiles - means[route], axis=1)
        s_jsd = np.array([DC.compute_jsd(means[route[i]], profiles[i])
                          for i in range(len(profiles))])
        knn_key = next(k for k in summ if "k-NN-GMM" in k)
        k_val = int(re.search(r"k=(\d+)", knn_key).group(1))
        s_knn = knn_gmm_scores(profiles, series, means, covs, weights, k=k_val)

        aucs = [roc_auc_score(labels[mask], s[mask])
                for s in (s_mahal, s_euclid, s_jsd, s_knn)]
        test_avg.append(float(np.mean(aucs)))
        test_per_det[K] = [float(a) for a in aucs]
        # sanity: incl-GN pooled mahal AUC must match the stored summary value
        auc_all = roc_auc_score(labels, s_mahal)
        stored = next(v["auc"] for key, v in summ.items()
                      if isinstance(v, dict) and "Mahalanobis (GMM" in key)
        if abs(auc_all - stored) > 1e-3:
            raise RuntimeError(f"K={K}: mahal-GMM {auc_all:.4f} != stored {stored:.4f}")

    # K = 1 (no clustering) leads the sweep, so the figure shows what the
    # mixture buys rather than only how the mixture varies.
    Ks = [1] + Ks
    val_avg = [float(np.mean(single["val"]))] + val_avg
    test_avg = [float(np.mean(single["test"]))] + test_avg
    test_per_det[1] = single["test"]

    print(f"  [info] K=1 (single Gaussian): val ex-GN {val_avg[0]:.4f}  "
          f"test ex-GN {test_avg[0]:.4f}")
    print(f"  [info] K={SELECTED_K}: val ex-GN {val_avg[Ks.index(SELECTED_K)]:.4f}  "
          f"test ex-GN {test_avg[Ks.index(SELECTED_K)]:.4f}")

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    lo = min(min(val_avg), min(test_avg)) - 0.012
    hi = max(max(val_avg), max(test_avg)) + 0.012
    ax.axvline(SELECTED_K, color="0.78", linewidth=0.8,
               linestyle=(0, (2, 2)), zorder=1)
    ax.plot(Ks, val_avg, color="0.15", linewidth=1.9,
            marker="o", markersize=3.4, markeredgewidth=0, zorder=3)
    ax.plot(Ks, test_avg, color="0.15", linewidth=1.25,
            linestyle=(0, (1, 1.6)), marker="o", markersize=2.6,
            markeredgewidth=0, zorder=2)

    # Circle each curve's own best K, the way knn_k_selection circles its two
    # selected k.  The validation circle is the actual selection; the test one
    # is where the selection would have landed with hindsight.
    for ys in (val_avg, test_avg):
        i_best = int(np.argmax(ys))
        ax.scatter([Ks[i_best]], [ys[i_best]], s=64, facecolor="none",
                   edgecolor="0.15", linewidths=1.1, zorder=4)

    ax.text(SELECTED_K + 0.35, lo + 0.003, f"$K={SELECTED_K}$",
            fontsize=7.5, color=MUTED, ha="left", va="bottom")
    ax.text(1, lo + 0.003, "no clustering",
            fontsize=7.5, color=MUTED, ha="center", va="bottom")

    ax.set_xlabel("GMM components $K$")
    ax.set_ylabel("Mean AUROC (GMM detectors)")
    ax.set_xticks([1] + list(range(2, 21, 2)))
    ax.set_xlim(0.4, 20.6)
    ax.set_ylim(lo, hi)

    handles = [
        Line2D([], [], color="0.15", linewidth=1.9, label="validation"),
        Line2D([], [], color="0.15", linewidth=1.25, linestyle=(0, (1, 1.6)),
               label="test"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor="none", markeredgecolor="0.15",
               markeredgewidth=1.1, label="best $K$ on that split"),
    ]
    _fig_legend(fig, handles, ncol=3)

    save_figure(fig, OUT_DIR, "auroc_val_test_vs_K")

    val_avg = [float(v) for v in val_avg]
    offsets = [v - t for v, t in zip(val_avg, test_avg)]
    i_sel = Ks.index(SELECTED_K)
    i_val_best, i_test_best = int(np.argmax(val_avg)), int(np.argmax(test_avg))

    notes = FigureNotes(
        "auroc_val_test_vs_K",
        title="K selection: mean GMM-detector AUROC on the validation set "
              "against its test-set counterpart",
        source=str(RESULTS_ROOT) + "/<K> clusters/atoms_analysis_mode_2")
    notes.line(f"    Sweep: {len(Ks)} values of K, {min(Ks)} to {max(Ks)} "
               f"({', '.join(str(k) for k in Ks)}).")
    notes.line(f"    K = 1 is the single-Gaussian, NO-CLUSTERING baseline. It is "
               f"not a snapshot directory, it is computed from the run's stored "
               f"mahal_detector.npz on the same four families and the same "
               f"ex-Gaussian-noise pool, with the k-NN pool being the whole "
               f"baseline at the val-selected single k = {k_single}. Its "
               f"incl-Gaussian-noise Mahalanobis AUROC was checked against the "
               f"stored results_ATOMs-Mahalanobis_single_Gaussian.json.")
    notes.line(f"    BOTH curves: {AGG_POOLED_EXGN}")
    notes.line(f"    {AGG_IDENTITY} Here the pool holds three perturbations, so "
               "each detector's value is the mean of its brightness, camera loss "
               "and PGD AUROCs, and the plotted curve is the mean of those four "
               "detector values.")
    notes.line(f"    {AGG_MEAN_OVER_DETECTORS} The four are Mahalanobis, "
               "Euclidean, JSD and k-NN, all GMM-pooled; k-NN uses each K's own "
               "val-selected k.")
    notes.line("    The validation curve is the actual selection criterion "
               "(summary.json __val_auc_gmm_avg__); the test curve is recomputed "
               "here on the same footing so the two are like-for-like.")
    notes.line("    Gaussian noise is EXCLUDED from both pools. The other "
               "AUROC figures in the chapter include it, so their numbers are "
               "not comparable with these.")
    notes.line(f"    The same 1000 validation and 1000 test frames are reused at "
               f"every K, so the {len(Ks)} points are not independent "
               f"comparisons and a consistent offset is not evidence about its "
               f"cause.")
    notes.line("    No rise/plateau/shape descriptors: those are defined for a "
               "smooth sweep over a severity knob, which K is not. The per-K "
               "listing carries the shape.")

    notes.section("Validation (the selection criterion)")
    notes.curve("mean GMM AUROC vs K", Ks, val_avg, rise=False, plateau=False, shape=False)
    notes.value("peak", val_avg[i_val_best], note=f"at K = {Ks[i_val_best]}")
    notes.value("minimum", min(val_avg),
                note=f"at K = {Ks[int(np.argmin(val_avg))]}")
    notes.value(f"at the selected K = {SELECTED_K}", val_avg[i_sel])
    _note_listing(notes, "per K", Ks, val_avg)

    notes.section("Test")
    notes.curve("mean GMM AUROC vs K", Ks, test_avg, rise=False, plateau=False, shape=False)
    notes.value("peak", test_avg[i_test_best], note=f"at K = {Ks[i_test_best]}")
    notes.value("minimum", min(test_avg),
                note=f"at K = {Ks[int(np.argmin(test_avg))]}")
    notes.value(f"at the selected K = {SELECTED_K}", test_avg[i_sel])
    notes.value("cost of selecting on validation",
                test_avg[i_test_best] - test_avg[i_sel],
                note=f"test AUROC at its own best K = {Ks[i_test_best]} minus "
                     f"test AUROC at the validation-selected K = {SELECTED_K}")
    _note_listing(notes, "per K", Ks, test_avg)

    notes.section("Validation minus test")
    notes.value("sign", "validation above test at every K"
                if all(o > 0 for o in offsets) else
                f"validation above test at {sum(o > 0 for o in offsets)} of "
                f"{len(offsets)} values of K")
    notes.value("offset range", f"{min(offsets):.4f} to {max(offsets):.4f}")
    notes.value("median offset", float(np.median(offsets)))
    notes.value(f"offset at K = {SELECTED_K}", offsets[i_sel])
    _note_listing(notes, "per K", Ks, offsets, y_fmt="{:+.4f}")

    notes.section("What the clustering buys, against K = 1")
    notes.value("validation at K = 1", val_avg[0])
    notes.value("test at K = 1", test_avg[0])
    notes.value(f"validation gain at the selected K = {SELECTED_K}",
                val_avg[i_sel] - val_avg[0])
    notes.value(f"test gain at the selected K = {SELECTED_K}",
                test_avg[i_sel] - test_avg[0])
    notes.value("values of K beating K = 1 on validation",
                f"{sum(v > val_avg[0] for v in val_avg[1:])} of {len(Ks) - 1}")
    notes.value("values of K beating K = 1 on test",
                f"{sum(t > test_avg[0] for t in test_avg[1:])} of {len(Ks) - 1}")
    notes.value("per detector at K = 1, validation",
                ", ".join(f"{PLAIN_NAME[f]} {v:.4f}" for f, v in
                          zip(("mahalanobis", "euclidean", "jsd", "knn"),
                              single["val"])))
    notes.value("per detector at K = 1, test",
                ", ".join(f"{PLAIN_NAME[f]} {v:.4f}" for f, v in
                          zip(("mahalanobis", "euclidean", "jsd", "knn"),
                              single["test"])))

    notes.section("Flatness")
    tail = [i for i, K in enumerate(Ks) if K >= 12]
    notes.value("validation spread over K >= 12",
                max(val_avg[i] for i in tail) - min(val_avg[i] for i in tail))
    notes.value("test spread over K >= 12",
                max(test_avg[i] for i in tail) - min(test_avg[i] for i in tail))
    notes.value("validation spread over the whole sweep",
                max(val_avg) - min(val_avg))
    notes.value("test spread over the whole sweep", max(test_avg) - min(test_avg))

    notes.section("Test, per detector")
    det_names = [PLAIN_NAME[f] for f in ("mahalanobis", "euclidean", "jsd", "knn")]
    for di, dn in enumerate(det_names):
        ys = [test_per_det[K][di] for K in Ks]
        notes.value(dn, f"peak {max(ys):.4f} at K = {Ks[int(np.argmax(ys))]}, "
                        f"at K = {SELECTED_K}: {ys[i_sel]:.4f}")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 14 - AUROC per perturbation against the component count K
# --------------------------------------------------------------------------- #
def fig_auroc_per_perturbation_vs_k() -> None:
    """Per-perturbation AUROC against K, on both splits, one panel each.

    auroc_val_test_vs_K averages three perturbations and then four detectors,
    so a single curve carries a level set almost entirely by the brightness
    increase.  This figure decomposes it, because the interpretation reads the
    shape of that curve as a property of the reference model.

    Small multiples rather than four coloured lines: PERT_COLORS is a
    dark-to-light red ramp built for the stacked histogram, and two of its four
    steps are too pale to carry a line.  The panel title is the identity
    channel instead, and the shared y-axis is the point, since the panels sit
    at very different heights.

    Also carries the k-fallback audit.  The clustered k-th-NN entry inside the
    mean uses each K's own val-selected k, while the components shrink as K
    grows, so the entry silently changes character once a component holds fewer
    frames than k.  The sidecar reports the smallest component and the share of
    evaluation samples routed to a component below k, per K."""
    from sklearn.metrics import roc_auc_score
    from scipy.spatial.distance import cdist
    from ATOMs_Analysis.utils.distance_computer import DistanceComputer as DC

    Ks = sorted(set().union(*[set(v) for v in load_sweep().values()]))
    series = load_baseline_series()

    splits = {}
    for tag, d, prof, lab in (
            ("val", ROOT / "data" / "TFV6" / "val_data_alt", "val_profiles_2.npy",
             "val_labeled.npz"),
            ("test", TEST_DIR, "test_profiles_2.npy", "test_labeled.npz")):
        X = np.load(d / "attention" / prof).astype(np.float64)
        z = np.load(d / lab, allow_pickle=True)
        splits[tag] = (X, z["label"].astype(int), z["perturbation"].astype(str))

    def _per_pert(scores, labels, perts):
        """AUROC of each perturbation against the clean frames only."""
        out = {}
        for p in PERT_ORDER:
            m = (perts == p) | (labels == 0)
            out[p] = float(roc_auc_score(labels[m], scores[m]))
        return out

    def _norm(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

    # K = 1, the single-Gaussian baseline, exactly as fig_val_test_auc_vs_k
    # builds it: the run's stored single-Gaussian parameters and a k-NN pool
    # that is the whole baseline, at the val-selected single k.
    md = np.load(RUN_DIR / "mahal_detector.npz", allow_pickle=True)
    s_mean, s_cov = md["mean"].astype(np.float64), md["cov"].astype(np.float64)
    summ8 = json.loads((RUN_DIR / "summary.json").read_text())
    k1 = int(re.search(r"k=(\d+)",
                       next(k for k in summ8 if "k-NN" in k and "GMM" not in k)).group(1))

    curves = {(t, p): [] for t in splits for p in PERT_ORDER}
    base_n = _norm(series)
    for tag, (X, lab, prt) in splits.items():
        det = [_per_pert(s, lab, prt) for s in (
            mahalanobis_batch(X, s_mean, s_cov),
            np.linalg.norm(X - s_mean, axis=1),
            np.array([DC.compute_jsd(s_mean, x) for x in X]),
            np.sort(cdist(_norm(X), base_n), axis=1)[:, k1 - 1])]
        for p in PERT_ORDER:
            curves[(tag, p)].append(float(np.mean([d[p] for d in det])))

    smallest, fallback = {}, {}
    for K in Ks:
        run = RESULTS_ROOT / f"{K} clusters" / "atoms_analysis_mode_2"
        summ = json.loads((run / "summary.json").read_text())
        g = np.load(run / "gmm.npz", allow_pickle=True)
        means = g["means"].astype(np.float64)
        covs = g["covariances"].astype(np.float64)
        weights = g["weights"].astype(np.float64)
        k_val = int(re.search(r"k=(\d+)",
                              next(k for k in summ if "k-NN-GMM" in k)).group(1))

        sizes = np.bincount(gmm_predict(series, means, covs, weights),
                            minlength=len(weights))
        smallest[K] = (int(sizes.min()), k_val)

        for tag, (X, lab, prt) in splits.items():
            cd = np.stack([mahalanobis_batch(X, means[c], covs[c])
                           for c in range(len(weights))])
            route = cd.argmin(axis=0)
            det = [_per_pert(s, lab, prt) for s in (
                cd.min(axis=0),
                np.linalg.norm(X - means[route], axis=1),
                np.array([DC.compute_jsd(means[route[i]], X[i])
                          for i in range(len(X))]),
                knn_gmm_scores(X, series, means, covs, weights, k=k_val))]
            for p in PERT_ORDER:
                curves[(tag, p)].append(float(np.mean([d[p] for d in det])))
            if tag == "test":
                small = {c for c in range(len(weights)) if sizes[c] < k_val}
                fallback[K] = float(np.mean([r in small for r in route]))

    # sanity: the stored per-perturbation test AUROCs at the selected K must
    # reproduce, and the three-perturbation mean must equal the plotted curve
    # of auroc_val_test_vs_K at that K.
    stored = load_per_perturbation()
    i_sel = [1] + Ks
    j = i_sel.index(SELECTED_K)
    for p in PERT_ORDER:
        want = float(np.mean([stored[p][f"{f}_gmm"] for f in GMM_FAMILIES]))
        got = curves[("test", p)][j]
        if abs(want - got) > 1e-3:
            raise RuntimeError(f"K={SELECTED_K} {p}: {got:.4f} != stored {want:.4f}")
    print(f"  [check] per-perturbation test AUROC at K={SELECTED_K} reproduces "
          "results_per_perturbation.json")

    xs = [1] + Ks
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.2), sharex=True, sharey=True)
    for ax, p in zip(axes.ravel(), PERT_ORDER):
        ax.axhline(0.5, color="0.78", linewidth=0.8, zorder=1)
        ax.axvline(SELECTED_K, color="0.78", linewidth=0.8,
                   linestyle=(0, (2, 2)), zorder=1)
        ax.plot(xs, curves[("val", p)], color="0.15", linewidth=1.6,
                marker="o", markersize=2.6, markeredgewidth=0, zorder=3)
        ax.plot(xs, curves[("test", p)], color="0.15", linewidth=1.2,
                linestyle=(0, (1, 1.6)), marker="o", markersize=2.2,
                markeredgewidth=0, zorder=2)
        ax.set_title(PERT_LABELS[p], fontsize=8.5, pad=3)
        ax.set_xticks([1] + list(range(4, 21, 4)))
        ax.set_xlim(0.4, 20.6)
    for ax in axes[-1]:
        ax.set_xlabel("GMM components $K$")
    for ax in axes[:, 0]:
        ax.set_ylabel("AUROC")

    handles = [
        Line2D([], [], color="0.15", linewidth=1.6, label="validation"),
        Line2D([], [], color="0.15", linewidth=1.2, linestyle=(0, (1, 1.6)),
               label="test"),
        Line2D([], [], color="0.78", linewidth=0.8, label="chance"),
        Line2D([], [], color="0.78", linewidth=0.8, linestyle=(0, (2, 2)),
               label=f"selected $K={SELECTED_K}$"),
    ]
    _fig_legend(fig, handles, ncol=4)
    fig.tight_layout()
    save_figure(fig, OUT_DIR, "auroc_per_perturbation_vs_K")

    notes = FigureNotes(
        "auroc_per_perturbation_vs_K",
        title="AUROC per perturbation against the component count K, "
              "validation and test",
        source=str(RESULTS_ROOT) + "/<K> clusters/atoms_analysis_mode_2 "
               "+ the labelled val/test splits")
    notes.line(f"    Sweep: K = 1 and {min(Ks)} to {max(Ks)}. K = 1 is the "
               "single-Gaussian, no-clustering baseline, built the same way as "
               "in auroc_val_test_vs_K.txt.")
    notes.line("    A panel is one perturbation's frames against the clean "
               "frames only, so a panel is NOT a pool and the four panels do "
               "not average to the K-selection curve, which drops Gaussian "
               "noise. The mean of the other three panels does.")
    notes.line("    Each curve is the unweighted mean over the four GMM "
               "detectors, the same averaging step as auroc_val_test_vs_K. "
               "k-NN-GMM uses each K's own val-selected k.")
    notes.line("    Shared y-axis on purpose: the panels sit at very different "
               "heights and that is the finding.")
    notes.line("    The same 1000 frames per split are scored at every K, so "
               "the points are not independent comparisons.")
    notes.line("    No rise/plateau/shape descriptors: K is not a severity knob.")

    mean3 = [float(np.mean([curves[("test", p)][i] for p in PERT_ORDER
                            if p != "gaussian_noise"])) for i in range(len(xs))]
    for p in PERT_ORDER:
        notes.section(PERT_LABELS[p])
        for tag in ("val", "test"):
            y = curves[(tag, p)]
            notes.curve(f"{tag} AUROC vs K", xs, y,
                        rise=False, plateau=False, shape=False)
            notes.value(f"{tag}: counts of K beating the K = 1 value",
                        sum(1 for v in y[1:] if v > y[0]),
                        note=f"of {len(Ks)} clustered counts")
            below = [xs[i] for i, v in enumerate(y) if v < 0.5]
            notes.value(f"{tag}: K with AUROC below chance",
                        ", ".join(str(b) for b in below) if below else "none")
        notes.value("test: rank correlation with the plotted K-selection curve",
                    spearman(curves[("test", p)], mean3))

    notes.section("The k-th-NN fallback across the sweep")
    notes.line("    The clustered k-th-NN entry in the mean uses each K's own "
               "val-selected k, while the components shrink as K grows. Once a "
               "component holds fewer frames than k, every sample routed to it "
               "is scored against the full baseline instead, so the entry "
               "changes character across the sweep.")
    for K in Ks:
        s, kv = smallest[K]
        notes.line(f"    K = {K:>2}: smallest component {s:>4} frames, "
                   f"k = {kv:>3}, test samples routed below k "
                   f"{fallback[K]:.1%}")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figure 13 — kNN neighbour-count selection on the validation set
# --------------------------------------------------------------------------- #
def fig_knn_k_selection() -> None:
    from scipy.spatial.distance import cdist
    from sklearn.metrics import roc_auc_score

    KS_NN = [1, 5, 10, 25, 50, 100, 250]
    series = load_baseline_series()
    vprof = np.load(ROOT / "data" / "TFV6" / "val_data_alt" / "attention"
                    / "val_profiles_2.npy").astype(np.float64)
    # k is selected on the FULL val set (incl. gaussian noise), exactly as
    # run_analysis.py does for results_knn(_gmm)_val_by_k.
    vlab = np.load(ROOT / "data" / "TFV6" / "val_data_alt"
                   / "val_labeled.npz")["label"].astype(int)

    def norm(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

    # The TEST curves are overlaid since 2026-09-03.  Validation-only, this
    # figure looked as though it contradicted the test-set figures; it does not,
    # it is a different split, and showing both is what makes the selection
    # failure legible in one place.  Same pool on both splits: the full set,
    # Gaussian noise included, exactly as run_analysis.py selects k.
    tprof = np.load(TEST_DIR / "attention" / "test_profiles_2.npy").astype(np.float64)
    tlab = np.load(TEST_DIR / "test_labeled.npz",
                   allow_pickle=True)["label"].astype(int)

    means, covs, weights, _ = load_gmm()

    d_single = np.sort(cdist(norm(vprof), norm(series)), axis=1)
    d_single_t = np.sort(cdist(norm(tprof), norm(series)), axis=1)
    curves = {
        ("single", "val"):  [roc_auc_score(vlab, d_single[:, k - 1]) for k in KS_NN],
        ("single", "test"): [roc_auc_score(tlab, d_single_t[:, k - 1]) for k in KS_NN],
        ("gmm", "val"):     [roc_auc_score(vlab, knn_gmm_scores(
                                 vprof, series, means, covs, weights, k=k))
                             for k in KS_NN],
        ("gmm", "test"):    [roc_auc_score(tlab, knn_gmm_scores(
                                 tprof, series, means, covs, weights, k=k))
                             for k in KS_NN],
    }
    auc_single, auc_gmm = curves[("single", "val")], curves[("gmm", "val")]

    VAL_STYLE, TEST_STYLE = "solid", (0, (1, 1.6))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for (variant, split), aucs in curves.items():
        color = KNN_SINGLE_COLOR if variant == "single" else METRIC_COLORS["knn"]
        ax.plot(KS_NN, aucs, color=color,
                linewidth=1.6 if split == "val" else 1.25, alpha=0.95,
                linestyle=VAL_STYLE if split == "val" else TEST_STYLE,
                marker="o", markersize=3.4 if split == "val" else 2.6,
                markeredgewidth=0)
        k_best = KS_NN[int(np.argmax(aucs))]
        ax.scatter([k_best], [max(aucs)], s=64, facecolor="none",
                   edgecolor=color, linewidths=1.1, zorder=4)
        print(f"  [info] best k = {k_best:3d} ({variant:6s} {split:4s}) "
              f"AUROC {max(aucs):.4f}")

    ax.set_xscale("log")
    ax.set_xticks(KS_NN)
    ax.set_xticklabels([str(k) for k in KS_NN])
    ax.minorticks_off()
    ax.set_xlabel("Neighbour count $k$")
    ax.set_ylabel("AUROC")

    handles = [
        Line2D([], [], color=KNN_SINGLE_COLOR, linewidth=1.6,
               label=METRIC_LEGEND["knn"] + " (single)"),
        Line2D([], [], color=METRIC_COLORS["knn"], linewidth=1.6,
               label=METRIC_LEGEND["knn"] + " (GMM)"),
        Line2D([], [], color="0.35", linewidth=1.6, linestyle=VAL_STYLE,
               label="validation"),
        Line2D([], [], color="0.35", linewidth=1.25, linestyle=TEST_STYLE,
               label="test"),
    ]
    _fig_legend(fig, handles, ncol=4)

    save_figure(fig, OUT_DIR, "knn_k_selection")

    _, _, weights_k, _ = load_gmm()
    comp_sizes = np.sort(np.bincount(
        gmm_predict(series, means, covs, weights), minlength=len(weights)))

    notes = FigureNotes(
        "knn_k_selection",
        title="Neighbour-count selection for the k-th-NN attention distance on "
              "the validation set",
        source=f"{ROOT}/data/TFV6/val_data_alt")
    notes.line("    BOTH SPLITS since 2026-09-03. Validation solid, test dotted. "
               "Validation-only, this figure read as though it contradicted the "
               "test-set figures; it does not, it is a different split, and "
               "k is selected on the validation curve alone.")
    notes.line("    AUROC here is a single pooled AUROC over the FULL validation "
               "set INCLUDING Gaussian noise, per k, exactly as run_analysis.py "
               "selects k. It is one detector, not a mean over detectors, and it "
               "includes Gaussian noise, so it is NOT the criterion used for K "
               "in auroc_val_test_vs_K.")
    notes.line(f"    {AGG_IDENTITY} Here the pool holds all four perturbations.")
    notes.line(f"    'single' scores against the whole {len(series)}-frame "
               f"baseline. 'GMM' routes each sample to its nearest of the "
               f"{len(weights_k)} components and scores within that component's "
               f"pool only, falling back to the full baseline when the pool "
               f"holds fewer than k frames.")
    notes.value("component sizes ascending",
                ", ".join(str(int(s)) for s in comp_sizes))
    notes.value("k values exceeding the smallest component",
                ", ".join(str(k) for k in KS_NN if k > comp_sizes[0]) or "none",
                note="those k trigger the full-baseline fallback for that "
                     "component's samples")
    notes.value("components smaller than k = 100",
                int((comp_sizes < 100).sum()), note=f"of {len(comp_sizes)}")
    notes.value("components smaller than k = 250",
                int((comp_sizes < 250).sum()), note=f"of {len(comp_sizes)}")

    notes.section("Routing, and how many samples the fallback actually touches")
    notes.line("    Counting components exceeded understates the fallback: what "
               "matters is how many samples are ROUTED to a component too small "
               "for k, since those are scored against a reference set of a "
               "different size and therefore on a different scale, mixed into "
               "one ranking with the rest.")
    sizes_by_c = np.bincount(gmm_predict(series, means, covs, weights),
                             minlength=len(weights))
    vd = np.stack([mahalanobis_batch(vprof, means[c], covs[c])
                   for c in range(len(weights))])
    routed = np.bincount(vd.argmin(axis=0), minlength=len(weights))
    notes.counts("baseline frames per component",
                 {str(c): int(sizes_by_c[c]) for c in range(len(weights))})
    notes.counts("validation samples routed to each component",
                 {str(c): int(routed[c]) for c in range(len(weights))})
    for c in range(len(weights)):
        notes.value(f"    component {c}",
                    f"{sizes_by_c[c] / len(series):.1%} of the baseline, "
                    f"{routed[c] / len(vprof):.1%} of the validation samples")
    for k in KS_NN:
        n_fb = int(sum(routed[c] for c in range(len(weights))
                       if sizes_by_c[c] < k))
        if n_fb:
            notes.value(f"validation samples falling back at k = {k}", n_fb,
                        note=f"{n_fb / len(vprof):.1%}, from "
                             f"{int((sizes_by_c < k).sum())} component(s)")
    notes.line("    Fallback rule, verified in run_analysis.py step 9b.3 "
               "(`if len(_pool) < _k: _pool = baseline_series`): the pool is "
               "replaced by the FULL baseline. It is not clamped and not padded, "
               "and DistanceComputer.compute_knn_distance raises rather than "
               "clamping, so the caller is the only place this is handled. "
               "Same rule as the Atari implementation.")

    for (variant, split), aucs in curves.items():
        label = ("single (full baseline)" if variant == "single"
                 else "GMM (component pools)") + f" — {split}"
        notes.section(label)
        kb = int(np.argmax(aucs))
        notes.value("best k", KS_NN[kb], note=f"AUROC {aucs[kb]:.4f}")
        notes.value("worst k", KS_NN[int(np.argmin(aucs))],
                    note=f"AUROC {min(aucs):.4f}")
        notes.value("spread over the sweep", max(aucs) - min(aucs))
        _note_listing(notes, "per k", KS_NN, aucs, x_fmt="k={:d}")

    notes.section("Selection, and whether it transfers")
    for variant in ("single", "gmm"):
        kv = KS_NN[int(np.argmax(curves[(variant, "val")]))]
        kt = KS_NN[int(np.argmax(curves[(variant, "test")]))]
        i_v = KS_NN.index(kv)
        notes.value(f"{variant}: k chosen on validation", kv,
                    note=f"test AUROC there {curves[(variant, 'test')][i_v]:.4f}")
        notes.value(f"{variant}: k that would have won on test", kt,
                    note=f"test AUROC {max(curves[(variant, 'test')]):.4f}, "
                         f"cost of selecting on validation "
                         f"{max(curves[(variant, 'test')]) - curves[(variant, 'test')][i_v]:.4f}")
    kv_g = KS_NN[int(np.argmax(curves[("gmm", "val")]))]
    kv_s = KS_NN[int(np.argmax(curves[("single", "val")]))]
    notes.value("validation prefers", 
                f"GMM at k = {kv_g} ({max(curves[('gmm', 'val')]):.4f}) over "
                f"single at k = {kv_s} ({max(curves[('single', 'val')]):.4f})")
    notes.value("test at those same two settings",
                f"GMM k = {kv_g}: {curves[('gmm', 'test')][KS_NN.index(kv_g)]:.4f}, "
                f"single k = {kv_s}: {curves[('single', 'test')][KS_NN.index(kv_s)]:.4f}",
                note="the selection reverses between the splits")

    notes.section("The two variants against each other, on validation")
    notes.value("difference at k = 1", auc_gmm[0] - auc_single[0],
                note="they should nearly coincide here, since the single "
                     "nearest baseline frame is usually inside the sample's own "
                     "component")
    for i, k in enumerate(KS_NN):
        notes.value(f"k = {k}", f"single {auc_single[i]:.4f}, "
                                f"GMM {auc_gmm[i]:.4f}",
                    note=f"GMM minus single {auc_gmm[i] - auc_single[i]:+.4f}")
    gb, sb = int(np.argmax(auc_gmm)), int(np.argmax(auc_single))
    notes.value("selected GMM peak over its neighbours",
                f"k={KS_NN[max(gb - 1, 0)]}: {auc_gmm[max(gb - 1, 0)]:.4f}, "
                f"k={KS_NN[gb]}: {auc_gmm[gb]:.4f}, "
                f"k={KS_NN[min(gb + 1, len(KS_NN) - 1)]}: "
                f"{auc_gmm[min(gb + 1, len(KS_NN) - 1)]:.4f}",
                note="how isolated the selected point is")
    notes.value("GMM best minus single best", auc_gmm[gb] - auc_single[sb])
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Figures 9-11 — live perturbation: detector scores over time
# --------------------------------------------------------------------------- #
# One live-run variant per perturbation (the runs picked for the thesis).
LIVE_VARIANTS = {
    "brightness_scale": "20260623_3front_000",
    "gaussian_noise":   "20260622_224036_000",
    "pgd":              "20260630_weak_000",
}


def _speed_entropy(logits: np.ndarray) -> np.ndarray:
    """PEOC score: Shannon entropy of the softmaxed 8-bin speed logits —
    matches ActionEntropyDetector(from_logits=True).score_batch."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def _predicted_speed(logits: np.ndarray) -> np.ndarray:
    """The agent's predicted target speed in m/s: the two-hot decoding of the
    softmaxed 8-bin speed logits, which is the expectation over the bins
    (planning_decoder.decode_two_hot)."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(axis=1, keepdims=True)
    return p @ SPEED_BINS


def _note_live_trace(notes: FigureNotes, label: str, y: np.ndarray,
                     injection: int, unit: str = "") -> None:
    """Pre/post-injection description of one trace, plus the largest step in
    each phase.  The step matters because a rise the clean twin also shows is
    the scene changing, not the perturbation."""
    y = np.asarray(y, float)
    pre, post = y[:injection], y[injection:]
    u = f" {unit}" if unit else ""
    notes.line(f"    {label}:")
    pad = "        "
    for name, seg, off in (("pre-injection", pre, 0),
                           ("post-injection", post, injection)):
        if len(seg) == 0:
            continue
        notes.line(f"{pad}{name} (frames {off}-{off + len(seg) - 1}, "
                   f"n = {len(seg)}): range {seg.min():.4g} to {seg.max():.4g}"
                   f"{u}, median {np.median(seg):.4g}, mean {seg.mean():.4g}")
    if len(pre) and len(post):
        notes.line(f"{pad}median shift across the injection: "
                   f"{np.median(post) - np.median(pre):+.4g}{u}")
        notes.line(f"{pad}one-frame change at the injection "
                   f"(frame {injection - 1} -> {injection}): "
                   f"{y[injection] - y[injection - 1]:+.4g}{u}")
        overlap = (post.min() <= pre.max()) and (pre.min() <= post.max())
        notes.line(f"{pad}the two phases {'overlap' if overlap else 'are disjoint'}"
                   f" in range")
    d = np.diff(y)
    if len(d):
        for name, lo, hi in (("pre-injection", 0, max(injection - 1, 0)),
                             ("post-injection", max(injection, 0), len(d))):
            if hi <= lo:
                continue
            seg = d[lo:hi]
            j = int(np.argmax(np.abs(seg)))
            notes.line(f"{pad}largest single-frame step {name}: "
                       f"{seg[j]:+.4g}{u} at frame {lo + j + 1}")


def fig_live_scores(pert: str) -> None:
    variant = LIVE_VARIANTS[pert]
    att = TEST_DIR / "attention" / "live_pert" / pert
    frames_npz = TEST_DIR / "live_pert_frames" / f"run_{pert}_live_pert_{variant}.npz"

    injection = int(np.argmax(np.load(frames_npz)["is_perturbed"]))

    # Mahalanobis-GMM from the cached ATOMs profiles, scored with the SAME
    # K=10 uniform-shrinkage model as every other thesis figure (the original
    # run_online_analysis figures used an older GMM fit).
    means, covs, _, _ = load_gmm()
    s_mahal = gmm_min_mahalanobis(
        np.load(att / f"live_pert_profiles_{variant}_2.npy").astype(np.float64),
        means, covs)
    c_mahal = gmm_min_mahalanobis(
        np.load(att / f"live_pert_profiles_{variant}_clean_2.npy").astype(np.float64),
        means, covs)

    mdx_path = att / f"live_pert_mdx_scores_{variant}.npy"
    if not mdx_path.exists():
        raise FileNotFoundError(
            f"{mdx_path} not found — run cache_live_mdx_scores.py "
            "(PCLA env, needs torch/timm) first.")
    s_mdx = np.load(mdx_path)
    c_mdx = np.load(att / f"live_pert_mdx_scores_{variant}_clean.npy")

    s_logits = np.load(att / f"live_pert_speed_logits_{variant}_2.npy")
    c_logits = np.load(att / f"live_pert_speed_logits_{variant}_clean_2.npy")
    s_peoc, c_peoc = _speed_entropy(s_logits), _speed_entropy(c_logits)

    fig, axs = plt.subplots(1, 3, figsize=(TEXT_WIDTH_IN, 2.2), sharex=True)
    panels = [("mahalanobis", s_mahal, c_mahal),
              ("mdx", s_mdx, c_mdx),
              ("peoc", s_peoc, c_peoc)]
    for ax, (fam, s_pert, s_clean) in zip(axs, panels):
        ax.axvline(injection, color="0.25", linestyle=(0, (1, 1.6)),
                   linewidth=0.9, zorder=1)
        ax.plot(np.arange(len(s_clean)), s_clean, color="0.55",
                linestyle=(0, (3, 1.8)), linewidth=0.9, zorder=2)
        ax.plot(np.arange(len(s_pert)), s_pert, color=METRIC_COLORS[fam],
                linewidth=1.4, zorder=3)
        ax.set_ylabel(METRIC_AXIS[fam])
        ax.margins(x=0.02)
        # Start every panel at zero. The three scores differ by two orders of
        # magnitude, and on a truncated axis a trace that wanders across its own
        # whole range looks no more variable than one that barely moves. How far
        # a score travels relative to its own level is the point of these panels.
        ax.set_ylim(bottom=0)

    fig.supxlabel("Frame index")
    save_figure(fig, OUT_DIR, f"live_scores_{pert}")

    notes = FigureNotes(
        f"live_scores_{pert}",
        title=f"Change-point detection, {PERT_LABELS[pert]}: Mahalanobis-GMM, "
              f"MDX and PEOC over one live run",
        source=str(att) + f" (variant {variant})")
    notes.line(f"    One recorded trajectory. {PERT_LABELS[pert]} is injected at "
               f"frame {injection} and stays on for the rest of the run.")
    notes.line(f"    Coloured trace: the perturbed run. Grey dashed trace: the "
               f"same frames as the agent would have seen them unperturbed (the "
               f"clean twin), so a move both traces make is the scene changing "
               f"and not the perturbation.")
    notes.line(f"    Mahalanobis-GMM uses the same K = {SELECTED_K} model as "
               "every other figure in the chapter.")
    notes.line("    A case study, not a statistical evaluation: one trajectory "
               "per perturbation.")
    notes.line("    No rise/plateau/shape descriptors: they are defined for a "
               "smooth sweep and would merge a noisy 100-frame trace into one "
               "segment. The phase blocks and the largest single-frame steps "
               "carry the shape instead.")
    notes.value("frames, perturbed run", len(s_mahal))
    notes.value("frames, clean twin", len(c_mahal))
    notes.value("injection frame", injection,
                note="first frame with is_perturbed set")

    for title, fam, s_p, s_c, unit in (
            ("Mahalanobis distance (left panel)", "mahalanobis",
             s_mahal, c_mahal, ""),
            ("MDX distance (middle panel)", "mdx", s_mdx, c_mdx, ""),
            ("PEOC entropy (right panel)", "peoc", s_peoc, c_peoc, "nats")):
        notes.section(title)
        _note_live_trace(notes, "perturbed", s_p, injection, unit)
        _note_live_trace(notes, "clean twin", s_c, injection, unit)
        pre_p = s_p[:injection]
        post_p, post_c = s_p[injection:], s_c[injection:injection + len(s_c)]
        n = min(len(post_p), len(post_c))
        if n:
            notes.value("perturbed minus clean twin, post-injection median",
                        float(np.median(post_p[:n]) - np.median(post_c[:n])))
            notes.value("frames where the perturbed trace is above its twin",
                        f"{int((post_p[:n] > post_c[:n]).sum())} of {n}",
                        note="post-injection only")
        if len(pre_p) and len(post_p):
            notes.value("post-injection minimum against the pre-injection maximum",
                        f"{post_p.min():.4g} vs {pre_p.max():.4g}",
                        note="the perturbed trace leaves its own pre-injection "
                             "band for good"
                        if post_p.min() > pre_p.max() else
                        "the perturbed trace returns into its own pre-injection band")
        notes.curve("perturbed, whole run", np.arange(len(s_p)), s_p,
                    unit=unit, rise=False, plateau=False, shape=False)
        notes.curve("clean twin, whole run", np.arange(len(s_c)), s_c,
                    unit=unit, rise=False, plateau=False, shape=False)

    notes.section("Predicted target speed")
    notes.line("    The quantity PEOC scores the entropy of. Softmax over the "
               "8 target-speed bins [0, 4, 8, 10, 13.9, 16, 17.8, 20] m/s, "
               "decoded to a scalar by the model's own two-hot decoding.")
    for label, lg in (("perturbed", s_logits), ("clean twin", c_logits)):
        spd = _predicted_speed(lg)
        _note_live_trace(notes, label, spd, injection, "m/s")
        p = np.exp(lg - lg.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        brake = p[:, 0]
        notes.line(f"        probability on the zero-speed (brake) bin: "
                   f"pre-injection median {np.median(brake[:injection]):.4g}, "
                   f"post-injection median {np.median(brake[injection:]):.4g}, "
                   f"post-injection max {brake[injection:].max():.4g}")
        notes.line(f"        frames with the brake bin above 0.999: "
                   f"{int((brake[injection:] > 0.999).sum())} of "
                   f"{len(brake) - injection} post-injection, "
                   f"{int((brake[:injection] > 0.999).sum())} of {injection} "
                   f"before")
    notes.write(OUT_DIR)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    global OUT_DIR
    ap = argparse.ArgumentParser(description="CARLA-chapter thesis figures.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where to write the figures and sidecars "
                         "(default: thesis_figures/, the committed copies)")
    OUT_DIR = ap.parse_args().out_dir
    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/13] GMM AUROC vs K ...")
    fig_gmm_auc_vs_k()
    print("[2/13] Baseline PCA (run vs GMM) ...")
    fig_pca_run_vs_gmm()
    print("[3/13] Score distributions (per perturbation) ...")
    fig_score_distributions()
    print("[4/13] AUROC per perturbation ...")
    fig_auroc_per_perturbation()
    print("[5/13] GMM vs single-Gaussian parity ...")
    fig_gmm_vs_single_parity()
    print("[6/13] Attention per cluster ...")
    fig_attention_per_cluster()
    print("[7/13] Attention per cluster with representative frames ...")
    fig_attention_per_cluster_frames()
    print("[8/13] Attention by cluster (single plot) ...")
    fig_attention_by_cluster()
    for i, pert in enumerate(LIVE_VARIANTS, start=9):
        print(f"[{i}/13] Live scores: {pert} ...")
        fig_live_scores(pert)
    print("[12/14] Val vs test mean AUROC over K ...")
    fig_val_test_auc_vs_k()
    print("[13/14] kNN k selection on val ...")
    fig_knn_k_selection()
    print("[14/14] AUROC per perturbation over K ...")
    fig_auroc_per_perturbation_vs_k()
    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
