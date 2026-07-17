#!/usr/bin/env python
"""
make_thesis_figures.py — CARLA-chapter thesis figures in the shared thesis style.

Recreates the key result figures of the alternative-split TFV6 experiment
(`EXPERIMENT_VARIANT = "alternative"`, mode 2, val-selected K=8) following
`documents/14_thesis_figure_style.md` / `thesis_style.py`, so the CARLA chapter
matches the ATOMs (Atari) chapter visually.

Figures written to `thesis_figures/` (each as .pdf + .png):

  1. gmm_auc_vs_K
       Test AUROC of every GMM detector vs cluster count K, plus their mean;
       the val-selected K=8 is marked.  (Thesis version of
       results_summary_alt/curve_meanGMM_vs_K_TFV6.png.)
  2. pca_baseline_run_vs_gmm
       1x2: the same baseline PCA, left coloured by collection run, right by
       GMM component (K=8) with component means.  (Combines
       pca/pca_baseline_by_run.png and pca/pca_baseline_clusters.png.)
  3. score_dist_mahal_gmm_vs_knn_single
       1x2: score distributions of the two strongest detectors on the
       labelled test set — Mahalanobis-GMM (left) vs plain single kNN over
       the full baseline (right; kNN-GMM is dominated by it, see figure 4).
       Scores are recomputed from the saved detector parameters; AUCs are
       checked against the stored results JSON.
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

Run with any env that has numpy / matplotlib / sklearn — both conda `PCLA`
(numpy 1.x) and `atoms3` (numpy 2.x) work; a shim below handles the
numpy-2-pickled object arrays in the alt-split npz files:

    python make_thesis_figures.py
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
SELECTED_K   = 8            # val-selected winner (max __val_auc_gmm_avg__)
RUN_DIR      = RESULTS_ROOT / f"{SELECTED_K} clusters" / "atoms_analysis_mode_2"
BASELINE_NPZ = ROOT / "data" / "TFV6" / "baseline_data_alt" / "baseline_2.npz"
FRAMES_DIR   = ROOT / "data" / "TFV6" / "baseline_data_alt" / "frames"
TEST_DIR     = ROOT / "data" / "TFV6" / "test_data_alt"
OUT_DIR      = ROOT / "thesis_figures"

# GMM detector families (canonical thesis_style keys) -> matcher on the raw
# summary.json / results_per_perturbation.json detector names.
GMM_FAMILIES = ["mahalanobis", "euclidean", "knn", "jsd"]

PERT_LABELS = {                       # display names for perturbation types
    "brightness_scale": "Brightness",
    "camera_loss":      "Camera loss",
    "gaussian_noise":   "Gaussian noise",
    "pgd":              "PGD",
}
PERT_ORDER = ["brightness_scale", "camera_loss", "gaussian_noise", "pgd"]


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
    for run_id, f in enumerate(sorted(FRAMES_DIR.glob("run_*.npz"))):
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

    run_cmap = plt.get_cmap("tab20")
    axs[0].scatter(proj[:, 0], proj[:, 1],
                   c=run_cmap(run_ids % 20), s=6, alpha=0.5, linewidths=0,
                   rasterized=True)

    K = len(weights)
    clus_cmap = plt.get_cmap("tab10")
    axs[1].scatter(proj[:, 0], proj[:, 1],
                   c=clus_cmap(labels % 10), s=6, alpha=0.5, linewidths=0,
                   rasterized=True)
    for k in range(K):
        axs[1].scatter(*cent[k], marker="*", s=120, facecolor=clus_cmap(k % 10),
                       edgecolor="black", linewidths=0.6, zorder=3)

    fig.supxlabel(f"PC1 ({var[0]:.1f} %)")
    fig.supylabel(f"PC2 ({var[1]:.1f} %)")

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=clus_cmap(k % 10), markeredgewidth=0,
                      label=str(k)) for k in range(K)]
    handles.append(Line2D([], [], marker="*", linestyle="none", markersize=9,
                          markerfacecolor="0.85", markeredgecolor="0.15",
                          markeredgewidth=0.6, label="cluster mean"))
    _fig_legend(fig, handles, ncol=K + 1)

    save_figure(fig, OUT_DIR, "pca_baseline_run_vs_gmm")


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
    labels = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)["label"].astype(int)

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

    clean_face, clean_edge = to_rgba("0.5", 0.45), "0.35"
    pert_face, pert_edge = to_rgba("#b71c1c", 0.35), "#b71c1c"

    # Different distance units per panel — independent x axes, shared y label.
    fig, axs = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))
    for ax, scores, res, xlabel, box in (
            (axs[0], s_mahal, res_mahal, METRIC_AXIS["mahalanobis"],
             f"AUROC = {res_mahal['auc']:.3f}"),
            (axs[1], s_knn, res_knn, METRIC_AXIS["knn"],
             f"AUROC = {res_knn['auc']:.3f}\n$k = {best_k}$")):
        bins = np.linspace(0.0, float(scores.max()) * 1.02, 48)
        ax.hist(scores[labels == 0], bins=bins, density=True,
                histtype="stepfilled", facecolor=clean_face,
                edgecolor=clean_edge, linewidth=0.9)
        ax.hist(scores[labels == 1], bins=bins, density=True,
                histtype="stepfilled", facecolor=pert_face,
                edgecolor=pert_edge, linewidth=0.9)
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0.0)
        _stats_box(ax, box)

    fig.supylabel("Density")

    handles = [
        Patch(facecolor=clean_face, edgecolor=clean_edge, linewidth=0.9,
              label="clean"),
        Patch(facecolor=pert_face, edgecolor=pert_edge, linewidth=0.9,
              label="perturbed"),
    ]
    _fig_legend(fig, handles, ncol=2)

    save_figure(fig, OUT_DIR, "score_dist_mahal_gmm_vs_knn_single")


# --------------------------------------------------------------------------- #
# Figure 4 — test AUROC per perturbation (GMM detectors + MDX / PEOC)
# --------------------------------------------------------------------------- #
def fig_auroc_per_perturbation() -> None:
    per_pert = load_per_perturbation()
    overall = load_overall()

    # "knn_single" = plain kNN over the full baseline (no clustering) — kNN is
    # the one detector for which the GMM pooling is conceptually questionable,
    # so its single variant is reported alongside.  Drawn outlined instead of
    # filled to mark it as the non-GMM variant.
    detectors = ["mahalanobis", "euclidean", "knn", "knn_single", "jsd", "mdx", "peoc"]
    hatches = {"mdx": "///", "peoc": "xxx"}   # bar analogue of the dashed lines

    def value(src: dict[str, float], fam: str) -> float:
        if fam in ("mdx", "peoc", "knn_single"):
            return src[fam] if fam in ("mdx", "peoc") else src["knn_single"]
        return src[f"{fam}_gmm"]

    groups = PERT_ORDER + ["overall"]
    group_labels = [PERT_LABELS[p] for p in PERT_ORDER] + ["Overall"]
    sources = [per_pert[p] for p in PERT_ORDER] + [overall]

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))

    n_det = len(detectors)
    width = 0.8 / n_det
    x0 = np.arange(len(groups))
    for j, fam in enumerate(detectors):
        xs = x0 + (j - (n_det - 1) / 2) * width
        ys = np.array([value(src, fam) for src in sources])
        if fam == "knn_single":
            ax.bar(xs, ys - 0.5, bottom=0.5, width=width * 0.92,
                   facecolor="none", edgecolor=METRIC_COLORS["knn"],
                   linewidth=1.0)
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
            handles.append(Patch(facecolor="none", edgecolor=METRIC_COLORS["knn"],
                                 linewidth=1.0,
                                 label=METRIC_LEGEND["knn"] + " (single)"))
        else:
            handles.append(Patch(facecolor=METRIC_COLORS[f], edgecolor="white",
                                 linewidth=0.0, hatch=hatches.get(f, None),
                                 label=METRIC_LEGEND[f]))
    _fig_legend(fig, handles, ncol=4)

    save_figure(fig, OUT_DIR, "auroc_per_perturbation_gmm")


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


# --------------------------------------------------------------------------- #
# Figures 6-8 — per-cluster ATOMs attention profiles
# --------------------------------------------------------------------------- #
CLUSTER_CMAP = plt.get_cmap("tab10")   # cluster k -> tab10(k), as in figure 2


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


def _cluster_bars(ax, series, labels, k, order_asc, with_whiskers=True):
    """Horizontal mean-attention bars (min-max whiskers) for one cluster."""
    rows = series[labels == k]
    mean, lo, hi = rows.mean(axis=0), rows.min(axis=0), rows.max(axis=0)
    vals = mean[order_asc]
    xerr = np.vstack([np.clip(vals - lo[order_asc], 0, None),
                      np.clip(hi[order_asc] - vals, 0, None)]) if with_whiskers else None
    ax.barh(np.arange(len(order_asc)), vals, height=0.72,
            color=CLUSTER_CMAP(k % 10), linewidth=0,
            xerr=xerr, error_kw=dict(ecolor="0.35", elinewidth=0.8,
                                     capsize=2.0, capthick=0.8))
    ax.text(0.97, 0.05, f"Cluster {k} (n = {(labels == k).sum()})",
            transform=ax.transAxes, fontsize=8.5, color=MUTED,
            ha="right", va="bottom")
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(axis="y", visible=False)


def fig_attention_per_cluster() -> None:
    series, labels, names, K, _, order_desc = cluster_attention_stats()
    order_asc = order_desc[::-1]          # barh: largest class ends up on top

    fig, axs = plt.subplots(4, 2, figsize=(TEXT_WIDTH_IN, 6.9),
                            sharex=True, sharey=True)
    for k, ax in enumerate(axs.ravel()):
        _cluster_bars(ax, series, labels, k, order_asc)
    axs[0, 0].set_yticks(np.arange(len(order_asc)))
    axs[0, 0].set_yticklabels([names[j] for j in order_asc])

    fig.supxlabel("Normalized attention")
    save_figure(fig, OUT_DIR, "attention_per_cluster")


def representative_frames(series, labels, K) -> dict[int, np.ndarray]:
    """Per cluster: the baseline frame closest (L2) to the cluster mean —
    same rule as run_analysis.py step 6b.5.  Loads only the two npz members
    needed (frame counts, then the one wide_rgb array per hit)."""
    files = sorted(FRAMES_DIR.glob("run_*.npz"))
    counts = np.array([np.load(f)["frame_idx"].shape[0] for f in files])
    bounds = np.concatenate([[0], np.cumsum(counts)])
    imgs = {}
    for k in range(K):
        mask = labels == k
        mean = series[mask].mean(axis=0)
        gidx = int(np.where(mask)[0][
            np.argmin(np.linalg.norm(series[mask] - mean, axis=1))])
        fi = int(np.searchsorted(bounds, gidx, side="right") - 1)
        img = np.load(files[fi])["wide_rgb"][gidx - bounds[fi]]   # [3, H, W]
        imgs[k] = np.transpose(img, (1, 2, 0))
    return imgs


def fig_attention_per_cluster_frames() -> None:
    series, labels, names, K, _, order_desc = cluster_attention_stats()
    order_asc = order_desc[::-1]
    imgs = representative_frames(series, labels, K)

    # 4 card rows x 2 columns; each card = frame strip (6:1) above its bars.
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 8.1))
    gs = fig.add_gridspec(8, 2, height_ratios=[1.0, 2.9] * 4)

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
        if r < 3:
            ax_bar.tick_params(labelbottom=False)
        bar_axs.append(ax_bar)

    fig.supxlabel("Normalized attention")
    save_figure(fig, OUT_DIR, "attention_per_cluster_frames")


def fig_attention_by_cluster() -> None:
    _, labels, names, K, cluster_mean, order_desc = cluster_attention_stats()

    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.9))
    x0 = np.arange(len(order_desc))
    width = 0.8 / K
    for k in range(K):
        ax.bar(x0 + (k - (K - 1) / 2) * width, cluster_mean[k][order_desc],
               width=width * 0.92, color=CLUSTER_CMAP(k % 10), linewidth=0)

    ax.set_xticks(x0)
    ax.set_xticklabels([names[j] for j in order_desc], rotation=30, ha="right")
    ax.set_ylabel("Normalized attention")
    ax.grid(axis="x", visible=False)

    handles = [Patch(facecolor=CLUSTER_CMAP(k % 10),
                     label=f"Cluster {k} (n = {(labels == k).sum()})")
               for k in range(K)]
    _fig_legend(fig, handles, ncol=4)

    save_figure(fig, OUT_DIR, "attention_by_cluster")


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


def fig_live_scores(pert: str) -> None:
    variant = LIVE_VARIANTS[pert]
    att = TEST_DIR / "attention" / "live_pert" / pert
    frames_npz = TEST_DIR / "live_pert_frames" / f"run_{pert}_live_pert_{variant}.npz"

    injection = int(np.argmax(np.load(frames_npz)["is_perturbed"]))

    # Mahalanobis-GMM from the cached ATOMs profiles, scored with the SAME
    # K=8 uniform-shrinkage model as every other thesis figure (the original
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

    s_peoc = _speed_entropy(np.load(att / f"live_pert_speed_logits_{variant}_2.npy"))
    c_peoc = _speed_entropy(np.load(att / f"live_pert_speed_logits_{variant}_clean_2.npy"))

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

    fig.supxlabel("Frame index")
    save_figure(fig, OUT_DIR, f"live_scores_{pert}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/11] GMM AUROC vs K ...")
    fig_gmm_auc_vs_k()
    print("[2/11] Baseline PCA (run vs GMM) ...")
    fig_pca_run_vs_gmm()
    print("[3/11] Score distributions (Mahalanobis-GMM vs single kNN) ...")
    fig_score_distributions()
    print("[4/11] AUROC per perturbation ...")
    fig_auroc_per_perturbation()
    print("[5/11] GMM vs single-Gaussian parity ...")
    fig_gmm_vs_single_parity()
    print("[6/11] Attention per cluster ...")
    fig_attention_per_cluster()
    print("[7/11] Attention per cluster with representative frames ...")
    fig_attention_per_cluster_frames()
    print("[8/11] Attention by cluster (single plot) ...")
    fig_attention_by_cluster()
    for i, pert in enumerate(LIVE_VARIANTS, start=9):
        print(f"[{i}/11] Live scores: {pert} ...")
        fig_live_scores(pert)
    print(f"\nDone -> {OUT_DIR}")


if __name__ == "__main__":
    main()
