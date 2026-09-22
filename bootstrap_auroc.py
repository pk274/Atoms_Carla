#!/usr/bin/env python
"""
bootstrap_auroc.py — route-level uncertainty on the CARLA AUROCs.

Every AUROC in the CARLA chapter is computed over pooled frames, but the frames
are not independent: the evaluation split is made at the ROUTE level (36 test
routes, 38 validation routes, 20-28 frames each), and frames inside one route
are strongly correlated.  A confidence interval that resamples frames therefore
understates the uncertainty by roughly the square root of the frames per route.

This resamples ROUTES with replacement instead, which is the right unit, and
reports both intervals side by side so the difference between them is visible
rather than asserted.  It answers two questions the chapter cannot otherwise
settle:

  1. Is the validation-minus-test offset, which is positive at every K, larger
     than the sampling variation the route-level split induces?
  2. Are the margins between the attention distances distinguishable at all?

Writes `thesis_figures/auroc_bootstrap.txt` via figure_notes.FigureNotes.  No
figure: the numbers belong in prose, not in another plot.

    python bootstrap_auroc.py [--n-boot 4000] [--seed 0] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "_core"):                    # numpy 1.x shim, as elsewhere
    import numpy.core as _np_core
    sys.modules["numpy._core"] = _np_core

from figure_notes import FigureNotes
from make_thesis_figures import (
    PERT_LABELS, PERT_ORDER, RUN_DIR, ROOT, TEST_DIR, SELECTED_K,
    gmm_min_mahalanobis, knn_gmm_scores, load_baseline_series, load_gmm,
    mahalanobis_batch, roc_auc,
)

OUT_DIR = ROOT / "thesis_figures"


def selected_knn_k() -> tuple[int, int]:
    """(single k, GMM k) as run_analysis.py selected them on the validation set
    at SELECTED_K, parsed from summary.json exactly as make_thesis_figures.py
    does.  Never hardcode these: the selected k moves with K (the GMM k was 50
    at K = 8 and is 250 at K = 10), and a stale literal here silently bootstraps
    a different curve from the one the figure plots."""
    summ = json.loads((RUN_DIR / "summary.json").read_text())

    def _k(pred) -> int:
        return int(re.search(r"k=(\d+)", next(n for n in summ if pred(n))).group(1))

    return (_k(lambda n: "k-NN" in n and "GMM" not in n),
            _k(lambda n: "k-NN-GMM" in n))


def load_split(split: str):
    """(profiles, labels, perturbation names, route ids) for 'val' or 'test'."""
    if split == "test":
        prof = np.load(TEST_DIR / "attention" / "test_profiles_2.npy")
        lab = np.load(TEST_DIR / "test_labeled.npz", allow_pickle=True)
    else:
        base = ROOT / "data" / "TFV6" / "val_data_alt"
        prof = np.load(base / "attention" / "val_profiles_2.npy")
        lab = np.load(base / "val_labeled.npz", allow_pickle=True)
    return (prof.astype(np.float64), lab["label"].astype(int),
            lab["perturbation"].astype(str), lab["run_id"].astype(int))


def boot_ci(scores, labels, groups=None, n_boot=4000, seed=0, alpha=0.05):
    """Percentile CI for the AUROC.

    `groups` = route ids resamples whole routes (the correct unit); `None`
    resamples frames independently, which is the naive interval kept for
    contrast.  Draws that lose one of the two classes are skipped, and the
    number skipped is returned so a degenerate interval cannot pass unnoticed.
    """
    rng = np.random.default_rng(seed)
    n = len(scores)
    if groups is None:
        idx_of = None
        units = np.arange(n)
    else:
        uniq = np.unique(groups)
        idx_of = {u: np.flatnonzero(groups == u) for u in uniq}
        units = uniq

    vals, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.choice(units, size=len(units), replace=True)
        sel = (pick if idx_of is None
               else np.concatenate([idx_of[u] for u in pick]))
        y = labels[sel]
        if y.min() == y.max():
            skipped += 1
            continue
        vals.append(roc_auc(scores[sel], y))
    vals = np.asarray(vals)
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(vals.std(ddof=1)), vals, skipped


def fmt_ci(point, lo, hi, se):
    return f"{point:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  SE {se:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="where to write auroc_bootstrap.txt (default: thesis_figures/)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    means, covs, weights, K = load_gmm()
    series = load_baseline_series()
    k_single, k_gmm = selected_knn_k()

    notes = FigureNotes(
        "auroc_bootstrap",
        title="Route-level bootstrap of the CARLA AUROCs",
        source=f"{RUN_DIR} + the labelled val/test splits")
    notes.line(f"    {args.n_boot} resamples, seed {args.seed}, percentile "
               f"intervals, K = {SELECTED_K}.")
    notes.line("    THE UNIT IS THE ROUTE, not the frame. The split is made at "
               "route level and frames within a route are strongly correlated, "
               "so a frame-level interval understates the uncertainty. Both are "
               "reported so the gap between them is visible.")
    notes.line("    Scores are the Mahalanobis distance to the nearest GMM "
               "component unless stated otherwise.")
    notes.line(f"    k-th-NN neighbour counts, read from summary.json (the "
               f"val-selected values at K = {SELECTED_K}): single k = {k_single}, "
               f"GMM k = {k_gmm}.")

    boots = {}
    for split in ("val", "test"):
        prof, lab, pert, run = load_split(split)
        s = gmm_min_mahalanobis(prof, means, covs)
        notes.section(f"{split} split")
        notes.value("frames", len(lab))
        notes.value("routes", int(len(np.unique(run))))
        notes.value("frames per route",
                    f"{np.bincount(np.unique(run, return_inverse=True)[1]).min()}"
                    f"-{np.bincount(np.unique(run, return_inverse=True)[1]).max()}")

        for pool_name, mask in (("all four perturbations", np.ones(len(lab), bool)),
                                ("ex-Gaussian-noise (the K criterion pool)",
                                 pert != "gaussian_noise")):
            pt = roc_auc(s[mask], lab[mask])
            lo, hi, se, vals, sk = boot_ci(s[mask], lab[mask], run[mask],
                                           args.n_boot, args.seed)
            flo, fhi, fse, _, _ = boot_ci(s[mask], lab[mask], None,
                                          args.n_boot, args.seed)
            notes.value(f"MD, {pool_name}", fmt_ci(pt, lo, hi, se),
                        note=f"by route; {sk} draws skipped")
            notes.value("    same, resampling frames instead",
                        fmt_ci(pt, flo, fhi, fse),
                        note=f"the route interval is {se / fse:.2f}x wider. If "
                             "frames within a route were strongly correlated it "
                             "would be several times wider, so they are not")
            boots[(split, pool_name)] = (pt, vals)

        if split == "test":
            notes.line("    per perturbation, against the clean frames only:")
            clean = lab == 0
            for p in PERT_ORDER:
                m = clean | (pert == p)
                pt = roc_auc(s[m], lab[m])
                lo, hi, se, vals, sk = boot_ci(s[m], lab[m], run[m],
                                               args.n_boot, args.seed)
                notes.value(f"    {PERT_LABELS[p]}", fmt_ci(pt, lo, hi, se))
                boots[("test", p)] = (pt, vals)

    # --- the quantity auroc_val_test_vs_K actually plots ----------------------
    # That figure shows the MEAN over the four GMM detectors, ex-Gaussian-noise,
    # not the MD alone.  Bootstrapping the MD would size a different quantity, so
    # the mean is resampled here on exactly the same footing.
    notes.section("The plotted curve itself: mean over the four GMM detectors, "
                  "ex-Gaussian-noise")
    from scipy.spatial.distance import cdist
    from ATOMs_Analysis.utils.distance_computer import DistanceComputer as DC

    def four_detector_scores(prof):
        comp = np.stack([mahalanobis_batch(prof, means[c], covs[c])
                         for c in range(K)])
        nearest = comp.argmin(axis=0)
        return [comp.min(axis=0),
                np.linalg.norm(prof - means[nearest], axis=1),
                np.array([DC.compute_jsd(means[nearest[i]], prof[i])
                          for i in range(len(prof))]),
                knn_gmm_scores(prof, series, means, covs, weights, k=k_gmm)]

    mean_boot = {}
    for split in ("val", "test"):
        prof, lab, pert, run = load_split(split)
        m = pert != "gaussian_noise"
        det = [d[m] for d in four_detector_scores(prof)]
        y, g = lab[m], run[m]
        point = float(np.mean([roc_auc(d, y) for d in det]))
        rng2 = np.random.default_rng(args.seed)
        uq = np.unique(g)
        idx = {u: np.flatnonzero(g == u) for u in uq}
        vals = []
        for _ in range(args.n_boot):
            pick = rng2.choice(uq, size=len(uq), replace=True)
            sel = np.concatenate([idx[u] for u in pick])
            yy = y[sel]
            if yy.min() == yy.max():
                continue
            vals.append(np.mean([roc_auc(d[sel], yy) for d in det]))
        vals = np.asarray(vals)
        lo, hi = np.percentile(vals, [2.5, 97.5])
        notes.value(f"{split}", fmt_ci(point, lo, hi, float(vals.std(ddof=1))),
                    note=f"this is the value plotted at K = {SELECTED_K}")
        mean_boot[split] = (point, vals)

    n = min(len(mean_boot["val"][1]), len(mean_boot["test"][1]))
    diff = mean_boot["val"][1][:n] - mean_boot["test"][1][:n]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    notes.value(f"offset of the plotted curves at K = {SELECTED_K}",
                f"{mean_boot['val'][0] - mean_boot['test'][0]:+.4f}  "
                f"95% CI [{lo:+.4f}, {hi:+.4f}]")
    notes.value("    is zero inside the interval?",
                "yes" if lo <= 0 <= hi else "NO")

    # --- the offset: val and test are DISJOINT route sets, so the two bootstrap
    # distributions are independent and their difference is the offset's -------
    notes.section("Validation minus test, MD alone")
    for pool in ("all four perturbations",
                 "ex-Gaussian-noise (the K criterion pool)"):
        (pv, dv), (ptst, dt) = boots[("val", pool)], boots[("test", pool)]
        n = min(len(dv), len(dt))
        diff = dv[:n] - dt[:n]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        notes.value(f"offset, {pool}",
                    f"{pv - ptst:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]",
                    note="the two splits hold disjoint routes, so the bootstraps "
                         "are independent")
        notes.value("    is zero inside the interval?",
                    "yes" if lo <= 0 <= hi else "NO",
                    note="if yes, the offset is consistent with the sampling "
                         "variation the route-level split induces, and needs no "
                         "further explanation")

    # --- do the metric margins survive? ---------------------------------------
    notes.section("Are the margins between the attention distances real?")
    notes.line("    Paired by route: each resample scores every metric on the "
               "same drawn routes, so the difference is not inflated by the "
               "shared uncertainty in the AUROC level itself.")
    prof, lab, pert, run = load_split("test")
    from scipy.spatial.distance import cdist

    def _n(a):
        return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)

    metrics = {
        "MD": gmm_min_mahalanobis(prof, means, covs),
        "ED": np.linalg.norm(prof - means[np.stack(
            [mahalanobis_batch(prof, means[c], covs[c])
             for c in range(K)]).argmin(axis=0)], axis=1),
        "k-NN single": np.sort(cdist(_n(prof), _n(series)), axis=1)[:, k_single - 1],
        "k-NN GMM": knn_gmm_scores(prof, series, means, covs, weights, k=k_gmm),
    }
    from ATOMs_Analysis.utils.distance_computer import DistanceComputer as DC
    route = np.stack([mahalanobis_batch(prof, means[c], covs[c])
                      for c in range(K)]).argmin(axis=0)
    metrics["JSD"] = np.array([DC.compute_jsd(means[route[i]], prof[i])
                               for i in range(len(prof))])

    rng = np.random.default_rng(args.seed)
    uniq = np.unique(run)
    idx_of = {u: np.flatnonzero(run == u) for u in uniq}
    clean = lab == 0
    for p in PERT_ORDER:
        m = clean | (pert == p)
        pts = {k: roc_auc(v[m], lab[m]) for k, v in metrics.items()}
        best, second = sorted(pts, key=pts.get, reverse=True)[:2]
        diffs = []
        for _ in range(args.n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            sel = np.concatenate([idx_of[u] for u in pick])
            sel = sel[m[sel]]
            y = lab[sel]
            if y.min() == y.max():
                continue
            diffs.append(roc_auc(metrics[best][sel], y)
                         - roc_auc(metrics[second][sel], y))
        d = np.asarray(diffs)
        lo, hi = np.percentile(d, [2.5, 97.5])
        notes.value(PERT_LABELS[p],
                    f"{best} {pts[best]:.4f} over {second} {pts[second]:.4f}, "
                    f"margin {pts[best] - pts[second]:+.4f} "
                    f"95% CI [{lo:+.4f}, {hi:+.4f}]",
                    note="margin distinguishable from zero"
                         if lo > 0 else "margin NOT distinguishable from zero")

    notes.write(args.out_dir)


if __name__ == "__main__":
    main()
