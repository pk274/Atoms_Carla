#!/usr/bin/env python
"""
summarize_results.py — compile & summarize OOD-detection results across the
cluster-count sweep for both agents (TFV6, WOR).

The cluster sweep lives under::

    data/<AGENT>/results/<K> clusters/atoms_analysis_mode_<1|2>/
    data/<AGENT>/results/atoms_analysis[_mode_<1|2>]/     (top-level mirrors)

Each run directory holds a ``summary.json`` (AUC + Youden-J for 12 detectors),
a ``results_per_perturbation.json`` (per-perturbation AUC), and a family of
``results_knn_k*.json`` files behind the "best-k" picks.

Key structural fact this script leans on: for a fixed (agent, mode) the
non-GMM detectors are *identical* across every cluster folder — only the five
GMM-based detectors move with K. So the "which cluster count" question only
concerns the GMM variants.

The script:
  * discovers every run, de-duplicating the top-level mirror folders;
  * builds a detector x K AUC matrix per (agent, mode);
  * works out the best K per GMM detector and ranks distances by robustness;
  * breaks results down per perturbation (best achievable over K);
  * extracts the k-NN k-sweep;
  * inventories the qualitative live-perturbation plots (no AUC there);
  * writes a Markdown report + PNG heatmaps.

Usage::

    python summarize_results.py                 # defaults: --data-root data --out results_summary
    python summarize_results.py --data-root data --out results_summary

Run it with an env that has matplotlib (e.g. the conda PCLA / atoms3 env).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------- #
# Canonical detector naming
# --------------------------------------------------------------------------- #
# Display order; GMM-dependent detectors are flagged in GMM_DETECTORS below.
DETECTOR_ORDER = [
    "Mahalanobis",
    "Mahalanobis-GMM",
    "Euclidean",
    "Euclidean-GMM",
    "JSD",
    "JSD-GMM",
    "Wasserstein",
    "Wasserstein-GMM",
    "k-NN",
    "k-NN-GMM",
    "MDX",
    "PEOC/Entropy",
]
GMM_DETECTORS = {d for d in DETECTOR_ORDER if d.endswith("-GMM")}


def canon_detector(name: str) -> str:
    """Map a raw detector key (from summary.json or per_perturbation.json) to a
    canonical, K-agnostic name."""
    n = name.lower()
    is_gmm = "gmm" in n
    if "mdx" in n:
        return "MDX"
    if "peoc" in n or "entropy" in n:
        return "PEOC/Entropy"
    if "k-nn" in n or "knn" in n:
        return "k-NN-GMM" if is_gmm else "k-NN"
    if "mahalanobis" in n:
        return "Mahalanobis-GMM" if is_gmm else "Mahalanobis"
    if "euclidean" in n:
        return "Euclidean-GMM" if is_gmm else "Euclidean"
    if "jsd" in n:
        return "JSD-GMM" if is_gmm else "JSD"
    if "wasserstein" in n:
        return "Wasserstein-GMM" if is_gmm else "Wasserstein"
    return name  # unknown — keep raw


def extract_best_k(name: str) -> int | None:
    """Pull the chosen k from a k-NN detector key like 'ATOMs-k-NN (k=10, best)'."""
    m = re.search(r"k=(\d+)", name)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Run:
    agent: str
    mode: str          # "1", "2", or "?"
    K: int             # GMM cluster count
    path: Path
    auc: dict          # canonical detector -> auc
    youden: dict       # canonical detector -> youden_j
    knn_best_k: dict   # {"k-NN": k, "k-NN-GMM": k}
    snapshot: bool = True   # True = deliberate "<K> clusters/" folder; False = scratch
    per_pert: dict = field(default_factory=dict)   # pert -> {canon detector: auc}
    knn_sweep: dict = field(default_factory=dict)  # {"plain": {k: auc}, "gmm": {k: auc}}


def _gmm_k_from_keys(summary: dict) -> int | None:
    for key in summary:
        m = re.search(r"K=(\d+)", key)
        if m:
            return int(m.group(1))
    return None


def load_summary(path: Path) -> tuple[dict, dict, dict]:
    raw = json.loads(path.read_text())
    auc, youden, knn_best = {}, {}, {}
    for key, val in raw.items():
        c = canon_detector(key)
        auc[c] = val["auc"]
        youden[c] = val.get("youden_j")
        if c in ("k-NN", "k-NN-GMM"):
            bk = extract_best_k(key)
            if bk is not None:
                knn_best[c] = bk
    return auc, youden, knn_best


def load_per_pert(path: Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for pert, entries in raw.items():
        d = {}
        for entry in entries:  # list of single-key dicts
            for k, v in entry.items():
                name = k.split("|")[0].strip()
                d[canon_detector(name)] = v
        out[pert] = d
    return out


def load_knn_sweep(run_dir: Path) -> dict:
    sweep = {"plain": {}, "gmm": {}}
    for f in run_dir.glob("results_knn_*.json"):
        m = re.search(r"results_knn_(gmm_)?k(\d+)\.json$", f.name)
        if not m:
            continue
        kind = "gmm" if m.group(1) else "plain"
        k = int(m.group(2))
        try:
            sweep[kind][k] = json.loads(f.read_text())["auc"]
        except (KeyError, json.JSONDecodeError):
            pass
    return sweep


def _nongmm_fingerprint(auc: dict) -> tuple:
    """K-invariant signature of a run: its non-GMM detector AUCs (sentinel -1 for
    missing). Constant across cluster counts within a (agent, mode)."""
    return tuple(round(auc.get(d, -1.0), 6) for d in DETECTOR_ORDER
                 if d not in GMM_DETECTORS)


def discover_runs(data_root: Path) -> list[Run]:
    """Find every summary.json. Named '<K> clusters/' folders are authoritative
    snapshots; the bare 'atoms_analysis*' folders are last-run scratch dirs whose
    K reflects whatever was run most recently. Bare folders with no mode suffix
    get their mode inferred from the K-invariant fingerprint. Runs are then
    de-duplicated by (agent, mode, K), preferring snapshot folders."""
    cands: list[Run] = []
    for summ in sorted(data_root.glob("*/results/**/summary.json")):
        agent = summ.relative_to(data_root).parts[0]
        mm = re.search(r"mode_(\d)", str(summ))
        mode = mm.group(1) if mm else "?"
        run_dir = summ.parent
        auc, youden, knn_best = load_summary(summ)
        K = _gmm_k_from_keys(json.loads(summ.read_text())) or 0
        named = "clusters" in str(run_dir).lower()
        cands.append(Run(
            agent=agent, mode=mode, K=K, path=run_dir,
            auc=auc, youden=youden, knn_best_k=knn_best, snapshot=named,
            per_pert=load_per_pert(run_dir / "results_per_perturbation.json"),
            knn_sweep=load_knn_sweep(run_dir),
        ))

    # infer mode for bare folders by matching the K-invariant fingerprint
    fps: dict[tuple, str] = {}
    for r in cands:
        if r.mode in ("1", "2"):
            fps.setdefault((r.agent, r.mode), _nongmm_fingerprint(r.auc))
    for r in cands:
        if r.mode == "?":
            fp = _nongmm_fingerprint(r.auc)
            for (a, m), known in fps.items():
                if a == r.agent and fp == known:
                    r.mode = m
                    break

    # Drop bare 'atoms_analysis' folders whose mode could not be resolved: these
    # are independent scratch runs (fingerprint matches no labelled mode) and only
    # add an ambiguous one-column group.
    dropped = [r for r in cands if r.mode == "?"]
    if dropped:
        print("Skipped mode-ambiguous scratch runs: "
              + ", ".join(f"{r.agent}/{r.path.name}(K={r.K})" for r in dropped))
    cands = [r for r in cands if r.mode != "?"]

    # de-dup by (agent, mode, K), preferring deliberate snapshot folders
    best: dict[tuple, Run] = {}
    for r in cands:
        key = (r.agent, r.mode, r.K)
        if key not in best or (r.snapshot and not best[key].snapshot):
            best[key] = r
    return list(best.values())


def inventory_live(data_root: Path) -> dict:
    """Map agent -> {perturbation -> sorted list of detector plot stems}.
    Live runs hold only score-distribution PNGs (no AUC)."""
    live: dict[str, dict] = defaultdict(lambda: defaultdict(set))
    for png in data_root.glob("*/results/*live*/**/*.png"):
        agent = png.relative_to(data_root).parts[0]
        stem = png.stem  # e.g. "mahalanobis_gmm_pgd"
        m = re.search(r"_(pgd|phantom_obstacle|brightness_scale|camera_loss|gaussian_noise)$", stem)
        if not m:
            continue
        pert = m.group(1)
        det = stem[: m.start()]
        live[agent][pert].add(det)
    return {a: {p: sorted(v) for p, v in sorted(d.items())} for a, d in live.items()}


# --------------------------------------------------------------------------- #
# Aggregation / analysis
# --------------------------------------------------------------------------- #
def group_by_agent_mode(runs: list[Run]) -> dict:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for r in runs:
        groups[(r.agent, r.mode)].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r.K)
    return dict(sorted(groups.items()))


def matrix_for_group(group_runs: list[Run]):
    """Return (Ks, detectors_present, matrix[det -> {K: auc}])."""
    Ks = sorted({r.K for r in group_runs})
    mat: dict[str, dict[int, float]] = defaultdict(dict)
    for r in group_runs:
        for det, v in r.auc.items():
            mat[det][r.K] = v
    dets = [d for d in DETECTOR_ORDER if d in mat]
    return Ks, dets, mat


def best_k_per_gmm(group_runs: list[Run]) -> dict:
    Ks, dets, mat = matrix_for_group(group_runs)
    out = {}
    for det in dets:
        if det not in GMM_DETECTORS:
            continue
        vals = mat[det]
        if not vals:
            continue
        bestK = max(vals, key=vals.get)
        out[det] = (bestK, vals[bestK])
    return out


def best_per_perturbation(group_runs: list[Run]):
    """For each (perturbation, canonical detector) return (best_auc, K_at_best)."""
    acc: dict[str, dict[str, tuple]] = defaultdict(dict)
    for r in group_runs:
        for pert, dd in r.per_pert.items():
            for det, v in dd.items():
                cur = acc[pert].get(det)
                if cur is None or v > cur[0]:
                    acc[pert][det] = (v, r.K)
    return acc


def mode_comparison_data(groups: dict) -> dict:
    """For each agent, extract the best AUC per (detector, mode).

    GMM detectors use their best-K AUC; non-GMM detectors are K-invariant.
    Returns {agent: {detector: {'1': auc, '2': auc}}}
    """
    by_agent: dict[str, dict] = defaultdict(lambda: defaultdict(dict))
    for (agent, mode), gr in groups.items():
        if mode not in ("1", "2"):
            continue
        Ks, dets, mat = matrix_for_group(gr)
        for det in dets:
            vals = mat[det]
            v = max(vals.values()) if det in GMM_DETECTORS else next(iter(vals.values()))
            by_agent[agent][det][mode] = v
    return {a: dict(dd) for a, dd in by_agent.items()}


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #
def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "—"


def safe_mode(mode: str) -> str:
    """Filename-safe mode token ('?' -> 'unk')."""
    s = re.sub(r"[^0-9A-Za-z]+", "", str(mode))
    return s or "unk"


def build_markdown(groups: dict, live: dict) -> str:
    L: list[str] = []
    w = L.append
    w("# OOD-Detection Results Summary")
    w("")
    w("_Auto-generated by `summarize_results.py`. Metric = ROC-AUC unless noted "
      "(1.0 perfect, 0.5 chance). Within each (agent, mode) the non-GMM detectors "
      "are constant across cluster counts; only the `-GMM` detectors depend on K._")
    w("")
    w("> **Provenance.** `<K> clusters/` folders are deliberate snapshots and are "
      "preferred. The bare `atoms_analysis*` folders are last-run scratch dirs — "
      "their K reflects whatever was run most recently, so re-run this script after "
      "new analyses. A bare folder only contributes a (agent, mode, K) point not "
      "already covered by a snapshot.")
    w("")

    # ---- 1. headline digest -------------------------------------------------
    w("## 1. Headline")
    w("")
    w("| Agent | Mode | Ks swept | Best detector | AUC | @K | Best feature-space (MDX) |")
    w("|-------|------|----------|---------------|-----|----|--------------------------|")
    for (agent, mode), gr in groups.items():
        Ks, dets, mat = matrix_for_group(gr)
        # best detector = max over all detectors and Ks
        best = (None, -1, None)
        for det in dets:
            for K, v in mat[det].items():
                if v > best[1]:
                    best = (det, v, K)
        mdx = mat.get("MDX", {})
        mdx_v = next(iter(mdx.values())) if mdx else None
        w(f"| {agent} | {mode} | {Ks} | **{best[0]}** | {fmt(best[1])} | "
          f"{best[2]} | {fmt(mdx_v)} |")
    w("")

    # ---- 2. per (agent, mode) full matrix ----------------------------------
    w("## 2. AUC matrix — detector × cluster count (K)")
    w("")
    for (agent, mode), gr in groups.items():
        Ks, dets, mat = matrix_for_group(gr)
        w(f"### {agent} — mode {mode}")
        w("")
        header = "| Detector | " + " | ".join(f"K={K}" for K in Ks) + " | K-dep? | best K |"
        sep = "|" + "---|" * (len(Ks) + 3)
        w(header)
        w(sep)
        for det in dets:
            row_vals = mat[det]
            is_gmm = det in GMM_DETECTORS
            best_cell = max(row_vals.values()) if row_vals else None
            cells = []
            for K in Ks:
                v = row_vals.get(K)
                if v is None:
                    cells.append("—")
                elif is_gmm and v == best_cell:
                    cells.append(f"**{fmt(v)}**")
                else:
                    cells.append(fmt(v))
            kdep = "yes" if is_gmm else "no"
            bestK = max(row_vals, key=row_vals.get) if (is_gmm and row_vals) else "—"
            w(f"| {det} | " + " | ".join(cells) + f" | {kdep} | {bestK} |")
        w("")

    # ---- 3. best cluster count ---------------------------------------------
    w("## 3. Which cluster count (K) works best?")
    w("")
    w("_Only the `-GMM` detectors depend on K. For each one, the K that maximizes "
      "AUC within that (agent, mode):_")
    w("")
    w("| Agent | Mode | Detector | best K | AUC | vs. its non-GMM twin |")
    w("|-------|------|----------|--------|-----|----------------------|")
    for (agent, mode), gr in groups.items():
        _, _, mat = matrix_for_group(gr)
        bk = best_k_per_gmm(gr)
        for det, (K, v) in bk.items():
            twin = det.replace("-GMM", "")
            twin_v = next(iter(mat.get(twin, {}).values()), None)
            delta = (v - twin_v) if twin_v is not None else None
            darrow = "" if delta is None else (f"  (+{delta:.4f})" if delta >= 0 else f"  ({delta:.4f})")
            w(f"| {agent} | {mode} | {det} | {K} | {fmt(v)} | {fmt(twin_v)}{darrow} |")
    w("")
    # aggregate: which K maximizes mean GMM AUC per (agent, mode)
    w("_Aggregate — K that maximizes the **mean** AUC over all five GMM detectors:_")
    w("")
    w("| Agent | Mode | best mean-K | mean GMM AUC at best K |")
    w("|-------|------|-------------|------------------------|")
    for (agent, mode), gr in groups.items():
        Ks, dets, mat = matrix_for_group(gr)
        gdets = [d for d in dets if d in GMM_DETECTORS]
        per_k = {}
        for K in Ks:
            vals = [mat[d][K] for d in gdets if K in mat[d]]
            if vals:
                per_k[K] = float(np.mean(vals))
        if per_k:
            bk = max(per_k, key=per_k.get)
            w(f"| {agent} | {mode} | {bk} | {fmt(per_k[bk])} |")
    w("")

    # ---- 4. distance robustness --------------------------------------------
    w("## 4. Which distance is most robust?")
    w("")
    w("_Per agent, averaged over modes. For GMM detectors the best-K value is used. "
      "`mean AUC` = average over all configs; `worst-pert AUC` = lowest AUC across "
      "perturbation types (worst case → robustness)._")
    w("")
    by_agent: dict[str, list] = defaultdict(list)
    for (agent, mode), gr in groups.items():
        by_agent[agent].append((mode, gr))
    for agent, items in by_agent.items():
        w(f"### {agent}")
        w("")
        # gather per-detector overall AUC (best-K for GMM) and worst-pert AUC
        overall: dict[str, list] = defaultdict(list)
        worstpert: dict[str, list] = defaultdict(list)
        for mode, gr in items:
            Ks, dets, mat = matrix_for_group(gr)
            bp = best_per_perturbation(gr)
            for det in dets:
                vals = mat[det]
                v = max(vals.values()) if det in GMM_DETECTORS else next(iter(vals.values()))
                overall[det].append(v)
                # worst perturbation for this detector (max over K already inside bp)
                pvals = [bp[p][det][0] for p in bp if det in bp[p]]
                if pvals:
                    worstpert[det].append(min(pvals))
        rows = []
        for det in DETECTOR_ORDER:
            if det not in overall:
                continue
            mo = float(np.mean(overall[det]))
            wp = float(np.mean(worstpert[det])) if worstpert[det] else None
            rows.append((det, mo, wp))
        rows.sort(key=lambda x: x[1], reverse=True)
        w("| Rank | Detector | mean AUC | worst-pert AUC |")
        w("|------|----------|----------|----------------|")
        for i, (det, mo, wp) in enumerate(rows, 1):
            w(f"| {i} | {det} | {fmt(mo)} | {fmt(wp)} |")
        w("")

    # ---- 5. per perturbation -----------------------------------------------
    w("## 5. Per-perturbation breakdown (best AUC achievable over K)")
    w("")
    for (agent, mode), gr in groups.items():
        bp = best_per_perturbation(gr)
        if not bp:
            continue
        perts = list(bp.keys())
        dets = [d for d in DETECTOR_ORDER if any(d in bp[p] for p in perts)]
        w(f"### {agent} — mode {mode}")
        w("")
        w("| Detector | " + " | ".join(perts) + " |")
        w("|" + "---|" * (len(perts) + 1))
        for det in dets:
            cells = []
            for p in perts:
                cell = bp[p].get(det)
                cells.append(fmt(cell[0]) if cell else "—")
            w(f"| {det} | " + " | ".join(cells) + " |")
        # winner per perturbation
        winners = []
        for p in perts:
            best = max(bp[p].items(), key=lambda kv: kv[1][0])
            winners.append(f"**{p}** → {best[0]} ({best[1][0]:.3f})")
        w("")
        w("_Best detector per perturbation: " + "; ".join(winners) + "._")
        w("")

    # ---- 6. k-NN k sweep ----------------------------------------------------
    w("## 6. k-NN neighbour-count (k) sweep")
    w("")
    w("_AUC vs. number of neighbours k (plain k-NN in ATOMs space). The chosen "
      "'best' k is what feeds the headline tables._")
    w("")
    for (agent, mode), gr in groups.items():
        # use the run whose folder is a named cluster run with a populated sweep
        sweep_run = next((r for r in gr if r.knn_sweep["plain"]), None)
        if sweep_run is None:
            continue
        ks = sorted(sweep_run.knn_sweep["plain"])
        chosen = sweep_run.knn_best_k.get("k-NN")
        w(f"**{agent} mode {mode}** (chosen k={chosen}): "
          + ", ".join(f"k={k}:{sweep_run.knn_sweep['plain'][k]:.3f}" for k in ks))
        w("")

    # ---- 7. live perturbation inventory ------------------------------------
    w("## 7. Live-perturbation runs (qualitative only)")
    w("")
    w("_These online runs produce per-detector score-distribution PNGs — there is "
      "**no AUC / labelled evaluation** here, so they are inventoried, not scored._")
    w("")
    if not live:
        w("_None found._")
    for agent, perts in live.items():
        w(f"### {agent}")
        for p, dets in perts.items():
            w(f"- **{p}**: {len(dets)} detector plots ({', '.join(dets)})")
        w("")

    # ---- 8. mode 1 vs mode 2 comparison ------------------------------------
    comp = mode_comparison_data(groups)
    any_both = any(
        any("1" in v and "2" in v for v in dmap.values())
        for dmap in comp.values()
    )
    if any_both:
        w("## 8. Mode 1 vs Mode 2 comparison")
        w("")
        w("_Mode 1 = **node-level** (LRP1 → filter top-K nodes by relevance → LRP2 per "
          "node; paper default). Mode 2 = **layer-level** (single FC→input map). "
          "For GMM detectors the best-K AUC is shown; non-GMM detectors are K-invariant._")
        w("")

        # --- 8a. per-detector comparison table per agent ---
        w("### 8a. Overall detector comparison (best AUC per mode)")
        w("")
        for agent in sorted(comp.keys()):
            dmap = comp[agent]
            has_both = any("1" in v and "2" in v for v in dmap.values())
            if not has_both:
                continue
            w(f"**{agent}**")
            w("")
            w("| Detector | Mode 1 | Mode 2 | Δ (2−1) | Better |")
            w("|----------|--------|--------|---------|--------|")
            mode2_wins = mode1_wins = ties = 0
            for det in DETECTOR_ORDER:
                if det not in dmap:
                    continue
                v = dmap[det]
                m1, m2 = v.get("1"), v.get("2")
                s1 = fmt(m1) if m1 is not None else "—"
                s2 = fmt(m2) if m2 is not None else "—"
                if m1 is not None and m2 is not None:
                    delta = m2 - m1
                    ds = (f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}")
                    if abs(delta) < 0.005:
                        better = "≈ tie"
                        ties += 1
                    elif delta > 0:
                        better = "**Mode 2**"
                        mode2_wins += 1
                    else:
                        better = "**Mode 1**"
                        mode1_wins += 1
                else:
                    ds = "—"
                    better = "—"
                w(f"| {det} | {s1} | {s2} | {ds} | {better} |")
            total = mode1_wins + mode2_wins + ties
            w("")
            if total:
                w(f"_Mode 2 wins {mode2_wins}/{total} detectors, "
                  f"Mode 1 wins {mode1_wins}/{total}, {ties} near-tie (|Δ|<0.005)._")
            w("")

        # --- 8b. per-perturbation mode comparison ---
        w("### 8b. Per-perturbation mode comparison (best AUC over K)")
        w("")
        for agent in sorted(comp.keys()):
            mode1_gr = next((gr for (a, m), gr in groups.items() if a == agent and m == "1"), None)
            mode2_gr = next((gr for (a, m), gr in groups.items() if a == agent and m == "2"), None)
            if mode1_gr is None or mode2_gr is None:
                continue
            bp1 = best_per_perturbation(mode1_gr)
            bp2 = best_per_perturbation(mode2_gr)
            perts = sorted(set(list(bp1.keys()) + list(bp2.keys())))
            if not perts:
                continue
            w(f"**{agent}**")
            w("")
            for p in perts:
                dets_here = [d for d in DETECTOR_ORDER
                             if d in bp1.get(p, {}) or d in bp2.get(p, {})]
                if not dets_here:
                    continue
                w(f"_Perturbation: `{p}`_")
                w("")
                w("| Detector | Mode 1 | Mode 2 | Δ (2−1) |")
                w("|----------|--------|--------|---------|")
                for det in dets_here:
                    v1 = bp1.get(p, {}).get(det)
                    v2 = bp2.get(p, {}).get(det)
                    s1 = fmt(v1[0]) if v1 else "—"
                    s2 = fmt(v2[0]) if v2 else "—"
                    if v1 and v2:
                        d = v2[0] - v1[0]
                        ds = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"
                    else:
                        ds = "—"
                    w(f"| {det} | {s1} | {s2} | {ds} |")
                w("")
            w("")

    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _heatmap(ax, data, row_labels, col_labels, title, vmin=0.40, vmax=0.75):
    arr = np.array([[np.nan if v is None else v for v in row] for row in data], float)
    im = ax.imshow(arr, aspect="auto", cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if not np.isnan(arr[i, j]):
                ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center",
                        fontsize=7, color="black")
    return im


def plot_k_heatmaps(groups: dict, out: Path):
    for (agent, mode), gr in groups.items():
        Ks, dets, mat = matrix_for_group(gr)
        data = [[mat[d].get(K) for K in Ks] for d in dets]
        fig, ax = plt.subplots(figsize=(3.0 + 0.7 * len(Ks), 0.45 * len(dets) + 1.8))
        im = _heatmap(ax, data, dets, [f"K={K}" for K in Ks],
                      f"{agent} — mode {mode}: AUC by detector × K")
        ax.set_xlabel("GMM cluster count K\n(non-GMM rows are K-invariant)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ROC-AUC")
        p = out / f"heatmap_K_{agent}_mode{safe_mode(mode)}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_perturbation_heatmaps(groups: dict, out: Path):
    for (agent, mode), gr in groups.items():
        bp = best_per_perturbation(gr)
        if not bp:
            continue
        perts = list(bp.keys())
        dets = [d for d in DETECTOR_ORDER if any(d in bp[p] for p in perts)]
        data = [[(bp[p][d][0] if d in bp[p] else None) for p in perts] for d in dets]
        fig, ax = plt.subplots(figsize=(3.2 + 1.1 * len(perts), 0.45 * len(dets) + 1.8))
        im = _heatmap(ax, data, dets, perts,
                      f"{agent} — mode {mode}: best AUC per perturbation (over K)")
        ax.set_xlabel("perturbation type")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ROC-AUC")
        p = out / f"heatmap_pert_{agent}_mode{safe_mode(mode)}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_k_curves(groups: dict, out: Path):
    """One figure per agent: mean GMM-detector AUC vs K, one line per mode."""
    by_agent: dict[str, list] = defaultdict(list)
    for (agent, mode), gr in groups.items():
        by_agent[agent].append((mode, gr))
    for agent, items in by_agent.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode, gr in items:
            Ks, dets, mat = matrix_for_group(gr)
            gdets = [d for d in dets if d in GMM_DETECTORS]
            xs, ys = [], []
            for K in Ks:
                vals = [mat[d][K] for d in gdets if K in mat[d]]
                if vals:
                    xs.append(K)
                    ys.append(np.mean(vals))
            ax.plot(xs, ys, "o-", label=f"mode {mode}")
        ax.axhline(0.5, ls="--", c="gray", lw=1, label="chance")
        ax.set_xlabel("GMM cluster count K")
        ax.set_ylabel("mean AUC over 5 GMM detectors")
        ax.set_title(f"{agent}: GMM detector quality vs cluster count")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.savefig(out / f"curve_meanGMM_vs_K_{agent}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_mode_comparison(groups: dict, out: Path):
    """Scatter plot: mode 1 AUC (x-axis) vs mode 2 AUC (y-axis) per detector.

    Points above the diagonal y=x mean mode 2 is better; below means mode 1 wins.
    One subplot per agent.  Skipped entirely if fewer than two modes exist.
    """
    comp = mode_comparison_data(groups)
    agents = [a for a in sorted(comp.keys())
              if any("1" in v and "2" in v for v in comp[a].values())]
    if not agents:
        return

    fig, axes = plt.subplots(1, len(agents), figsize=(5.5 * len(agents), 4.8), squeeze=False)
    for ax, agent in zip(axes[0], agents):
        dmap = comp[agent]
        xs, ys, labels = [], [], []
        for det in DETECTOR_ORDER:
            if det not in dmap:
                continue
            v = dmap[det]
            if "1" not in v or "2" not in v:
                continue
            xs.append(v["1"])
            ys.append(v["2"])
            labels.append(det)
        if not xs:
            ax.set_visible(False)
            continue
        xs_arr, ys_arr = np.array(xs), np.array(ys)
        colors = ["tab:green" if y > x + 0.005 else
                  ("tab:red" if x > y + 0.005 else "tab:gray")
                  for x, y in zip(xs_arr, ys_arr)]
        ax.scatter(xs_arr, ys_arr, c=colors, s=70, zorder=3)
        for lbl, x, y in zip(labels, xs_arr, ys_arr):
            ax.annotate(lbl, (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7)
        lo = min(xs_arr.min(), ys_arr.min()) - 0.03
        hi = max(xs_arr.max(), ys_arr.max()) + 0.03
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, zorder=2, label="parity (y=x)")
        ax.axhline(0.5, ls=":", c="gray", lw=0.8)
        ax.axvline(0.5, ls=":", c="gray", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Mode 1 AUC (node-level)", fontsize=10)
        ax.set_ylabel("Mode 2 AUC (layer-level)", fontsize=10)
        ax.set_title(f"{agent}: mode 1 vs mode 2\n"
                     "green=mode2 better  red=mode1 better", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "mode_comparison_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="data", type=Path)
    ap.add_argument("--out", default="results_summary", type=Path)
    args = ap.parse_args()

    data_root = args.data_root.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # clear previously-generated artifacts so stale figures don't linger
    for old in list(out.glob("*.png")) + list(out.glob("SUMMARY.md")):
        old.unlink()

    runs = discover_runs(data_root)
    groups = group_by_agent_mode(runs)
    live = inventory_live(data_root)

    print(f"Discovered {len(runs)} runs across {len(groups)} (agent, mode) groups:")
    for (agent, mode), gr in groups.items():
        print(f"  {agent:5s} mode {mode}: Ks = {sorted(r.K for r in gr)}")

    md = build_markdown(groups, live)
    (out / "SUMMARY.md").write_text(md, encoding="utf-8")
    print(f"\nWrote {out / 'SUMMARY.md'}  ({len(md.splitlines())} lines)")

    plot_k_heatmaps(groups, out)
    plot_perturbation_heatmaps(groups, out)
    plot_k_curves(groups, out)
    plot_mode_comparison(groups, out)
    pngs = sorted(out.glob("*.png"))
    print(f"Wrote {len(pngs)} figures:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
