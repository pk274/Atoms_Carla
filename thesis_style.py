"""
thesis_style.py
---------------
The single source of truth for how thesis figures look. Self-contained and
project-agnostic: copy this file into any other project (e.g. the CARLA
environment) and import it there — it only needs matplotlib.

Usage:
    from thesis_style import apply_thesis_style, save_figure, \
        METRIC_COLORS, METRIC_DASHES, METRIC_LEGEND, METRIC_AXIS, \
        METRIC_ORDER, REWARD_COLOR, MUTED, TEXT_WIDTH_IN, \
        metric_legend_handles

    apply_thesis_style()          # once, before creating any figure
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 2.7))
    ...
    save_figure(fig, out_dir, "my_figure", ["pdf", "png"])

The full written specification (layout rules per figure type, rationale,
CVD validation numbers) lives in documentation/14_thesis_figure_style.md.

Core rules (short form):
* no plot titles — the LaTeX caption carries that information,
* multi-panel figures are ONE matplotlib figure: shared axes, axis labels
  only on the outer edges (fig.supxlabel / fig.supylabel), exactly one
  legend and one colorbar per figure,
* panel identification (column headers via ax.set_title on the top row,
  row labels on the right margin) only where a caption cannot compactly
  identify the panels — i.e. only in grids with >= 6 panels,
* serif font stack matching the thesis body text, TrueType-embedded PDFs,
* CVD-safe metric palette (every pairwise distance >= 14 dE under
  protanopia/deuteranopia/tritanopia simulation); MDX and PEOC — the two
  metrics that do not use attention profiles — are additionally dashed.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── Layout constants ───────────────────────────────────────────────────────────

# LaTeX \textwidth of the thesis in inches. Full-width figures use exactly
# this width so LaTeX never rescales them (rescaling changes font sizes).
TEXT_WIDTH_IN = 6.3

# ── Metric palette ─────────────────────────────────────────────────────────────
# CVD-validated (protan/deutan/tritan, Machado 2009 simulation, CIE76 dE):
# worst pairwise distance 14.3. Hues keep their established families from the
# exploratory plots (MD red, ED blue, kNN green, JSD purple) but darkened /
# shifted for print contrast and CVD separation. Do NOT swap individual hex
# values without re-validating the whole set pairwise.

METRIC_COLORS = {
    "mahalanobis": "#b71c1c",   # dark red      (primary metric)
    "euclidean":   "#0d47a1",   # dark blue
    "knn":         "#2e7d32",   # green
    "jsd":         "#9467bd",   # purple
    "mdx":         "#c26565",   # rose — Mahalanobis family, non-attention
    "peoc":        "#9e7c0c",   # dark gold     — non-attention
}

# MDX / PEOC do not use the attention profile: dashed as a secondary encoding
# (semantic cue + line style survives grayscale printing and CVD).
METRIC_DASHES = {
    "mdx":  (0, (4.2, 1.8)),            # plain dash
    "peoc": (0, (4.2, 1.4, 1.1, 1.4)),  # dash-dot
}

METRIC_LEGEND = {
    "mahalanobis": "Mahalanobis",
    "euclidean":   "Euclidean",
    "knn":         "$k^\\mathrm{th}$-NN",
    "jsd":         "Jensen–Shannon",
    "mdx":         "MDX",
    "peoc":        "PEOC",
}

METRIC_AXIS = {
    "mahalanobis": "Mahalanobis distance",
    "euclidean":   "Euclidean distance",
    "knn":         "$k^\\mathrm{th}$-NN distance",
    "jsd":         "Jensen–Shannon divergence",
    "mdx":         "MDX distance",
    "peoc":        "PEOC entropy",
}

# Canonical order of metrics in legends and multi-metric plots.
METRIC_ORDER = ["mahalanobis", "euclidean", "knn", "jsd", "mdx", "peoc"]

REWARD_COLOR = "#1f77b4"   # reward / performance curves (matplotlib C0 blue)
MUTED = "0.35"             # in-panel tags, secondary annotation text

# Sequential colormap for intensity-coded scatters (0 -> max intensity).
INTENSITY_CMAP = "viridis"


# ── rcParams ───────────────────────────────────────────────────────────────────

def apply_thesis_style() -> None:
    """Set global rcParams for thesis figures. Call once before plotting."""
    plt.rcParams.update({
        # Serif stack blending with the thesis body text. First hit wins;
        # DejaVu Serif is the always-available fallback.
        "font.family":       "serif",
        "font.serif":        ["Palatino Linotype", "TeX Gyre Pagella",
                              "Georgia", "DejaVu Serif"],
        "mathtext.fontset":  "dejavuserif",
        # Sizes chosen for figures placed at \textwidth without rescaling:
        # body text 9.5 pt ~ \small relative to 11 pt thesis text.
        "font.size":         9.5,
        "axes.labelsize":    9.5,
        "axes.titlesize":    9.5,   # column headers in grids (only title use)
        "axes.titleweight":  "normal",
        "xtick.labelsize":   8.5,
        "ytick.labelsize":   8.5,
        "legend.fontsize":   8.5,
        # Light frame: thin lines, no top/right spines, subtle solid grid.
        "axes.linewidth":    0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "0.87",
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",
        "axes.axisbelow":    True,
        "legend.frameon":    False,
        "figure.constrained_layout.use": True,
        # Output: 300 dpi rasters, TrueType-embedded (editable, no Type-3).
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_figure(fig: plt.Figure, out_dir: Path, name: str,
                formats: List[str] = ("pdf", "png")) -> None:
    """Save one figure under every requested extension and close it.

    bbox_inches="tight" is mandatory: legends anchored below the figure
    (see metric_legend_handles) live outside the canvas until save time.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  [OK] {path}")
    plt.close(fig)


def metric_legend_handles(metrics: List[str] = None) -> List[Line2D]:
    """Proxy handles for a shared figure-level metric legend.

    Attach with:
        fig.legend(handles=metric_legend_handles(), loc="upper center",
                   ncol=6, bbox_to_anchor=(0.5, -0.002),
                   bbox_transform=fig.transFigure,
                   columnspacing=1.4, handlelength=1.9)
    which anchors the legend just BELOW the figure so it can never collide
    with fig.supxlabel; save_figure's bbox_inches="tight" grows the canvas
    to include it.
    """
    metrics = metrics or METRIC_ORDER
    return [Line2D([], [], color=METRIC_COLORS[m],
                   linestyle=METRIC_DASHES.get(m, "-"), linewidth=1.6,
                   label=METRIC_LEGEND[m]) for m in metrics]
