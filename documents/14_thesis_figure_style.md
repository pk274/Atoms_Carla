# 14 — Thesis Figure Style Specification

This document is the exact, complete specification of how figures for the
thesis must look. It exists so the same style can be reproduced in other
projects (in particular the **CARLA environment** project) without reverse-
engineering the ATOMs plotting code.

**Machine-readable counterpart:** `scripts/thesis_style.py` — a
self-contained module (only depends on matplotlib) holding the rcParams,
the palette, and the shared helpers. **To port the style: copy that one
file into the other project and import it.** Do *not* copy
`src/data_generation/viz_config.py` — that file was never changed; it still
holds the old palette and is only used by the exploratory pipeline plots.

Applied by: `scripts/make_thesis_figures.py` → `results/thesis_figures/`.

---

## 1. Hard rules (apply to every thesis figure)

1. **No plot titles.** Neither figure-level (`fig.suptitle`) nor per-axes
   (`ax.set_title`) titles that *describe* the plot. The LaTeX caption
   carries that information. The **only** permitted `ax.set_title` use is
   as a *column header* in a panel grid (rule 4).
2. **One figure = one matplotlib figure.** Panels that appear side by side
   in the thesis are never separate image files stitched in LaTeX; they are
   subplots of a single figure. This is what guarantees consistent font
   sizes, aligned axes, and shared decorations.
3. **No redundant decorations.** Per figure there is at most:
   - **one legend** (figure-level, not per panel),
   - **one colorbar** (figure-level, `fig.colorbar(..., ax=all_axes)`),
   - **one x-axis label** and **one y-axis label** per shared dimension —
     use `fig.supxlabel(...)` / `fig.supylabel(...)` when the label applies
     to all panels; per-axes labels only when panels genuinely measure
     different quantities (then: left panel keeps its own y-label, etc.).
   - Shared axes (`sharex=True, sharey=True`) whenever panels show the same
     quantity, so tick labels appear only on the outer edges **and** the
     panels are directly comparable (this fixed the clean-vs-perturbed
     figure, whose two panels previously had different y-scales).
4. **Panel identification** (so the caption can reference panels) only when
   a caption cannot compactly do it — in practice: grids with ≥ 6 panels.
   - Column headers: `ax.set_title("...", pad=6)` on the **top row only**,
     normal weight (styled via rcParams, 9.5 pt).
   - Row labels: right margin, rotated —
     `axs[r, -1].annotate(text, xy=(1.06, 0.5), xycoords="axes fraction",
     rotation=270, ha="left", va="center", fontsize=9, color="0.15",
     annotation_clip=False)`.
   - Alternatively small in-panel tags, bottom-left:
     `ax.text(0.03, 0.04, tag, transform=ax.transAxes, fontsize=8.5,
     color=MUTED)` (used in the correlation grid).
   - 1×2 figures get **no** panel labels — the caption says "left: …,
     right: …".
5. **Legends may be omitted entirely** when the caption already explains
   the encodings (e.g. "thin lines: individual frames; thick line: mean").
   Never add a legend for a single line.
6. **Output**: every figure is saved as `.pdf` (vector, goes into LaTeX)
   **and** `.png` (300 dpi, for quick viewing), always with
   `bbox_inches="tight"`. PDFs embed fonts as TrueType (`pdf.fonttype: 42`)
   so text stays selectable/editable and no Type-3 fonts appear.

## 2. Fonts and sizes

| Setting | Value |
|---|---|
| Family | serif, stack: `Palatino Linotype → TeX Gyre Pagella → Georgia → DejaVu Serif` |
| Math text | `mathtext.fontset: dejavuserif` |
| Base / axis labels / column headers | **9.5 pt** |
| Tick labels / legend | **8.5 pt** |
| Row labels (margin) | 9 pt, color `0.15` |
| In-panel annotation boxes (e.g. r-values) | 7.5 pt |

Sizes assume the figure is placed at **exactly `\textwidth` = 6.3 in**
(`TEXT_WIDTH_IN` in `thesis_style.py`) — LaTeX must never rescale the
figure, because rescaling changes the effective font size. If the CARLA
thesis chapter uses a different `\textwidth`, change `TEXT_WIDTH_IN` once
and keep everything else.

## 3. Frame, grid, layout

| Setting | Value |
|---|---|
| Spines | top + right **off**; left + bottom 0.7 pt |
| Tick width | 0.7 pt |
| Grid | on, solid, color `0.87`, 0.6 pt, drawn **below** data (`axes.axisbelow`) |
| Legend frame | off (`legend.frameon: False`) |
| Layout engine | `figure.constrained_layout.use: True` |

Canonical figure sizes (width fixed at 6.3 in unless noted):

| Figure type | figsize |
|---|---|
| 1×2 panel pair | (6.3, 2.7–2.8) |
| Grid, R rows × 3 cols | (6.3, 1.75·R + 0.9) |
| Single panel | (5.0, 3.0) |
| 3×2 scatter grid + right colorbar | (6.1, 7.6) — slightly narrower so the tight-bbox'd colorbar lands at ≈ 6.3 total |

## 4. Color palette

### 4.1 Metric colors (CVD-validated)

Validated with Machado-2009 CVD simulation + CIE76 ΔE: **every pairwise
distance ≥ 14.3 ΔE** under normal vision, protanopia, deuteranopia and
tritanopia. Hue families are kept from the old exploratory palette so
readers of both recognize the metrics. **Never swap a single hex value
without re-validating the whole set pairwise.**

| Metric | Hex | Role / rationale |
|---|---|---|
| Mahalanobis | `#b71c1c` | dark red — primary metric |
| Euclidean | `#0d47a1` | dark blue |
| kNN | `#2e7d32` | green |
| JSD | `#9467bd` | purple |
| MDX | `#c26565` | rose — visually "Mahalanobis family", but non-attention |
| PEOC | `#9e7c0c` | dark gold — non-attention |

### 4.2 Line-style secondary encoding

Metrics that do **not** use attention profiles (MDX, PEOC) are dashed —
a semantic cue that also survives grayscale printing and CVD:

| Metric | linestyle |
|---|---|
| all attention-based | `"-"` (solid) |
| MDX | `(0, (4.2, 1.8))` (dash) |
| PEOC | `(0, (4.2, 1.4, 1.1, 1.4))` (dash-dot) |

### 4.3 Other colors

| Use | Value |
|---|---|
| Reward / performance curves | `#1f77b4` |
| Min–max band around reward mean | same color, `alpha=0.18`, `linewidth=0` |
| Intensity-coded scatter | colormap `viridis`, `vmin=0, vmax=<max intensity>` (fixed, not data-dependent, so all panels/figures share the mapping). **viridis is reserved for perturbation intensity** — do not reuse it for other quantities |
| Signed correlation maps | diverging `RdBu_r`, `vmin=-1, vmax=1` (blue = expected negative relation, red = inverted); masked/no-data cells gray `0.88` |
| Muted in-panel text | gray `0.35` |
| Legend/caption text for canonical metric names | Mahalanobis, Euclidean, $k^\mathrm{th}$-NN, Jensen–Shannon, MDX, PEOC — in this order (k-NN renamed to $k^\mathrm{th}$-NN thesis-wide 2026-07-16) |

## 5. Line and marker conventions

| Element | Spec |
|---|---|
| Mean curve (hero line) | `linewidth 1.9` |
| Curves in multi-metric grids | `linewidth 1.25, alpha 0.95` |
| Spaghetti: per-frame lines (static, 100 frames) | metric color, `alpha 0.15, linewidth 0.5, zorder 1`; mean on top `zorder 2` |
| Spaghetti: per-agent lines (live, 10 agents) | metric color, `alpha 0.3, linewidth 0.7, zorder 1`; mean on top `zorder 2` |
| Scatter points | `s=26, edgecolor="black", linewidths=0.3` |
| Stats annotation box | top-right, `xy=(0.97, 0.97)` axes fraction, `boxstyle="round,pad=0.32"`, white `alpha 0.85`, edge `0.8` at 0.5 pt |
| Legend handles (figure legend) | `linewidth 1.6, handlelength 1.9, columnspacing 1.4` |
| Normalized-distance axes | `ylim (-0.04, 1.04)`, yticks `[0, 0.5, 1]` |
| Intensity axes | `xlim (0, max)`, ticks every 2 |

## 6. Figure-level legend below the figure (the collision-safe recipe)

matplotlib 3.10's `loc="outside lower center"` collides with
`fig.supxlabel`. The working pattern (encapsulated in
`thesis_style.metric_legend_handles`):

```python
fig.legend(handles=metric_legend_handles(), loc="upper center", ncol=6,
           bbox_to_anchor=(0.5, -0.002), bbox_transform=fig.transFigure,
           columnspacing=1.4, handlelength=1.9)
```

This anchors the legend just *below* the canvas; `bbox_inches="tight"` at
save time grows the page to include it. Handles are `Line2D` proxies (so
dashed metrics show their dash pattern), never taken from a single axes.

## 7. Colorbar recipe (one per figure)

```python
fig.colorbar(scatter, ax=axs, shrink=0.5, aspect=28, pad=0.02,
             label="Perturbation intensity")
```

Passing the whole axes array as `ax` makes constrained_layout steal space
from **all** panels equally; `shrink=0.5` keeps it visually subordinate.

## 8. Porting checklist for the CARLA project

1. Copy `scripts/thesis_style.py` (unchanged) into the CARLA repo.
2. In every thesis-figure script: `from thesis_style import *`-equivalents,
   call `apply_thesis_style()` once before plotting, save via
   `save_figure(fig, out_dir, name)`.
3. Reuse `METRIC_COLORS` / `METRIC_DASHES` / `METRIC_LEGEND` for the same
   metrics so ATOMs and CARLA chapters are visually consistent. New
   CARLA-only series need colors validated against the existing six
   (pairwise ΔE ≥ ~14 under protan/deutan/tritan) before being added.
4. Keep `TEXT_WIDTH_IN` in sync with the thesis `\textwidth`.
5. Follow the hard rules in §1 — especially: no titles, one legend, one
   colorbar, shared axes with outer-edge labels only.
