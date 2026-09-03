"""
figure_notes.py
---------------
Companion ``.txt`` files for thesis figures: the exact numbers behind every
visual feature a caption or a body paragraph would describe, written beside the
``.pdf``/``.png`` so the prose can be drafted off one file instead of by
searching the summary CSVs again.

Self-contained and project-agnostic (numpy only): copy this file into any other
project (e.g. the CARLA environment) verbatim, exactly like ``thesis_style.py``.
It is the single source of truth for *numbers* the way ``thesis_style.py`` is
for *looks*.

Usage:
    from figure_notes import FigureNotes

    notes = FigureNotes("perframe_distance_gaussian_noise_A2C_a1_gmm3",
                        title="Static per-frame distances — A2C agent 1, gmm3",
                        source="data/perturbed_datasets/breakout_gaussian_noise/...")
    notes.section("Mahalanobis")
    notes.curve("mean distance", intensities, mean_curve)
    notes.value("highest clean frame at intensity 0", 4.34)
    notes.write(out_dir)          # -> out_dir/perframe_..._gmm3.txt

Two kinds of entry, and the file keeps them visibly apart:

* **measurements** — read straight off the data (endpoints, extrema, counts).
* **derived descriptors** — quantities that only exist relative to a rule, such
  as "where the rise starts" or "where the curve plateaus". These are marked
  ``[def]`` and the rule is printed in the file header, because the thesis
  quotes them as if they were observations ("a rise starts around intensity
  5.5", "stabilizes around 2.05 above 4.3") and a reader must be able to see
  what threshold produced the number.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Defaults for the derived descriptors ──────────────────────────────────────
# Chosen so the emitted numbers land close to the ones already eyeballed in the
# draft; they are conventions, not measurements, and are echoed into every file.

RISE_FRAC = 0.10      # "rise onset": first sustained crossing of y0 + 10% of the climb
PLATEAU_TOL = 0.05    # "plateau": MAD around the level within 5% of the total range
PLATEAU_RHO_MAX = 0.3 # "plateau": and no residual trend beyond this |Spearman rho|
SHAPE_WINDOW = 9      # smoothing window (samples) for the monotonicity segmentation
SHAPE_MIN_FRAC = 0.12 # a segment must span >= 12% of the x-range to be named

_INDENT = "    "


# ── Formatting ─────────────────────────────────────────────────────────────────

def fmt(value: Optional[float], sig: int = 4) -> str:
    """Number at `sig` significant digits; 'n/a' for None/NaN."""
    if value is None:
        return "n/a"
    v = float(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{sig}g}"


def fmt_x(value: Optional[float]) -> str:
    """An x-position (perturbation intensity): one decimal, matching the 0.1 grid."""
    if value is None:
        return "n/a"
    v = float(value)
    return "n/a" if not np.isfinite(v) else f"{v:.1f}"


# ── Rank correlation (numpy only, so the module stays dependency-light) ────────

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged — the same convention as scipy's rankdata."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average the ranks within each group of equal values
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation. NaN when either side is constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = _rankdata(x[ok]), _rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Curve descriptors ──────────────────────────────────────────────────────────

def extremum(x: Sequence[float], y: Sequence[float],
             kind: str = "max") -> Tuple[float, float]:
    """(value, x-position) of the min or max. The position matters: the draft
    writes "maximum of 2.11 through a fluctuation at intensity 8"."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not np.isfinite(y).any():
        return float("nan"), float("nan")
    i = int(np.nanargmax(y) if kind == "max" else np.nanargmin(y))
    return float(y[i]), float(x[i])


def rise_onset(x: Sequence[float], y: Sequence[float],
               frac: float = RISE_FRAC) -> float:
    """[def] Lowest x from which y stays above ``y[0] + frac * (max - y[0])``
    for the whole remainder of the sweep.

    "Sustained" rather than "first crossing", matching ``sustained_onset`` in
    compute_detection_summary.py, so a single noisy sample cannot set it.
    NaN when the curve never rises that far.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) < 2 or not np.isfinite(y).any():
        return float("nan")
    climb = np.nanmax(y) - y[0]
    if climb <= 0:
        return float("nan")
    above = y >= y[0] + frac * climb
    for i in range(len(above)):
        if above[i] and above[i:].all():
            return float(x[i])
    return float("nan")


def plateau_onset(x: Sequence[float], y: Sequence[float],
                  tol: float = PLATEAU_TOL,
                  rho_max: float = PLATEAU_RHO_MAX) -> Tuple[float, float]:
    """[def] ``(x, level)`` of the lowest x from which the curve has stopped
    trending: the remainder has weak rank correlation with x (|rho| <= rho_max)
    *and* a median absolute deviation from its own median within ``tol * range``.

    Answers "stabilizes around 2.05 for intensities greater than 4.3". The
    deviation test uses the MAD, not the max, on purpose: the same sentence goes
    on to say "reaching a maximum of 2.11 through a fluctuation at intensity 8",
    so a short excursion has to sit *inside* the plateau rather than end it.
    A max-based band would push the onset past every such fluctuation.

    Returns (NaN, NaN) when the curve never settles.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    span = np.nanmax(y) - np.nanmin(y)
    if len(y) < 5 or span <= 0:
        return float("nan"), float("nan")
    for i in range(len(y) - 4):
        tail = y[i:]
        level = float(np.nanmedian(tail))
        mad = float(np.nanmedian(np.abs(tail - level)))
        if mad > tol * span:
            continue
        rho = spearman(x[i:], tail)
        if not np.isfinite(rho) or abs(rho) <= rho_max:
            return float(x[i]), level
    return float("nan"), float("nan")


def dip_below_start(x: Sequence[float], y: Sequence[float]
                    ) -> Optional[Tuple[float, float, float]]:
    """``(x_first, x_last, deepest_value)`` of the excursion below ``y[0]``, or
    None if the curve never drops under its own starting value.

    Answers "with the ED temporarily falling below its value at intensity 0".
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    below = np.where(y < y[0])[0]
    if below.size == 0:
        return None
    return float(x[below[0]]), float(x[below[-1]]), float(np.nanmin(y[below]))


def shape_segments(x: Sequence[float], y: Sequence[float],
                   window: int = SHAPE_WINDOW,
                   min_frac: float = SHAPE_MIN_FRAC) -> List[Tuple[str, float, float]]:
    """[def] Monotone segments as ``(direction, x_start, x_end)`` after
    smoothing with a moving average of `window` samples.

    Gives the "MD roughly forms an N shape, following an increase with a
    decrease and another increase" style of description something to rest on.
    Segments shorter than `min_frac` of the x-range are absorbed into their
    neighbour so noise does not fragment the description.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) < max(4, window):
        return []
    kernel = np.ones(window) / window
    smooth = np.convolve(y, kernel, mode="same")
    # convolve's edges are averaged against zeros — restore them from the raw curve
    half = window // 2
    smooth[:half], smooth[len(smooth) - half:] = y[:half], y[len(y) - half:]

    sign = np.sign(np.diff(smooth))
    sign[sign == 0] = 1                      # flat counts as continuing
    segments: List[Tuple[str, float, float]] = []
    start = 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[start]:
            segments.append(("rise" if sign[start] > 0 else "fall",
                             float(x[start]), float(x[i])))
            start = i

    span = float(x[-1] - x[0])
    if span <= 0:
        return segments
    merged: List[Tuple[str, float, float]] = []
    for direction, a, b in segments:
        if merged and (b - a) < min_frac * span:
            merged[-1] = (merged[-1][0], merged[-1][1], b)   # absorb into previous
        elif merged and merged[-1][0] == direction:
            merged[-1] = (direction, merged[-1][1], b)       # extend same direction
        else:
            merged.append((direction, a, b))
    return merged


def shape_word(segments: Sequence[Tuple[str, float, float]]) -> str:
    """A short name for the segment pattern ('monotone rise', 'N shape', ...)."""
    dirs = [d for d, _, _ in segments]
    if not dirs:
        return "undetermined"
    if len(dirs) == 1:
        return "monotone rise" if dirs[0] == "rise" else "monotone fall"
    if dirs == ["rise", "fall"]:
        return "rise then fall (peak)"
    if dirs == ["fall", "rise"]:
        return "fall then rise (bowl)"
    if dirs == ["rise", "fall", "rise"]:
        return "N shape (rise-fall-rise)"
    if dirs == ["fall", "rise", "fall"]:
        return "inverted N (fall-rise-fall)"
    return "-".join(dirs)


# ── The notes file ─────────────────────────────────────────────────────────────

class FigureNotes:
    """Accumulates labelled sections and key/value lines, then writes them to
    ``<out_dir>/<name>.txt``.

    Mirrors ``thesis_style.save_figure``: build it up next to the plotting code
    so the numbers can never drift from the figure they describe.
    """

    def __init__(self, name: str, title: str = "", source: str = "") -> None:
        self.name = name
        self.title = title
        self.source = source
        self._lines: List[str] = []
        self._used_derived = False

    # -- structure ------------------------------------------------------------

    def section(self, heading: str) -> "FigureNotes":
        if self._lines:
            self._lines.append("")
        self._lines.append(f"{heading}")
        self._lines.append("-" * len(heading))
        return self

    def line(self, text: str = "") -> "FigureNotes":
        self._lines.append(text)
        return self

    def value(self, label: str, value, unit: str = "",
              note: str = "", derived: bool = False) -> "FigureNotes":
        """One ``label: value`` line. `unit` is appended, `note` parenthesised."""
        shown = value if isinstance(value, str) else fmt(value)
        text = f"{_INDENT}{label}: {shown}"
        if unit:
            text += f" {unit}"
        if derived:
            text += "  [def]"
            self._used_derived = True
        if note:
            text += f"  ({note})"
        self._lines.append(text)
        return self

    def counts(self, label: str, mapping: Dict[str, int]) -> "FigureNotes":
        """A labelled tally, e.g. the 8/2 split of agents by response sign."""
        parts = ", ".join(f"{k} = {v}" for k, v in mapping.items())
        self._lines.append(f"{_INDENT}{label}: {parts}")
        return self

    # -- the standard curve block --------------------------------------------

    def curve(self, label: str, x: Sequence[float], y: Sequence[float],
              unit: str = "", *, shape: bool = True,
              rise: bool = True, plateau: bool = True) -> "FigureNotes":
        """Everything the thesis normally says about one curve: endpoints,
        extrema with their positions, Spearman rho against x, and the derived
        rise/plateau/shape descriptors."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        if len(y) == 0 or not np.isfinite(y).any():
            self._lines.append(f"{_INDENT}{label}: no finite data")
            return self

        self._lines.append(f"{_INDENT}{label}:")
        pad = _INDENT * 2
        u = f" {unit}" if unit else ""
        self._lines.append(f"{pad}at x={fmt_x(x[0])}: {fmt(y[0])}{u}")
        self._lines.append(f"{pad}at x={fmt_x(x[-1])}: {fmt(y[-1])}{u}")

        hi, hi_at = extremum(x, y, "max")
        lo, lo_at = extremum(x, y, "min")
        self._lines.append(f"{pad}max {fmt(hi)}{u} at x={fmt_x(hi_at)}")
        self._lines.append(f"{pad}min {fmt(lo)}{u} at x={fmt_x(lo_at)}")
        self._lines.append(f"{pad}Spearman rho vs x: {fmt(spearman(x, y), 3)}")

        if rise:
            self._lines.append(f"{pad}rise onset: x={fmt_x(rise_onset(x, y))}  [def]")
            self._used_derived = True
        if plateau:
            p_at, p_level = plateau_onset(x, y)
            self._lines.append(
                f"{pad}plateau: from x={fmt_x(p_at)} at {fmt(p_level)}{u}  [def]")
            self._used_derived = True
        dip = dip_below_start(x, y)
        if dip is not None:
            self._lines.append(
                f"{pad}dips below its x={fmt_x(x[0])} value between "
                f"x={fmt_x(dip[0])} and x={fmt_x(dip[1])}, "
                f"deepest {fmt(dip[2])}{u}")
        if shape:
            segs = shape_segments(x, y)
            self._lines.append(f"{pad}shape: {shape_word(segs)}  [def]")
            for direction, a, b in segs:
                self._lines.append(f"{pad}{_INDENT}{direction} x={fmt_x(a)} -> {fmt_x(b)}")
            self._used_derived = True
        return self

    # -- output ---------------------------------------------------------------

    def render(self) -> str:
        head = [f"# {self.title or self.name}", f"# figure: {self.name}"]
        if self.source:
            head.append(f"# source: {self.source}")
        head.append(
            "# Generated by figure_notes.py — do not edit; regenerate with the figure.")
        if self._used_derived:
            head += [
                "#",
                "# [def] marks a DERIVED descriptor: it depends on the rule below,",
                "#       not only on the data. Quote it as such.",
                f"#   rise onset  = lowest x from which the curve stays above "
                f"y(x0) + {RISE_FRAC:.0%} of its total climb, for all larger x",
                f"#   plateau     = lowest x from which the curve stops trending "
                f"(|rho| <= {PLATEAU_RHO_MAX}) and its MAD around that level "
                f"stays within {PLATEAU_TOL:.0%} of the range",
                f"#   shape       = monotone segments after a {SHAPE_WINDOW}-sample "
                f"moving average, segments < {SHAPE_MIN_FRAC:.0%} of the x-range merged",
            ]
        head.append("")
        return "\n".join(head + self._lines) + "\n"

    def write(self, out_dir: Path | str, ext: str = ".txt") -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}{ext}"
        path.write_text(self.render(), encoding="utf-8")
        print(f"  [OK] {path}")
        return path
