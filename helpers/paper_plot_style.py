"""House style, taste helpers, and automatic audit. Every gen_fig*.py imports it.

    from paper_plot_style import *
    fig, ax = plt.subplots(figsize=figsize("column"))
    ... plot with method_style(key) ...
    finish(ax)                 # ticks, range frame, number format
    label_lines(ax)            # direct labels instead of a legend
    save_fig(fig, "fig2_curves", width="column")

Source: the `paper-figure` skill (Appendix A), copied verbatim except for
(1) the constants blocks (VENUE) and (2) the backend line: inside a Jupyter
notebook the inline backend is kept so figures still display in the notebook.
"""
import json
import os
import shutil
import subprocess

import numpy as np
import matplotlib
try:                                   # EDIT (2): keep the inline backend in Jupyter
    get_ipython  # noqa: F821
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

# ---------------------------------------------------------------- geometry --
# (full text width in, column width in, body font pt). Measured values beat
# this table: set the overrides from \the\textwidth / \the\columnwidth
# (pt / 72.27 = inches).
VENUES = {
    "neurips": (5.50, 5.50, 10),
    "iclr":    (5.50, 5.50, 10),
    "icml":    (6.75, 3.25, 10),
    "acl":     (6.30, 3.03, 11),
    "cvpr":    (6.875, 3.28, 10),
    "report":  (6.30, 6.30, 11),   # A4 article, ~2.5 cm margins: MEASURE it
}
VENUE = "report"               # EDIT: key of VENUES (A4 report; not yet measured)
TEXT_WIDTH_OVERRIDE = None     # EDIT: inches, measured
COLUMN_WIDTH_OVERRIDE = None
FIG_DIR = "figures"

TEXT_WIDTH, COLUMN_WIDTH, BODY_PT = VENUES[VENUE]
TEXT_WIDTH = TEXT_WIDTH_OVERRIDE or TEXT_WIDTH
COLUMN_WIDTH = COLUMN_WIDTH_OVERRIDE or COLUMN_WIDTH
MIN_PT = 6.0            # nothing prints smaller than this

# Two type sizes only: hierarchy comes from colour and weight, not from
# a ladder of sizes.
LABEL_PT = BODY_PT - 1  # axis labels, direct labels of the key series
TICK_PT = BODY_PT - 2   # ticks, other labels, annotations

# ------------------------------------------------------------- palette -----
# Ink hierarchy: data > labels > axes > grid.
INK = "#161616"         # axis labels, annotation text
MUTED = "#505050"       # tick labels, secondary text
RULE = "#9A9A9A"        # spines, ticks, leader lines
GRID = "#EBEBEB"        # gridlines (rarely needed)
# Palette A ("editorial"), chosen by the author over the skill's muted default:
# one saturated signal red for the claim, a heavy petrol for the main rival,
# a light slate and a dark oxblood for context series, gold for boxes drawn on
# radiographs. Red and petrol differ strongly in lightness, so they stay
# distinct in grayscale and for colour-blind readers.
PETROL, RED, SLATE, OXBLOOD, GOLD = (
    "#16425B", "#D7301F", "#7FA3B3", "#8C1C13", "#FFB000")
PALETTE = [PETROL, RED, SLATE, OXBLOOD, GOLD]
ACCENT = RED                                    # the claim's series ("ours")
NEUTRALS = ["#4D4D4D", "#8C8C8C", "#BDBDBD"]    # context series
SEQ_CMAP, DIV_CMAP = "cividis", "RdBu_r"

MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
LINESTYLES = ["-", "--", "-.", ":"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
    "font.size": LABEL_PT,
    "axes.labelsize": LABEL_PT,
    "axes.titlesize": LABEL_PT,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 4,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
    "legend.borderaxespad": 0.3,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.6,
    "axes.labelpad": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "xtick.color": RULE,
    "ytick.color": RULE,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.2,
    "lines.markersize": 3.5,
    "lines.markeredgewidth": 0.6,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "patch.linewidth": 0,
    "errorbar.capsize": 0,
    "scatter.edgecolors": "none",
    "figure.constrained_layout.use": True,
    "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42,        # TrueType, never Type 3 (venues reject Type 3)
    "ps.fonttype": 42,
    "image.cmap": SEQ_CMAP,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})


def figsize(width="column", aspect=0.62, nrows=1, ncols=1, rel=1.0):
    """Final printed size in inches. width: 'column' | 'full'."""
    w = (COLUMN_WIDTH if width == "column" else TEXT_WIDTH) * rel
    return (w, w * aspect * nrows / ncols)


# ---------------------------------------------------- method registry -------
# figures/methods.json: {"ours": {"label": "Ours", "role": "focus", ...}}
# role: "focus" (the claim's series), "baseline", or "reference" (random,
# oracle, human). Per-method keys (color, marker, linestyle, lw) override.
ROLE_DEFAULTS = {
    "focus":     {"color": ACCENT, "lw": 1.8, "zorder": 4, "marker": "o",
                  "linestyle": "-"},
    "baseline":  {"lw": 1.1, "zorder": 3},
    "reference": {"color": NEUTRALS[1], "lw": 0.9, "zorder": 2,
                  "linestyle": ":", "marker": ""},
}
_METHODS = None


def method_style(key):
    """Same label, colour, marker, linestyle, weight for a method in EVERY
    figure. Returns a dict; pass s['plot'] as **kwargs to ax.plot."""
    global _METHODS
    if _METHODS is None:
        path = os.path.join(FIG_DIR, "methods.json")
        _METHODS = json.load(open(path)) if os.path.exists(path) else {}
    if key not in _METHODS:
        raise KeyError(f"'{key}' missing from {FIG_DIR}/methods.json; add it "
                       "there instead of styling it inline.")
    m = dict(_METHODS[key])
    s = {**ROLE_DEFAULTS.get(m.get("role", "baseline"), {}), **m}
    s.setdefault("color", PETROL)
    s.setdefault("marker", "o")
    s.setdefault("linestyle", "-")
    s["plot"] = dict(color=s["color"], lw=s["lw"], zorder=s["zorder"],
                     linestyle=s["linestyle"], label=s["label"])
    s["markers"] = dict(marker=s["marker"], markersize=3.5,
                        markeredgecolor="white", markeredgewidth=0.5)
    return s


# ------------------------------------------------------------ taste kit -----
def text_color(c, max_lum=0.55):
    """Darken a light series colour so text in that colour stays legible."""
    rgb = np.array(to_rgb(c))
    for _ in range(12):
        if _lum(rgb) <= max_lum:
            break
        rgb = rgb * 0.85
    return to_hex(rgb)


def band(ax, x, mean, lo, hi, s, markevery=None):
    """Mean line over a quiet uncertainty band (no band edge)."""
    x, mean = np.asarray(x), np.asarray(mean)
    ax.fill_between(x, lo, hi, color=s["color"], alpha=0.15, lw=0,
                    zorder=s["zorder"] - 1)
    kw = dict(s["plot"])
    if markevery:
        kw.update(s["markers"], markevery=markevery)
    return ax.plot(x, mean, **kw)[0]


def _fmt(kind):
    def f(v, _):
        if kind == "pct":
            return f"{v:g}%"
        if kind == "pct1":                      # data in [0, 1]
            return f"{100 * v:g}%"
        if kind == "k":
            a = abs(v)
            if a >= 1e9: return f"{v / 1e9:g}B"
            if a >= 1e6: return f"{v / 1e6:g}M"
            if a >= 1e3: return f"{v / 1e3:g}k"
            return f"{v:g}"
        if kind == "x":
            return f"{v:g}×"
        return f"{v:g}"
    return FuncFormatter(f)


def finish(ax, x=None, y=None, nticks=5, range_frame=True, offset=4,
           grid=None, categorical=None, snap=True):
    """Final touches that separate a designed plot from a default one.
    x / y: tick format, one of None, 'pct', 'pct1', 'k', 'x', 'plain'.
    grid: None | 'y' | 'x' (hairline, behind data; only if values are read).
    categorical: 'x' or 'y' for the category axis of bar/dot plots.
    snap: end each axis on round ticks just beyond the data.
    Call after all data is plotted and before label_lines/callout."""
    for axis, kind, name in ((ax.xaxis, x, "x"), (ax.yaxis, y, "y")):
        scale = ax.get_xscale() if name == "x" else ax.get_yscale()
        if name == categorical:
            axis.set_tick_params(length=0, pad=4)
            continue
        if scale != "linear":
            continue
        loc = MaxNLocator(nbins=nticks, steps=[1, 2, 5, 10])
        axis.set_major_formatter(_fmt(kind or "plain"))
        dl = ax.dataLim
        dmin, dmax = (dl.x0, dl.x1) if name == "x" else (dl.y0, dl.y1)
        if not (snap and np.isfinite([dmin, dmax]).all() and dmax > dmin):
            axis.set_major_locator(loc)
            continue
        # Axis ends on round ticks just outside the data (finished look).
        # Data within 3% of a tick counts as reaching it (e.g. a band
        # dipping to -0.4 still snaps to 0).
        tol = 0.03 * (dmax - dmin)
        t = loc.tick_values(dmin + tol, dmax - tol)
        below, above = t[t <= dmin + tol], t[t >= dmax - tol]
        lo_t = below.max() if len(below) else dmin
        hi_t = above.min() if len(above) else dmax
        ticks = t[(t >= lo_t - 1e-12) & (t <= hi_t + 1e-12)]
        axis.set_major_locator(FixedLocator(ticks))
        m = (hi_t - lo_t) * 0.02
        (ax.set_xlim if name == "x" else ax.set_ylim)(
            min(lo_t - m, dmin), max(hi_t + m, dmax))
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", offset))
    if categorical == "x":
        ax.spines["bottom"].set_visible(False)
    if categorical == "y":
        ax.spines["left"].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        (ax.spines["left"] if grid == "y" else ax.spines["bottom"]
         ).set_visible(False)
        (ax.yaxis if grid == "y" else ax.xaxis).set_tick_params(length=0)
    if range_frame:
        for side, axis, lim in (("bottom", ax.xaxis, ax.get_xlim()),
                                ("left", ax.yaxis, ax.get_ylim())):
            if not ax.spines[side].get_visible() or \
                    (side == "bottom" and categorical == "x") or \
                    (side == "left" and categorical == "y"):
                continue
            lo, hi = sorted(lim)
            t = [v for v in axis.get_majorticklocs() if lo <= v <= hi]
            if len(t) >= 2:
                ax.spines[side].set_bounds(t[0], t[-1])
    return ax


def label_lines(ax, lines=None, pad_pt=4, focus_bold=True):
    """Direct labels at the right end of each line (replaces the legend).
    Labels are nudged apart vertically, coloured like their line."""
    fig = ax.figure
    fig.canvas.draw()                              # settle layout first
    lines = lines or [l for l in ax.lines
                      if l.get_label() and not l.get_label().startswith("_")]
    items = []
    for l in lines:
        xy = np.asarray(l.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(1)]
        if not len(xy):
            continue
        px = ax.transData.transform(xy[-1])
        items.append([px[1], px[1], l, xy[-1]])
    if not items:
        return
    gap = TICK_PT * 1.25 * fig.dpi / 72
    items.sort(key=lambda it: it[0])
    for i in range(1, len(items)):                 # push up
        items[i][1] = max(items[i][1], items[i - 1][1] + gap)
    top = ax.bbox.y1 + gap * 0.5
    over = items[-1][1] - top
    if over > 0:                                   # shift down, keep gaps
        for it in items:
            it[1] -= over
        for i in range(len(items) - 2, -1, -1):
            items[i][1] = min(items[i][1], items[i + 1][1] - gap)
    for y0, y1, l, end in items:
        dy = (y1 - y0) * 72 / fig.dpi
        heavy = focus_bold and l.get_linewidth() >= 1.6
        ax.annotate(l.get_label(), xy=end, xycoords="data",
                    xytext=(pad_pt, dy), textcoords="offset points",
                    va="center", ha="left", color=text_color(l.get_color()),
                    fontsize=TICK_PT, fontweight="bold" if heavy else "normal",
                    annotation_clip=False)._direct_label = True
    if ax.get_legend():
        ax.get_legend().remove()


def callout(ax, text, xy, xytext=(-30, 12), **kw):
    """Say the takeaway on the plot: short text + hairline leader."""
    kw = {"ha": "center", "va": "bottom", **kw}
    return ax.annotate(text, xy=xy, xytext=xytext, textcoords="offset points",
                       fontsize=TICK_PT, color=INK,
                       arrowprops=dict(arrowstyle="-", lw=0.5, color=RULE,
                                       shrinkA=1, shrinkB=2), **kw)


def ylabel_top(ax, text):
    """Horizontal y-label above the axis: readable without head-tilting.
    Good for single-panel plots with short labels."""
    ax.set_ylabel("")
    ax.set_title(text, loc="left", fontsize=TICK_PT, color=MUTED, pad=6)
    ax._ylabel_top = True


def contact_sheet(names, out="_preview/_sheet.png", gap_in=0.3):
    """Stack previews at true relative size, as they will sit in the paper,
    to judge the figure SET as one family (palette, weights, type, style)."""
    imgs = [plt.imread(os.path.join(FIG_DIR, "_preview", f"{n}.png"))[..., :3]
            for n in names]
    w = max(im.shape[1] for im in imgs)
    gap = np.ones((int(gap_in * 220), w, 3))
    rows = []
    for im in imgs:
        pad = np.ones((im.shape[0], w - im.shape[1], 3))
        rows += [np.concatenate([im, pad], axis=1), gap]
    p = os.path.join(FIG_DIR, out)
    plt.imsave(p, np.clip(np.concatenate(rows[:-1]), 0, 1))
    return p


# --------------------------------------------------------------- audit -------
def _texts(fig):
    hidden = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = (hi - lo) * 1e-9
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - eps <= tick.get_loc() <= hi + eps):
                    hidden.update({id(tick.label1), id(tick.label2)})
    out = []
    for t in fig.findobj(Text):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is not None and not t.axes.get_visible():
            continue
        out.append(t)
    return out


def _line_pts(ax, l):
    xy = ax.transData.transform(np.asarray(l.get_xydata(), float))
    xy = xy[np.isfinite(xy).all(1)]
    if 1 < len(xy) < 2000:
        t = np.linspace(0, 1, 12)[:, None, None]
        xy = (xy[:-1] * (1 - t) + xy[1:] * t).reshape(-1, 2)
    return xy


def _lum(c):
    r, g, b = to_rgb(c)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def audit(fig, expected_width=None):
    """Deterministic checks on the figure at its final printed size.
    Returns a list of {severity, check, detail}."""
    issues = []

    def add(sev, check, detail):
        issues.append({"severity": sev, "check": check, "detail": detail})

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox

    w, h = fig.get_size_inches()
    if expected_width and abs(w - expected_width) / expected_width > 0.02:
        add("error", "size", f"figure is {w:.2f} in wide, must be "
            f"{expected_width:.2f} in (it will be rescaled and fonts change)")

    texts = _texts(fig)
    owner = {id(t): ax for ax in fig.axes for t in ax.findobj(Text)}
    boxes = []
    for t in texts:
        s = t.get_text().strip()[:40]
        if t.get_fontsize() < MIN_PT - 1e-6:
            add("error", "font-size", f"'{s}' is {t.get_fontsize():.1f} pt "
                f"(< {MIN_PT} pt at print size)")
        bb = t.get_window_extent(r)
        if (bb.x0 < fb.x0 - 1 or bb.y0 < fb.y0 - 1 or
                bb.x1 > fb.x1 + 1 or bb.y1 > fb.y1 + 1):
            add("error", "clipped", f"'{s}' extends outside the figure")
        elif t.axes is not None and t in t.axes.texts and not \
                getattr(t, "_direct_label", False) and not \
                t.axes.bbox.padded(1).contains(bb.x1, bb.y1):
            add("warn", "annotation", f"'{s}' sits outside its axes")
        boxes.append((s, bb))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1].padded(-1), boxes[j][1].padded(-1)
            if a.overlaps(b):
                add("error", "text-overlap",
                    f"'{boxes[i][0]}' overlaps '{boxes[j][0]}'")
            elif owner.get(id(texts[i])) is not owner.get(id(texts[j])) \
                    and owner.get(id(texts[i])) is not None \
                    and owner.get(id(texts[j])) is not None \
                    and boxes[i][1].padded(
                    TICK_PT * 1.2 * fig.dpi / 72).overlaps(boxes[j][1]):
                add("taste", "crowded", f"'{boxes[i][0]}' and "
                    f"'{boxes[j][0]}' (different panels) nearly touch; "
                    "add wspace/hspace or use fewer ticks")

    if fig._suptitle is not None and fig._suptitle.get_text().strip():
        add("error", "title", "figure-level title; put it in the caption")
    n_axes = len([a for a in fig.axes if a.get_visible()])

    for ax in fig.axes:
        if not ax.get_visible():
            continue
        title = " ".join(ax.get_title(loc) for loc in ("left", "center",
                                                      "right")).strip()
        if title and n_axes == 1 and \
                not getattr(ax, "_ylabel_top", False):
            add("error", "title", f"axes title '{title}' on a "
                "single-panel figure; move it to the caption")
        for lab in (ax.get_xlabel(), ax.get_ylabel()):
            if "_" in lab and "$" not in lab:
                add("warn", "label", f"'{lab}' looks like a variable name")
        has_data = bool(ax.lines or ax.collections or ax.patches or ax.images)
        has_sup = fig._supxlabel is not None and fig._supxlabel.get_text()
        if has_data and not ax.get_xlabel() and not has_sup \
                and ax.name != "polar" \
                and any(t.get_visible() and t.get_text()
                        for t in ax.get_xticklabels()):
            add("warn", "label", "x-axis has tick labels but no axis label")

        # annotations / direct labels sitting on top of data
        for t in ax.texts:
            if not t.get_visible() or not t.get_text().strip():
                continue
            tb = t.get_window_extent(r).padded(1)
            hit = sum(tb.contains(x, y) for l in ax.lines
                      for x, y in _line_pts(ax, l))
            if hit:
                add("error", "text-on-data", f"'{t.get_text()[:30]}' is "
                    "drawn over data; move it into empty space")

        # jet / rainbow colormaps
        for m in list(ax.images) + list(ax.collections):
            cm = getattr(m, "cmap", None)
            if cm is not None and cm.name in {"jet", "rainbow", "hsv",
                                             "gist_rainbow", "nipy_spectral"}:
                add("error", "colormap", f"'{cm.name}' is not perceptually "
                    "uniform; use viridis/cividis (or RdBu/PuOr if diverging)")

        # bars must not be truncated (axis must include the bar baseline)
        bars = [p for p in ax.patches if type(p).__name__ == "Rectangle"
                and p.get_width() > 0 and p.get_height() > 0]
        if bars:
            bottoms = {round(p.get_y(), 9) for p in bars}
            lefts = {round(p.get_x(), 9) for p in bars}
            if len(bottoms) == 1 and ax.get_yscale() == "linear":
                base, lim = bottoms.pop(), min(ax.get_ylim())
            elif len(lefts) == 1 and ax.get_xscale() == "linear":
                base, lim = lefts.pop(), min(ax.get_xlim())
            else:
                base = lim = None
            if base is not None and lim > base + 1e-9:
                add("error", "axis", "bar axis starts above the bar baseline, "
                    "so bar lengths lie; include the baseline or use a dot "
                    "plot / difference plot")

        # series indistinguishable in grayscale with no redundant encoding
        lines = [l for l in ax.lines if l.get_label() and
                 not l.get_label().startswith("_")]
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                same_enc = (a.get_linestyle() == b.get_linestyle() and
                            a.get_marker() == b.get_marker())
                if same_enc and abs(_lum(a.get_color()) -
                                    _lum(b.get_color())) < 0.12:
                    add("error", "grayscale", f"'{a.get_label()}' and "
                        f"'{b.get_label()}' look identical in grayscale; vary "
                        "linestyle or marker")

        # legend covering data
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            lb = leg.get_window_extent(r)
            hit = 0
            for l in ax.lines:
                hit += sum(lb.contains(x, y) for x, y in _line_pts(ax, l))
            for c in ax.collections:
                try:
                    off = c.get_offset_transform().transform(c.get_offsets())
                    hit += sum(lb.contains(x, y) for x, y in off)
                except Exception:
                    pass
            for p in bars:
                if p.get_window_extent(r).overlaps(lb):
                    hit += 1
            if hit:
                add("error", "legend", f"legend covers data ({hit} points); "
                    "move it, label lines directly, or put it outside")
    # ------------------------------------------------ taste (default smells)
    tab10 = {to_hex(c) for c in plt.get_cmap("tab10").colors}
    for t in texts:
        if _lum(t.get_color()) > 0.6:
            add("taste", "contrast", f"'{t.get_text()[:30]}' is too light to "
                "read; use text_color()")
    for t in texts:
        if "DejaVu" in t.get_fontname():
            add("taste", "font", "default DejaVu font in use; the serif "
                "stack did not resolve for some text")
            break
    for ax in fig.axes:
        if not ax.get_visible() or ax.get_label() == "<colorbar>":
            continue
        cols = [to_hex(l.get_color()) for l in ax.lines]
        cols += [to_hex(p.get_facecolor()) for p in ax.patches]
        if tab10 & set(cols):
            add("taste", "palette", "matplotlib default (tab10) colours; use "
                "method_style() / PALETTE")
        line_cols = {to_hex(l.get_color()) for l in ax.lines
                     if not l.get_label().startswith("_")}
        if len(line_cols) > 6:
            add("taste", "palette", f"{len(line_cols)} colours in one panel; "
                "group, facet, or grey out context series")
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            n = sum(lo <= v <= hi for v in axis.get_majorticklocs())
            if n > 7:
                add("taste", "ticks", f"{n} major ticks; 3-6 is calmer "
                    "(finish(ax) sets this)")
        if ax.spines["top"].get_visible() and ax.spines["right"].get_visible():
            add("taste", "frame", "boxed axes; drop top/right spines")
        leg = ax.get_legend()
        if leg is not None and leg.get_frame_on():
            add("taste", "legend", "legend has a frame")
        labelled = [l for l in ax.lines if not l.get_label().startswith("_")]
        if leg is not None and 2 <= len(labelled) <= 5:
            add("taste", "legend", "legend for <=5 lines; direct labels "
                "(label_lines) read faster")
        if len(labelled) >= 3 and len({round(l.get_linewidth(), 2)
                                       for l in labelled}) == 1:
            add("taste", "hierarchy", "every series has equal weight; give "
                "the claim's series role 'focus' in methods.json")
        for c in ax.collections:
            if type(c).__name__.endswith("PolyCollection"):
                a = c.get_alpha()
                if a is None:
                    fc = c.get_facecolor()
                    a = fc[0][3] if len(fc) else 1
                if a > 0.35:
                    add("taste", "band", f"uncertainty band alpha {a:.2f}; "
                        "keep bands quiet (~0.15)")
                    break
        for p in ax.patches:
            if p.get_linewidth() > 0 and p.get_edgecolor()[3] > 0 and \
                    sum(p.get_edgecolor()[:3]) < 0.3:
                add("taste", "bars", "black outlines on bars; drop them")
                break
        bb = ax.get_window_extent(r)
        ratio = bb.height / max(bb.width, 1)
        if ax.lines and (ratio < 0.2 or ratio > 1.6):
            add("taste", "aspect", f"panel aspect {ratio:.2f}; lines read "
                "best around 0.5-0.8")
        for g in ax.get_xgridlines() + ax.get_ygridlines():
            if g.get_visible() and (g.get_linewidth() > 0.6 or
                                    _lum(g.get_color()) < 0.8):
                add("taste", "grid", "gridlines too heavy; hairline, light "
                    "grey, behind data")
                break
    return issues


def _pdf_checks(pdf):
    issues = []
    if shutil.which("pdffonts"):
        out = subprocess.run(["pdffonts", pdf], capture_output=True,
                             text=True).stdout
        if "Type 3" in out:
            issues.append({"severity": "error", "check": "fonts",
                           "detail": "Type 3 font embedded"})
    if os.path.getsize(pdf) > 1_000_000:
        issues.append({"severity": "warn", "check": "filesize",
                       "detail": "PDF > 1 MB; use rasterized=True on dense "
                       "scatter/heatmap artists (keeps text vector)"})
    return issues


def _preview(fig, pdf, name):
    """PNG of what LaTeX will see + a grayscale copy, for visual review."""
    d = os.path.join(FIG_DIR, "_preview")
    os.makedirs(d, exist_ok=True)
    png = os.path.join(d, f"{name}.png")
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "220", "-singlefile", pdf,
                        png[:-4]], check=True)
    else:
        fig.savefig(png, dpi=220)
    import numpy as np
    img = plt.imread(png)[..., :3]
    gray = img @ np.array([0.2126, 0.7152, 0.0722])
    plt.imsave(png[:-4] + "_gray.png", gray, cmap="gray", vmin=0, vmax=1)
    return png


def save_fig(fig, name, width="column", rel=1.0, formats=("pdf",)):
    """Audit, save vector PDF, render previews, write the audit log.
    Prints PASS/FAIL. Fix every 'error' before the figure is done."""
    os.makedirs(FIG_DIR, exist_ok=True)
    expected = (COLUMN_WIDTH if width == "column" else TEXT_WIDTH) * rel
    issues = audit(fig, expected)
    paths = []
    for fmt in formats:
        p = os.path.join(FIG_DIR, f"{name}.{fmt}")
        fig.savefig(p)          # no bbox='tight': it silently changes size
        paths.append(p)
    pdf = next((p for p in paths if p.endswith(".pdf")), None)
    if pdf:
        issues += _pdf_checks(pdf)
    png = _preview(fig, pdf or paths[0], name)
    os.makedirs(os.path.join(FIG_DIR, "_audit"), exist_ok=True)
    with open(os.path.join(FIG_DIR, "_audit", f"{name}.json"), "w") as f:
        json.dump({"figure": name, "width_in": expected,
                   "preview": png, "issues": issues}, f, indent=2)
    n_err = sum(i["severity"] == "error" for i in issues)
    n_taste = sum(i["severity"] == "taste" for i in issues)
    print(f"{'FAIL' if n_err else 'PASS'} {name}: {n_err} errors, "
          f"{len(issues) - n_err - n_taste} warnings, {n_taste} taste notes "
          f"-> {png}")
    shown = {}
    for i in issues:                      # full list is in the JSON log
        k = (i["severity"], i["check"])
        shown[k] = shown.get(k, 0) + 1
        if shown[k] <= 3:
            print(f"  [{i['severity']}] {i['check']}: {i['detail']}")
    for (sev, chk), n in shown.items():
        if n > 3:
            print(f"  [{sev}] {chk}: ... {n - 3} more (see _audit/{name}.json)")
    plt.close(fig)
    return issues
