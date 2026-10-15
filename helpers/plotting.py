"""
Project figure helpers for the ENG2440 Assignment 2 notebook.

The general house style (fonts, colours, tick/frame rules, the figure audit) lives in
`helpers/paper_plot_style.py` (copied from Assignment 1). This file adds what 3D MRI needs on top:
cutting an axial, coronal or sagittal plane out of an RAS+ volume in a fixed display convention, drawing
reference / prediction contours and true-positive / false-positive / false-negative error maps, and
styling notebook tables.

Display convention (stated in every scan caption): radiological. Axial and coronal planes show the
patient's right on the viewer's left; sagittal planes show anterior on the left; superior / anterior is up.
Orientation letters are drawn at the panel edges, so the reader never has to trust array index order.

Nothing in this file touches the data or the models.
"""
import matplotlib as mpl
import matplotlib.patches as mpatches
import numpy as np
from IPython.display import display

from helpers.paper_plot_style import *            # noqa: F401,F403  (house style + audit)
from helpers.paper_plot_style import (GOLD, RED, PETROL, SLATE, INK, MUTED, RULE, TICK_PT,
                                      save_fig, text_color)

REF = GOLD                     # reference contours drawn on scans
PRED = RED                     # prediction contours
# Error-map colours (chosen by the author from three options): correct tumour recedes as light grey, errors stand
# out; the three differ in lightness, so they stay distinct in grayscale and for colour-blind readers.
ERR = {"TP": "#E8E8E8", "FP": "#D7301F", "FN": "#2E9BD6"}


def tint(color, strength):
    """Mix `color` with white: strength 1 gives the full colour, 0 gives white (quiet fills, shaded cells)."""
    return tuple(1 - strength * (1 - c) for c in mpl.colors.to_rgb(color))


def readable_on(background):
    """White or dark ink, whichever contrasts more with `background` (WCAG relative luminance)."""
    def luminance(color):
        linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in mpl.colors.to_rgb(color)]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
    bg = luminance(background)
    return "white" if (1.05) / (bg + 0.05) > (bg + 0.05) / (luminance(INK) + 0.05) else INK


def save_and_show(fig, name, width="full", rel=1.0):
    """Audit + save the figure (vector PDF and preview PNG under figures/), then show it in the notebook."""
    save_fig(fig, name, width=width, rel=rel)
    display(fig)


def place_text(ax, text, prefer=(0.03, 0.97), **kw):
    """Put `text` in an empty part of `ax`, as close as possible to `prefer` (axes fractions, 0-1).

    Candidate positions on a 15 x 15 grid are tried from the nearest to the farthest; the first one whose
    text box stays inside the axes and touches no plotted line, no bar and no other text is kept.
    This uses the same geometry as the figure audit, so labels never need hand-tuned coordinates.
    """
    from helpers.paper_plot_style import _line_pts
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = [p for line in ax.lines for p in _line_pts(ax, line)]
    boxes = [t.get_window_extent(renderer) for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    boxes += [p.get_window_extent(renderer) for p in ax.patches if isinstance(p, mpatches.Rectangle)]
    grid = [(x, y) for x in np.linspace(0.02, 0.98, 15) for y in np.linspace(0.03, 0.97, 15)]
    grid.sort(key=lambda xy: (xy[0] - prefer[0]) ** 2 + (xy[1] - prefer[1]) ** 2)
    kw = {"fontsize": TICK_PT, "color": INK, **kw}
    for x, y in grid:
        t = ax.text(x, y, text, transform=ax.transAxes, ha="left" if x < 0.5 else "right",
                    va="top" if y > 0.5 else "bottom", **kw)
        bb = t.get_window_extent(renderer)
        inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
        free = not any(bb.padded(2).contains(px, py) for px, py in points) and not any(bb.overlaps(b) for b in boxes)
        if inside and free:
            return t
        t.remove()
    print(f"place_text: no empty spot for {text[:30]!r}; placed at the preferred position")
    return ax.text(*prefer, text, transform=ax.transAxes, ha="left" if prefer[0] < 0.5 else "right",
                   va="top" if prefer[1] > 0.5 else "bottom", **kw)



# ------------------------------------------------------------------- planes --
PLANES = ("axial", "coronal", "sagittal")
# letters at (left, right, top, bottom) of a displayed plane, radiological convention
EDGE_LETTERS = {"axial": ("R", "L", "A", "P"), "coronal": ("R", "L", "S", "I"), "sagittal": ("A", "P", "S", "I")}


def plane(volume, which, index):
    """2D display array of one plane of an RAS+ volume [X, Y, Z] (use with origin='lower').

    axial    z = index: columns run patient right -> left, rows posterior -> anterior
    coronal  y = index: columns right -> left, rows inferior -> superior
    sagittal x = index: columns anterior -> posterior, rows inferior -> superior
    """
    if which == "axial":
        return volume[::-1, :, index].T
    if which == "coronal":
        return volume[::-1, index, :].T
    if which == "sagittal":
        return volume[index, ::-1, :].T
    raise ValueError(which)


def plane_index(point, which):
    """The array index of the plane through a voxel `point` (x, y, z)."""
    return int(point[{"axial": 2, "coronal": 1, "sagittal": 0}[which]])


def show_plane(ax, image2d, window=None, letters=None, tag=None, title=None, origin="lower"):
    """Grey-scale MRI plane, no axes, optional orientation letters and a corner tag.

    Voxels that are exactly 0 are outside the brain (also after z-scoring, which keeps the background at 0) and are
    drawn black, so a normalised image looks like the raw one."""
    lo, hi = window if window is not None else np.percentile(image2d[image2d != 0], [0.5, 99.5]) \
        if (image2d != 0).any() else (0, 1)
    cmap = mpl.colormaps["gray"].with_extremes(bad="black")
    ax.imshow(np.ma.masked_equal(image2d, 0), cmap=cmap, vmin=lo, vmax=hi, origin=origin, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if letters:
        l, r, t, b = letters
        kw = dict(transform=ax.transAxes, fontsize=TICK_PT - 1, color="white", fontweight="bold")
        ax.text(0.02, 0.5, l, ha="left", va="center", **kw)
        ax.text(0.98, 0.5, r, ha="right", va="center", **kw)
        ax.text(0.5, 0.98, t, ha="center", va="top", **kw)
        ax.text(0.5, 0.02, b, ha="center", va="bottom", **kw)
    if tag:
        ax.text(0.03, 0.97, tag, transform=ax.transAxes, fontsize=TICK_PT, color=INK, ha="left", va="top",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))
    if title:
        ax.set_title(title, fontsize=TICK_PT, pad=2)


def frame(img2d, centre_rc, size_rc):
    """Cut a fixed-size window centred on (row, column) out of a 2D display array, zero-padding outside."""
    out = np.zeros(size_rc, dtype=img2d.dtype)
    r0, c0 = centre_rc[0] - size_rc[0] // 2, centre_rc[1] - size_rc[1] // 2
    rs, cs = slice(max(r0, 0), min(r0 + size_rc[0], img2d.shape[0])), slice(max(c0, 0), min(c0 + size_rc[1], img2d.shape[1]))
    out[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0] = img2d[rs, cs]
    return out


def display_centre(point, which, shape):
    """(row, column) of a voxel point in the display array returned by `plane` for this plane type."""
    x, y, z = point
    return {"axial": (y, shape[0] - 1 - x), "coronal": (z, shape[0] - 1 - x), "sagittal": (z, shape[1] - 1 - y)}[which]


def contour(ax, mask2d, color, lw=0.7, ls="-", origin="lower"):
    """Outline of a binary 2D mask."""
    if mask2d.any():
        ax.contour(mask2d.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls,
                   origin=origin)


def error_overlay(ax, ref2d, pred2d, colors=None, alpha=0.75):
    """Colour TP / FP / FN voxels of a plane (true negatives stay transparent)."""
    colors = colors or ERR
    rgba = np.zeros(ref2d.shape + (4,))
    ref2d, pred2d = ref2d.astype(bool), pred2d.astype(bool)
    for key, m in (("TP", ref2d & pred2d), ("FP", ~ref2d & pred2d), ("FN", ref2d & ~pred2d)):
        rgba[m, :3] = mpl.colors.to_rgb(colors[key])
        rgba[m, 3] = alpha
    ax.imshow(rgba, origin="lower", interpolation="nearest")


def error_handles(colors=None):
    colors = colors or ERR
    names = {"TP": "true positive", "FP": "false positive", "FN": "false negative"}
    return [mpatches.Patch(facecolor=colors[k], edgecolor="none", label=names[k]) for k in ("TP", "FP", "FN")]


def crop_box(mask, margin=8):
    """Slices of the bounding box of a 3D mask plus a margin (for zoomed-in panels)."""
    idx = np.nonzero(mask)
    return tuple(slice(max(int(i.min()) - margin, 0), int(i.max()) + margin + 1) for i in idx)


def style_table(df, precision=3, caption=None):
    """Render a pandas DataFrame as a clean HTML table in the notebook."""
    styler = (df.style
              .format(precision=precision, thousands=",", na_rep="–")
              .set_table_styles([
                  {"selector": "th", "props": [("background", "#f4f4f4"), ("color", INK),
                                                ("font-weight", "600"), ("text-align", "center"),
                                                ("padding", "6px 10px"), ("border-bottom", f"1px solid {RULE}")]},
                  {"selector": "td", "props": [("padding", "5px 10px"), ("text-align", "center"),
                                                ("border-bottom", "1px solid #ebebeb"),
                                                ("font-variant-numeric", "tabular-nums")]},
                  {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                                     ("font-weight", "600"), ("color", INK),
                                                     ("padding-bottom", "6px")]},
              ]))
    if caption:
        styler = styler.set_caption(caption)
    return styler


def label_point(ax, xy, text, radii=(9, 14, 20, 28, 38, 50, 64, 80), **kw):
    """Label one data point with `text` in nearby empty space, joined to it by a thin leader line.

    Every candidate position on rings around the point (in points) whose text box stays inside the axes and touches
    no marker, line or other text is scored by the leader's length plus a penalty for each data point or line sample
    the leader passes close to; the lowest score wins. Checks run on a freshly drawn figure, so layout changes made
    by earlier labels are taken into account.
    """
    from helpers.paper_plot_style import _line_pts
    fig = ax.figure
    kw = {"fontsize": TICK_PT, "color": INK, **kw}
    px_per_pt = fig.dpi / 72

    def geometry():
        fig.canvas.draw()
        pts = [c.get_offset_transform().transform(c.get_offsets()) for c in ax.collections
               if hasattr(c, "get_offsets") and len(c.get_offsets())]
        line_pts = [p for line in ax.lines for p in _line_pts(ax, line)]
        for o in ax.texts:                                   # leader lines drawn earlier are obstacles too
            if getattr(o, "arrow_patch", None) is not None:
                verts = o.arrow_patch.get_path().vertices
                verts = o.arrow_patch.get_transform().transform(verts) if len(verts) else verts
                for a_, b_ in zip(verts[:-1], verts[1:]):
                    line_pts += [a_ + (b_ - a_) * f for f in np.linspace(0, 1, 12)]
        pts = np.concatenate(pts + ([np.array(line_pts)] if line_pts else [])) if pts or line_pts else np.empty((0, 2))
        return fig.canvas.get_renderer(), pts

    renderer, pts = geometry()
    target = ax.transData.transform(xy)
    others = [o.get_window_extent(renderer) for o in fig.findobj(mpl.text.Text) if o.get_visible() and o.get_text().strip()]
    best = None
    for r in radii:
        for ang in np.deg2rad(np.arange(0, 360, 22.5)):
            dx, dy = r * np.cos(ang), r * np.sin(ang)
            ha = "left" if dx > 1 else "right" if dx < -1 else "center"
            va = "bottom" if dy > 1 else "top" if dy < -1 else "center"
            t = ax.text(*xy, text, ha=ha, va=va, transform=ax.transData + mpl.transforms.ScaledTranslation(
                dx / 72, dy / 72, fig.dpi_scale_trans), **kw)
            bb = t.get_window_extent(renderer)
            t.remove()
            inside = ax.bbox.contains(bb.x0, bb.y0) and ax.bbox.contains(bb.x1, bb.y1)
            pad = bb.padded(3 * px_per_pt)
            if not inside or any(pad.contains(px, py) for px, py in pts) or any(bb.padded(2 * px_per_pt).overlaps(o) for o in others):
                continue
            # leader from the marker edge to the text anchor: penalise data it passes within 3 pt of
            end = target + np.array([dx, dy]) * px_per_pt
            seg = end - target
            u = np.clip(((pts - target) @ seg) / max(seg @ seg, 1e-9), 0, 1)
            near = np.linalg.norm(pts - (target + u[:, None] * seg), axis=1) < 3 * px_per_pt
            far_from_target = np.linalg.norm(pts - target, axis=1) > 6 * px_per_pt
            # a leader must not run through another label
            crosses = any(o.padded(1.5 * px_per_pt).contains(*(target + seg * f)) for o in others for f in np.linspace(0.15, 0.9, 12))
            if crosses:
                continue
            score = 2 * r + 25 * int((near & far_from_target).sum())
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va)
    if best is None:
        print(f"label_point: no empty spot for {text!r}")
        t = ax.annotate(text, xy, xytext=(6, 6), textcoords="offset points", **kw)
        return t, None, 1e6
    score, dx, dy, ha, va = best
    t = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **kw)
    leader = ax.annotate("", xy, xytext=(dx, dy), textcoords="offset points",   # the leader, as a text-free annotation
                         arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1.5, shrinkB=4.5))
    return t, leader, score


def label_points(ax, items, **kw):
    """Label several points [(xy, text), ...], trying every placement order and keeping the one whose labels sit
    closest to their points with the fewest leader crossings (at most 4 labels: 24 orders)."""
    from itertools import permutations
    best = None
    for order in permutations(range(len(items))):
        placed, total = [], 0.0
        for i in order:
            t, leader, score = label_point(ax, items[i][0], items[i][1], **kw)
            placed += [t] + ([leader] if leader is not None else [])
            total += score
        if best is None or total < best[0]:
            best = (total, order)
        for artist in placed:
            artist.remove()
    for i in best[1]:
        label_point(ax, items[i][0], items[i][1], **kw)
