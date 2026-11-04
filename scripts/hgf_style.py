#!/usr/bin/env python3
"""Helmholtz house style for every matplotlib figure in this deck.

**Vendored from ~/projects/slides/scripts/hgf_style.py.** Copied verbatim except
for FONT_DIR, which points at this repo's own `fonts/` rather than the Beamer
template submodule. Keep it a copy, not a fork: if the house style changes there,
re-copy and re-apply the FONT_DIR line. The deck-specific helpers (likert, the
vitiligo ROLE assignments) are kept rather than pruned so a diff against the
source stays readable.

Import and call `use_hgf()` before drawing. It registers the brand OTFs that ship with
`beamer-template-2024/fonts/` and sets rcParams so a plain matplotlib call already comes
out on-brand — the fonts are rasterised into the PNG, so they render correctly whether or
not anyone has them installed.

    from hgf_style import use_hgf, HGF, hermann, halvar, slab, chevron_slab
    use_hgf()

The visual language, taken from `theme/beamerouterthemehelmholtzai.sty` rather than
invented: **square corners, flat fills, 45-degree cuts.** The theme's signature shapes are
the stepped "slider" bars along the slide edges and the nested corner chevrons — both are
angled polygons, and neither is a rounded rectangle. Headings are Hermann Bold; everything
else is HalvarMittel.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle

FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

# --- the exact palette from theme/beamercolorthemehelmholtzai.sty ----------------------
HGF = {
    "blue": "#005AA0",        # RGB 0, 90, 160    — primary
    "darkblue": "#002864",    # RGB 0, 40, 100    — text, dark blocks
    "lightblue": "#14C8FF",   # RGB 20, 200, 255  — accent
    "green": "#8CB423",       # RGB 140, 180, 35
    "gray": "#5A696E",        # RGB 90, 105, 110  — secondary text
    "highlight": "#CDEEFB",   # RGB 205, 238, 251 — fills
    "mint": "#05E5BA",        # RGB 5, 229, 186
    "pale": "#ECFBFD",        # RGB 236, 251, 253 — panel grounds
    # research fields
    "aerospace": "#50C8AA",
    "earth": "#326469",
    "energy": "#FFD228",
    "health": "#D23264",      # this project's field
    "information": "#A0235A",
    "matter": "#F0781E",
    "white": "#FFFFFF",
}

# Convenience aliases, so figure code reads cleanly.
BLUE = HGF["blue"]
DARK = HGF["darkblue"]
LIGHT = HGF["lightblue"]
GRAY = HGF["gray"]
HIGHLIGHT = HGF["highlight"]
MINT = HGF["mint"]
PALE = HGF["pale"]
HEALTH = HGF["health"]
WHITE = HGF["white"]

# --- accent palette -------------------------------------------------------------------
# Beyond the brand blues, for figures that need more colour. Derived from the user's
# PowerPoint "Recent Colors" strip.
#
# **Settled 2026-09-08 — this palette is final; stop revisiting it.**
#
# These six are the values every figure in the deck was drawn and reviewed against. They
# began as reads off a screenshot of the user's PowerPoint "Recent Colors" strip, so they
# are not byte-identical to those swatches. That was checked rather than assumed: the
# exact stored values were recovered from a slide the user filled from the swatches
# themselves, and they are
#
#     orange #FF8200   purple #832CB0   pink #FD79CA
#     lime   #EBE300   peach  #FDB477   gold #FFCD79
#
# — every one of them a little bluer here than there, which is the signature of a
# Display-P3 screenshot read as sRGB. **The user's decision is to keep the values below**,
# because the deck is composed around them and the brighter swatch orange costs contrast
# where orange sets type (2.49:1 against white, from 2.77:1). The exact hexes are kept
# here only so nobody has to go and measure them a third time.
ACCENTS = {
    "peach": "#E8A87C",
    "orange": "#ED7D31",
    "purple": "#7030A0",
    "pink": "#EE72B8",
    "lime": "#DCE035",
    "gold": "#F2C57C",
}
# Chart-level additions from the Claude Design spec (styles/chart_style_claude_design.html).
# CYAN_LIGHT and BLUE_MID are sequential-ramp stops; ORANGE_DARK is the dark pole of the
# diverging ramp and the only orange dark enough to set type in.
ACCENTS.update({
    "cyan_light": "#7FDEFF",
    "blue_mid": "#0B87C8",
    "orange_dark": "#A34A12",
})
CYAN_LIGHT = ACCENTS["cyan_light"]
BLUE_MID = ACCENTS["blue_mid"]
ORANGE_DARK = ACCENTS["orange_dark"]

PEACH = ACCENTS["peach"]
ORANGE = ACCENTS["orange"]
PURPLE = ACCENTS["purple"]
PINK = ACCENTS["pink"]
LIME = ACCENTS["lime"]
GOLD = ACCENTS["gold"]

# A categorical order that keeps adjacent series distinguishable.
ACCENT_CYCLE = [ORANGE, PURPLE, PINK, LIME, PEACH, GOLD]


# Shades, mirroring the theme's hgfblue10..90 (colour mixed with white).
def shade(hex_color: str, pct: int) -> str:
    """`pct` percent of the colour, the rest white — the theme's hgfXXX<pct> colours."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    f = pct / 100.0
    return "#%02X%02X%02X" % tuple(round(c * f + 255 * (1 - f)) for c in (r, g, b))


_registered = False

# The bundled OTFs do not agree on a family name — HalvarMittel-Rg registers as
# "Helmholtz Halvar Mittelschrift" while HalvarMittel-Bd registers as plain "Halvar
# Mittelschrift". Asking for one family therefore silently loses the other weight, so
# every face is bound by file path instead and the family list is only a fallback.
_FACES = {
    ("halvar", False, False): "HalvarMittel-Rg.otf",
    ("halvar", True, False): "HalvarMittel-Bd.otf",
    ("halvar", False, True): "HalvarMittel-RgSlanted.otf",
    ("halvar", True, True): "HalvarMittel-BdSlanted.otf",
    ("hermann", True, False): "Hermann-Bold.otf",
    ("hermann", False, False): "Hermann-Bold.otf",
    ("mono", False, False): "TexGyreCursor-Regular.otf",
    ("mono", True, False): "TexGyreCursor-Bold.otf",
}

_FAMILY_FALLBACK = [
    "Helmholtz Halvar Mittelschrift",  # HalvarMittel-Rg
    "Halvar Mittelschrift",            # HalvarMittel-Bd and the slanted faces
    "DejaVu Sans",
]

# What rcParams gets, and it is deliberately *not* the list above.
#
# matplotlib probes each family in a list individually, and "Halvar Mittelschrift" holds
# only the bold upright face — the vendor OTFs register regular and bold under different
# family names. So including it makes every process emit
#   findfont: Failed to find font weight normal, now using 700.
# on the first draw, even though resolution then succeeds. Bold axes text is rare, and
# where it is wanted `halvar(..., bold=True)` binds the file directly and never warns.
_RC_FAMILY = ["Helmholtz Halvar Mittelschrift", "DejaVu Sans"]


def register_fonts() -> list[str]:
    """Add the bundled OTFs to matplotlib's font cache. Idempotent."""
    global _registered
    added = []
    if not FONT_DIR.is_dir():
        return added
    for path in sorted(FONT_DIR.glob("*.otf")):
        try:
            font_manager.fontManager.addfont(str(path))
            added.append(path.name)
        except Exception:  # a malformed face should not take the whole figure down
            pass
    _registered = True
    return added


def _face(kind: str, size: float, bold: bool = False, italic: bool = False) -> FontProperties:
    if not _registered:
        register_fonts()
    name = _FACES.get((kind, bold, italic)) or _FACES[(kind, False, False)]
    path = FONT_DIR / name
    if path.exists():
        prop = FontProperties(fname=str(path))
        prop.set_size(size)
        return prop
    return FontProperties(family=_FAMILY_FALLBACK, size=size,
                          weight="bold" if bold else "normal")


def hermann(size: float = 10.0) -> FontProperties:
    """Hermann Bold — headings only, as in the Beamer theme's \\hermann{}."""
    return _face("hermann", size, bold=True)


def halvar(size: float = 10.0, bold: bool = False, italic: bool = False) -> FontProperties:
    """HalvarMittel — body text, the theme's main and sans font."""
    return _face("halvar", size, bold=bold, italic=italic)


def mono(size: float = 10.0, bold: bool = False) -> FontProperties:
    """TeX Gyre Cursor — the theme's monospace face."""
    return _face("mono", size, bold=bold)


def use_hgf(base_size: float = 10.0) -> None:
    """Set rcParams so an unstyled matplotlib call is already on-brand."""
    register_fonts()
    matplotlib.rcParams.update({
        "font.family": _RC_FAMILY,
        "font.size": base_size,
        "text.color": DARK,
        "axes.labelcolor": DARK,
        "axes.edgecolor": GRAY,
        "axes.titlecolor": DARK,
        "xtick.color": GRAY,
        "ytick.color": GRAY,
        # Spec: all spines off — the single bottom baseline is drawn by baseline(),
        # not by a spine, so it can be extended past the plot box.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        # Spec: no gridlines by default. Every plotted value is labelled directly, so a
        # grid only adds junk. Where a chart has >6 categories, one 1 pt gray-22 line per
        # major tick is permitted — never lighter.
        "axes.grid": False,
        "grid.color": shade(GRAY, 22),
        "grid.linewidth": 1.0,
        "ytick.left": False,
        "ytick.labelleft": False,
        "xtick.bottom": False,
        # Categories are drawn by category_label(), not by the x axis, so the tick
        # labels have to go too — hiding only the ticks leaves the numbers behind.
        "xtick.labelbottom": False,
        "lines.linewidth": 2.5,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.0,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "axes.prop_cycle": matplotlib.cycler(
            color=[BLUE, LIGHT, ORANGE, PURPLE, PINK, LIME, MINT, GOLD]
        ),
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": FIGURE_DPI,
    })
    _register_colormaps()


# ------------------------------------------------------------------- the theme's shapes
# Raster resolution for the PNG that gets placed on the slide. 300 is indistinguishable
# from vector on a projector and prints cleanly; these are flat-colour figures, so the
# files stay small (~150 KB).
FIGURE_DPI = 300

# House rule from 2026-09-07: figures use a rounded aesthetic — no harsh corners on
# panels, rows or text boxes. RADIUS is in figure units and is the single dial for it.
# 1.1 to match deck_common.ROUND_R, which the chart spec names as the bar-corner radius,
# so a rounded bar and a rounded panel carry the same curvature.
RADIUS = 1.1


def slab(ax, x, y, w, h, color, z=2, alpha=1.0, radius=None, sharp=False):
    """A flat block. Rounded by default; pass sharp=True for a true rectangle."""
    if sharp:
        ax.add_patch(Rectangle((x, y), w, h, fc=color, ec="none", zorder=z, alpha=alpha))
        return
    r = min(RADIUS if radius is None else radius, w / 2, h / 2)
    ax.add_patch(FancyBboxPatch(
        (x + r, y + r), max(w - 2 * r, 1e-6), max(h - 2 * r, 1e-6),
        boxstyle=f"round,pad={r:.4f},rounding_size={r:.4f}",
        fc=color, ec="none", zorder=z, alpha=alpha))


def chevron_slab(ax, x, y, w, h, color, cut=None, side="right", z=2):
    """A block with one edge cut at 45 degrees — the theme's chevron motif.

    `side` is which edge is angled: "right", "left", or "both".
    """
    cut = h * 0.5 if cut is None else cut
    left_in = cut if side in ("left", "both") else 0.0
    right_in = cut if side in ("right", "both") else 0.0
    pts = [
        (x + left_in, y),
        (x + w, y),
        (x + w - right_in, y + h),
        (x, y + h),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))


def stepped_bar(ax, x, y, w, color, tall=0.9, short=0.45, split=0.25, step=0.45, z=1):
    """The theme's edge slider: a bar that steps down partway along its length."""
    xs = x + w * split
    pts = [
        (x, y), (x, y + tall), (xs, y + tall),
        (xs + step, y + tall - (tall - short)),
        (x + w, y + short), (x + w, y),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))


def corner_chevrons(ax, x, y, size, outer=DARK, inner=LIGHT, z=1, thickness=0.45):
    """The nested 45-degree corner mark used on the Beamer section slides."""
    for i, color in enumerate((outer, inner)):
        o = i * thickness * 1.5
        pts = [
            (x - size + o, y), (x + o, y),
            (x + o, y - size), (x + o - thickness, y - size),
            (x + o - thickness, y - thickness),
            (x - size + o, y - thickness),
        ]
        ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))


def arrow_chevron(ax, x, y, h, color=BLUE, w=None, z=5):
    """A solid chevron pointing right — the theme's way of saying 'then'."""
    w = h * 0.55 if w is None else w
    t = h * 0.30
    pts = [
        (x, y - h / 2), (x + w, y), (x, y + h / 2),
        (x, y + h / 2 - t), (x + w - t * 1.7, y), (x, y - h / 2 + t),
    ]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec="none", zorder=z))


def likert(ax, x, y, w, h=0.9, n=5, filled=None, color=None, gap=None, z=5):
    """A discrete response scale as `n` square ticks — the deck's questionnaires are all
    5-point, and squares are the theme's vocabulary. `filled` highlights one position."""
    color = color or BLUE
    gap = w * 0.055 if gap is None else gap
    tw = (w - gap * (n - 1)) / n
    for i in range(n):
        on = filled is not None and i == filled
        slab(ax, x + i * (tw + gap), y, tw, h,
             color if on else shade(color, 22), z=z)
    return tw


# ------------------------------------------------------------------------------- saving
def save(fig, stem, formats=("png", "svg"), pad=0.04, transparent=True, dpi=None):
    """Write a figure deterministically. **Use this for every figure.**

    Two formats by default, and each has a job:

    * **PNG at 300 dpi** — the copy `deck_common.figure_slide()` places on the slide.
      python-pptx cannot insert SVG (`add_picture` raises `UnidentifiedImageError`), and
      embedding one programmatically means hand-writing the `svgBlip` extension Office
      uses, which is more PowerPoint plumbing than it saves.
    * **SVG with real `<text>`** — not placed automatically. It is there for the case
      where one figure needs hand-tuning: drag it into PowerPoint yourself and use
      *Graphics Format → Convert to Shape* for full native editability, on demand rather
      than by default.

    Deterministic, so a rebuild is a no-op in git: matplotlib stamps a CreationDate into
    PDF and a Date into SVG, and both are cleared here. Without it every rebuild rewrote
    the file with identical drawing but different bytes.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight", "pad_inches": pad, "transparent": transparent}
        if ext in ("pdf", "eps", "ps"):
            kwargs["metadata"] = {"CreationDate": None}
        elif ext == "svg":
            kwargs["metadata"] = {"Date": None}
        if ext in ("png", "jpg", "jpeg", "tif", "tiff"):
            kwargs["dpi"] = dpi or FIGURE_DPI
        fig.savefig(path, **kwargs)
        written.append(path)
    return written


# =====================================================================================
# Chart-level design system
# =====================================================================================
# Implements styles/chart_style_claude_design.html. Three rules underpin all of it:
#
#   1. No axis a label can replace. If every plotted value is labelled, the y axis, its
#      ticks and the grid are junk.
#   2. Weight, not hue, carries the claim. The hues are owned by semantics, so emphasis
#      is type size (7.6 -> 14), stroke weight (1.75 -> 3 pt) and a pale ground.
#   3. Nothing below 7.6, nothing lighter than gray 22. The projector eats the rest.

GRAY22 = shade(GRAY, 22)   # #DBDEDF  gridlines, rules
GRAY45 = shade(GRAY, 45)   # #B5BCBE  null/sham fills, estimated keylines
GRAY85 = shade(GRAY, 85)   # #738084  axis, baseline, tick and caveat text
BLUE45 = shade(BLUE, 45)   # #8CB5D4  diverging cool mid
BLUE60 = shade(BLUE, 60)   # #669CC6  context (non-focus) series
BLUE85 = shade(BLUE, 85)   # #2673AE  second human rater
WASH = shade(HIGHLIGHT, 45)  # #F6FDFE  region washes

#: What each hue *means*. Pick colours from here, never by eye — a reader decodes a hue
#: once and it has to hold across the whole deck.
ROLE = {
    "patient_average": BLUE,          # the per-patient average series
    "pooled_average": LIGHT,          # swatch only — never set type in it
    "model": ORANGE,                  # the model, and the "after" side
    "human": DARK,                    # the human rater, and the "before" side
    "human_second": BLUE85,           # a second human rater
    "endpoint": PURPLE,               # the endpoint, brackets, spread annotation
    "ethics": PINK,                   # ethics-requested series
    "shared_scale": MINT,             # shared patient/physician instrument
    "null": GRAY45,                   # sham / null distribution, at alpha 0.55
    "context": BLUE60,                # non-focus series in small multiples
    "axis": GRAY85,                   # axis, baseline, ticks, caveats
    "claim_ground": HIGHLIGHT,        # filled claim row / focus panel ground
    "band_ground": PALE,              # band ground
}

NULL_ALPHA = 0.55

#: Type roles, in the same units the figure modules already pass to hermann()/halvar().
#: CHART_MIN_SIZE is a hard floor: the 6.2-6.4 tier used elsewhere dies on a projector.
# Rescaled 2026-09-07 to the measured placement. A figure is drawn 124 u wide and placed
# 11.50 in wide, so the scale is 0.927 and a size in figure units reads at ~0.93x on the
# slide. The design spec's original tier (tick 7.6 -> 7.0 pt) was too small on a
# projector; these land body text at ~12 pt, matching the three built diagrams.
TYPE = {
    "tick": 11.0,          # halvar regular            -> ~10.2 pt on slide
    "gloss": 11.0,         # halvar regular, sub-label -> ~10.2 pt
    "category": 13.0,      # hermann caps              -> ~12.1 pt
    "axis_title": 13.0,    # hermann caps              -> ~12.1 pt
    "value": 13.5,         # halvar bold, non-claim    -> ~12.5 pt
    "panel_header": 14.0,  # hermann caps              -> ~13.0 pt
    "claim": 20.0,         # hermann bold, the value carrying the figure -> ~18.5 pt
    "hero": 28.0,          # hermann bold, one figure-carrying numeral   -> ~26 pt
}
CHART_MIN_SIZE = 11.0

#: Stroke weights, in points.
WEIGHT = {
    "data": 2.5,
    "focus": 3.0,
    "context": 1.75,
    "reference": 2.0,
    "bracket": 1.75,
    "baseline": 1.5,
    "grid": 1.0,
    "estimated_edge": 1.5,
}
REFERENCE_DASH = (6, 4)

#: Geometry, in figure units (10 u == 1 in).
# 124 x 44.5 u is the aspect (2.787) that exactly fills the slide's figure band,
# 11.50 x 4.13 in — measured, not chosen. Anything taller is scaled down and loses type
# size; anything shorter leaves the band unfilled. Keep new figures on this canvas.
GEOM = {
    "canvas": (124.0, 44.5),
    "canvas_small_multiples": (124.0, 44.5),
    "margins": (9.0, 4.0, 6.0, 7.0),   # left, right, top, bottom
    "plot": (111.0, 31.5),
    "bar_width_grouped": 4.2,
    "bar_gap_intra": 0.8,
    "bar_width_single": 8.0,
    "cap_width": 1.6,
    "panel": (53.5, 16.2),
    "gutter": 3.7,
}

# --- colormaps ------------------------------------------------------------------------
SEQ_STOPS = [PALE, HIGHLIGHT, CYAN_LIGHT, LIGHT, BLUE_MID, BLUE, DARK]
SEQ_STOPS_DISCRETE = [PALE, HIGHLIGHT, LIGHT, BLUE, DARK]
DIV_STOPS = [ORANGE_DARK, ORANGE, GOLD, "#FFFFFF", BLUE45, BLUE, DARK]


def _register_colormaps() -> None:
    """`hgf_seq` and `hgf_div`. Idempotent."""
    from matplotlib.colors import LinearSegmentedColormap

    for name, stops in (("hgf_seq", SEQ_STOPS),
                        ("hgf_seq_d", SEQ_STOPS_DISCRETE),
                        ("hgf_div", DIV_STOPS)):
        if name in matplotlib.colormaps:
            continue
        cmap = LinearSegmentedColormap.from_list(name, stops)
        matplotlib.colormaps.register(cmap, name=name)
        matplotlib.colormaps.register(cmap.reversed(), name=f"{name}_r")


def darken(color: str) -> str:
    """One step darker, for an error bar against its own bar. Never gray — gray reads
    as 'disabled' here."""
    ladder = {BLUE: DARK, LIGHT: BLUE, ORANGE: ORANGE_DARK, MINT: BLUE,
              PURPLE: DARK, PINK: PURPLE, BLUE60: BLUE, BLUE85: DARK}
    return ladder.get(color, DARK)


# --- canvas ---------------------------------------------------------------------------
def chart_axes(canvas=None, margins=None, dpi=None):
    """A figure and axes at the spec's canvas and margins, in figure units.

    Default 124 x 44 u with 9/4/6/7 margins, giving a 111 x 31 u plot box. Spines and
    ticks are stripped here; add the single baseline with `baseline(ax)`.
    """
    import matplotlib.pyplot as plt

    cw, ch = canvas or GEOM["canvas"]
    ml, mr, mt, mb = margins or GEOM["margins"]
    fig = plt.figure(figsize=(cw / 10, ch / 10), dpi=dpi or FIGURE_DPI)
    ax = fig.add_axes((ml / cw, mb / ch, (cw - ml - mr) / cw, (ch - mt - mb) / ch))
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False, length=0)
    ax.grid(False)
    ax.set_axisbelow(True)
    return fig, ax


def baseline(ax, color=None, lw=None, extend=0.009):
    """The single bottom baseline, extended a little past the plot box. The only spine."""
    ax.plot([0 - extend, 1 + extend], [0, 0], transform=_blend(ax),
            color=color or GRAY85, lw=lw or WEIGHT["baseline"],
            solid_capstyle="butt", clip_on=False, zorder=3)


def _blend(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transAxes, ax.transData)


def _blend_y(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


# --- bars -----------------------------------------------------------------------------
def rounded_bars(ax, xs, heights, color, width=None, radius=None, zorder=4, alpha=1.0):
    """Bars with the top corners rounded at ROUND_R, bottom square.

    Drawn as a FancyBboxPatch that starts `radius` below zero and is clipped away by
    `ax.set_ylim(0, ...)`, so only the top corners survive. Square bars would be the only
    hard corner in the deck besides the header chevron.

    Call `ax.set_ylim(0, top)` before this so the aspect used for the corner is right.
    """
    from matplotlib.patches import FancyBboxPatch

    width = GEOM["bar_width_single"] if width is None else width
    r = RADIUS if radius is None else radius
    xs = list(xs)
    heights = list(heights)

    # A boxstyle radius is applied in data space, so on a non-square axes the corner
    # would come out elliptical. mutation_aspect corrects for the data-per-inch ratio.
    ax.figure.canvas.draw_idle()
    bbox = ax.get_window_extent()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px_per_x = bbox.width / max(x1 - x0, 1e-9)
    px_per_y = bbox.height / max(y1 - y0, 1e-9)
    aspect = px_per_x / px_per_y if px_per_y else 1.0

    r_x = r * (x1 - x0) / GEOM["plot"][0]          # radius in x data units
    patches = []
    for x, h in zip(xs, heights):
        r_y = r_x / aspect
        patch = FancyBboxPatch(
            (x - width_to_data(ax, width) / 2, -r_y),
            width_to_data(ax, width), h + r_y,
            boxstyle=f"round,pad=0,rounding_size={r_x:.6f}",
            mutation_aspect=aspect, fc=color, ec="none", alpha=alpha, zorder=zorder)
        ax.add_patch(patch)
        patches.append(patch)
    return patches


def width_to_data(ax, units):
    """Convert a width in figure units to x data units for the current plot box."""
    x0, x1 = ax.get_xlim()
    return units * (x1 - x0) / GEOM["plot"][0]


def error_bars(ax, xs, heights, err, color, cap_units=None, lw=None):
    """Whiskers in the darkened hue of their own bar."""
    cap = width_to_data(ax, cap_units or GEOM["cap_width"]) / 2
    c = darken(color)
    w = lw or WEIGHT["baseline"]
    for x, h in zip(xs, heights):
        ax.plot([x, x], [h - err, h + err], color=c, lw=w, solid_capstyle="butt",
                zorder=6)
        for y in (h - err, h + err):
            ax.plot([x - cap, x + cap], [y, y], color=c, lw=w, solid_capstyle="butt",
                    zorder=6)


# --- annotation -----------------------------------------------------------------------
def direct_label(ax, x, y, text, claim=False, color=None, dy=6, ha="center",
                 size=None, caps=None):
    """A value label that replaces the y axis.

    Non-claim values are 9.5 Halvar Bold in gray 85; the one or two values carrying the
    figure are promoted to 14 Hermann Bold in darkblue. That promotion — not a different
    hue — is how emphasis works here.
    """
    if claim:
        size = size or TYPE["claim"]
        font = hermann(size)
        color = color or DARK
    else:
        size = size or TYPE["value"]
        font = halvar(size, bold=True)
        color = color or GRAY85
    if size < CHART_MIN_SIZE:
        raise ValueError(f"{size} is below the {CHART_MIN_SIZE} chart floor")
    return ax.annotate(text.upper() if caps else text, (x, y), textcoords="offset points",
                       xytext=(0, dy), ha=ha, va="bottom", color=color,
                       fontproperties=font, clip_on=False, zorder=7)


def series_label(ax, x, y, text, color, size=None, dy=6, ha="center"):
    """A series name set at that curve's own peak, in that curve's hue — so no legend."""
    return ax.annotate(text.upper(), (x, y), textcoords="offset points",
                       xytext=(0, dy), ha=ha, va="bottom", color=color,
                       fontproperties=hermann(size or TYPE["panel_header"]),
                       clip_on=False, zorder=7)


def category_label(ax, x, text, gloss=None, y=-0.012, color=None):
    """A category label in Hermann caps with an optional Halvar gloss beneath it.

    The gloss is what removes the legend: "inter-rater" never has to be spelled out in a
    key if it sits under the category it belongs to.
    """
    tr = _blend_y(ax)
    ax.text(x, y, text.upper(), transform=tr, ha="center", va="top",
            color=color or DARK, fontproperties=hermann(TYPE["category"]),
            clip_on=False, zorder=7)
    if gloss:
        ax.text(x, y - 0.055, gloss, transform=tr, ha="center", va="top",
                color=GRAY85, fontproperties=halvar(TYPE["gloss"]), clip_on=False,
                zorder=7)


def axis_title(ax, text, x=0.0, y=1.0, dy=10):
    """One line of Hermann caps, top left, unboxed. Replaces the axis label."""
    return ax.annotate(text.upper(), (x, y), xycoords="axes fraction",
                       textcoords="offset points", xytext=(0, dy), ha="left",
                       va="bottom", color=DARK,
                       fontproperties=hermann(TYPE["axis_title"]), clip_on=False,
                       zorder=7)


# --- reference line -------------------------------------------------------------------
def reference_line(ax, y, label=None, color=None, lw=None, dash=None, size=None,
                   label_side="right", dy=4):
    """A 2 pt dashed rule with its label at one end.

    Gray by default and deliberately: a floor or a ceiling is *context*, not the claim,
    so it must not compete with the darkblue the claim is set in.
    """
    color = color or GRAY85
    ax.axhline(y, color=color, lw=lw or WEIGHT["reference"],
               ls=(0, dash or REFERENCE_DASH), zorder=3)
    if label:
        x, ha = (1.0, "right") if label_side == "right" else (0.0, "left")
        ax.annotate(label.upper(), (x, y), xycoords=_blend(ax),
                    textcoords="offset points", xytext=(0, dy), ha=ha, va="bottom",
                    color=color, fontproperties=hermann(size or TYPE["tick"]),
                    clip_on=False, zorder=7)


def bracket(ax, x0, x1, y, label=None, color=None, lw=None, tick=None, size=None,
            dy=5, hero=False):
    """A span bracket with end ticks pointing down, and its label above the centre.

    The endpoint colour by default, and it should be the only purple on the plate — a
    bracket is a claim about a relationship, which is what purple means here.
    """
    color = color or ROLE["endpoint"]
    lw = lw or WEIGHT["bracket"]
    drop = (tick or GEOM["cap_width"]) / GEOM["plot"][1] * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.plot([x0, x1], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=6,
            clip_on=False)
    for x in (x0, x1):
        ax.plot([x, x], [y, y - drop], color=color, lw=lw, solid_capstyle="butt",
                zorder=6, clip_on=False)
    if label:
        ax.annotate(label.upper(), ((x0 + x1) / 2, y), textcoords="offset points",
                    xytext=(0, dy), ha="center", va="bottom", color=color,
                    fontproperties=hermann(size or (TYPE["claim"] if hero
                                                    else TYPE["panel_header"])),
                    clip_on=False, zorder=7)


def percentile_marker(ax, x, label, color=None, wash_side="left", size=None):
    """A full-height dashed marker with a labelled top, and a wash on one side.

    Used for the sham null's 95th percentile: the wash makes the null a *zone* rather
    than a line, which is the honest reading.
    """
    color = color or DARK
    ax.axvline(x, color=color, lw=WEIGHT["reference"], ls=(0, REFERENCE_DASH), zorder=5)
    x0, x1 = ax.get_xlim()
    if wash_side:
        span = (x0, x) if wash_side == "left" else (x, x1)
        ax.axvspan(*span, color=WASH, lw=0, zorder=0)
    ax.annotate(label.upper(), (x, 1.0), xycoords=_blend_y(ax),
                textcoords="offset points", xytext=(0, 5), ha="center", va="bottom",
                color=color, fontproperties=hermann(size or TYPE["value"]),
                clip_on=False, zorder=7)


def focus_ground(ax, x0, x1, color=None, bleed=0.0, zorder=0):
    """The pale ground that says "read this one". Not a border — this system has no
    outlines except meaningful keylines."""
    ax.axvspan(x0 - bleed, x1 + bleed, color=color or ROLE["claim_ground"], lw=0,
               zorder=zorder)


# --- badges and chips -----------------------------------------------------------------
BADGE_STYLES = {
    # averaging scope — every metric column names its own, because three averages are in
    # circulation and they differ by up to 0.44 on one patient
    "pooled": dict(fc="#FFFFFF", tc=DARK, ec=None),
    "patient": dict(fc=HIGHLIGHT, tc=DARK, ec=None),
    "macro": dict(fc=ORANGE, tc="#FFFFFF", ec=None),
    # scope caveat — "one patient · pilot", "v8nf", "one fold"
    "caveat": dict(fc=PALE, tc=GRAY85, ec=None),
    # a border, not a colour, so it never competes with a semantic hue
    "estimated": dict(fc="#FFFFFF", tc=GRAY85, ec=GRAY45),
}


def badge(ax, x, y, text, kind="caveat", size=None, ha="left", va="bottom",
          xycoords="axes fraction", pad=(0.9, 0.2)):
    """A fully rounded pill in Hermann caps.

    `kind` picks the semantics: pooled / patient / macro name an averaging scope,
    caveat carries a scope note, estimated flags modelled data.

    Place scope badges **top right or bottom right**. `axis_title()` owns the top left,
    and a badge at (0, 1) will sit on top of it.
    """
    style = BADGE_STYLES[kind]
    size = size or TYPE["tick"]
    if size < CHART_MIN_SIZE:
        raise ValueError(f"{size} is below the {CHART_MIN_SIZE} chart floor")
    boxstyle = f"round,pad={pad[1]/10:.3f},rounding_size={pad[1]/10:.3f}"
    return ax.annotate(
        " ".join(text.upper()) if False else text.upper(), (x, y), xycoords=xycoords,
        ha=ha, va=va, color=style["tc"], fontproperties=hermann(size),
        clip_on=False, zorder=8,
        bbox=dict(boxstyle=boxstyle, fc=style["fc"],
                  ec=style["ec"] or "none",
                  lw=WEIGHT["estimated_edge"] if style["ec"] else 0))


def scope_caveat(ax, text, kind="caveat", x=1.0, y=-0.16, size=None):
    """The scope note that belongs bottom-right of every figure carrying a soft number."""
    return badge(ax, x, y, text, kind=kind, size=size, ha="right", va="top")


ESTIMATED_HATCH = dict(hatch="////", edgecolor=GRAY45, linewidth=0.75)


# --- small multiples ------------------------------------------------------------------
def small_multiples(nrows=2, ncols=2, canvas=None, margins=None, panel=None,
                    gutter=None, dpi=None, sharey=True):
    """A grid of panels on a shared y range, y ticks left column only, x ticks bottom row.

    Returns (fig, axes) with axes as a nested list. Panels carry no frame; mark the focus
    panel with `focus_ground`, never with a border.
    """
    import matplotlib.pyplot as plt

    cw, ch = canvas or GEOM["canvas_small_multiples"]
    ml, mr, mt, mb = margins or GEOM["margins"]
    pw, ph = panel or GEOM["panel"]
    g = GEOM["gutter"] if gutter is None else gutter
    fig = plt.figure(figsize=(cw / 10, ch / 10), dpi=dpi or FIGURE_DPI)
    axes = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            left = ml + c * (pw + g)
            bottom = ch - mt - (r + 1) * ph - r * g
            ax = fig.add_axes((left / cw, bottom / ch, pw / cw, ph / ch),
                              sharey=row[0] if (sharey and row) else
                              (axes[0][0] if (sharey and axes) else None))
            for side in ("top", "right", "left", "bottom"):
                ax.spines[side].set_visible(False)
            ax.tick_params(left=False, labelleft=False, bottom=False, length=0)
            ax.grid(False)
            row.append(ax)
        axes.append(row)
    return fig, axes


def panel_header(ax, text, gloss=None, y=1.0, dy=6):
    """Hermann caps above a small-multiples panel: patient id, then the one fact that
    distinguishes it."""
    label = text.upper() if gloss is None else f"{text.upper()} · {gloss.upper()}"
    return ax.annotate(label, (0.0, y), xycoords="axes fraction",
                       textcoords="offset points", xytext=(0, dy), ha="left",
                       va="bottom", color=DARK,
                       fontproperties=hermann(TYPE["panel_header"]), clip_on=False,
                       zorder=7)
