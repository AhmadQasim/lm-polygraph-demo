#!/usr/bin/env python3
"""Shared palette, shape helpers and chrome for the slide-building scripts.

**Vendored verbatim from ~/projects/slides/scripts/deck_common.py.** Keep it a copy, not a
fork: if it changes there, re-copy it.

Every generated shape is named with GEN_PREFIX so a rebuild can strip what it added and
re-add it. That is what makes the build re-runnable rather than append-only.
"""

from __future__ import annotations

from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls
from pptx.util import Inches, Pt

# --- the official Helmholtz palette --------------------------------------------------
# Source of truth is beamer-template-2024/theme/beamercolorthemehelmholtzai.sty, which is
# what hgf_style.py uses for figures. The pptx theme carries the same colours rounded by
# one or two units (dk1 #002764 vs #002864, accent1 #10C5FF vs #14C8FF); the difference is
# imperceptible, and using one set everywhere is worth more than matching the rounding.
NAVY = RGBColor(0x00, 0x28, 0x64)   # hgfdarkblue  RGB 0, 40, 100
BLUE = RGBColor(0x00, 0x5A, 0xA0)   # hgfblue      RGB 0, 90, 160
CYAN = RGBColor(0x14, 0xC8, 0xFF)   # hgflightblue RGB 20, 200, 255
MINT = RGBColor(0x05, 0xE5, 0xBA)   # hgfmint      RGB 5, 229, 186
PALE = RGBColor(0xCD, 0xEE, 0xFB)   # hgfhighlight RGB 205, 238, 251
GHOST = RGBColor(0xEC, 0xFB, 0xFD)  # hgfpale      RGB 236, 251, 253
GREEN = RGBColor(0x8C, 0xB4, 0x23)  # hgfgreen     RGB 140, 180, 35
HEALTH = RGBColor(0xD2, 0x32, 0x64) # hgfhealth    RGB 210, 50, 100 — this project's field
GREY = RGBColor(0xE9, 0xE7, 0xEA)   # light rule / inactive fill
MUTE = RGBColor(0x5A, 0x69, 0x6E)   # hgfgray      RGB 90, 105, 110 — secondary text
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# --- accent palette, mirroring hgf_style.ACCENTS --------------------------------------
# Read off the user's PowerPoint "Recent Colors" strip, not sampled — see hgf_style.
PEACH = RGBColor(0xE8, 0xA8, 0x7C)
ORANGE = RGBColor(0xED, 0x7D, 0x31)
PURPLE = RGBColor(0x70, 0x30, 0xA0)
PINK = RGBColor(0xEE, 0x72, 0xB8)
LIME = RGBColor(0xDC, 0xE0, 0x35)
GOLD = RGBColor(0xF2, 0xC5, 0x7C)
ACCENT_CYCLE = [ORANGE, PURPLE, PINK, LIME, PEACH, GOLD]

# Corner radius for rounded rectangles, in *figure units* (the coordinate system the
# shapes_*.py modules draw in, 10 units to the inch at figure scale).
#
# Absolute, not proportional, and that matters: PowerPoint's roundRect adjustment is a
# fraction of the shorter side, so one adjustment value gives a 3 mm radius on a small
# row and a blob on a large panel. hgf_pptx.slab converts this to a per-shape adjustment
# so every corner in the deck has the same curvature.
ROUND_R = 1.1

# The brand faces (Hermann Bold, HalvarMittel) ship as OTFs in beamer-template-2024/fonts
# but are NOT installed on this machine, so naming them here would make PowerPoint
# substitute something arbitrary. Figures rasterise the real fonts and are unaffected.
# Install the OTFs via Font Book and set FONT = "Helmholtz Halvar Mittelschrift" to get
# brand typography in native pptx text too.
FONT = "Aptos"
FONT_HEADING = FONT

GEN_PREFIX = "GEN_"

# Slide geometry, inches. The master puts the logo at x 0.14-2.54, the slide-number
# placeholder at x 9.99+, the title at y 0.78-1.81 and a footer bar at y 7.01.
SLIDE_W, SLIDE_H = 13.333, 7.5
BODY_L, BODY_R = 0.92, 12.42
TITLE_Y, TITLE_H = 0.78, 1.06


def clear_generated(slide, prefix: str = GEN_PREFIX) -> int:
    """Remove shapes a previous build added, so re-running never stacks."""
    removed = 0
    for shape in list(slide.shapes):
        if shape.name.startswith(prefix):
            shape._element.getparent().remove(shape._element)
            removed += 1
    return removed


def style_text(shape, lines, size=11, color=NAVY, bold=False, align=PP_ALIGN.LEFT,
               anchor=MSO_ANCHOR.TOP, wrap=True, spacing=None):
    """Fill a text frame. `lines` is a string, or a list of str / (text, size, colour, bold)."""
    tf = shape.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.04)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align
        if spacing and i:
            para.space_before = Pt(spacing)
        text, sz, col, bd = line if isinstance(line, tuple) else (line, size, color, bold)
        run = para.add_run()
        run.text = text
        run.font.size = Pt(sz)
        run.font.bold = bd
        run.font.color.rgb = col
        run.font.name = FONT
    return shape


def label(slide, x, y, w, h, lines, name="TEXT", **kw):
    box_ = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    box_.name = f"{GEN_PREFIX}{name}"
    style_text(box_, lines, **kw)
    return box_


def box(slide, x, y, w, h, fill=PALE, line=None, name="BOX",
        shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.12):
    sp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.name = f"{GEN_PREFIX}{name}"
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid()
        sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(1.0)
    sp.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            sp.adjustments[0] = radius
        except Exception:
            pass
    sp.text_frame.text = ""
    return sp


def picture(slide, path, x, y, w=None, h=None, name="PIC"):
    pic = slide.shapes.add_picture(
        str(path), Inches(x), Inches(y),
        Inches(w) if w else None, Inches(h) if h else None,
    )
    pic.name = f"{GEN_PREFIX}{name}"
    return pic


def set_title(slide, text, subtitle=None):
    """Fill the layout's own title placeholder, so branding and position are inherited."""
    if slide.shapes.title is None:
        label(slide, BODY_L, TITLE_Y, BODY_R - BODY_L, TITLE_H,
              [(text, 20, NAVY, True)], name="TITLE")
        return
    tf = slide.shapes.title.text_frame
    tf.text = text
    if subtitle:
        para = tf.add_paragraph()
        run = para.add_run()
        run.text = subtitle
        run.font.size = Pt(12)
        run.font.color.rgb = BLUE
        run.font.name = FONT


# ------------------------------------------------------------------------ slide numbers
_FLD = (
    '<a:fld %s id="{1D2E3F40-0000-4000-9000-00000000%04d}" type="slidenum">'
    '<a:rPr lang="en-US" sz="900"><a:solidFill><a:srgbClr val="7A8896"/></a:solidFill>'
    '<a:latin typeface="Aptos"/></a:rPr><a:t>%d</a:t></a:fld>'
)


def add_slide_number(slide, cached: int, seq: int):
    """A real slidenum *field*, so PowerPoint keeps it correct when slides move.

    The <a:t> is only a cached value; PowerPoint recomputes it on open. A plain string
    would go stale the first time a slide is inserted.
    """
    box_ = slide.shapes.add_textbox(
        Inches(11.30), Inches(0.28), Inches(1.30), Inches(0.30)
    )
    box_.name = f"{GEN_PREFIX}SLIDENUM"
    tf = box_.text_frame
    tf.word_wrap = False
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.RIGHT
    para._p.append(parse_xml(_FLD % (nsdecls("a"), seq, cached)))
    return box_


def number_slides(prs, skip_layouts=("title slide",)) -> int:
    """Number every slide except the ones whose layout is in `skip_layouts`."""
    stamped = 0
    for i, slide in enumerate(prs.slides):
        for shape in list(slide.shapes):
            if shape.name == f"{GEN_PREFIX}SLIDENUM":
                shape._element.getparent().remove(shape._element)
        name = (slide.slide_layout.name or "").lower()
        if any(k in name for k in skip_layouts):
            continue
        add_slide_number(slide, cached=i + 1, seq=i + 1)
        stamped += 1
    return stamped


# ------------------------------------------------------------------- locating a slide
def find_chapter_slide(prs, chapter: str, nth: int = 0) -> int:
    """Index of the `nth` content slide inside the chapter whose divider title matches.

    Slide modules address their target by chapter rather than by absolute index, so
    inserting a slide earlier in the deck does not silently move a figure into the wrong
    section. Matching is case-insensitive and ignores surrounding whitespace, because the
    divider titles are typed by hand in PowerPoint.
    """
    import nav_bar  # local import: deck_common must not require it at module level

    groups = nav_bar.detect_chapters(prs)
    wanted = " ".join(chapter.lower().split())
    for group in groups:
        if group["divider"] is None:
            continue
        if " ".join((group["name"] or "").lower().split()) != wanted:
            continue
        body = [s for s in group["slides"] if s != group["divider"]]
        if nth >= len(body):
            raise IndexError(
                f"chapter {chapter!r} has {len(body)} content slide(s); "
                f"asked for #{nth + 1}. Add an empty slide after its divider."
            )
        return body[nth]
    names = [g["name"] for g in groups if g["divider"] is not None]
    raise LookupError(f"no chapter titled {chapter!r}; found {names}")


# --------------------------------------------------------------- the figure-slide layout
FOOTER_Y = 7.01   # the master's footer bar; nothing may cross it
LEAD_Y = 1.80
LEAD_H = 0.32
# Where a figure starts when the slide has no lead line. 1.89 rather than 2.14 is the
# hand-edit of 2026-09-08: with the lead gone the figure was pulled up to just under the
# title, and the three lead-less slides were aligned to each other there.
FIG_TOP = 1.89
# The full body width. 10.80 left 0.70 in unused, which cost every figure about 6% of
# its type size for nothing.
FIG_MAX_W = 11.50


def layer_paths(figure):
    """A figure's click layers, or the figure itself when it has none.

    `scripts/hgf_layers.py` writes `<stem>_l0.png`, `_l1.png`, ... next to the figure.
    Resolved at build time rather than named in the slide script, so a figure that gains
    or loses a panel does not need its slide edited.
    """
    figure = Path(figure)
    stem = figure.with_suffix("")
    layers = sorted(stem.parent.glob(f"{stem.name}_l*.png"),
                    key=lambda p: int(p.stem.rsplit("_l", 1)[1]))
    return layers or [figure]


def figure_slide(slide, title, figure, lead=None, caption=None, subtitle=None,
                 clicks=False, verbose=True):
    """Title, an optional lead line, a centred figure, an optional caption.

    The figure is scaled to fit *both* the width and the space left above the footer bar,
    so a taller figure shrinks rather than running off the slide. Idempotent: clears the
    shapes a previous build added before placing new ones.

    `figure` may be a **list** of transparent PNGs rendered on one canvas, which is how a
    figure gets a click-by-click build: they are placed at the same box, in order, named
    GEN_FIGURE, GEN_FIGURE_1, ... so an animation can reveal them one at a time. The
    first one sets the box, and they must therefore be the same size — see
    `fig_study.build(layer=...)` for how that is guaranteed.
    """
    from PIL import Image  # only needed to read the aspect ratio

    if isinstance(figure, (list, tuple)):
        layers = [Path(f) for f in figure]
    else:
        layers = layer_paths(figure) if clicks else [Path(figure)]
    figure = layers[0]
    for path in layers:
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing — run its scripts/fig_*.py")

    clear_generated(slide)
    set_title(slide, title, subtitle)

    top = FIG_TOP
    if lead:
        label(slide, BODY_L, LEAD_Y, BODY_R - BODY_L, LEAD_H, [(lead, 11, BLUE, True)],
              name="LEAD", align=PP_ALIGN.LEFT)
        top = LEAD_Y + LEAD_H + 0.06

    caption_h = 0.52 if caption else 0.0
    caption_gap = 0.12 if caption else 0.0
    avail_h = FOOTER_Y - 0.06 - top - caption_h - caption_gap

    with Image.open(figure) as im:
        aspect = im.width / im.height

    w = min(FIG_MAX_W, avail_h * aspect)
    h = w / aspect
    x = (BODY_L + BODY_R) / 2 - w / 2
    for i, path in enumerate(layers):
        picture(slide, path, x, top, w=w, name="FIGURE" if i == 0 else f"FIGURE_{i}")

    if caption:
        label(slide, BODY_L, top + h + caption_gap, BODY_R - BODY_L, caption_h,
              [(caption, 10.5, NAVY, False)], name="CAPTION", align=PP_ALIGN.LEFT)

    bottom = top + h + caption_gap + caption_h
    if bottom > FOOTER_Y - 0.03:
        raise AssertionError(f"content runs to {bottom:.2f}in, into the footer bar")

    if clicks and len(layers) > 1:
        from hgf_anim import appear_on_click

        placed = {sh.name: sh for sh in slide.shapes}
        appear_on_click(slide, [[placed[f"GEN_FIGURE_{i}"]]
                                for i in range(1, len(layers))], verbose=verbose)
    return slide


def shapes_slide(slide, title, draw, fig_w, fig_h, lead=None, caption=None,
                 subtitle=None, verbose=True):
    """Same layout as figure_slide, but the figure is drawn as native shapes.

    `draw(slide, frame)` receives an hgf_pptx.Frame mapping the figure's own units onto
    the slide, so a native port can reuse the matplotlib version's coordinates unchanged.
    """
    import hgf_pptx

    clear_generated(slide)
    set_title(slide, title, subtitle)

    top = FIG_TOP
    if lead:
        label(slide, BODY_L, LEAD_Y, BODY_R - BODY_L, LEAD_H, [(lead, 11, BLUE, True)],
              name="LEAD", align=PP_ALIGN.LEFT)
        top = LEAD_Y + LEAD_H + 0.06

    caption_h = 0.52 if caption else 0.0
    caption_gap = 0.12 if caption else 0.0
    avail_h = FOOTER_Y - 0.06 - top - caption_h - caption_gap

    aspect = fig_w / fig_h
    w = min(FIG_MAX_W, avail_h * aspect)
    h = w / aspect
    x = (BODY_L + BODY_R) / 2 - w / 2

    frame = hgf_pptx.Frame(x, top, w, fig_w, fig_h)
    draw(slide, frame, verbose=verbose)

    if caption:
        label(slide, BODY_L, top + h + caption_gap, BODY_R - BODY_L, caption_h,
              [(caption, 10.5, NAVY, False)], name="CAPTION", align=PP_ALIGN.LEFT)

    bottom = top + h + caption_gap + caption_h
    if bottom > FOOTER_Y - 0.03:
        raise AssertionError(f"content runs to {bottom:.2f}in, into the footer bar")
    return slide


# ------------------------------------------------------------------- inserting a slide
DEFAULT_CONTENT_LAYOUT = "Title Only"


def insert_slide(prs, after: int, layout_name: str = DEFAULT_CONTENT_LAYOUT):
    """Add a slide immediately after slide `after` (1-based, as the template numbers it).

    The template holds only the title slide and the chapter dividers — the content slides
    are created here. `after` is an *absolute* slide number in the template, which is the
    convention adopted 2026-09-07: it is stable against anything the builder itself
    inserts, unlike a running position.
    """
    want = layout_name.lower()
    layout = next((l for l in prs.slide_layouts if (l.name or "").lower() == want), None)
    if layout is None:
        names = [l.name for l in prs.slide_layouts]
        raise LookupError(f"no layout named {layout_name!r}; have {names}")
    slide = prs.slides.add_slide(layout)          # appended
    lst = prs.slides._sldIdLst
    element = list(lst)[-1]
    lst.remove(element)
    lst.insert(after, element)                    # "after slide N" == 0-based index N
    return slide
