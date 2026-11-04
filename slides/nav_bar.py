#!/usr/bin/env python3
"""Stamp a chapter navigation bar and side marker onto a Helmholtz PowerPoint deck.

**Vendored verbatim from ~/projects/slides/scripts/nav_bar.py.** Keep it a copy, not a
fork: if it changes there, re-copy it.

PowerPoint has no field that knows "which chapter am I in", so this cannot be live the
way Beamer's \\section{} is. Instead the bar is *generated*: run this after any edit and
it rebuilds itself from the deck's own structure. Idempotent — every shape it adds is
named with a NAV_PREFIX and deleted before the next pass, so re-running never stacks.

Chapters are detected with no configuration file:

  * a slide on a chapter-divider layout starts a new chapter
  * the chapter's name is that divider slide's own title text
  * the chapter contains the divider plus every slide up to the next divider
  * slides before the first divider are front matter and get no bar

So renaming a chapter means editing the divider slide's title; reordering chapters means
dragging slides. Both are what you would do anyway, and the bar follows.

    python scripts/nav_bar.py deck.pptx --dry-run     # show detected structure, write nothing
    python scripts/nav_bar.py deck.pptx               # stamp, saving to deck.nav.pptx
    python scripts/nav_bar.py deck.pptx -o deck.pptx  # stamp in place
    python scripts/nav_bar.py deck.pptx --labels scripts/nav_labels.json

Close PowerPoint first when writing in place.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

# --- brand palette, read from the deck's own theme (theme1.xml clrScheme) -------------
NAVY = RGBColor(0x00, 0x28, 0x64)  # hgfdarkblue
BLUE = RGBColor(0x00, 0x5A, 0xA0)  # accent3 — active pill
CYAN = RGBColor(0x14, 0xC8, 0xFF)  # hgflightblue — within-chapter progress
PALE = RGBColor(0xCD, 0xEE, 0xFB)  # hgfhighlight — visited pills
GREY = RGBColor(0xE9, 0xE7, 0xEA)  # accent5 — unvisited pills
MUTE = RGBColor(0x5A, 0x69, 0x6E)  # hgfgray — inactive pill text
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

# --- geometry, in inches on the 13.333 x 7.5 canvas -----------------------------------
# The master puts the logo at x 0.14-2.54 and the slide-number placeholder at x 9.99+,
# both in the y 0.20-0.62 band. The gap between them is where the bar goes.
NAV_X0, NAV_X1 = 2.70, 9.85
NAV_Y, NAV_H = 0.33, 0.20
NAV_GAP = 0.05
PROGRESS_H = 0.030
PROGRESS_DY = 0.045
SIDE_X, SIDE_CY, SIDE_LEN = 0.34, 3.95, 3.10

NAV_PREFIX = "NAV_"


def _nav_font() -> str:
    """The brand body face when it is installed, else the template's Aptos.

    Only the navigation chrome resolves this. Slide titles, lead lines, captions and slide
    numbers stay on the template's own theme font by explicit decision (2026-09-07).
    """
    try:
        import hgf_pptx
        return hgf_pptx.resolve_fonts(verbose=False)[1]
    except Exception:
        return "Aptos"

# Layouts that begin a chapter. Matched case-insensitively as substrings.
DIVIDER_LAYOUTS = (
    "heading chapter",
    "section header",
    "computing layout",
    "energy_custom",
    "health_custom",
    "space_custom",
    "information_custom",
    "earth_custom",
    "matter_custom",
)
# Layouts that never receive chrome, even inside a chapter.
BARE_LAYOUTS = ("title slide", "agenda")

SECTION_EXT_URI = "{521415D9-36F7-43E2-AB2F-B90AF26B5E84}"
P14 = "http://schemas.microsoft.com/office/powerpoint/2010/main"


# --------------------------------------------------------------------------- detection
def _layout_name(slide) -> str:
    try:
        return (slide.slide_layout.name or "").lower()
    except Exception:
        return ""


def _title_text(slide) -> str:
    """The slide's title, preferring the real title placeholder."""
    try:
        if slide.shapes.title is not None and slide.shapes.title.has_text_frame:
            txt = slide.shapes.title.text_frame.text.strip()
            if txt:
                return " ".join(txt.split())
    except Exception:
        pass
    for shape in slide.shapes:  # fall back to the first non-empty text on the slide
        if shape.name.startswith(NAV_PREFIX) or not shape.has_text_frame:
            continue
        txt = shape.text_frame.text.strip()
        if txt:
            return " ".join(txt.split())
    return ""


def is_divider(slide) -> bool:
    name = _layout_name(slide)
    return any(k in name for k in DIVIDER_LAYOUTS)


def detect_chapters(prs) -> list[dict]:
    """Group slides into chapters. Returns [{name, slides: [idx, ...]}, ...]."""
    chapters: list[dict] = []
    for idx, slide in enumerate(prs.slides):
        if is_divider(slide) or not chapters:
            if is_divider(slide):
                name = _title_text(slide) or f"Chapter {len(chapters) + 1}"
                chapters.append({"name": name, "slides": [idx], "divider": idx})
                continue
            # slides before the first divider are front matter
            chapters.append({"name": "", "slides": [idx], "divider": None})
            continue
        chapters[-1]["slides"].append(idx)
    if chapters and chapters[0]["divider"] is None:
        front = chapters.pop(0)
        return [{"name": "", "slides": front["slides"], "divider": None}] + chapters
    return chapters


def short_label(name: str, labels: dict[str, str], limit: int = 11) -> str:
    """Pill text: an explicit override, else the first word(s) that fit."""
    if name in labels:
        return labels[name]
    words = name.replace("—", " ").replace("–", " ").split()
    drop = {"the", "a", "an", "of", "and", "in", "on", "for", "to", "is", "it", "we"}
    kept = [w for w in words if w.lower() not in drop] or words
    out = ""
    for w in kept:
        cand = (out + " " + w).strip()
        if len(cand) > limit:
            break
        out = cand
    return out or (kept[0][:limit] if kept else "")


# ------------------------------------------------------------------------------ drawing
def clear_nav(slide) -> int:
    """Remove every shape this script previously added. Makes re-runs idempotent."""
    removed = 0
    for shape in list(slide.shapes):
        if shape.name.startswith(NAV_PREFIX):
            shape._element.getparent().remove(shape._element)
            removed += 1
    return removed


def _style_text(shape, text, size, color, bold=False):
    tf = shape.text_frame
    tf.word_wrap = False
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = _nav_font()


def _flat(shape, fill):
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False


def add_nav(slide, chapters, current, position, total_in_chapter, labels):
    """Pill row across the top, active pill filled, with a within-chapter progress bar."""
    n = len(chapters)
    if n == 0:
        return
    span = NAV_X1 - NAV_X0
    pill_w = (span - NAV_GAP * (n - 1)) / n
    for i, chapter in enumerate(chapters):
        left = NAV_X0 + i * (pill_w + NAV_GAP)
        active = i == current
        # Rounded, matching the house aesthetic adopted 2026-09-07.
        pill = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(left), Inches(NAV_Y), Inches(pill_w), Inches(NAV_H),
        )
        pill.name = f"{NAV_PREFIX}PILL_{i}"
        pill.adjustments[0] = 0.5
        _flat(pill, BLUE if active else (PALE if i < current else GREY))
        _style_text(
            pill,
            short_label(chapter["name"], labels),
            7.5,
            WHITE if active else MUTE,
            bold=active,
        )
        if active and total_in_chapter > 1:
            frac = max(0.06, position / total_in_chapter)
            bar = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                Inches(left), Inches(NAV_Y + NAV_H + PROGRESS_DY),
                Inches(pill_w * frac), Inches(PROGRESS_H),
            )
            bar.name = f"{NAV_PREFIX}PROGRESS"
            bar.adjustments[0] = 0.5
            _flat(bar, CYAN)


def add_side_marker(slide, name, position, total_in_chapter):
    """Rotated chapter name down the left margin, with n/N inside the chapter."""
    text = name if total_in_chapter <= 1 else f"{name}   ·   {position}/{total_in_chapter}"
    box = slide.shapes.add_textbox(
        Inches(SIDE_X - SIDE_LEN / 2), Inches(SIDE_CY - 0.14),
        Inches(SIDE_LEN), Inches(0.28),
    )
    box.name = f"{NAV_PREFIX}SIDE"
    box.rotation = 270
    _style_text(box, text, 7.5, MUTE)


# ----------------------------------------------------------------- native pptx sections
def write_sections(prs, chapters) -> None:
    """Mirror the detected chapters into PowerPoint's own Sections panel.

    Purely a convenience: it makes the structure visible and reorderable in the UI. Ids
    are hashed from the chapter name so repeated runs produce identical XML.
    """
    sld_id_lst = prs.slides._sldIdLst
    ids = [s.get("id") for s in sld_id_lst.findall(qn("p:sldId"))]

    root = prs._element
    ext_lst = root.find(qn("p:extLst"))
    if ext_lst is None:
        ext_lst = root.makeelement(qn("p:extLst"), {})
        root.append(ext_lst)
    for ext in ext_lst.findall(qn("p:ext")):
        if ext.get("uri") == SECTION_EXT_URI:
            ext_lst.remove(ext)

    ext = ext_lst.makeelement(qn("p:ext"), {"uri": SECTION_EXT_URI})
    section_lst = ext.makeelement(f"{{{P14}}}sectionLst", {})
    for chapter in chapters:
        digest = hashlib.md5((chapter["name"] or "front").encode()).hexdigest().upper()
        guid = f"{{{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}}}"
        section = section_lst.makeelement(
            f"{{{P14}}}section",
            {"name": chapter["name"] or "Front matter", "id": guid},
        )
        holder = section.makeelement(f"{{{P14}}}sldIdLst", {})
        for idx in chapter["slides"]:
            if idx < len(ids):
                holder.append(
                    holder.makeelement(f"{{{P14}}}sldId", {"id": ids[idx]})
                )
        section.append(holder)
        section_lst.append(section)
    ext.append(section_lst)
    ext_lst.append(ext)


# --------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("deck", type=Path)
    ap.add_argument("-o", "--out", type=Path, help="default: <deck>.nav.pptx")
    ap.add_argument("--labels", type=Path, help="JSON map of chapter title -> short pill text")
    ap.add_argument("--dry-run", action="store_true", help="print the structure, write nothing")
    ap.add_argument("--no-side", action="store_true", help="skip the rotated side marker")
    ap.add_argument("--no-sections", action="store_true", help="do not write PowerPoint Sections")
    ap.add_argument("--on-dividers", action="store_true", help="also stamp the bar on chapter dividers")
    args = ap.parse_args(argv)

    if not args.deck.exists():
        print(f"no such deck: {args.deck}", file=sys.stderr)
        return 1
    labels = json.loads(args.labels.read_text()) if args.labels and args.labels.exists() else {}

    prs = Presentation(str(args.deck))
    groups = detect_chapters(prs)
    chapters = [c for c in groups if c["divider"] is not None]
    front = [c for c in groups if c["divider"] is None]

    print(f"{args.deck.name}: {len(prs.slides)} slides, {len(chapters)} chapters detected")
    if front:
        print(f"  front matter        slides {[i + 1 for i in front[0]['slides']]}")
    if not chapters:
        print(
            "\n  No chapter dividers found. Give each chapter a divider slide on the\n"
            "  'Heading chapter slide' layout and put the chapter name in its title.",
            file=sys.stderr,
        )
    for i, chapter in enumerate(chapters):
        pill = short_label(chapter["name"], labels)
        print(
            f"  {i + 1}. {chapter['name'][:40]:40s} pill={pill!r:16s} "
            f"{len(chapter['slides'])} slides {[s + 1 for s in chapter['slides']]}"
        )
    if args.dry_run:
        return 0

    stamped = cleared = 0
    for i, chapter in enumerate(chapters):
        body = [s for s in chapter["slides"] if s != chapter["divider"]]
        for slide_idx in chapter["slides"]:
            slide = prs.slides[slide_idx]
            cleared += clear_nav(slide)
            if any(k in _layout_name(slide) for k in BARE_LAYOUTS):
                continue
            if slide_idx == chapter["divider"] and not args.on_dividers:
                continue
            position = body.index(slide_idx) + 1 if slide_idx in body else 0
            add_nav(slide, chapters, i, position, len(body), labels)
            if not args.no_side and chapter["name"]:
                add_side_marker(slide, chapter["name"], position, len(body))
            stamped += 1
    for chapter in front:  # strip chrome from front matter if it was ever stamped
        for slide_idx in chapter["slides"]:
            cleared += clear_nav(prs.slides[slide_idx])

    if not args.no_sections and chapters:
        write_sections(prs, groups)

    out = args.out or args.deck.with_suffix(".nav.pptx")
    prs.save(str(out))
    print(f"\nstamped {stamped} slides ({cleared} old nav shapes removed) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
