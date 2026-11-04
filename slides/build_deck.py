#!/usr/bin/env python3
"""Build the whole journal club deck from the template, the figures and the markdowns.

`journal_club_template.pptx` is read-only source: it holds the title slide and the
chapter dividers (written by `set_chapters.py`), and nothing else. Every run copies it,
inserts the content slides after their chapter dividers, numbers the slides, stamps the
chapter navigation bar, and writes `build/journal_club_deck.pptx`. The template is never
modified, so a build can always be thrown away and repeated.

    .venv/bin/python slides/build_deck.py               # -> slides/build/journal_club_deck.pptx
    .venv/bin/python slides/build_deck.py --figures     # regenerate every figure first
    .venv/bin/python slides/build_deck.py --pdf         # also export a PDF with LibreOffice
    .venv/bin/python slides/build_deck.py --list        # show the running order and stop
    .venv/bin/python slides/build_deck.py --only uq_types consistency   # a quick partial build

Each content slide is one figure plus three pieces of text - title, blue lead, grey
caption - and the text comes from `markdowns/NN_slide_<name>.md`. Edit the words there
and rebuild; no Python changes. The figure is `figures/<name>.png`, written by
`fig_methods.py` (the opening and method slides) or `plot_bioasq.py` (the results).

Adding a slide: write its figure and its markdown, then add one line to SLIDES below.
The list is the running order.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from pptx import Presentation  # noqa: E402

import deck_common  # noqa: E402
import hgf_layers  # noqa: E402
import nav_bar  # noqa: E402
from hgf_copy import copy_for, flow  # noqa: E402

TEMPLATE = HERE / "journal_club_template.pptx"
BUILD_DIR = HERE / "build"
OUTPUT = BUILD_DIR / "journal_club_deck.pptx"
LABELS = HERE / "nav_labels.json"
CHAPTERS = HERE / "chapters.json"
FIGURE_DIR = HERE / "figures"
OUTLINE = HERE / "deck_outline.json"

# The chapter each slide belongs to, by its divider's title prefix in chapters.json.
# Resolved to a template slide number at build time, so retitling or reordering a
# chapter in chapters.json cannot quietly put a slide in the wrong section.
#
# (figure / markdown stem, chapter)  - in running order
SLIDES: list[tuple[str, str]] = [
    # why it matters
    ("confidently_wrong", "Problem"),
    ("selective_generation", "Problem"),
    # the method families. types_by_tier sits here, not in the example chapter,
    # because it is where T1-T4 are defined and every family slide below refers
    # to them. It replaced uq_types, which covered the same five families from
    # the literature without the tiers; its figure and markdown are still on
    # disk, so restoring it is one line.
    ("uq_map", "Methods"),
    ("types_by_tier", "Methods"),
    ("information_based", "Methods"),
    ("self_report", "Methods"),
    ("consistency", "Methods"),
    ("trained_probes", "Methods"),
    # the library
    ("lm_polygraph", "Tooling: LM-Polygraph"),
    ("polygraph_catalogue", "Tooling: LM-Polygraph"),
    ("wrap_model", "Tooling: Your model"),
    ("ue_manager", "Tooling: Your model"),
    ("custom_estimator", "Tooling: Your model"),
    # a worked example, kept short. One chapter, not two: the setup slide alone
    # did not justify a full-screen divider in front of it. The template still
    # holds the "Example: What it showed" divider; drop_empty_chapters() takes it
    # out of the build, so the template itself stays untouched.
    ("dataset", "Example: A small"),
    ("auroc_by_method", "Example: A small"),
    ("auroc_by_tier", "Example: A small"),
    ("limitations", "Conclusions"),
    ("takeaways", "Conclusions"),
    ("references", "Conclusions"),
    # detail for questions
    ("uncertainty_by_correctness", "Backup"),
    ("model_size_comparison", "Backup"),
]

# The runs the results figures are drawn from. The figures say which model they show;
# keep these in step with the captions in plot_bioasq.SLIDE_COPY.
CACHE_SMALL = ROOT / "outputs" / "bioasq_Qwen2.5-3B-Instruct_n1000_seed0_t4.json"
CACHE_BIG = ROOT / "outputs" / "bioasq_Qwen2.5-7B-Instruct_n1000_seed0_t4.json"
# Other families, for the backup model comparison only. manual_labels.json covers
# their non-exact answers too (each label records the runs it came from).
# Ordered so the two size ladders (Qwen 3B->7B, Llama 3B->8B) sit side by side;
# the slopegraph's x axis is categorical, so the order is purely for reading.
CACHE_OTHERS = [ROOT / "outputs" / "bioasq_Llama-3.2-3B-Instruct_n1000_seed0_t4.json",
                ROOT / "outputs" / "bioasq_Llama-3.1-8B-Instruct_n1000_seed0_t4.json",
                ROOT / "outputs" / "bioasq_Mistral-7B-Instruct-v0.3_n1000_seed0_t4.json",
                ROOT / "outputs" / "bioasq_OLMo-2-1124-7B-Instruct_n1000_seed0_t4.json"]

# Figure generators, run under --figures, in this order. Each writes its PNGs into
# figures/ and leaves an already-edited markdown alone.
FIGURES: list[tuple[str, list[str]]] = [
    ("opening and method slides", [str(HERE / "fig_methods.py")]),
    ("library slides", [str(HERE / "fig_polygraph.py")]),
    ("results slides", [str(HERE / "plot_bioasq.py"), str(CACHE_SMALL),
                        "--compare", str(CACHE_BIG), *map(str, CACHE_OTHERS),
                        "--labels", "Qwen2.5-3B,Qwen2.5-7B,Llama-3.2-3B,Llama-3.1-8B,"
                        "Mistral-7B,OLMo-2-7B"]),
]


# =====================================================================================
def chapter_anchors() -> dict[str, int]:
    """{chapter title: its divider's slide number in the template}, from chapters.json."""
    spec = json.loads(CHAPTERS.read_text())
    return {ch["title"]: int(ch["slide"]) for ch in spec["chapters"]}


def anchor_for(chapter: str, anchors: dict[str, int]) -> int:
    """The divider slide a chapter key names: an exact title, or a unique prefix of one."""
    if chapter in anchors:
        return anchors[chapter]
    hits = [t for t in anchors if t.split(":")[0].strip() == chapter
            or t.startswith(chapter)]
    if len(hits) != 1:
        raise LookupError(f"chapter {chapter!r} matches {hits or 'nothing'} in "
                          f"{CHAPTERS.name}; have {list(anchors)}")
    return anchors[hits[0]]


def plan(keys: list[str] | None = None) -> list[tuple[str, str, int, int]]:
    """(stem, chapter, insert_at, final slide number) for each slide to build.

    Every insertion pushes the slides after it down by one, so a template anchor is only
    right for the first slide placed there. `insert_at` corrects each anchor for the
    insertions already made at or before it; slides sharing an anchor land in list order.
    """
    anchors = chapter_anchors()
    # Dividers no slide is assigned to are dropped from the build, so the final
    # number is one lower for every such divider at or before this slide's anchor.
    # `insert_at` is unaffected: insertion happens while they are still there.
    used = {anchor_for(ch, anchors) for _, ch in SLIDES}
    gone = sorted(n for n in anchors.values() if n not in used)
    out, inserted = [], []
    for stem, chapter in SLIDES:
        if keys is not None and stem not in keys:
            continue
        anchor = anchor_for(chapter, anchors)
        # After the divider AND after the slides already placed in that chapter.
        insert_at = anchor + sum(1 for a in inserted if a <= anchor)
        shift = sum(1 for n in gone if n <= anchor)
        out.append((stem, chapter, insert_at, insert_at + 1 - shift))
        inserted.append(anchor)
    return out


def slide_text(stem: str) -> tuple[str, str | None, str | None]:
    """Title, lead and caption from markdowns/NN_slide_<stem>.md."""
    T = copy_for(f"slide_{stem}")
    title = T("title")
    lead = flow(T("lead")) or None
    caption = flow(T("caption")) or None
    return title, lead, caption


def stamp_nav(prs) -> tuple[int, int]:
    """The chapter bar and side marker, using nav_bar's own detection and drawing."""
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    groups = nav_bar.detect_chapters(prs)
    chapters = [c for c in groups if c["divider"] is not None]
    stamped = 0
    for i, chapter in enumerate(chapters):
        body = [s for s in chapter["slides"] if s != chapter["divider"]]
        for idx in chapter["slides"]:
            slide = prs.slides[idx]
            nav_bar.clear_nav(slide)
            layout = (slide.slide_layout.name or "").lower()
            if any(k in layout for k in nav_bar.BARE_LAYOUTS) or idx == chapter["divider"]:
                continue
            position = body.index(idx) + 1 if idx in body else 0
            nav_bar.add_nav(slide, chapters, i, position, len(body), labels)
            if chapter["name"]:
                nav_bar.add_side_marker(slide, chapter["name"], position, len(body))
            stamped += 1
    for group in groups:
        if group["divider"] is None:
            for idx in group["slides"]:
                nav_bar.clear_nav(prs.slides[idx])
    if chapters:
        nav_bar.write_sections(prs, groups)
    return stamped, len(chapters)


def drop_empty_chapters(prs) -> list[str]:
    """Remove divider slides that no content slide was assigned to.

    chapters.json has to list every divider in the template, because that is how
    set_chapters.py finds the slide to write a title into. Merging two chapters
    therefore leaves a divider with nothing behind it, which would otherwise
    render as a full-screen section header for an empty section.

    Detection is structural - a divider with no body slides after it - rather
    than read from chapters.json, so this keeps working if the running order
    changes again. The template is never touched: this runs on the build copy.
    """
    empty = [c for c in nav_bar.detect_chapters(prs)
             if c["divider"] is not None and len(c["slides"]) == 1]
    rel_ns = ("{http://schemas.openxmlformats.org/officeDocument/"
              "2006/relationships}id")
    lst = prs.slides._sldIdLst
    # Highest index first, so earlier indices stay valid as entries are removed.
    for chapter in sorted(empty, key=lambda c: -c["divider"]):
        entry = list(lst)[chapter["divider"]]
        prs.part.drop_rel(entry.get(rel_ns))
        lst.remove(entry)
    return [c["name"] for c in empty]


TITLE_INK = "002864"   # hgf_style.DARK


def finish_front_matter(prs) -> str:
    """Fix two things in the copy that the template gets wrong for this deck.

    1. The title slide sets its title white, which on the pale band behind it is
       barely readable. It is set to the deck's dark blue.
    2. Chapter titles that do not fit on one line wrap wherever PowerPoint puts
       them, which left "...BioASQ / study". They get an explicit break: after
       the "Category:" prefix when the rest then fits, else at the most balanced
       word boundary. Measured in Hermann Bold, the font the master sets.
    """
    from lxml import etree
    from PIL import ImageFont
    from pptx.oxml.ns import qn

    def ink(run):
        rpr = run._r.get_or_add_rPr()
        for old in rpr.findall(qn("a:solidFill")):
            rpr.remove(old)
        fill = etree.SubElement(rpr, qn("a:solidFill"))
        etree.SubElement(fill, qn("a:srgbClr")).set("val", TITLE_INK)
        rpr.insert(0, fill)   # fill comes before latin/ea/cs in the schema

    title_runs = 0
    for shape in prs.slides[0].placeholders:
        if shape.placeholder_format.type == 3:   # CENTER_TITLE
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    ink(run)
                    title_runs += 1

    font = ImageFont.truetype(str(HERE / "fonts" / "Hermann-Bold.otf"), 540)
    width_in = lambda t: font.getlength(t) / 10 / 72          # at 54 pt
    broken = []
    for slide in prs.slides:
        if not slide.slide_layout.name.startswith("Earth_Custom"):
            continue
        for shape in slide.placeholders:
            if shape.placeholder_format.type != 1:   # TITLE
                continue
            text = shape.text_frame.text
            room = shape.width / 914400 - 0.2       # default 0.1 in insets
            if width_in(text) <= room:
                continue
            words = text.split()
            cuts = [k for k in range(1, len(words))
                    if max(width_in(" ".join(words[:k])),
                           width_in(" ".join(words[k:]))) <= room]
            if not cuts:
                continue
            colon = [k for k in cuts if words[k - 1].endswith(":")]
            k = colon[0] if colon else min(cuts, key=lambda c: max(
                width_in(" ".join(words[:c])), width_in(" ".join(words[c:]))))
            para = shape.text_frame.paragraphs[0]
            run = para.runs[0]
            for extra in para.runs[1:]:
                para._p.remove(extra._r)
            run.text = " ".join(words[:k])
            br = etree.SubElement(para._p, qn("a:br"))
            run._r.addnext(br)
            tail = copy.deepcopy(run._r)
            tail.find(qn("a:t")).text = " ".join(words[k:])
            br.addnext(tail)
            broken.append(" / ".join((" ".join(words[:k]), " ".join(words[k:]))))
    return f"title ink set on {title_runs} runs; breaks: " + "; ".join(broken)


def snapshot_outline(prs) -> int:
    """Record every slide's text, small enough for git, after the build.

    The template is untracked (it is 187 MB), so this is the reviewable record of what
    the deck says and in what order. Text only - not a rebuild.
    """
    slides = []
    for i, slide in enumerate(prs.slides, 1):
        texts = [sh.text_frame.text.strip() for sh in slide.shapes
                 if sh.has_text_frame and sh.text_frame.text.strip()
                 and not sh.name.startswith(("NAV_", f"{deck_common.GEN_PREFIX}SLIDENUM"))]
        slides.append({"slide": i, "layout": slide.slide_layout.name, "text": texts})
    OUTLINE.write_text(json.dumps(
        {"note": "What the built deck says, slide by slide. Written by "
                 "slides/build_deck.py on every full build.", "slides": slides},
        indent=2, ensure_ascii=False) + "\n")
    return len(slides)


def check_layers() -> int:
    """Composite every layered figure and refuse the build if one lost content.

    A band declared in clicks.json that misses part of its panel does not fail when
    the figure is drawn - the artist simply lands in no layer and disappears from the
    slide. Only compositing catches it, so it is checked here rather than trusted.
    """
    spec = hgf_layers.load_spec()
    if not spec:
        return 0
    bad = hgf_layers.verify(FIGURE_DIR, spec)
    return bad


def regenerate_figures() -> int:
    failed = 0
    for name, cmd in FIGURES:
        began = time.time()
        done = subprocess.run([sys.executable, *cmd], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if done.returncode:
            failed += 1
            print(f"  figures  {name:<28s} FAILED\n{done.stderr}")
            continue
        kept = done.stdout.count("(kept: edited copy)")
        print(f"  figures  {name:<28s} {time.time() - began:5.1f}s"
              + (f"  ({kept} markdown(s) kept as edited)" if kept else ""))
    return failed


def export_pdf(pptx: Path) -> Path | None:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        print("  no LibreOffice (soffice) on PATH - skipped")
        return None
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir",
                    str(pptx.parent), str(pptx)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return pptx.with_suffix(".pdf")


# =====================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--figures", action="store_true",
                    help="regenerate every figure before building")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF (LibreOffice)")
    ap.add_argument("--only", nargs="+", metavar="NAME",
                    help="build only these slides (by figure name), for a fast loop")
    ap.add_argument("--list", action="store_true", help="print the running order and stop")
    ap.add_argument("--no-nav", action="store_true", help="skip the navigation bar")
    ap.add_argument("--no-clicks", action="store_true",
                    help="place every figure whole, with no click-by-click build")
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("-o", "--out", type=Path, default=OUTPUT)
    args = ap.parse_args(argv)

    if args.only:
        unknown = set(args.only) - {s for s, _ in SLIDES}
        if unknown:
            print(f"unknown slide(s) {sorted(unknown)}; see --list", file=sys.stderr)
            return 1

    full = plan()
    if args.list:
        print("slides, in running order:")
        for stem, chapter, _at, pos in full:
            print(f"  slide {pos:2d}  {chapter:<30s} {stem}")
        return 0

    if not args.template.exists():
        print(f"template not found: {args.template}\n"
              "  cp ~/projects/slides/vitiligo_voucher.pptx slides/journal_club_template.pptx\n"
              "  .venv/bin/python slides/strip_template.py && "
              ".venv/bin/python slides/set_chapters.py", file=sys.stderr)
        return 1

    started = time.time()
    BUILD_DIR.mkdir(exist_ok=True)

    if args.figures:
        print("figures")
        if regenerate_figures():
            return 1
        if not args.no_clicks:
            print("\nclick layers")
            if check_layers():
                print("a figure lost content when its layers were composited; fix its "
                      "bands in clicks.json, or build with --no-clicks",
                      file=sys.stderr)
                return 1

    # Check every slide's inputs before touching the deck, so a missing figure or a
    # markdown with no caption fails in one line rather than halfway through a build.
    todo = plan(args.only)
    texts, missing = {}, []
    for stem, *_ in todo:
        if not (FIGURE_DIR / f"{stem}.png").exists():
            missing.append(f"figures/{stem}.png")
        try:
            texts[stem] = slide_text(stem)
        except Exception as exc:  # noqa: BLE001 - report every problem, then stop
            missing.append(f"markdowns/*_slide_{stem}.md: {exc}")
    if missing:
        print("cannot build; missing:\n  " + "\n  ".join(missing)
              + "\nrun with --figures to regenerate them", file=sys.stderr)
        return 1

    # The template is source, never a target: copy it, then only touch the copy.
    working = BUILD_DIR / f"_working_{args.out.name}"
    shutil.copy2(args.template, working)
    prs = Presentation(str(working))
    print(f"\ntemplate: {args.template.name}  ({len(prs.slides._sldIdLst)} slides)")

    print("slides")
    clicks = 0
    for stem, chapter, insert_at, pos in todo:
        title, lead, caption = texts[stem]
        figure = FIGURE_DIR / f"{stem}.png"
        slide = deck_common.insert_slide(prs, insert_at, "Title Only")
        deck_common.figure_slide(slide, title, figure, lead=lead, caption=caption,
                                 clicks=not args.no_clicks, verbose=False)
        # layer_paths falls back to the figure itself, so a figure with no layers
        # reports zero rather than counting its own single PNG as a click.
        n = 0 if args.no_clicks else len(deck_common.layer_paths(figure)) - 1
        clicks += n
        built = f"{n} click{'s' if n != 1 else ''}" if n else "-"
        print(f"  slide {pos:2d}  {chapter:<30s} {stem:<28s} {built}")
    if args.only is None:
        # Only on a full build: a partial build legitimately leaves chapters empty.
        for name in drop_empty_chapters(prs):
            print(f"  dropped empty chapter divider: {name!r}")
    if not args.no_clicks:
        print(f"\nclicks: {clicks} across {len(todo)} slides")

    print(f"front matter: {finish_front_matter(prs)}")
    n = deck_common.number_slides(prs)
    print(f"\nnumbering: {n} slides")
    if not args.no_nav:
        stamped, chapters = stamp_nav(prs)
        print(f"navigation: {chapters} chapters, {stamped} slides stamped")
    if not args.only:
        print(f"outline: {snapshot_outline(prs)} slides -> {OUTLINE.relative_to(ROOT)}")

    prs.save(str(args.out))
    working.unlink(missing_ok=True)
    size = args.out.stat().st_size / 1e6
    print(f"\n{args.out.relative_to(ROOT)}  ({size:.1f} MB, {time.time() - started:.1f}s)")

    if args.pdf:
        pdf = export_pdf(args.out)
        if pdf:
            print(f"{pdf.relative_to(ROOT)}")
    print(f"{args.template.name} was not modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
