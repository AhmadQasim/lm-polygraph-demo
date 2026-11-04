#!/usr/bin/env python3
"""Empty the vitiligo deck's content out of `journal_club_template.pptx`.

The template was copied from ~/projects/slides, where it is the vitiligo talk's
source, so it arrives carrying that talk: a title slide naming the study and
eight chapter dividers ("GOAL: VITILIGO ASSESSMENT", "Methods: Model training",
...). The Helmholtz layouts, masters, theme and background media are what we
want; the words on the nine slides are not.

**Surgical on purpose**, the same way `rename_chapter.py` is in the slides repo.
The pptx is opened as a zip and all 102 entries are copied across with their
original compression and attributes; only the nine slide parts and the slide
titles cached in `docProps/app.xml` are rewritten. A python-pptx round-trip
would re-serialise a 196 MB hand-authored package to blank a few dozen
characters, and nobody would be able to say what else moved. The run verifies
afterwards that every untouched entry still has its original CRC.

The slides and their placeholders are KEPT, emptied rather than deleted: a
pristine Helmholtz template has no chapter dividers at all, and `nav_bar.py`
reads the navigation labels off them, so a deck built on one with none gets no
chapters. Nine empty slides is the useful skeleton. Delete the ones this talk
does not need in PowerPoint.

No backup is written, because the pristine original is still sitting at
~/projects/slides/vitiligo_voucher.pptx and this file is a copy of it:

    cp ~/projects/slides/vitiligo_voucher.pptx slides/journal_club_template.pptx

Run:  python3 slides/strip_template.py            # --check first, then strip
      python3 slides/strip_template.py --check    # report, change nothing
"""

from __future__ import annotations

import argparse
import re
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "journal_club_template.pptx"
PRISTINE = Path.home() / "projects" / "slides" / "vitiligo_voucher.pptx"

SLIDE_PART = re.compile(r"^ppt/slides/slide\d+\.xml$")
TXBODY = re.compile(r"(<p:txBody>)(.*?)(</p:txBody>)", re.S)
PARAS = re.compile(r"(<a:p[ >].*</a:p>|<a:p/>)", re.S)
RUN_TEXT = re.compile(r"<a:t>([^<]*)</a:t>")

# What PowerPoint itself writes for an empty placeholder. Keeping the paragraph
# rather than dropping the whole txBody is what makes the placeholder show its
# "Click to add title" prompt instead of vanishing.
EMPTY_PARA = '<a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p>'


def slide_titles(z: zipfile.ZipFile) -> dict[str, list[str]]:
    out = {}
    for name in sorted(z.namelist()):
        if SLIDE_PART.match(name):
            out[name] = RUN_TEXT.findall(z.read(name).decode("utf-8", "replace"))
    return out


def blank_slide(xml: str) -> str:
    """Replace every paragraph in every text body with one empty paragraph."""
    def fix(m):
        open_tag, body, close_tag = m.groups()
        return open_tag + PARAS.sub(EMPTY_PARA, body, count=1) + close_tag
    return TXBODY.sub(fix, xml)


def blank_app_titles(xml: str, n_slides: int) -> str:
    """Blank the slide titles cached in docProps/app.xml.

    TitlesOfParts lists fonts, then themes, then one entry per slide; the counts
    are in HeadingPairs. PowerPoint rebuilds this on save, but until someone
    opens the file the old talk's titles are still sitting in its properties -
    and that is where Finder and SharePoint read a deck's name from.
    """
    m = re.search(r"(<TitlesOfParts>.*?</TitlesOfParts>)", xml, re.S)
    if not m:
        return xml
    block = m.group(1)
    entries = re.findall(r"<vt:lpstr>([^<]*)</vt:lpstr>", block)
    if len(entries) < n_slides:
        return xml
    keep = entries[:-n_slides]                       # fonts and theme names
    rebuilt = "".join(f"<vt:lpstr>{e}</vt:lpstr>" for e in keep)
    rebuilt += "<vt:lpstr></vt:lpstr>" * n_slides
    return xml.replace(
        block,
        f'<TitlesOfParts><vt:vector size="{len(entries)}" baseType="lpstr">'
        f"{rebuilt}</vt:vector></TitlesOfParts>")


def strip(template: Path, check_only: bool) -> int:
    with zipfile.ZipFile(template) as z:
        entries = z.infolist()
        titles = slide_titles(z)
        n_slides = len(titles)

        print(f"{template.name}  -  {len(entries)} parts, {n_slides} slides\n")
        for part, runs in titles.items():
            shown = " | ".join(r for r in runs if r.strip()) or "(already empty)"
            print(f"  {part:26s} {shown}")

        if check_only:
            print("\n--check: nothing written")
            return 0

        patched: dict[str, bytes] = {}
        for part in titles:
            xml = z.read(part).decode("utf-8")
            new = blank_slide(xml)
            left = [t for t in RUN_TEXT.findall(new) if t.strip()]
            if left:
                raise SystemExit(f"{part}: text survived the blanking: {left}")
            patched[part] = new.encode("utf-8")

        app = "docProps/app.xml"
        if app in z.namelist():
            patched[app] = blank_app_titles(
                z.read(app).decode("utf-8"), n_slides).encode("utf-8")

        tmp = template.with_suffix(".stripping.pptx")
        with zipfile.ZipFile(tmp, "w") as out:
            for item in entries:
                data = patched.get(item.filename) or z.read(item.filename)
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.create_system = item.create_system
                out.writestr(info, data)
        originals = {i.filename: i.CRC for i in entries}

    # --- verify before anything replaces the file we were given ----------------------
    with zipfile.ZipFile(tmp) as c:
        if c.testzip() is not None:
            raise SystemExit("the rewritten template does not read back cleanly")
        if len(c.namelist()) != len(entries):
            raise SystemExit("the rewritten template lost an entry")
        moved = [i.filename for i in c.infolist()
                 if i.filename not in patched and originals[i.filename] != i.CRC]
        if moved:
            raise SystemExit(f"entries changed that should not have: {moved[:5]}")
        for part in titles:
            left = [t for t in RUN_TEXT.findall(c.read(part).decode("utf-8"))
                    if t.strip()]
            if left:
                raise SystemExit(f"{part} still carries text: {left}")
        for part in c.namelist():
            blob = c.read(part).lower()
            for word in (b"vitiligo", b"depigment", b"3dviz"):
                if word in blob:
                    raise SystemExit(f"{word.decode()!r} survives in {part}")

    shutil.move(str(tmp), str(template))
    kept = len(entries) - len(patched)
    print(f"\n  blanked {len(titles)} slides and the cached titles in docProps")
    print(f"  {kept} of {len(entries)} parts copied across unchanged, CRCs verified")
    print(f"  layouts, masters, theme and background media untouched")
    if PRISTINE.exists():
        print(f"\n  to start over:  cp {PRISTINE} {template}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--check", action="store_true",
                    help="print what is on the slides and stop")
    args = ap.parse_args(argv)
    if not args.template.exists():
        raise SystemExit(f"{args.template} not found")
    return strip(args.template, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
