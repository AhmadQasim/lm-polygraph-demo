#!/usr/bin/env python3
"""Write the chapter divider titles into `journal_club_template.pptx`.

The titles live in `slides/chapters.json`, not here, so retitling a chapter is
a data edit. Running this also regenerates `slides/nav_labels.json`, the
{title: pill} map `nav_bar.py` reads in the slides repo - a title without a pill
gets whatever prefix happens to fit, which is rarely the useful word.

Same surgical zip rewrite as `strip_template.py`: all 102 entries copied across
with their original compression, only the touched slide parts rewritten, and
every untouched entry's CRC checked afterwards. The template is never opened by
python-pptx, which would re-serialise 196 MB to change a few dozen characters.

Why the template and not a build script: a pristine Helmholtz template has no
chapter dividers, and `nav_bar.detect_chapters()` finds chapters by reading the
divider titles. A deck built on a template whose dividers say nothing gets no
chapters at all, so these strings have to be in the file itself.

    python3 slides/set_chapters.py            # write them
    python3 slides/set_chapters.py --check    # show what would change

To start over:  cp ~/projects/slides/vitiligo_voucher.pptx slides/journal_club_template.pptx
                python3 slides/strip_template.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "journal_club_template.pptx"
CHAPTERS = HERE / "chapters.json"
NAV_LABELS = HERE / "nav_labels.json"

SHAPE = re.compile(r"<p:sp>.*?</p:sp>", re.S)
PH_TYPE = re.compile(r'<p:ph\b[^>]*\btype="([^"]+)"')
TXBODY = re.compile(r"(<p:txBody>)(.*?)(</p:txBody>)", re.S)
PARAS = re.compile(r"(<a:p[ >].*</a:p>|<a:p/>)", re.S)
RUN_TEXT = re.compile(r"<a:t>([^<]*)</a:t>")

TITLE_PH = {"title", "ctrTitle"}
SUBTITLE_PH = {"subTitle"}


def paragraph(text: str) -> str:
    """One paragraph holding one run, or an empty one when the line is blank."""
    if not text:
        return '<a:p><a:endParaRPr lang="en-US" dirty="0"/></a:p>'
    return ('<a:p><a:r><a:rPr lang="en-US" dirty="0"/>'
            f"<a:t>{escape(text)}</a:t></a:r></a:p>")


def set_placeholder(xml: str, wanted: set[str], lines: list[str]) -> tuple[str, bool]:
    """Replace the text of the first placeholder of a matching type.

    Reports whether the placeholder was FOUND, not whether the bytes changed.
    Writing a title that is already there is a no-op on the XML, and treating
    that as "no placeholder" made a second run of an unedited chapters.json
    fail - which is exactly the run you make after changing one chapter.
    """
    found = False

    def fix_shape(m):
        nonlocal found
        shape = m.group(0)
        kind = PH_TYPE.search(shape)
        if found or not kind or kind.group(1) not in wanted:
            return shape
        body = "".join(paragraph(t) for t in lines) or paragraph("")
        # lambda replacements, so a backslash or a \1 in a title is literal
        new, n = TXBODY.subn(
            lambda b: b.group(1) + PARAS.sub(lambda _: body, b.group(2), count=1)
            + b.group(3), shape, count=1)
        if n:
            found = True
        return new

    return SHAPE.sub(fix_shape, xml), found


def plan(spec: dict) -> dict[int, list[tuple[set[str], list[str]]]]:
    """{slide number: [(placeholder types, lines), ...]} for everything to write."""
    out: dict[int, list[tuple[set[str], list[str]]]] = {}
    ts = spec.get("title_slide") or {}
    if ts.get("title") or ts.get("subtitle"):
        out[int(ts["slide"])] = [
            (TITLE_PH, [ts.get("title", "")]),
            (SUBTITLE_PH, list(ts.get("subtitle") or [])),
        ]
    for ch in spec["chapters"]:
        out.setdefault(int(ch["slide"]), []).append((TITLE_PH, [ch["title"]]))
    return out


def run(template: Path, spec: dict, check_only: bool) -> int:
    todo = plan(spec)
    with zipfile.ZipFile(template) as z:
        names = set(z.namelist())
        patched: dict[str, bytes] = {}
        for number, jobs in sorted(todo.items()):
            part = f"ppt/slides/slide{number}.xml"
            if part not in names:
                raise SystemExit(f"{part} is not in the template")
            xml = z.read(part).decode("utf-8")
            before = [t for t in RUN_TEXT.findall(xml) if t.strip()]
            for wanted, lines in jobs:
                if not any(l for l in lines):
                    continue
                xml, ok = set_placeholder(xml, wanted, lines)
                if not ok:
                    raise SystemExit(
                        f"{part}: no {'/'.join(sorted(wanted))} placeholder to write "
                        f"{lines[0]!r} into")
            after = [t for t in RUN_TEXT.findall(xml) if t.strip()]
            print(f"  slide {number}: {before or '(empty)'} -> {after}")
            patched[part] = xml.encode("utf-8")

        if check_only:
            print("\n--check: nothing written")
            return 0

        entries = z.infolist()
        originals = {i.filename: i.CRC for i in entries}
        tmp = template.with_suffix(".settingchapters.pptx")
        with zipfile.ZipFile(tmp, "w") as out:
            for item in entries:
                data = patched.get(item.filename) or z.read(item.filename)
                info = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                info.compress_type = item.compress_type
                info.external_attr = item.external_attr
                info.create_system = item.create_system
                out.writestr(info, data)

    with zipfile.ZipFile(tmp) as c:
        if c.testzip() is not None:
            raise SystemExit("the rewritten template does not read back cleanly")
        if len(c.namelist()) != len(entries):
            raise SystemExit("the rewritten template lost an entry")
        moved = [i.filename for i in c.infolist()
                 if i.filename not in patched and originals[i.filename] != i.CRC]
        if moved:
            raise SystemExit(f"entries changed that should not have: {moved[:5]}")
        for ch in spec["chapters"]:
            part = f"ppt/slides/slide{ch['slide']}.xml"
            if ch["title"] not in c.read(part).decode("utf-8"):
                raise SystemExit(f"{part} does not carry {ch['title']!r}")
    shutil.move(str(tmp), str(template))

    labels = {ch["title"]: ch["pill"] for ch in spec["chapters"] if ch.get("pill")}
    NAV_LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n")
    print(f"\n  wrote {len(patched)} slides; "
          f"{len(entries) - len(patched)} parts unchanged, CRCs verified")
    print(f"  nav pills -> {NAV_LABELS}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", type=Path, default=TEMPLATE)
    ap.add_argument("--chapters", type=Path, default=CHAPTERS)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    spec = json.loads(args.chapters.read_text())
    return run(args.template, spec, args.check)


if __name__ == "__main__":
    raise SystemExit(main())
