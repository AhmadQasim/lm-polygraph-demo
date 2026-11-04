#!/usr/bin/env python3
"""Every word a figure draws, in a markdown file you can edit without touching Python.

A figure's wording used to live inside its matplotlib script, which meant changing a
heading was a code edit. Now each figure has a companion file in `markdowns/`, named for
the slide it lands on, and the script asks it for its strings by key:

    from hgf_copy import copy_for
    T = copy_for("fig_metrics")
    ...
    ax.text(x, y, T("pooled.title"), ...)

Edit `markdowns/26_fig_metrics.md`, re-run the build, and the slide changes. The key is
the contract; the words are yours. The number in the filename is the slide, and it is
there so the folder reads in the order the talk does — the file is found by its module
stem, so renumbering is a rename and nothing else.

## The format

    ## Left panel · pooled Dice          <- a heading, for your eyes only

    - **pooled.title** — card heading, small caps, centred over the value
      > POOLED DICE
    - **pooled.gloss** — two lines under the number
      > every lesion pixel in the
      > cohort counted once

A `- **key** — note` line starts an entry; the `>` lines under it are the text, one per
line as it appears on the slide. Everything else — headings, paragraphs, tables — is
documentation and is ignored, so explain as much as you like. The note after the em dash
is where the text sits on the figure, so you can find it without rendering.

**Numbers stay in the code.** Where a line carries a measured value the copy holds a
`{placeholder}` and the figure fills it:

    - **counts.fn** — the "of SS's 2,994" line beside the big number
      > of SS's {ref}

That way a wording change cannot silently overwrite a number read from a report.

## What is checked, and when

`copy_for` raises on a key the file does not define, and names the near misses. The
reverse — a key defined but never drawn — is not an error while a figure is being
edited, but `python scripts/copy_check.py` reports it, along with any text that now
overflows its canvas or collides with its neighbour. **Run it after editing copy**: the
figures are laid out by hand and a longer line does not reflow, it overlaps.

    python scripts/copy_check.py               # every figure
    python scripts/copy_check.py fig_metrics   # one of them
"""

from __future__ import annotations

import difflib
import re
from functools import lru_cache
from pathlib import Path

# Vendored from ~/projects/slides/scripts/hgf_copy.py; only this path differs:
# the markdowns sit beside this file in journal_club/slides/.
MARKDOWN_DIR = Path(__file__).resolve().parent / "markdowns"

#: `- **some.key** — a note about where it is`. The note is optional and ignored.
KEY_LINE = re.compile(r"^\s*[-*]\s+\*\*([a-z0-9][a-z0-9_.-]*)\*\*\s*(?:[—–-]\s*(.*))?$")
#: `  > the text`, one line of it. A bare `>` is an empty line inside a block.
TEXT_LINE = re.compile(r"^\s*>\s?(.*)$")


class MissingKey(SystemExit):
    pass


def parse(text: str, source: str = "<string>") -> tuple[dict[str, str], dict[str, str]]:
    """({key: text}, {key: note}) from a copy file, in file order.

    Strict on the two things that would otherwise fail silently: a key defined twice, and
    a key with no `>` text under it. Everything it does not recognise — headings, prose,
    tables — is ignored, so a copy file can carry as much explanation as it likes.
    """
    entries: dict[str, str] = {}
    notes: dict[str, str] = {}
    key, lines = None, []

    def flush(line_no):
        if key is None:
            return
        if not lines:
            raise SystemExit(f"{source}:{line_no}: **{key}** has no `>` text under it")
        entries[key] = "\n".join(lines).rstrip()

    for i, line in enumerate(text.splitlines(), 1):
        match = KEY_LINE.match(line)
        if match:
            flush(i)
            key, lines = match.group(1), []
            if key in entries:
                raise SystemExit(f"{source}:{i}: **{key}** is defined twice")
            notes[key] = (match.group(2) or "").strip()
            continue
        if key is None:
            continue
        text_match = TEXT_LINE.match(line)
        if text_match:
            lines.append(text_match.group(1).rstrip())
        elif line.strip() == "" and lines:
            flush(i)
            key, lines = None, []
        elif line.lstrip().startswith("#"):
            flush(i)
            key, lines = None, []
    flush(len(text.splitlines()) + 1)
    return entries, notes


class Copy:
    """One figure's words. Call it with a key; it tracks which keys were used."""

    def __init__(self, stem: str, entries: dict[str, str], notes: dict[str, str],
                 path: Path):
        self.stem, self.path = stem, path
        self._entries, self.notes = entries, notes
        self.used: set[str] = set()
        #: Every string handed out, after formatting. `copy_check.py` compares this
        #: against what the figure actually drew, which is how a line still hard-coded
        #: in Python gets found.
        self.served: set[str] = set()

    def __call__(self, key: str, **fmt) -> str:
        if key not in self._entries:
            near = difflib.get_close_matches(key, self._entries, n=3, cutoff=0.4)
            raise MissingKey(
                f"{self.path.name}: no entry **{key}**"
                + (f" — did you mean {', '.join('**' + k + '**' for k in near)}?"
                   if near else "")
                + f"\n  add it as:\n  - **{key}** — where it sits\n    > the text")
        self.used.add(key)
        value = self._entries[key]
        value = value.format(**fmt) if fmt else value
        self.served.add(value)
        return value

    def many(self, prefix: str) -> list[tuple[str, str]]:
        """Every `prefix.*` entry in file order, as (suffix, text). For repeated rows."""
        out = []
        for key, value in self._entries.items():
            if key.startswith(prefix + "."):
                self.used.add(key)
                self.served.add(value)
                out.append((key[len(prefix) + 1:], value))
        if not out:
            raise MissingKey(f"{self.path.name}: no entries under **{prefix}.**")
        return out

    @property
    def keys(self):
        return list(self._entries)

    @property
    def unused(self):
        return [k for k in self._entries if k not in self.used]


@lru_cache(maxsize=None)
def path_for(stem: str) -> Path:
    """`markdowns/26_fig_metrics.md` from `fig_metrics`, found rather than constructed.

    The leading number is the slide the figure lands on, and it is decoration: the file
    is located by its module stem, so renumbering one when the deck grows is a rename
    and nothing else. Two files claiming the same stem is an error worth stopping for —
    it means a rename left a copy behind, and one of the two would be silently ignored.
    """
    matches = sorted(MARKDOWN_DIR.glob(f"*_{stem}.md"))
    if not matches:
        raise SystemExit(
            f"no markdown for {stem} in {MARKDOWN_DIR.name}/ — every figure needs one.\n"
            f"  create it with:  python scripts/copy_extract.py --convert {stem}")
    if len(matches) > 1:
        raise SystemExit(f"{stem} has {len(matches)} markdown files: "
                         + ", ".join(m.name for m in matches))
    return matches[0]


@lru_cache(maxsize=None)
def copy_for(stem: str) -> Copy:
    """The markdown for a figure module, by its stem — `copy_for("fig_metrics")`.

    Cached, because a figure that draws itself once per animation frame would otherwise
    re-read and re-parse the file forty times.
    """
    path = path_for(stem)
    entries, notes = parse(path.read_text(), source=str(path))
    return Copy(stem, entries, notes, path)


def flow(text: str) -> str:
    """One paragraph out of however many lines it was written on.

    A figure's `>` lines are lines on the canvas — nothing wraps, so the break matters.
    A slide's lead and caption are the opposite: PowerPoint wraps them in a text frame,
    and a newline there is a hard break in the middle of a sentence. So those are written
    across as many `>` lines as reads well in the file and flowed back to one paragraph
    on the way out.
    """
    return " ".join(text.split())


def reload(stem: str) -> Copy:
    """Drop the cache for one figure. For a REPL, or a watcher."""
    copy_for.cache_clear()
    path_for.cache_clear()
    return copy_for(stem)


def stems() -> list[str]:
    """Every figure that has a markdown, by module stem, in slide order."""
    out = []
    for path in sorted(MARKDOWN_DIR.glob("*_*.md")):
        prefix, _, stem = path.stem.partition("_")
        if prefix.replace("-", "").isdigit():
            out.append(stem)
    return out
