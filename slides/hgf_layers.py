#!/usr/bin/env python3
"""Split a figure into click-by-click layers, without rewriting the figure.

A PNG cannot be revealed in parts, so a slide that builds up its panels needs one
transparent PNG per step, all the same size, stacked at the same box. `deck_common.
figure_slide(..., clicks=True)` places every `<stem>_l*.png` it finds and asks
`hgf_anim` to reveal them one click at a time. This is what writes those files.

**It works from the finished figure, not from gated draw calls.** Hand-gating every
draw in thirteen figures would be thirteen chances to get it wrong. Instead the figure
is built exactly as it is, each artist is placed in a band, and everything outside the
band being saved is hidden.

**Bands are declared, not guessed**, in `clicks.json`:

    "uq_types":   {"split_x": [41.5, 82.5]}        three columns, left to right
    "self_report": {"rects": [[0, 0, 62, 44.5], [62, 0, 124, 44.5]]}

in figure units - 10 to the inch, origin bottom left, so the whole canvas is
0..124 x 0..44.5, the same numbers the figure scripts lay out in. The slides repo
detects its panels by looking for `slab()` grounds; half the figures here are charts
with no slab under them, so detection would quietly find one panel and write nothing.
Declaring them is a line of JSON and it cannot fail silently.

**Anything outside every band stays visible in all layers.** That is what you want for
the furniture - the figure's heading and the grey scope line under it belong to the
slide, not to any one panel. It also means a band only has to be roughly right.

**Every layer is saved with the same bounding box**, measured from the whole figure
before anything is hidden, so the layers line up exactly when stacked. Getting this
from a tight bbox per layer would shift each one.

    from hgf_layers import layered_save as save      # in place of hgf_style.save

`layered_save` writes the figure as `save()` always did, then writes its layers if
`clicks.json` names it. The figure scripts each change one import line and nothing else.

    python3 slides/hgf_layers.py            # what every figure would split into
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
from matplotlib.transforms import Bbox

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hgf_style  # noqa: E402
from hgf_style import FIGURE_DPI  # noqa: E402

HERE = Path(__file__).resolve().parent
SPEC_FILE = HERE / "clicks.json"

#: Padding around the measured bbox, in inches. Matches hgf_style.save's default.
PAD = 0.04


# =====================================================================================
# the spec
# =====================================================================================
def load_spec(path: Path | None = None) -> dict:
    path = path or SPEC_FILE
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def bands_for(stem: str, spec: dict | None = None) -> list[tuple]:
    """The bands for a figure, as (x0, y0, x1, y1) rectangles in figure units."""
    spec = load_spec() if spec is None else spec
    entry = spec.get(stem)
    if not entry:
        return []
    if "rects" in entry:
        return [tuple(r) for r in entry["rects"]]
    if "split_x" in entry:
        cw, ch = entry.get("canvas", hgf_style.GEOM["canvas"])
        edges = [0.0, *entry["split_x"], float(cw)]
        return [(edges[i], 0.0, edges[i + 1], float(ch))
                for i in range(len(edges) - 1)]
    raise SystemExit(f"{SPEC_FILE.name}: {stem!r} needs 'split_x' or 'rects'")


def groups_for(stem: str, n_bands: int, spec: dict | None = None) -> list[list[int]]:
    """Which bands share a click. Default: one band per click, in order."""
    spec = load_spec() if spec is None else spec
    declared = (spec.get(stem) or {}).get("groups")
    if not declared:
        return [[i] for i in range(n_bands)]
    groups = [list(g) for g in declared]
    seen = sorted(i for g in groups for i in g)
    if seen != list(range(n_bands)):
        raise SystemExit(f"{SPEC_FILE.name}: {stem!r} groups cover {seen}, but there "
                         f"are {n_bands} bands - each band goes in exactly one group")
    return groups


# =====================================================================================
# measuring
# =====================================================================================
def _units(fig, bbox):
    """A pixel bbox as figure units: 10 to the inch, origin at the canvas corner."""
    s = 10.0 / fig.dpi
    return (bbox.x0 * s, bbox.y0 * s, bbox.x1 * s, bbox.y1 * s)


def _finite(bb):
    return bb is not None and all(map(math.isfinite, (bb.x0, bb.y0, bb.x1, bb.y1)))


def _extent(artist, renderer):
    """An artist's pixel extent, or None when it genuinely has none.

    A PathCollection (ax.scatter) reports an infinite window extent, and a zero-area
    box is still a box - a vertical rule is exactly that. Both were silently dropped
    at first, which leaves the artist in no band and therefore visible in every layer.
    """
    try:
        bb = artist.get_window_extent(renderer)
    except Exception:
        return None
    if _finite(bb):
        return _clipped(artist, bb)
    datalim = getattr(artist, "get_datalim", None)
    if datalim is not None and getattr(artist, "axes", None) is not None:
        try:
            db = datalim(artist.axes.transData)
        except Exception:
            return None
        if _finite(db):
            return _clipped(artist, Bbox(artist.axes.transData.transform(db.get_points())))
    return None


def _clipped(artist, bb):
    """The part of `bb` the artist can actually draw in.

    A clipped artist whose data runs past its axes limits - a strip of dots with a
    few outliers above the view - measures far beyond the canvas, so its centre lands
    outside every band and it became furniture, on screen before its own click.
    """
    if not artist.get_clip_on():
        return bb
    clip = artist.get_clip_box()
    if clip is None and getattr(artist, "axes", None) is not None:
        clip = artist.axes.bbox
    if clip is None:
        return bb
    inter = Bbox.intersection(bb, clip)
    return inter if inter is not None else bb


def content(fig):
    """Every artist that can be hidden: the axes' children, plus figure-level text.

    Spines, the axes patch and the axes themselves are skipped - hiding an axes would
    take its whole band's content with it whatever band an individual artist is in.
    """
    out = []
    for ax in fig.axes:
        for artist in ax.get_children():
            if isinstance(artist, (matplotlib.spines.Spine, matplotlib.axis.Axis)):
                continue
            if artist is ax.patch or isinstance(artist, matplotlib.axes.Axes):
                continue
            out.append(artist)
    out.extend(fig.texts)
    return out


def _separation(bands):
    """Which way the bands are laid out: "x", "y", or "both"."""
    ys = {(round(b[1], 3), round(b[3], 3)) for b in bands}
    xs = {(round(b[0], 3), round(b[2], 3)) for b in bands}
    if len(ys) == 1 and len(xs) > 1:
        return "x"
    if len(xs) == 1 and len(ys) > 1:
        return "y"
    return "both"


#: An artist narrower than this much of the narrowest band is placed by its centre;
#: anything wider is placed by overlap, so it can be recognised as spanning.
SMALL = 0.5
#: A wide artist needs this much of itself inside one band to belong to it. Below
#: that it spans bands and is furniture - a chance line ruled across every column.
CLAIM = 0.6
#: ...and may run out of that band by at most this much of the band's own width.
#: CLAIM alone let a gridline ruled across a two-band figure join the wider band,
#: and the click that revealed it then drew the line over the other band's dots.
SPILL = 0.25


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def assign(fig, bands, renderer):
    """([artists] per band, [furniture]).

    Membership is by overlap along the axis the bands are separated on, not by centre
    alone. Centre alone put a rule drawn across all four tier columns in whichever
    column its midpoint happened to land in, and it then vanished until that column's
    click. An artist that no single band covers `CLAIM` of is furniture: it belongs to
    the plate rather than to a panel, and stays on screen for every click.

    Furniture is also what an artist outside every band becomes, which is how the
    figure's heading and the grey scope line under it stay put - declare bands that
    do not reach them and they are handled.
    """
    axis = _separation(bands)
    groups = [[] for _ in bands]
    furniture = []
    span = (lambda b: (b[0], b[2])) if axis != "y" else (lambda b: (b[1], b[3]))
    narrowest = min(hi - lo for lo, hi in map(span, bands)) or 1.0

    for artist in content(fig):
        bb = _extent(artist, renderer)
        if bb is None:
            furniture.append(artist)
            continue
        x0, y0, x1, y1 = _units(fig, bb)
        inside = [i for i, (bx0, by0, bx1, by1) in enumerate(bands)
                  if bx0 <= (x0 + x1) / 2 <= bx1 and by0 <= (y0 + y1) / 2 <= by1]
        a0, a1 = (x0, x1) if axis != "y" else (y0, y1)
        if axis == "both" or (a1 - a0) <= SMALL * narrowest:
            (groups[inside[0]] if inside else furniture).append(artist)
            continue
        # wide enough to straddle: it belongs to a band only if that band holds most
        # of it, and the centre has to agree - a long caption low under one column
        # is not that column's.
        fracs = [_overlap(a0, a1, *span(b)) / max(a1 - a0, 1e-9) for b in bands]
        best = max(range(len(bands)), key=lambda i: fracs[i])
        width = span(bands[best])[1] - span(bands[best])[0]
        spill = (a1 - a0) - _overlap(a0, a1, *span(bands[best]))
        if fracs[best] >= CLAIM and best in inside and spill <= SPILL * width:
            groups[best].append(artist)
        else:
            furniture.append(artist)
    return groups, furniture


# =====================================================================================
# writing
# =====================================================================================
def frame(fig) -> Bbox:
    """The one bounding box every layer - and the whole figure - is saved with.

    Measured before anything is hidden. A per-layer tight bbox would frame each
    layer differently and the stack would jitter on the slide.
    """
    fig.canvas.draw()
    return fig.get_tightbbox(fig.canvas.get_renderer()).padded(PAD)


def render_layers(fig, stem: Path, spec: dict | None = None, box: Bbox | None = None,
                  verbose=False):
    """Write one transparent PNG per click. Returns the paths, in reveal order.

    Called with the figure still open, straight after it was saved whole, so nothing
    is drawn twice.
    """
    stem = Path(stem)
    bands = bands_for(stem.name, spec)
    if len(bands) < 2:
        _clear_stale(stem, keep=())
        return []

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    groups, furniture = assign(fig, bands, renderer)
    layers = groups_for(stem.name, len(bands), spec)
    box = frame(fig) if box is None else box

    paths = []
    for i, wanted in enumerate(layers):
        hidden = []
        # Furniture goes in the first layer only. The layers are stacked and the first
        # stays on screen, so furniture repeated in a later layer is painted on top of
        # the panels revealed before it: a gridline across the tier-1 dots, an opaque
        # ground spanning every column hiding text an earlier click put there.
        hide = [g for j, g in enumerate(groups) if j not in wanted]
        if i > 0:
            hide.append(furniture)
        for group in hide:
            for artist in group:
                if artist.get_visible():
                    artist.set_visible(False)
                    hidden.append(artist)
        path = stem.with_name(f"{stem.name}_l{i}.png")
        fig.savefig(path, bbox_inches=box, transparent=True, dpi=FIGURE_DPI)
        paths.append(path)
        for artist in hidden:
            artist.set_visible(True)

    # A figure that loses a band leaves its last layer behind, and figure_slide places
    # every _l*.png it finds - so a slide keeps showing a panel the figure stopped
    # drawing. Delete the tail rather than trusting it.
    stale = _clear_stale(stem, keep=paths)
    if verbose:
        print(f"    {len(bands)} bands -> {len(layers)} layer(s), "
              f"{len(layers) - 1} click(s)"
              + (f", {stale} stale removed" if stale else ""))
    return paths


def _clear_stale(stem: Path, keep) -> int:
    keep = {Path(p) for p in keep}
    gone = 0
    for path in sorted(stem.parent.glob(f"{stem.name}_l*.png")):
        if path not in keep:
            path.unlink()
            gone += 1
    return gone


def layered_save(fig, stem, formats=("png", "svg"), pad=PAD, transparent=True,
                 dpi=None):
    """`hgf_style.save`, plus the click layers when clicks.json names this figure.

    Drop-in: the figure scripts import this in place of `save` and change nothing else.

    A figure with layers is written with an explicit box rather than through
    `bbox_inches="tight"`, and the layers get the same one. Left to resolve "tight"
    itself, matplotlib rounded the whole figure and the layers to sizes a pixel
    apart on two of the twelve - harmless on the slide, since only the layers are
    placed there, but it means the PNG and its layers are not the same frame, and
    it made the composite check unable to compare them.
    """
    stem = Path(stem)
    if len(bands_for(stem.name)) < 2:
        _clear_stale(stem, keep=())
        return hgf_style.save(fig, stem, formats=formats, pad=pad,
                              transparent=transparent, dpi=dpi)

    box = frame(fig)
    written = []
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        path = stem.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": box, "transparent": transparent}
        if ext == "svg":                       # drop the Date stamp, as save() does,
            kwargs["metadata"] = {"Date": None}   # so a rebuild is a no-op in git
        if ext in ("png", "jpg", "jpeg", "tif", "tiff"):
            kwargs["dpi"] = dpi or FIGURE_DPI
        fig.savefig(path, **kwargs)
        written.append(path)
    render_layers(fig, stem, box=box)
    return written


# =====================================================================================
def verify(figdir: Path, spec: dict) -> int:
    """Composite each figure's layers and compare with the figure drawn whole.

    The failure that matters is an artist in no band and no furniture list: it is
    then in no layer either, and the slide quietly loses it. Compositing catches
    that, where counting layers does not.

    The second failure is occlusion: a later layer painting over what an earlier
    click revealed - a gridline across dots, an opaque ground hiding a note. The
    stack is therefore also flattened onto white, the way the slide shows it after
    the last click, and compared colour for colour with the whole figure. Furniture
    sits in the first layer only, so nothing is drawn twice and the two match exactly
    up to rounding.
    """
    from PIL import Image
    import numpy as np

    bad = 0
    print(f"{'figure':<30}{'layers':>8}{'missing px':>12}{'covered px':>12}   size")
    for stem in spec:
        base = figdir / f"{stem}.png"
        layers = sorted(figdir.glob(f"{stem}_l*.png"),
                        key=lambda q: int(q.stem.rsplit("_l", 1)[1]))
        if not base.exists() or not layers:
            print(f"  {stem:<28}{'-':>8}{'no figures':>12}{'':>12}   "
                  f"run build_deck.py --figures")
            bad += 1
            continue
        whole = Image.open(base).convert("RGBA")
        sizes = {Image.open(q).size for q in layers}
        if sizes != {whole.size}:
            print(f"  {stem:<28}{len(layers):>8}{'MISMATCH':>12}{'':>12}   "
                  f"whole {whole.size}, layers {sorted(sizes)}")
            bad += 1
            continue
        stack = Image.new("RGBA", whole.size, (0, 0, 0, 0))
        for q in layers:
            stack = Image.alpha_composite(stack, Image.open(q).convert("RGBA"))
        # a pixel the whole figure inks solidly but the stack does not ink at all
        wa = np.asarray(whole)[..., 3].astype(int)
        sa = np.asarray(stack)[..., 3].astype(int)
        missing = int(((wa > 40) & (sa <= 8)).sum())
        # flattened onto white: ink the whole figure shows that the stack paints over
        white = Image.new("RGBA", whole.size, (255, 255, 255, 255))
        wf = np.asarray(Image.alpha_composite(white, whole))[..., :3].astype(int)
        sf = np.asarray(Image.alpha_composite(white, stack))[..., :3].astype(int)
        covered = int((np.abs(wf - sf).max(-1) > 40).sum())
        flag = ("" if missing == 0 else "   <-- CONTENT LOST") + \
               ("" if covered == 0 else "   <-- PAINTED OVER")
        print(f"  {stem:<28}{len(layers):>8}{missing:>12}{covered:>12}   {whole.size}{flag}")
        bad += missing > 0 or covered > 0
    print(f"\n  {len(spec)} figures, {bad} with a problem")
    return 1 if bad else 0


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", type=Path, default=SPEC_FILE)
    ap.add_argument("--verify", action="store_true",
                    help="composite each figure's layers and check nothing was lost")
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    args = ap.parse_args(argv)
    if args.verify:
        return verify(args.figures, load_spec(args.spec))
    spec = load_spec(args.spec)
    if not spec:
        print(f"{args.spec.name} declares no figures")
        return 0
    total = 0
    print(f"{'figure':<30}{'bands':>7}{'clicks':>8}   how")
    for stem, entry in spec.items():
        bands = bands_for(stem, spec)
        clicks = len(groups_for(stem, len(bands), spec)) - 1
        how = ("split_x " + str(entry["split_x"])) if "split_x" in entry \
            else f"{len(entry.get('rects', []))} rects"
        print(f"  {stem:<28}{len(bands):>7}{clicks:>8}   {how}")
        total += max(clicks, 0)
    print(f"\n  {len(spec)} figures, {total} clicks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
