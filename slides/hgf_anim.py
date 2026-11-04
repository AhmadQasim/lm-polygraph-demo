#!/usr/bin/env python3
"""Click-by-click entrance animations, written straight into the slide XML.

**Vendored from ~/projects/slides/scripts/hgf_anim.py, unchanged.** It is slide-level
and knows nothing about any particular deck. Keep it a copy, not a fork.

python-pptx has no animation API — the `p:timing` tree simply is not modelled — so this
builds it as XML and appends it to the slide. The effect used is the plainest one there
is: **Appear**, `presetID 1`, which sets `style.visibility` to `visible`. A shape with an
entrance effect is hidden by PowerPoint until its effect runs, so putting a shape in a
click group is all that is needed to hide it beforehand.

    from hgf_anim import appear_on_click, split_by_x
    groups = split_by_x(slide, [5.90, 9.40])      # three bands, left to right
    appear_on_click(slide, groups[1:])            # band 0 is on screen from the start

One click advances one group, and every shape in a group appears together: the first
carries `nodeType="clickEffect"` and the rest `withEffect`, which is how PowerPoint's own
files express "with previous".

**What this does not do.** No exit effects, no motion, no timing beyond instant. That is
deliberate: a panel that flies in reads as a slide transition, and the point here is
only to withhold a panel until it is being talked about.

Exporting elsewhere: LibreOffice (`build_deck.py --pdf`) ignores `p:timing` and
flattens every layer, so the PDF handout shows each slide finished rather than one
page per click — which is what a handout wants.
"""

from __future__ import annotations

from lxml import etree
from pptx.util import Emu

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"

#: Never animated: the slide furniture that should be on screen from the first frame.
ALWAYS_VISIBLE = ("NAV_", "GEN_SLIDENUM", "GEN_LEAD", "GEN_CAPTION", "Title")


def _is_furniture(shape) -> bool:
    name = shape.name or ""
    return name.startswith(ALWAYS_VISIBLE)


def split_by_x(slide, bounds, overrides=None):
    """Group the slide's content shapes into left-to-right bands.

    `bounds` are the x positions in inches where one band ends and the next begins, so
    two bounds give three bands. `overrides` maps a shape-name prefix to a band index,
    for the pieces whose position does not say which panel they belong to — a caption bar
    that spans the slide, or a return arrow drawn under all three.
    """
    overrides = overrides or {}
    groups = [[] for _ in range(len(bounds) + 1)]
    for shape in slide.shapes:
        if _is_furniture(shape):
            continue
        band = None
        for prefix, index in overrides.items():
            if (shape.name or "").startswith(prefix):
                band = index
                break
        if band is None:
            x = Emu(shape.left).inches if shape.left is not None else 0.0
            band = sum(1 for b in bounds if x >= b)
        groups[band].append(shape)
    return groups


def _el(tag, **attrs):
    return etree.SubElement if False else etree.Element(
        f"{{{P_NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attrs.items()})


def _appear(parent, shape_id, node_type, counter):
    """One shape's Appear effect, as PowerPoint writes it."""
    par = etree.SubElement(parent, f"{{{P_NS}}}par")
    ctn = etree.SubElement(par, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "presetID": "1", "presetClass": "entr",
        "presetSubtype": "0", "fill": "hold", "grpId": "0", "nodeType": node_type})
    st = etree.SubElement(ctn, f"{{{P_NS}}}stCondLst")
    etree.SubElement(st, f"{{{P_NS}}}cond", {"delay": "0"})
    child = etree.SubElement(ctn, f"{{{P_NS}}}childTnLst")
    set_ = etree.SubElement(child, f"{{{P_NS}}}set")
    cbhvr = etree.SubElement(set_, f"{{{P_NS}}}cBhvr")
    inner = etree.SubElement(cbhvr, f"{{{P_NS}}}cTn",
                             {"id": str(next(counter)), "dur": "1", "fill": "hold"})
    inner_st = etree.SubElement(inner, f"{{{P_NS}}}stCondLst")
    etree.SubElement(inner_st, f"{{{P_NS}}}cond", {"delay": "0"})
    tgt = etree.SubElement(cbhvr, f"{{{P_NS}}}tgtEl")
    etree.SubElement(tgt, f"{{{P_NS}}}spTgt", {"spid": str(shape_id)})
    attrs = etree.SubElement(cbhvr, f"{{{P_NS}}}attrNameLst")
    etree.SubElement(attrs, f"{{{P_NS}}}attrName").text = "style.visibility"
    to = etree.SubElement(set_, f"{{{P_NS}}}to")
    etree.SubElement(to, f"{{{P_NS}}}strVal", {"val": "visible"})


def play_media(slide, shape, auto=True, verbose=True):
    """Make an embedded movie play, either with the slide or on the first click.

    python-pptx's `add_movie` writes the shape and the media relationship but no
    playback: without a `p:timing` entry PowerPoint shows a still with a play button and
    waits to be poked. This writes the media call PowerPoint itself writes — a
    `playFrom(0.0)` command targeted at the shape — inside the same mainSeq shape as
    `appear_on_click`, so the two cannot both claim the slide's timing.

    `auto=True` starts it on the slide; `auto=False` starts it on the first click. Both
    then answer the arrow keys like any other build.
    """
    sld = slide._element
    for old in sld.findall(f"{{{P_NS}}}timing"):
        sld.remove(old)

    counter = iter(range(1, 10_000))
    timing = etree.SubElement(sld, f"{{{P_NS}}}timing")
    tnlst = etree.SubElement(timing, f"{{{P_NS}}}tnLst")
    root_par = etree.SubElement(tnlst, f"{{{P_NS}}}par")
    root = etree.SubElement(root_par, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "dur": "indefinite", "restart": "never",
        "nodeType": "tmRoot"})
    root_children = etree.SubElement(root, f"{{{P_NS}}}childTnLst")
    seq = etree.SubElement(root_children, f"{{{P_NS}}}seq",
                           {"concurrent": "1", "nextAc": "seek"})
    seq_ctn = etree.SubElement(seq, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "dur": "indefinite", "nodeType": "mainSeq"})
    seq_children = etree.SubElement(seq_ctn, f"{{{P_NS}}}childTnLst")

    outer_par = etree.SubElement(seq_children, f"{{{P_NS}}}par")
    outer = etree.SubElement(outer_par, f"{{{P_NS}}}cTn",
                             {"id": str(next(counter)), "fill": "hold"})
    outer_st = etree.SubElement(outer, f"{{{P_NS}}}stCondLst")
    etree.SubElement(outer_st, f"{{{P_NS}}}cond",
                     {"delay": "0" if auto else "indefinite"})
    outer_children = etree.SubElement(outer, f"{{{P_NS}}}childTnLst")
    inner_par = etree.SubElement(outer_children, f"{{{P_NS}}}par")
    inner = etree.SubElement(inner_par, f"{{{P_NS}}}cTn",
                             {"id": str(next(counter)), "fill": "hold"})
    inner_st = etree.SubElement(inner, f"{{{P_NS}}}stCondLst")
    etree.SubElement(inner_st, f"{{{P_NS}}}cond", {"delay": "0"})
    effects = etree.SubElement(inner, f"{{{P_NS}}}childTnLst")

    effect_par = etree.SubElement(effects, f"{{{P_NS}}}par")
    effect = etree.SubElement(effect_par, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "presetID": "1", "presetClass": "mediacall",
        "presetSubtype": "0", "fill": "hold", "nodeType":
        "afterEffect" if auto else "clickEffect"})
    effect_st = etree.SubElement(effect, f"{{{P_NS}}}stCondLst")
    etree.SubElement(effect_st, f"{{{P_NS}}}cond", {"delay": "0"})
    effect_children = etree.SubElement(effect, f"{{{P_NS}}}childTnLst")
    cmd = etree.SubElement(effect_children, f"{{{P_NS}}}cmd",
                           {"type": "call", "cmd": "playFrom(0.0)"})
    cbhvr = etree.SubElement(cmd, f"{{{P_NS}}}cBhvr")
    etree.SubElement(cbhvr, f"{{{P_NS}}}cTn",
                     {"id": str(next(counter)), "dur": "1", "fill": "hold"})
    tgt = etree.SubElement(cbhvr, f"{{{P_NS}}}tgtEl")
    etree.SubElement(tgt, f"{{{P_NS}}}spTgt", {"spid": str(shape.shape_id)})

    for tag, evt in ((f"{{{P_NS}}}prevCondLst", "onPrev"),
                     (f"{{{P_NS}}}nextCondLst", "onNext")):
        cond_lst = etree.SubElement(seq, tag)
        cond = etree.SubElement(cond_lst, f"{{{P_NS}}}cond", {"evt": evt, "delay": "0"})
        cond_tgt = etree.SubElement(cond, f"{{{P_NS}}}tgtEl")
        etree.SubElement(cond_tgt, f"{{{P_NS}}}sldTgt")

    if verbose:
        print(f"    video plays {'with the slide' if auto else 'on the first click'}")
    return 1


def appear_on_click(slide, groups, verbose=True):
    """Make each group of shapes appear on its own click. Returns the click count.

    Idempotent: an existing `p:timing` is replaced, so a rebuild does not stack two
    animations on one slide.
    """
    groups = [[s for s in g] for g in groups if g]
    sld = slide._element
    for old in sld.findall(f"{{{P_NS}}}timing"):
        sld.remove(old)
    if not groups:
        return 0

    counter = iter(range(1, 10_000))
    timing = etree.SubElement(sld, f"{{{P_NS}}}timing")
    tnlst = etree.SubElement(timing, f"{{{P_NS}}}tnLst")
    root_par = etree.SubElement(tnlst, f"{{{P_NS}}}par")
    root = etree.SubElement(root_par, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "dur": "indefinite", "restart": "never",
        "nodeType": "tmRoot"})
    root_children = etree.SubElement(root, f"{{{P_NS}}}childTnLst")
    seq = etree.SubElement(root_children, f"{{{P_NS}}}seq",
                           {"concurrent": "1", "nextAc": "seek"})
    seq_ctn = etree.SubElement(seq, f"{{{P_NS}}}cTn", {
        "id": str(next(counter)), "dur": "indefinite", "nodeType": "mainSeq"})
    seq_children = etree.SubElement(seq_ctn, f"{{{P_NS}}}childTnLst")

    for group in groups:
        click_par = etree.SubElement(seq_children, f"{{{P_NS}}}par")
        click = etree.SubElement(click_par, f"{{{P_NS}}}cTn",
                                 {"id": str(next(counter)), "fill": "hold"})
        click_st = etree.SubElement(click, f"{{{P_NS}}}stCondLst")
        etree.SubElement(click_st, f"{{{P_NS}}}cond", {"delay": "indefinite"})
        click_children = etree.SubElement(click, f"{{{P_NS}}}childTnLst")
        inner_par = etree.SubElement(click_children, f"{{{P_NS}}}par")
        inner = etree.SubElement(inner_par, f"{{{P_NS}}}cTn",
                                 {"id": str(next(counter)), "fill": "hold"})
        inner_st = etree.SubElement(inner, f"{{{P_NS}}}stCondLst")
        etree.SubElement(inner_st, f"{{{P_NS}}}cond", {"delay": "0"})
        effects = etree.SubElement(inner, f"{{{P_NS}}}childTnLst")
        for i, shape in enumerate(group):
            _appear(effects, shape.shape_id,
                    "clickEffect" if i == 0 else "withEffect", counter)

    # onPrev/onNext, so the sequence answers the arrow keys like any other build
    for tag, evt in ((f"{{{P_NS}}}prevCondLst", "onPrev"),
                     (f"{{{P_NS}}}nextCondLst", "onNext")):
        cond_lst = etree.SubElement(seq, tag)
        cond = etree.SubElement(cond_lst, f"{{{P_NS}}}cond",
                                {"evt": evt, "delay": "0"})
        tgt = etree.SubElement(cond, f"{{{P_NS}}}tgtEl")
        etree.SubElement(tgt, f"{{{P_NS}}}sldTgt")

    if verbose:
        sizes = " + ".join(str(len(g)) for g in groups)
        print(f"    {len(groups)} click(s), {sizes} shapes")
    return len(groups)
