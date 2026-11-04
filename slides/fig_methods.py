#!/usr/bin/env python3
"""Figures for the talk's opening: the problem, and the families of estimators.

These are the slides that come *before* the results in `plot_bioasq.py`:

  01 confidently_wrong     - four real wrong answers the model was sure of
  02 selective_generation  - what an uncertainty score buys: flag, keep, accuracy
  03 uq_types              - the five signal types, the anchor slide
  04 information_based     - reading the token probabilities (T1)
  05 self_report           - asking the model to judge itself (T2)
  06 consistency           - asking again: sampling, with and without NLI (T3, T4)
  07 trained_probes        - supervised detectors on hidden states (not run)
  08 lm_polygraph          - the library, and how it shares statistics
  09 types_by_tier         - every estimator run, by signal type and tier
  02b uq_map               - two questions place every family: access, and answers drawn
  09b dataset              - the benchmark: one real question, the sample, the grading
  15 takeaways            - five things to take home
  14 limitations           - one dataset, two small models: an introduction, not a benchmark

Run:  .venv/bin/python slides/fig_methods.py

SOURCES. The family names and the method names outside our own run follow the
taxonomy in Vazhentsev et al., "Uncertainty Quantification for Large Language
Diffusion Models", arXiv:2605.14570 (Sec. 2 and App. D.2). Only the categories
and names are taken from it - none of its results, which are on diffusion LMs.
Self-report is not one of its groups; it is added because p(True) is in our
run. Formulas were checked against the lm-polygraph 0.7.0 source in .venv.

Slides 01 and 02 use the measured 7B run, so they need its cache; the other
seven are diagrams and need nothing.

Drawn in the same house style as `plot_bioasq.py`, on the same 124 x 44.5 u
canvas, and with the same colour key: blue is tier 1, blue 60 is tiers 2-4,
orange is a wrong answer. Each figure gets its slide's title / lead / caption
written beside it as `slides/markdowns/{nn}_slide_{name}.md`.
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

SLIDES = pathlib.Path(__file__).resolve().parent
ROOT = SLIDES.parent
sys.path.insert(0, str(SLIDES))
sys.path.insert(0, str(ROOT / "scripts"))
import bioasq  # noqa: E402
from hgf_style import (  # noqa: E402
    BLUE, DARK, GRAY22, GRAY45, GRAY85, BLUE60, ORANGE, ORANGE_DARK, PALE,
    HIGHLIGHT, TYPE, WEIGHT, GEOM, REFERENCE_DASH, HGF, PURPLE,
    halvar, hermann, mono, shade, slab, use_hgf,
)
from hgf_layers import layered_save as save  # noqa: E402  # writes the click layers too
from plot_bioasq import PRETTY, TIER_GLOSS, load, wrap_measured  # noqa: E402

FIGDIR = SLIDES / "figures"
MDDIR = SLIDES / "markdowns"
CACHE_7B = ROOT / "outputs" / "bioasq_Qwen2.5-7B-Instruct_n1000_seed0_t4.json"
CACHE_3B = ROOT / "outputs" / "bioasq_Qwen2.5-3B-Instruct_n1000_seed0_t4.json"
MANUAL = ROOT / "manual_labels.json"

CW, CH = GEOM["canvas"]   # 124 x 44.5 u
TIER1, TIER_REST = BLUE, BLUE60
WRONG, WRONG_TEXT = ORANGE, ORANGE_DARK


# --- colour roles: one theme colour per idea, the same on every slide -------------
def deepen(hex_color, f):
    """`f` of the colour, the rest black: a text-safe ink for a light theme colour."""
    h = hex_color.lstrip("#")
    return "#%02X%02X%02X" % tuple(round(int(h[i:i + 2], 16) * f) for i in (0, 2, 4))


#: The five method families. `fill` marks and grounds, `ink` sets type (>= 4.5:1 on
#: white), `ground` is the pale panel colour. Used by the map, the cards, the family
#: slides, the type-by-tier grid and the library catalogue.
FAMILY = {
    "Information": BLUE,
    "Self-report": PURPLE,
    "Consistency": HGF["aerospace"],
    "Sampling + likelihood": HGF["information"],
    "Trained probe": HGF["green"],
}
_FAMILY_INK = {"Consistency": deepen(HGF["aerospace"], 0.6),
               "Trained probe": deepen(HGF["green"], 0.62)}


def fam_fill(name):
    return FAMILY[name]


def fam_ink(name):
    return _FAMILY_INK.get(name, FAMILY[name])


def fam_ground(name):
    return shade(FAMILY[name], 12)


#: The parts of LM-Polygraph, on the pipeline slide and in the code highlighting.
LIB = {
    "model": (HGF["earth"], HGF["earth"]),
    "stats": (HGF["mint"], deepen(HGF["mint"], 0.55)),
    "estimator": (BLUE, BLUE),
    "manager": (DARK, DARK),
    "metric": (HGF["energy"], deepen(HGF["energy"], 0.55)),
}
CAUTION, CAUTION_INK = HGF["energy"], deepen(HGF["energy"], 0.55)   # limitations
TAKEAWAY_INK = _FAMILY_INK["Trained probe"]                           # take-home
PERFECT_INK = deepen(HGF["green"], 0.62)                              # an oracle line

S = TYPE  # sizes, by role


# =====================================================================================
# drawing helpers - everything is placed in figure units on one full-canvas axes
# =====================================================================================
def board():
    fig = plt.figure(figsize=(CW / 10, CH / 10))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, CW)
    ax.set_ylim(0, CH)
    ax.axis("off")
    return fig, ax


def put(ax, x, y, s, font, color=DARK, ha="left", va="baseline", **kw):
    return ax.text(x, y, s, fontproperties=font, color=color, ha=ha, va=va, **kw)


def lines(fig, ax, x, y, text, font, width, color=DARK, pitch=None, max_lines=None):
    """Wrap `text` to `width` u (measured) and draw it downwards from baseline y.

    Returns the baseline of the next free line.
    """
    pitch = pitch or font.get_size() * 0.185
    out = wrap_measured(fig, text, font, width)
    if max_lines and len(out) > max_lines:
        raise ValueError(f"{text!r} needs {len(out)} lines at {width} u")
    for i, ln in enumerate(out):
        put(ax, x, y - i * pitch, ln, font, color)
    return y - len(out) * pitch


def tag(ax, x, y, text, color, w=None, ha="left"):
    """A small filled pill carrying a tier label, white Hermann on the tier hue."""
    w = w or 2.2 + 1.15 * len(text)
    x0 = x if ha == "left" else x - w
    slab(ax, x0, y - 0.9, w, 3.0, color, z=3, radius=0.6)
    put(ax, x0 + w / 2, y + 0.6, text, hermann(S["gloss"]), "white", ha="center",
        va="center", zorder=4)
    return w


def arrow(ax, x0, y0, x1, y1, color=GRAY85, lw=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=11, color=color,
                                 lw=lw or WEIGHT["context"], zorder=2))


def rule(ax, x0, x1, y, color=GRAY22, lw=None):
    ax.plot([x0, x1], [y, y], color=color, lw=lw or WEIGHT["grid"],
            solid_capstyle="butt", zorder=1)


def header(ax, x, y, text, color=GRAY85, ha="left"):
    put(ax, x, y, text.upper(), hermann(S["category"]), color, ha=ha)


def tier_hue(t):
    return TIER1 if t == 1 else TIER_REST


def method_table(fig, ax, x0, x1, top, rows, pitch=3.7, name_w=21.0, accent=GRAY45):
    """Right-hand method list: name | what it reads | tier pill (or 'not run').

    rows: (name, gloss, tier) with tier an int, a string such as "T3 · T4",
    or None for a method that is named but was not run here.
    """
    header(ax, x0, top, "method")
    header(ax, x0 + name_w, top, "reads")
    header(ax, x1, top, "tier", ha="right")
    y = top - 1.6
    rule(ax, x0, x1, y, accent, lw=WEIGHT["context"])
    gloss_w = x1 - x0 - name_w - 9.5
    for name, gloss, tier, *ink in rows:
        n_lines = len(wrap_measured(fig, gloss, halvar(S["tick"]), gloss_w))
        h = pitch + (n_lines - 1) * 2.05
        base = y - 2.65
        run = tier is not None
        put(ax, x0, base, name, halvar(S["tick"], bold=True),
            (ink[0] if ink else DARK) if run else GRAY85)
        lines(fig, ax, x0 + name_w, base, gloss, halvar(S["tick"]), gloss_w,
              DARK if run else GRAY85, pitch=2.05)
        if tier is None:
            put(ax, x1, base, "not run", halvar(S["tick"], italic=True), GRAY85,
                ha="right")
        else:
            label = f"T{tier}" if isinstance(tier, int) else tier
            hue = TIER1 if label == "T1" else TIER_REST
            tag(ax, x1, base, label, hue, ha="right")
        y -= h
        rule(ax, x0, x1, y)
    return y


def footnote(ax, text):
    put(ax, 0, 0.6, text, halvar(S["gloss"]), GRAY85)


# =====================================================================================
# 01. confidently wrong
# =====================================================================================
# Hand-picked from the 7B run: answers that are wrong on any reading, not grader
# misses. The string grader marks several right answers wrong ("IL-6" against
# "interleukin-6", "X-linked recessive" against "X- linked recessive"), and the
# most confident "errors" in the cache are mostly those - so every row below was
# read by hand. Matched by question prefix; the numbers come from the cache.
WRONG_EXAMPLES = [
    "PDQ39 questionnaires is design for which disease?",
    "How many tissue kallikrein genes are present in the human genome?",
    "Inhaled Molgramostim can be used for treatment of which disease?",
    "What disease can be treatment with Bulevirtide?",
]


def fig_confidently_wrong(d, y):
    msp = np.asarray(d["scores"]["MaximumSequenceProbability"])
    prob = np.exp(-msp)  # MSP = -log P(answer | question) in lm-polygraph
    rows = []
    for q in WRONG_EXAMPLES:
        i = next(k for k, qq in enumerate(d["questions"]) if qq.startswith(q))
        assert y[i] == 1, f"{q!r} is graded correct in the cache"
        rows.append((d["questions"][i], d["generations"][i], d["golds"][i][0],
                     prob[i]))

    fig, ax = board()
    xq, xa, xr, xp = 0.0, 62.0, 84.0, CW
    top = 42.5
    header(ax, xq, top, "BioASQ question")
    header(ax, xa, top, "model answered")
    header(ax, xr, top, "reference")
    header(ax, xp, top, "model's probability", ha="right")
    rule(ax, 0, CW, top - 1.6, GRAY45)

    pitch = 8.6
    y0 = top - 1.6
    for k, (q, gen, gold, p) in enumerate(rows):
        base = y0 - 3.6 - k * pitch
        lines(fig, ax, xq, base, q, halvar(S["tick"]), xa - xq - 3, pitch=2.1,
              max_lines=2)
        put(ax, xa, base, gen, halvar(S["value"], bold=True), WRONG_TEXT)
        lines(fig, ax, xr, base, gold, halvar(S["tick"]), xp - xr - 16, pitch=2.1,
              max_lines=2)
        put(ax, xp, base - 0.6, f"{p:.3f}", hermann(S["claim"]), WRONG_TEXT, ha="right")
        rule(ax, 0, CW, y0 - (k + 1) * pitch)

    footnote(ax, "Qwen2.5-7B-Instruct, greedy  ·  probability = P(answer | question), "
                 "the product of its token probabilities  ·  all four graded wrong "
                 "and checked by hand  ·  questions verbatim [sic]")
    out = save(fig, FIGDIR / "confidently_wrong", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 02. selective generation
# =====================================================================================
def kept_accuracy(u, y, rejected):
    """Accuracy of what is kept after flagging the `rejected` most uncertain."""
    order = np.argsort(u, kind="stable")  # most certain first
    correct = (y[order] == 0).astype(float)
    n = len(u)
    out = []
    for r in rejected:
        keep = max(1, int(round(n * (1 - r))))
        out.append(correct[:keep].mean())
    return np.array(out)


def fig_selective_generation(d, y):
    u = np.asarray(d["scores"]["MeanTokenEntropy"])
    acc = 1 - y.mean()
    r = np.linspace(0, 0.9, 91)
    curve = kept_accuracy(u, y, r)
    oracle = np.minimum(1.0, acc / (1 - r))

    fig, ax_b = board()
    ax = fig.add_axes((12 / CW, 8 / CH, 88 / CW, 32 / CH))
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0.2, 1.0)
    ax.axis("off")

    for t in (0.2, 0.4, 0.6, 0.8, 1.0):
        ax.plot([0, 0.9], [t, t], color=GRAY22, lw=WEIGHT["grid"], zorder=0)
        ax.text(-0.012, t, f"{t:.0%}", fontproperties=halvar(S["tick"]), color=GRAY85,
                ha="right", va="center")
    for t in (0, 0.25, 0.5, 0.75, 0.9):
        ax.text(t, 0.17, f"{t:.0%}", fontproperties=halvar(S["tick"]), color=GRAY85,
                ha="center", va="top")
    ax.plot([0, 0.9], [0.2, 0.2], color=GRAY85, lw=WEIGHT["baseline"])

    ax.plot(r, oracle, color=PERFECT_INK, lw=WEIGHT["reference"], dashes=REFERENCE_DASH)
    ax.plot([0, 0.9], [acc, acc], color=GRAY85, lw=WEIGHT["reference"],
            dashes=REFERENCE_DASH)
    ax.plot(r, curve, color=TIER1, lw=WEIGHT["focus"])

    half = kept_accuracy(u, y, [0.5])[0]
    ax.plot([0.5], [half], "o", color=TIER1, ms=8, zorder=5)

    lab = halvar(S["tick"])
    ax_b.text(101.5, 8 + (oracle[-1] - 0.2) / 0.8 * 32, "perfect ranking",
              fontproperties=lab, color=PERFECT_INK, va="center")
    ax_b.text(101.5, 8 + (curve[-1] - 0.2) / 0.8 * 32, "mean token entropy",
              fontproperties=halvar(S["tick"], bold=True), color=TIER1, va="center")
    ax_b.text(101.5, 8 + (acc - 0.2) / 0.8 * 32, "random flags",
              fontproperties=lab, color=GRAY85, va="center")

    hx = 12 + 0.5 / 0.9 * 88
    hy = 8 + (half - 0.2) / 0.8 * 32
    put(ax_b, hx + 1.0, hy - 4.6, f"{half:.0%}", hermann(S["claim"]), TIER1)
    put(ax_b, hx + 8.6, hy - 4.6, f"right after flagging half, from {acc:.0%}",
        halvar(S["tick"]), DARK)

    put(ax_b, 12, 42.4, "accuracy of the answers kept".upper(), hermann(S["axis_title"]),
        GRAY85)
    put(ax_b, 12 + 44, 1.4, "answers flagged for review, most uncertain first".upper(),
        hermann(S["axis_title"]), GRAY85, ha="center")

    out = save(fig, FIGDIR / "selective_generation", formats=("png", "svg"))
    plt.close(fig)
    return out[0], acc, half


# =====================================================================================
# 02b. the map: two questions place every family
# =====================================================================================
#: (row, column) -> [(family, examples)]. Rows: one answer / several answers.
#: Columns: what the method can see of the model.
UQ_MAP = {
    (0, 0): [("Self-report", "verbalized confidence")],
    (0, 1): [("Information", "MSP, perplexity, token entropy"),
             ("Self-report", "p(True)")],
    (0, 2): [("Trained probe", "SAPLMA, SATRMD, Lookback Lens; needs labels")],
    (1, 0): [("Consistency", "lexical similarity, DegMat, KLE")],
    (1, 1): [("Sampling + likelihood", "MC sequence entropy, semantic entropy")],
}
MAP_COLS = [("text only", "any API, black-box"),
            ("token probabilities", "an open model, white-box"),
            ("hidden states", "white-box, inside the model")]
MAP_ROWS = [("one answer", "the answer it gave"),
            ("several answers", "the question sampled again")]


def fig_uq_map():
    fig, ax = board()
    left, gap = 24.0, 1.6
    cw = (CW - left - 2 * gap) / 3
    top_hdr, row_h = 36.5, 15.6
    for c, (head, sub) in enumerate(MAP_COLS):
        x = left + c * (cw + gap)
        put(ax, x, 41.6, head.upper(), hermann(S["category"]), DARK)
        put(ax, x, 39.2, sub, halvar(S["tick"]), GRAY85)
    arrow(ax, left, 44.0, CW - 2, 44.0)
    put(ax, 0, 43.4, "more access", halvar(S["tick"], italic=True), GRAY85)
    for r, (head, sub) in enumerate(MAP_ROWS):
        y = top_hdr - r * (row_h + gap)
        put(ax, 0, y - 3.4, head.upper(), hermann(S["category"]), DARK)
        put(ax, 0, y - 5.8, sub, halvar(S["tick"]), GRAY85)
        for c in range(3):
            x = left + c * (cw + gap)
            cell = UQ_MAP.get((r, c))
            if not cell:
                continue
            slab(ax, x, y - row_h, cw, row_h,
                 fam_ground(cell[0][0]) if len(cell) == 1 else PALE, z=0)
            yy = y - 3.6
            for fam, ex in cell:
                # a family-coloured bar beside each family, so a shared cell reads as two
                slab(ax, x + 1.0, yy - 3.4, 0.7, 5.2, fam_fill(fam), z=1, radius=0.3)
                put(ax, x + 2.6, yy, fam.upper(), hermann(S["panel_header"]), fam_ink(fam))
                yy = lines(fig, ax, x + 2.6, yy - 2.6, ex, halvar(S["tick"]),
                           cw - 4.0, DARK, pitch=2.05) - 1.8
    out = save(fig, FIGDIR / "uq_map", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 03. the five signal types
# =====================================================================================
UQ_TYPES = [
    # name, reads, access, paper's method names, tier label
    ("Information",
     "the token probabilities of the answer, from the pass that wrote it",
     "white-box",
     ["Max sequence prob.", "Perplexity", "Mean token entropy", "CCP",
      "Attention score", "RAUQ"],
     "T1"),
    ("Self-report",
     "the model's own verdict when asked whether its answer is true",
     "white- or black-box",
     ["p(True)", "Verbalized confidence"],
     "T2"),
    ("Consistency",
     "how far answers sampled again agree, in words or in meaning",
     "black-box",
     ["Lexical similarity", "LUQ", "DegMat", "Eccentricity", "EigVal", "KLE"],
     "T3 · T4"),
    ("Sampling + likelihood",
     "sampled answers, weighted by the probabilities the model gave them",
     "white-box",
     ["MC sequence entropy", "Semantic entropy", "SAR", "Semantic density",
      "EigenScore", "CoCoA"],
     "T3 · T4"),
    ("Trained probe",
     "hidden states or attention, through a classifier trained on labelled answers",
     "white-box + labels",
     ["SAPLMA", "SATRMD", "Lookback Lens", "TAD"],
     None),
]


def fig_uq_types():
    fig, ax = board()
    n = len(UQ_TYPES)
    gap = 2.4
    w = (CW - (n - 1) * gap) / n
    pad = 1.4
    for k, (name, reads, access, methods, tier) in enumerate(UQ_TYPES):
        x = k * (w + gap)
        slab(ax, x, 3.2, w, CH - 3.2, fam_ground(name), z=0)
        slab(ax, x, CH - 1.2, w, 1.2, fam_fill(name), z=1, radius=0.5)
        tw = w - 2 * pad
        yy = lines(fig, ax, x + pad, CH - 4.4, name, hermann(S["panel_header"]), tw,
                   fam_ink(name), pitch=2.7)
        yy = lines(fig, ax, x + pad, yy - 0.6, reads, halvar(S["tick"]), tw, DARK,
                   pitch=2.05)
        put(ax, x + pad, yy - 0.8, access, halvar(S["tick"], italic=True), fam_ink(name))
        yy -= 3.6
        rule(ax, x + pad, x + w - pad, yy + 1.2, fam_fill(name), lw=WEIGHT["context"])
        for m in methods:
            yy -= 2.05
            put(ax, x + pad, yy + 0.4, m, halvar(S["tick"]), GRAY85)
        if tier:
            tag(ax, x + pad, 5.4, tier, TIER1 if tier == "T1" else TIER_REST)
        else:
            put(ax, x + pad, 5.6, "not run here", halvar(S["tick"], italic=True), GRAY85)
    footnote(ax, "")
    out = save(fig, FIGDIR / "uq_types", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 04-07. one slide per family: how it works on the left, the methods on the right
# =====================================================================================
LEFT_W = 56.0
TABLE_X0 = 61.0


def formula_block(ax, x, y, name, tex, gloss, ink=DARK):
    put(ax, x, y, name.upper(), hermann(S["category"]), ink)
    put(ax, x, y - 4.4, tex, halvar(S["value"] + 1.5), DARK)
    put(ax, x, y - 8.9, gloss, halvar(S["tick"]), GRAY85)


def fig_information():
    fam = "Information"
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, fam_ground(fam), z=0)
    slab(ax, 0, CH - 1.2, LEFT_W, 1.2, fam_fill(fam), z=1, radius=0.5)
    x = 2.0
    formula_block(ax, x, 40.5, "Max sequence probability",
                  r"$u = -\sum_{l}\ \log p(y_l \mid y_{<l}, x)$",
                  "the whole answer's log-probability", ink=fam_ink(fam))
    formula_block(ax, x, 27.5, "Perplexity",
                  r"$u = -\frac{1}{L}\sum_{l}\ \log p(y_l \mid y_{<l}, x)$",
                  "the same, per token, so length does not count", ink=fam_ink(fam))
    formula_block(ax, x, 14.5, "Mean token entropy",
                  r"$u = \frac{1}{L}\sum_{l}\ \mathcal{H}\left[p(\cdot \mid y_{<l}, x)\right]$",
                  "how spread every next-token choice was", ink=fam_ink(fam))
    put(ax, x, 1.6, "x question  ·  y answer, L tokens  ·  higher u = less sure",
        halvar(S["gloss"]), GRAY85)

    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("Max sequence prob.", "log-probability of the answer", 1),
        ("Perplexity", "the same, per token", 1),
        ("Mean token entropy", "spread of each next-token distribution", 1),
        ("Rényi-Neg · Fisher-Rao", "other measures of the same distributions", 1),
        ("Self-certainty", "distance from a uniform next-token choice", 1),
        ("Boosted prob.", "token probabilities, re-weighted", 1),
        ("Mean PMI · cond. PMI", "what the question adds; a 2nd pass without it", 1),
        ("CCP", "probability pooled over same-meaning tokens", None),
        ("Attention score · RAUQ", "attention maps of the same pass", None),
    ], pitch=3.55, accent=fam_fill(fam))
    out = save(fig, FIGDIR / "information_based", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_self_report():
    fam = "Self-report"
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, fam_ground(fam), z=0)
    slab(ax, 0, CH - 1.2, LEFT_W, 1.2, fam_fill(fam), z=1, radius=0.5)
    x = 2.0
    header(ax, x, 41.0, "a second prompt, after the answer", fam_ink(fam))
    slab(ax, x, 21.0, LEFT_W - 4, 17.0, "white", z=1)
    m = mono(S["tick"])
    for k, ln in enumerate([
        "Question: What disease can be",
        "  treatment with Bulevirtide?",
        "Possible answer: Hepatitis B",
        "Is the possible answer True or",
        "  False? The possible answer is:",
    ]):
        put(ax, x + 1.5, 35.2 - k * 2.9, ln, m, DARK, zorder=2)

    put(ax, x, 16.2, "P(TRUE)", hermann(S["category"]), fam_ink(fam))
    put(ax, x, 12.0, r'$u = -\log\, p(\mathrm{“True”})$', halvar(S["value"] + 1.5), DARK)
    put(ax, x + 27, 16.2, "P(TRUE) EMPIRICAL", hermann(S["category"]), fam_ink(fam))
    put(ax, x + 27, 12.0, "sample the verdict 10×,", halvar(S["tick"]), DARK)
    put(ax, x + 27, 9.9, "count how often “True”", halvar(S["tick"]), DARK)
    put(ax, x, 1.6, "prompt as lm-polygraph 0.7.0 builds it; the answer is the "
        "7B model's", halvar(S["gloss"]), GRAY85)

    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("p(True)", "log-probability the model puts on “True”", 2),
        ("p(True) empirical", "share of 10 sampled verdicts that say “True”", 2),
        ("Verbalized 1S", "a confidence the model states as a number", None),
    ], pitch=4.2, accent=fam_fill(fam))
    lines(fig, ax, TABLE_X0, 20.0,
          "Verbalized 1S was dropped: Qwen2.5 never stated a confidence that "
          "lm-polygraph could parse, so it has no score to rank by.",
          halvar(S["tick"], italic=True), CW - TABLE_X0, GRAY85, pitch=2.1)
    out = save(fig, FIGDIR / "self_report", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_consistency():
    fam = "Consistency"
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, fam_ground(fam), z=0)
    slab(ax, 0, CH - 1.2, LEFT_W, 1.2, fam_fill(fam), z=1, radius=0.5)
    x = 2.0
    header(ax, x, 41.0, "the same question, sampled again", fam_ink(fam))
    # five sampled answers, as letters: A' is A in other words
    chips = ["A", "A", "A′", "B", "A"]
    cx = [x + 1 + k * 10.4 for k in range(5)]
    for c, lab in zip(cx, chips):
        hue = WRONG if lab == "B" else fam_fill(fam)
        slab(ax, c, 32.0, 8.4, 5.0, hue, z=1, radius=0.8)
        put(ax, c + 4.2, 34.5, lab, halvar(S["panel_header"], bold=True), "white", ha="center",
            va="center", zorder=2)

    rows = [
        ("WORDS", "do the samples share words?", "ROUGE-L", 26.5),
        ("MEANING", "does each entail the other? A′ joins A", "NLI model", 19.5),
        ("+ LIKELIHOOD", "weight each sample by its probability", "white-box", 12.5),
    ]
    for head, what, how, yy in rows:
        # the likelihood row is the other family: its colour, not consistency's
        ink = fam_ink("Sampling + likelihood") if head.startswith("+") else fam_ink(fam)
        put(ax, x, yy, head, hermann(S["category"]), ink)
        put(ax, x, yy - 2.6, what, halvar(S["tick"]), DARK)
        put(ax, LEFT_W - 2, yy, how, halvar(S["tick"], italic=True), ink, ha="right")
    put(ax, x, 6.0, r"$SE = -\sum_{c}\ p(c \mid x)\,\log p(c \mid x)$",
        halvar(S["value"] + 1.0), DARK)
    put(ax, x, 0.9, "c: a cluster of samples with one meaning", halvar(S["gloss"]),
        GRAY85)

    sl, co = fam_ink("Sampling + likelihood"), fam_ink(fam)
    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("Lexical similarity", "word overlap between samples", 3, co),
        ("NumSemSets", "how many meaning clusters", 4, co),
        ("DegMat · Ecc. · EigVal", "graph of pairwise entailment", 4, co),
        ("Kernel lang. entropy", "entropy of a semantic kernel", 4, co),
        ("MC sequence entropy", "mean −log P over samples; also length-normalised", 3, sl),
        ("Semantic entropy", "probability mass over meaning clusters", 4, sl),
        ("SAR · Sem. density · CoCoA", "likelihood and meaning combined", None, sl),
    ], pitch=3.7, name_w=23.0, accent=fam_fill(fam))
    put(ax, TABLE_X0, 8.0, "Consistency", halvar(S["tick"], bold=True), fam_ink(fam))
    put(ax, TABLE_X0 + 13.5, 8.0, "Sampling + likelihood", halvar(S["tick"], bold=True), sl)
    put(ax, TABLE_X0 + 33.0, 8.0, "name colour = family", halvar(S["tick"]), GRAY85)
    out = save(fig, FIGDIR / "consistency", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_trained_probes():
    fam = "Trained probe"
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, fam_ground(fam), z=0)
    slab(ax, 0, CH - 1.2, LEFT_W, 1.2, fam_fill(fam), z=1, radius=0.5)
    x = 2.0
    header(ax, x, 41.0, "read the model's insides, learn the pattern", fam_ink(fam))
    # the LLM as a stack of layers
    for k in range(6):
        hue = fam_fill(fam) if k == 3 else shade(fam_fill(fam), 40)
        slab(ax, x + 1, 9.5 + k * 4.6, 14, 3.4, hue, z=1, radius=0.5)
    put(ax, x + 8, 6.2, "the LLM", halvar(S["tick"]), GRAY85, ha="center")
    put(ax, x + 17.5, 28.6, "hidden state", halvar(S["tick"]), DARK)
    put(ax, x + 17.5, 26.5, "of one layer", halvar(S["tick"]), DARK)
    arrow(ax, x + 15.6, 25.0, x + 28.5, 22.6)
    slab(ax, x + 29, 18.0, 12.5, 8.4, fam_ink(fam), z=1)
    put(ax, x + 35.25, 23.3, "classifier", halvar(S["tick"], bold=True), "white",
        ha="center", zorder=2)
    put(ax, x + 35.25, 20.4, "trained", halvar(S["tick"]), "white", ha="center",
        zorder=2)
    arrow(ax, x + 41.8, 22.2, x + 45.5, 22.2)
    put(ax, x + 46.0, 21.4, "p(wrong)", halvar(S["value"], bold=True), WRONG_TEXT)
    slab(ax, x + 22, 7.0, LEFT_W - x - 26, 7.2, "white", z=1)
    put(ax, x + 23.5, 11.4, "needs graded answers", halvar(S["tick"], bold=True), DARK,
        zorder=2)
    put(ax, x + 23.5, 8.9, "from your own task to train on", halvar(S["tick"]), DARK,
        zorder=2)

    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("SAPLMA", "a small network on one layer's hidden state", None),
        ("SATRMD", "token states' distance from those of right answers", None),
        ("Lookback Lens", "attention on the context against its own words", None),
        ("TAD", "attention features, with a trained head", None),
    ], pitch=4.2, accent=fam_fill(fam))
    lines(fig, ax, TABLE_X0, 17.5,
          "Left out on purpose: this comparison is between methods that need no "
          "labels. Lookback Lens and TAD also need attention weights, which force "
          "the slower eager attention path on transformers 5.x.",
          halvar(S["tick"], italic=True), CW - TABLE_X0, GRAY85, pitch=2.1)
    out = save(fig, FIGDIR / "trained_probes", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 08. LM-Polygraph
# =====================================================================================
def fig_lm_polygraph():
    fig, ax = board()
    stages = [  # (name, lines); colours from LIB, as in the code highlighting
        ("Model", ["WhiteboxModel: any", "Hugging Face LM", "BlackboxModel: an API"]),
        ("Stat calculators", ["greedy probabilities,", "samples, NLI matrix —",
                              "each computed once"]),
        ("Estimators", ["20 in this talk, one", "uncertainty score", "per answer"]),
        ("UEManager", ["runs all of it over", "a dataset, batched"]),
    ]
    n = len(stages)
    gap = 5.0
    w = (CW - (n - 1) * gap) / n
    top, bot = 41.0, 20.0
    roles = ["model", "stats", "estimator", "manager"]
    for k, (name, body) in enumerate(stages):
        x = k * (w + gap)
        fill, ink = LIB[roles[k]]
        slab(ax, x, bot, w, top - bot, shade(fill, 14), z=0)
        slab(ax, x, top - 1.2, w, 1.2, fill, z=1, radius=0.5)
        put(ax, x + 1.6, top - 4.6, name.upper(), hermann(S["panel_header"]), ink)
        for j, ln in enumerate(body):
            put(ax, x + 1.6, top - 9.4 - j * 2.4, ln, halvar(S["tick"]), DARK)
        if k < n - 1:
            arrow(ax, x + w + 0.6, (top + bot) / 2, x + w + gap - 0.6, (top + bot) / 2)

    put(ax, 0, 14.0, "SHARED STATISTICS", hermann(S["category"]), LIB["stats"][1])
    lines(fig, ax, 0, 10.8,
          "An estimator declares the statistics it needs, and each is computed "
          "once per answer. Every sampling estimator reads the same samples, and "
          "every NLI-based one the same entailment matrix, so estimators that "
          "share a statistic compute it once.",
          halvar(S["tick"]), 80.0, DARK, pitch=2.2)
    put(ax, CW, 14.0, "IN THIS TALK", hermann(S["category"]), GRAY85, ha="right")
    for j, ln in enumerate(["lm-polygraph 0.7.0", "Qwen2.5 3B and 7B; 4 more in backup",
                            "DeBERTa NLI for tier 4"]):
        put(ax, CW, 10.8 - j * 2.2, ln, halvar(S["tick"]), DARK, ha="right")
    out = save(fig, FIGDIR / "lm_polygraph", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 09. every estimator, by type and tier
# =====================================================================================
# The family each estimator in bioasq.TIERS belongs to. Asserted complete below,
# so a new estimator in TIERS cannot silently fall off this slide.
TYPE_OF = {
    "MaximumSequenceProbability": "Information",
    "Perplexity": "Information",
    "MeanTokenEntropy": "Information",
    "FisherRao": "Information",
    "RenyiNeg": "Information",
    "SelfCertainty": "Information",
    "BoostedProbSequence": "Information",
    "MeanPointwiseMutualInformation": "Information",
    "MeanConditionalPointwiseMutualInformation": "Information",
    "PTrue": "Self-report",
    "PTrueEmpirical": "Self-report",
    "LexicalSimilarity": "Consistency",
    "NumSemSets": "Consistency",
    "DegMat": "Consistency",
    "Eccentricity": "Consistency",
    "EigValLaplacian": "Consistency",
    "KernelLanguageEntropy": "Consistency",
    "MonteCarloSequenceEntropy": "Sampling + likelihood",
    "MonteCarloNormalizedSequenceEntropy": "Sampling + likelihood",
    "SemanticEntropy": "Sampling + likelihood",
}
SHORT = {  # PRETTY names where the cell is too narrow for them
    "MaximumSequenceProbability": "MSP",
    "MeanPointwiseMutualInformation": "PMI",
    "MeanConditionalPointwiseMutualInformation": "cond. PMI",
    "BoostedProbSequence": "Boosted prob.",
    "LexicalSimilarity": "Lexical similarity",
    "KernelLanguageEntropy": "KLE",
    "EigValLaplacian": "EigVal",
    "MonteCarloSequenceEntropy": "MC seq. entropy",
    "MonteCarloNormalizedSequenceEntropy": "MC norm. seq. entropy",
    "MeanTokenEntropy": "Mean token entropy",
}


def wrap_items(fig, items, font, max_u):
    """Wrap a comma list between items, never inside one ("MC norm. seq." / "ent.")."""
    out, cur = [], ""
    for k, item in enumerate(items):
        piece = item + ("," if k < len(items) - 1 else "")
        trial = f"{cur} {piece}".strip()
        if cur and len(wrap_measured(fig, trial, font, max_u)) > 1:
            out.append(cur)
            cur = piece
        else:
            cur = trial
    return out + ([cur] if cur else [])


def fig_types_by_tier():
    names = [n for t in bioasq.TIERS.values() for n in t]
    missing = set(names) - set(TYPE_OF)
    assert not missing, f"no type for {missing}"
    tier_of = {n: t for t, ns in bioasq.TIERS.items() for n in ns}

    fig, ax = board()
    label_w = 23.0
    gap = 1.2
    col_w = (CW - label_w - 4 * gap) / 4
    col_x = [label_w + gap + k * (col_w + gap) for k in range(4)]
    top = 44.0
    for k, t in enumerate((1, 2, 3, 4)):
        tag(ax, col_x[k], top - 2.2, f"T{t}", tier_hue(t))
        put(ax, col_x[k] + 5.2, top - 2.0, TIER_GLOSS[t], halvar(S["tick"]), GRAY85)

    types = ["Information", "Self-report", "Consistency", "Sampling + likelihood",
             "Trained probe"]
    font = halvar(S["tick"])
    y = top - 4.6
    for typ in types:
        cells = {t: [SHORT.get(n, PRETTY.get(n, n)) for n in names
                     if TYPE_OF[n] == typ and tier_of[n] == t] for t in (1, 2, 3, 4)}
        wrapped = {t: wrap_items(fig, v, font, col_w - 2.4)
                   for t, v in cells.items() if v}
        n_lines = max([len(v) for v in wrapped.values()] + [1])
        h = 2.0 + n_lines * 2.1
        slab(ax, 0, y - h + 0.3, 0.7, h - 0.6, fam_fill(typ), z=1, radius=0.3)
        put(ax, 1.6, y - 2.9, typ.upper(), hermann(S["category"]), fam_ink(typ))
        if typ == "Trained probe":
            # No fill: this row spans every tier column, so under click layering
            # an opaque ground is redrawn in every layer and hides its own note.
            ax.add_patch(plt.Rectangle((col_x[0], y - h), CW - col_x[0], h, fc="none",
                                       ec=fam_fill(typ), lw=WEIGHT["grid"], ls=(0, (4, 3))))
            put(ax, col_x[0] + 1.2, y - 2.9, "not run: needs labelled answers to "
                "train on", halvar(S["tick"], italic=True), GRAY85)
        for k, t in enumerate((1, 2, 3, 4)):
            if typ == "Trained probe":
                break
            filled = t in wrapped
            if filled:  # an empty cell stays transparent, not white
                slab(ax, col_x[k], y - h, col_w, h, fam_ground(typ), z=0)
            for j, ln in enumerate(wrapped.get(t, [])):
                put(ax, col_x[k] + 1.2, y - 2.9 - j * 2.1, ln, font, DARK)
        y -= h + gap
    out = save(fig, FIGDIR / "types_by_tier", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 09b. the benchmark: one real question, the sample, and how it was graded
# =====================================================================================
# A question where the grading matters: 3B is wrong, and 7B is right in a form no
# string rule matches ("JAK1" for "Janus kinase 1"), so a manual label decides it.
DATASET_EXAMPLE = "Which molecule is targeted by upadacitinib?"


def dataset_stats(d7, y7, d3, y3):
    """Every number the dataset slide shows, measured rather than typed."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the cached copy is the one we ran on
    from datasets import load_dataset

    split = load_dataset(bioasq.HF_DATASET, split="factoid")
    pool = bioasq.load_factoid(n=None, seed=0)
    assert d7["questions"] == d3["questions"], "the two runs must share their questions"
    qs, golds = d7["questions"], d7["golds"]
    challenges = sorted({r["asq_challenge"] for r in split if r["question"] in set(qs)})

    def grading(d, y):
        graded = [bioasq.grade(p, g) for p, g in zip(d["generations"], d["golds"])]
        exact = sum(g["em"] for g in graded)
        rules = sum(g["correct"] and not g["em"] for g in graded)
        right = int((y == 0).sum())
        # Net, not the count of changed verdicts: labels move answers both ways, and
        # only the net keeps the three rows summing to the final number of right answers.
        return {"exact": exact, "rules": rules, "manual": right - exact - rules,
                "right": right, "n": len(y)}

    return {
        "factoid": len(split),
        "pool": len(pool),
        "n": len(qs),
        "q_words": int(np.median([len(q.split()) for q in qs])),
        "gold_words": int(np.median([min(len(g.split()) for g in gs) for gs in golds])),
        "synonyms": np.mean([len(gs) > 1 for gs in golds]),
        "challenges": (min(challenges), max(challenges)),
        "3B": grading(d3, y3),
        "7B": grading(d7, y7),
    }


def fig_dataset(d7, y7, d3, y3, st):
    i = d7["questions"].index(DATASET_EXAMPLE)
    gold = d7["golds"][i][0]
    answers = [("Qwen2.5-3B", bioasq.answer_text(d3["generations"][i]), y3[i] == 0),
               ("Qwen2.5-7B", bioasq.answer_text(d7["generations"][i]), y7[i] == 0)]
    assert [ok for *_, ok in answers] == [False, True], "the example no longer grades 3B wrong, 7B right"

    fig, ax = board()
    # --- left: one question, as the models see it --------------------------------
    slab(ax, 0, 0, LEFT_W, CH, PALE, z=0)
    x = 2.0
    header(ax, x, 41.0, "one question, as the models see it", DARK)
    put(ax, x, 35.6, "QUESTION", hermann(S["category"]), GRAY85)
    put(ax, x, 32.4, DATASET_EXAMPLE, halvar(S["value"], bold=True), DARK)
    put(ax, x, 27.6, "REFERENCE", hermann(S["category"]), GRAY85)
    put(ax, x, 24.4, gold, halvar(S["value"]), DARK)
    for k, (model, ans, ok) in enumerate(answers):
        yy = 19.0 - k * 4.4
        put(ax, x, yy, model, halvar(S["tick"]), GRAY85)
        put(ax, x + 17, yy, ans, halvar(S["value"], bold=True), DARK if ok else WRONG_TEXT)
        put(ax, x + 27, yy, "right" if ok else "wrong", halvar(S["tick"], italic=True),
            DARK if ok else WRONG_TEXT)
    lines(fig, ax, x, 8.4,
          "Closed-book: each model sees the question after three hand-written "
          "examples, never BioASQ's snippets. “JAK1” is right, but no string "
          "rule can know that — a manual label decides it.",
          halvar(S["tick"]), LEFT_W - 4, DARK, pitch=2.1, max_lines=3)

    # --- right: the sample ----------------------------------------------------------
    x0 = TABLE_X0
    header(ax, x0, 41.0, "the sample")
    steps = [(f"{st['factoid']:,}", "factoid questions"),
             (f"{st['pool']:,}", "reference ≤ 5 words"),
             (f"{st['n']:,}", "sampled, seed 0")]
    col = (CW - x0) / 3
    for k, (num, what) in enumerate(steps):
        cx = x0 + k * col
        put(ax, cx, 34.6, num, hermann(S["claim"]), BLUE if k == 2 else DARK)
        put(ax, cx, 31.6, what, halvar(S["tick"]), GRAY85)
        if k < 2:
            arrow(ax, cx + 9.6, 35.5, cx + col - 1.2, 35.5)
    lo, hi = st["challenges"]
    put(ax, x0, 27.4,
        f"median question {st['q_words']} words  ·  median reference "
        f"{st['gold_words']} words  ·  {st['synonyms']:.0%} list synonyms",
        halvar(S["tick"]), DARK)
    put(ax, x0, 25.2, f"BioASQ challenges {lo}–{hi}  ·  the same 1,000 for both models",
        halvar(S["tick"]), DARK)

    # --- right: how each answer was graded ------------------------------------------
    header(ax, x0, 19.6, "graded in three passes")
    cx3, cx7 = CW - 12.0, CW
    put(ax, cx3, 19.6, "3B", hermann(S["category"]), GRAY85, ha="right")
    put(ax, cx7, 19.6, "7B", hermann(S["category"]), GRAY85, ha="right")
    rule(ax, x0, CW, 18.0, GRAY45)
    rows = [("exact match", "exact", False),
            ("+ token rules: case, punctuation, synonyms", "rules", False),
            ("+ manual labels, net", "manual", False)]
    for k, (label, key, _) in enumerate(rows):
        yy = 15.4 - k * 3.2
        put(ax, x0, yy, label, halvar(S["tick"]), DARK)
        sign = "+" if key != "exact" else ""
        put(ax, cx3, yy, f"{sign}{st['3B'][key]}", halvar(S["tick"]), DARK, ha="right")
        put(ax, cx7, yy, f"{sign}{st['7B'][key]}", halvar(S["tick"]), DARK, ha="right")
    rule(ax, x0, CW, 7.3)
    put(ax, x0, 4.6, "right in the end", halvar(S["tick"], bold=True), DARK)
    for cx, m in ((cx3, "3B"), (cx7, "7B")):
        g = st[m]
        assert g["exact"] + g["rules"] + g["manual"] == g["right"]
        put(ax, cx, 4.6, f"{g['right']}", halvar(S["value"], bold=True), BLUE, ha="right")
    put(ax, x0, 1.6, "out of 1,000; 3B and 7B answer the same questions",
        halvar(S["gloss"]), GRAY85)
    out = save(fig, FIGDIR / "dataset", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 14. limitations: what this is, and what it is not
# =====================================================================================
def limitations(n_judgement):
    return [
        ("One dataset",
         "BioASQ factoids: short, English, biomedical answers. Long answers or "
         "extraction may well rank the methods differently."),
        ("Six open models",
         "Qwen2.5 3B/7B, Llama 3.2-3B and 3.1-8B, Mistral-7B, OLMo-2-7B. Larger and "
         "hosted models were not tested."),
        ("One setting",
         "Greedy answers, one few-shot prompt, 10 samples per question, one seed."),
        ("Imperfect labels",
         f"Right or wrong comes from string rules plus LLM-assisted review, not "
         f"domain experts; {n_judgement} labels are judgement calls."),
        ("A subset of methods",
         "20 of LM-Polygraph's estimators. Attention-based methods and trained "
         "probes were not run."),
        ("One metric",
         "AUROC ranks right against wrong. It does not say where to set a threshold, "
         "or whether a score is calibrated."),
    ]


def count_judgement_calls():
    import json
    labels = json.loads(MANUAL.read_text())["labels"]
    flat = [l for v in labels.values() for l in (v if isinstance(v, list) else [v])]
    # Two label passes, two note prefixes: the first wrote "JUDGEMENT CALL", the
    # second "DEBATABLE". Both mean the same thing - a call that is not a plain
    # synonym match - so the card must count both or it understates the caveat.
    return sum(l.get("note", "").startswith(("JUDGEMENT CALL", "DEBATABLE"))
               for l in flat)


def fig_limitations():
    cards = limitations(count_judgement_calls())
    fig, ax = board()
    cols, gap = 3, 2.4
    w = (CW - (cols - 1) * gap) / cols
    h = (CH - gap) / 2
    pad = 1.6
    for k, (head, body) in enumerate(cards):
        r, c = divmod(k, cols)
        x = c * (w + gap)
        y = CH - (r + 1) * h - r * gap
        slab(ax, x, y, w, h, shade(CAUTION, 14), z=0)
        slab(ax, x, y + h - 1.2, w, 1.2, CAUTION, z=1, radius=0.5)
        put(ax, x + pad, y + h - 4.6, head.upper(), hermann(S["panel_header"]), CAUTION_INK)
        lines(fig, ax, x + pad, y + h - 9.0, body, halvar(S["value"]), w - 2 * pad, DARK,
              pitch=2.7, max_lines=5)
    out = save(fig, FIGDIR / "limitations", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 15. take-home: what to do on Monday
# =====================================================================================
TAKEAWAYS = [
    ("Fluency is not reliability",
     "Errors in this run carried model probabilities above 0.98. Correctness has to be "
     "scored separately from the answer."),
    ("Access determines the family",
     "An API exposes text: consistency, or a verbalized confidence. An open model adds "
     "token probabilities and p(True). Probes also require labels."),
    ("One pass is a fair default",
     "On BioASQ, mean token entropy matched the sampling-plus-NLI methods on four of "
     "six models. Sampling is worth adding where it helps."),
    ("Triage, not adjudication",
     "The least certain answers go to review. The threshold is set on the target data; "
     "AUROC does not determine it."),
    # Not "cheap": section 1 of CLAUDE.md rules out cost language on the slides.
    ("Repeating it is a short script",
     "A model is wrapped once and UEManager runs any set of estimators over a dataset. "
     "A new method is a short class."),
]


def fig_takeaways():
    fig, ax = board()
    n = len(TAKEAWAYS)
    row = CH / n
    for k, (head, body) in enumerate(TAKEAWAYS):
        top = CH - k * row
        put(ax, 0, top - 5.6, f"{k + 1}", hermann(S["hero"]), TAKEAWAY_INK)
        put(ax, 7.5, top - 3.0, head.upper(), hermann(S["panel_header"]), DARK)
        lines(fig, ax, 7.5, top - 5.8, body, halvar(S["tick"]), CW - 7.5, DARK,
              pitch=2.1, max_lines=2)
        if k < n - 1:
            rule(ax, 7.5, CW, top - row + 0.4, shade(HGF["green"], 45))
    out = save(fig, FIGDIR / "takeaways", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 16. references
# =====================================================================================
# (what it is cited for, its colour, the citation). Sections are separated by None.
# The key carries the concept colour of the idea it backs, so a family's paper reads
# in that family's colour. Venues for the library papers come from the citation block
# in lm_polygraph-0.7.0.dist-info/METADATA; the method credits follow its README table.
REFERENCES = [
    ("Semantic entropy", fam_ink("Sampling + likelihood"),
     "Kuhn, Gal, Farquhar. Semantic uncertainty: linguistic invariances for uncertainty "
     "estimation in natural language generation. ICLR 2023."),
    ("Semantic entropy", fam_ink("Sampling + likelihood"),
     "Farquhar, Kossen, Kuhn, Gal. Detecting hallucinations in large language models "
     "using semantic entropy. Nature 630, 625–630 (2024)."),
    None,
    ("LM-Polygraph", DARK,
     "Fadeeva et al. LM-Polygraph: uncertainty estimation for language models. "
     "EMNLP 2023, System Demonstrations."),
    ("LM-Polygraph", DARK,
     "Vashurin et al. Benchmarking uncertainty quantification methods for large "
     "language models with LM-Polygraph. TACL 13, 220–248 (2025)."),
    ("Method families", DARK,
     "Vazhentsev, Smirnov, Li, Panov, Baldwin, Shelmanov. Uncertainty quantification "
     "for large language diffusion models. arXiv:2605.14570 (2026)."),
    None,
    ("p(True)", fam_ink("Self-report"),
     "Kadavath et al. Language models (mostly) know what they know. "
     "arXiv:2207.05221 (2022)."),
    ("Lexical similarity", fam_ink("Consistency"),
     "Fomicheva et al. Unsupervised quality estimation for neural machine translation. "
     "TACL 8 (2020)."),
    ("DegMat · Ecc. · EigVal", fam_ink("Consistency"),
     "Lin, Trivedi, Sun. Generating with confidence: uncertainty quantification for "
     "black-box large language models. TMLR 2024."),
    ("KLE", fam_ink("Consistency"),
     "Nikitin, Kossen, Gal, Marttinen. Kernel language entropy: fine-grained "
     "uncertainty quantification for LLMs from semantic similarities. NeurIPS 2024."),
    ("SAPLMA", fam_ink("Trained probe"),
     "Azaria, Mitchell. The internal state of an LLM knows when it's lying. "
     "Findings of EMNLP 2023."),
    None,
    ("BioASQ", GRAY85,
     "Tsatsaronis et al. An overview of the BIOASQ large-scale biomedical semantic "
     "indexing and question answering competition. BMC Bioinformatics 16, 138 (2015)."),
    # One row for the six models: four rows would overflow the canvas.
    ("Models", GRAY85,
     "Qwen Team, Qwen2.5 technical report, arXiv:2412.15115 (2024). Grattafiori et "
     "al., The Llama 3 herd of models, arXiv:2407.21783 (2024). Jiang et al., Mistral "
     "7B, arXiv:2310.06825 (2023). Team OLMo, 2 OLMo 2 Furious, arXiv:2501.00656 (2025)."),
    ("NLI model", GRAY85,
     "He, Liu, Gao, Chen. DeBERTa: decoding-enhanced BERT with disentangled attention. "
     "ICLR 2021."),
]


def fig_references():
    fig, ax = board()
    key_w = 19.0
    text_w = CW - key_w
    font = halvar(S["tick"])
    pitch, row_gap, section_gap = 2.05, 0.5, 1.2
    y = CH - 1.6
    for ref in REFERENCES:
        if ref is None:
            rule(ax, 0, CW, y + 0.55)
            y -= section_gap
            continue
        key, colour, text = ref
        put(ax, 0, y, key, halvar(S["tick"], bold=True), colour)
        y = lines(fig, ax, key_w, y, text, font, text_w, DARK, pitch=pitch) - row_gap
    if y < 0.5:
        raise ValueError(f"references overflow the canvas by {0.5 - y:.1f} u")
    out = save(fig, FIGDIR / "references", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# the words that belong on the slide, not in the figure
# =====================================================================================
def slide_copy(acc, half, st=None):
    copy = {
        "confidently_wrong": (
            "01",
            "High-confidence errors",
            "A wrong answer can carry near-certain probability.",
            "Four BioASQ answers from the 7B model. Each is incorrect "
            "against the reference, and each was assigned a probability "
            "above 0.98 by the model itself. The text of an answer gives no "
            "indication of which to verify.",
        ),
        "selective_generation": (
            "02",
            "Selective answering",
            "Withholding the least certain answers raises the accuracy of the rest.",
            f"Accuracy of the answers that remain as the most uncertain are "
            f"progressively withheld, ranked by mean token entropy. Withholding half "
            f"raises the "
            f"accuracy of the remainder from {acc:.0%} to {half:.0%}. An uninformative "
            f"score follows the flat line; a perfect ranking follows the upper dashed "
            f"line. AUROC, reported throughout, measures how far a ranking separates "
            f"correct answers from incorrect ones.",
        ),
        "uq_types": (
            "03",
            "Five signal types",
            "Each estimator reads one of five signals, and each signal requires a "
            "different level of model access.",
            "White-box methods require the model's probabilities or internal states; "
            "black-box methods require only its text. Families and method names follow "
            "Vazhentsev et al. (arXiv:2605.14570); self-report is added here because "
            "p(True) is included in the run. T1–T4 denote the tiers used to group the "
            "results.",
        ),
        "information_based": (
            "04",
            "Information-based estimators",
            "These estimators need no sampling: the answering pass, plus a "
            "question-free pass for PMI.",
            "All three are computed from the generation pass itself, so they"
            " add no further computation. They require token probabilities, "
            "which many hosted APIs do not return.",
        ),
        "self_report": (
            "05",
            "Self-report estimators",
            "A second prompt asks the model to assess its own answer.",
            "p(True) takes the probability the model assigns to “True”; the "
            "empirical variant samples the verdict ten times and counts. "
            "Both require a second prompt per answer, which defines tier 2.",
        ),
        "consistency": (
            "06",
            "Sampling-based estimators",
            "Disagreement among resampled answers indicates uncertainty; "
            "some methods also weigh each sample by its probability.",
            "Two families share the samples. Consistency methods measure "
            "agreement by shared words or by meaning, with an NLI model "
            "testing whether two answers entail each other; sampling + "
            "likelihood methods also weight each sample by its probability. "
            "Sampling defines tier 3; the NLI model on top defines tier 4.",
        ),
        "trained_probes": (
            "07",
            "Trained probes",
            "A classifier on the model's hidden states can predict incorrectness, "
            "given labelled data.",
            "Probes require graded answers from the task they will be applied to, and "
            "retraining for each new task or model. They are included here for "
            "completeness and were not run.",
        ),
        "lm_polygraph": (
            "08",
            "LM-Polygraph",
            "One library implements most of these estimators behind one "
            "interface.",
            "Open source and pip-installable. An estimator returns one "
            "uncertainty score per answer, higher indicating less certainty;"
            " grading the answers and scoring the estimators against those "
            "grades are separate steps.",
        ),
        "types_by_tier": (
            "09",
            "Every estimator, by type and tier",
            "Type denotes the signal an estimator reads; tier, what it "
            "requires to run.",
            "The twenty estimators compared in this talk, placed by signal "
            "type and tier. Tiers are cumulative: T1 needs no sampling (the "
            "answering pass, plus a question-free pass for the two PMIs), T2"
            " adds a second prompt, T3 adds sampling, and T4 adds a DeBERTa "
            "NLI model over the samples. Consistency and sampling methods "
            "divide between T3 and T4 according to whether they require the "
            "NLI model. The later slides use these tiers to group the "
            "results.",
        ),
    }
    if st is not None:
        copy["dataset"] = (
            "09b",
            "The evaluation set: BioASQ factoids",
            f"{st['n']:,} biomedical factoid questions, each with a short reference "
            f"answer.",
            "Questions are drawn from the BioASQ challenges and retained only where the "
            "reference answer is at most five words, so correctness can be determined. "
            "Answers are graded by exact match, then by token rules, then by manual "
            "labels where no rule applies. The labels were proposed with LLM assistance "
            "and reviewed; they were not set by domain experts.",
        )
    copy["uq_map"] = (
        "02b",
        "Where the signal comes from",
        "Two properties locate every estimator: the model access it needs and the "
        "number of answers it draws.",
        "Position further right indicates greater access required: a hosted API "
        "exposes text only, an open model also exposes token probabilities and hidden "
        "states. The lower row samples the same question several times. The next "
        "slide places every estimator in these families, and the four after it take "
        "the families in turn, the two sampling families together.",
    )
    copy["takeaways"] = (
        "15",
        "Summary",
        "The families differ in the access they need; the ranking between them is "
        "task-specific.",
        "Point three is the only item drawn from the BioASQ results, and the preceding "
        "slide gives its range across models. The remainder is independent of the task.",
    )
    copy["references"] = (
        "16",
        "References",
        "Sources for the methods, the library and the data.",
        "The key on the left indicates what each paper supports, in the colour used "
        "for it on the method slides. The LM-Polygraph README cites a paper for each "
        "of its 38 methods.",
    )
    copy["limitations"] = (
        "14",
        "Scope and limitations",
        "An introduction rather than a benchmark: the ranking is specific to this "
        "setting.",
        "The ranking is conditioned on one dataset, one prompt and one grading "
        "procedure. Replicating the comparison on a target task and model is a short "
        "script in LM-Polygraph.",
    )
    return copy


def write_markdown(stem, number, title, lead, caption, force=False):
    import textwrap
    MDDIR.mkdir(exist_ok=True)
    existing = sorted(MDDIR.glob(f"*_slide_{stem}.md"))
    if existing and not force:
        # The markdown is where the slide's wording is edited, so a figure re-run
        # must not overwrite it. --rewrite-copy regenerates it from the Python.
        return existing[0], False
    path = MDDIR / f"{number}_slide_{stem}.md"
    path.write_text(
        f"# slide_{stem}\n\n"
        "The three pieces of text the *slide* carries, around the figure. Edit them\n"
        "here. The lead and the caption flow — they are wrapped by PowerPoint, so\n"
        "write them over as many `>` lines as reads well and they are joined back into\n"
        "one paragraph. Titles do not flow.\n\n"
        "## the slide\n\n"
        "- **title** — the slide's heading, top left\n"
        f"  > {title}\n\n"
        "- **lead** — the blue line under the title — one sentence, it wraps on its own\n"
        + "".join(f"  > {ln}\n" for ln in textwrap.wrap(lead, 84, break_on_hyphens=False))
        + "\n- **caption** — the grey paragraph under the figure — it wraps on its own\n"
        + "".join(f"  > {ln}\n" for ln in textwrap.wrap(caption, 84, break_on_hyphens=False))
    )
    return path, True


# =====================================================================================
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rewrite-copy", action="store_true",
                    help="regenerate the slide markdowns, overwriting edits made there")
    args = ap.parse_args()
    use_hgf()
    FIGDIR.mkdir(exist_ok=True)
    d, y = load(str(CACHE_7B), str(MANUAL))

    outs = [fig_confidently_wrong(d, y)]
    sel, acc, half = fig_selective_generation(d, y)
    outs += [sel, fig_uq_map(), fig_uq_types(), fig_information(), fig_self_report(),
             fig_consistency(), fig_trained_probes(), fig_lm_polygraph(),
             fig_types_by_tier()]
    d3, y3 = load(str(CACHE_3B), str(MANUAL))
    st = dataset_stats(d, y, d3, y3)
    outs += [fig_dataset(d, y, d3, y3, st), fig_limitations(), fig_takeaways(),
             fig_references()]

    copy = slide_copy(acc, half, st)
    for out in outs:
        md, wrote = write_markdown(out.stem, *copy[out.stem], force=args.rewrite_copy)
        print(f"wrote {out.relative_to(ROOT)}  +  {md.relative_to(ROOT)}"
              + ("" if wrote else "  (kept: edited copy)"))


if __name__ == "__main__":
    main()
