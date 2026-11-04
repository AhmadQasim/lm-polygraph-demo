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
    HIGHLIGHT, TYPE, WEIGHT, GEOM, REFERENCE_DASH,
    halvar, hermann, mono, slab, use_hgf,
)
from hgf_layers import layered_save as save  # noqa: E402  # writes the click layers too
from plot_bioasq import PRETTY, TIER_GLOSS, load, wrap_measured  # noqa: E402

FIGDIR = SLIDES / "figures"
MDDIR = SLIDES / "markdowns"
CACHE_7B = ROOT / "outputs" / "bioasq_Qwen2.5-7B-Instruct_n1000_seed0_t4.json"
MANUAL = ROOT / "manual_labels.json"

CW, CH = GEOM["canvas"]   # 124 x 44.5 u
TIER1, TIER_REST = BLUE, BLUE60
WRONG, WRONG_TEXT = ORANGE, ORANGE_DARK

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


def method_table(fig, ax, x0, x1, top, rows, pitch=3.7, name_w=21.0):
    """Right-hand method list: name | what it reads | tier pill (or 'not run').

    rows: (name, gloss, tier) with tier an int, a string such as "T3 · T4",
    or None for a method that is named but was not run here.
    """
    header(ax, x0, top, "method")
    header(ax, x0 + name_w, top, "reads")
    header(ax, x1, top, "tier", ha="right")
    y = top - 1.6
    rule(ax, x0, x1, y, GRAY45)
    gloss_w = x1 - x0 - name_w - 9.5
    for name, gloss, tier in rows:
        n_lines = len(wrap_measured(fig, gloss, halvar(S["tick"]), gloss_w))
        h = pitch + (n_lines - 1) * 2.05
        base = y - 2.65
        run = tier is not None
        put(ax, x0, base, name, halvar(S["tick"], bold=True), DARK if run else GRAY85)
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
                 "and checked by hand")
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

    ax.plot(r, oracle, color=GRAY45, lw=WEIGHT["reference"], dashes=REFERENCE_DASH)
    ax.plot([0, 0.9], [acc, acc], color=GRAY85, lw=WEIGHT["reference"],
            dashes=REFERENCE_DASH)
    ax.plot(r, curve, color=TIER1, lw=WEIGHT["focus"])

    half = kept_accuracy(u, y, [0.5])[0]
    ax.plot([0.5], [half], "o", color=TIER1, ms=8, zorder=5)

    lab = halvar(S["tick"])
    ax_b.text(101.5, 8 + (oracle[-1] - 0.2) / 0.8 * 32, "perfect ranking",
              fontproperties=lab, color=GRAY85, va="center")
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
        slab(ax, x, 3.2, w, CH - 3.2, PALE, z=0)
        tw = w - 2 * pad
        yy = lines(fig, ax, x + pad, CH - 3.6, name, hermann(S["panel_header"]), tw,
                   DARK, pitch=2.7)
        yy = lines(fig, ax, x + pad, yy - 0.6, reads, halvar(S["tick"]), tw, DARK,
                   pitch=2.05)
        put(ax, x + pad, yy - 0.8, access, halvar(S["tick"], italic=True), BLUE)
        yy -= 3.6
        rule(ax, x + pad, x + w - pad, yy + 1.2, GRAY45)
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


def formula_block(ax, x, y, name, tex, gloss):
    put(ax, x, y, name.upper(), hermann(S["category"]), DARK)
    put(ax, x, y - 4.4, tex, halvar(S["value"] + 1.5), DARK)
    put(ax, x, y - 8.9, gloss, halvar(S["tick"]), GRAY85)


def fig_information():
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, PALE, z=0)
    x = 2.0
    formula_block(ax, x, 40.5, "Max sequence probability",
                  r"$u = -\sum_{l}\ \log p(y_l \mid y_{<l}, x)$",
                  "the whole answer's log-probability")
    formula_block(ax, x, 27.5, "Perplexity",
                  r"$u = -\frac{1}{L}\sum_{l}\ \log p(y_l \mid y_{<l}, x)$",
                  "the same, per token, so length does not count")
    formula_block(ax, x, 14.5, "Mean token entropy",
                  r"$u = \frac{1}{L}\sum_{l}\ \mathcal{H}\left[p(\cdot \mid y_{<l}, x)\right]$",
                  "how spread every next-token choice was")
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
    ], pitch=3.55)
    out = save(fig, FIGDIR / "information_based", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_self_report():
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, PALE, z=0)
    x = 2.0
    header(ax, x, 41.0, "a second prompt, after the answer", DARK)
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

    put(ax, x, 16.2, "P(TRUE)", hermann(S["category"]), DARK)
    put(ax, x, 12.0, r'$u = -\log\, p(\mathrm{“True”})$', halvar(S["value"] + 1.5), DARK)
    put(ax, x + 27, 16.2, "P(TRUE) EMPIRICAL", hermann(S["category"]), DARK)
    put(ax, x + 27, 12.0, "sample the verdict 10×,", halvar(S["tick"]), DARK)
    put(ax, x + 27, 9.9, "count how often “True”", halvar(S["tick"]), DARK)
    put(ax, x, 1.6, "prompt as lm-polygraph 0.7.0 builds it; the answer is the "
        "7B model's", halvar(S["gloss"]), GRAY85)

    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("p(True)", "log-probability the model puts on “True”", 2),
        ("p(True) empirical", "share of 10 sampled verdicts that say “True”", 2),
        ("Verbalized 1S", "a confidence the model states as a number", None),
    ], pitch=4.2)
    lines(fig, ax, TABLE_X0, 20.0,
          "Verbalized 1S was dropped: Qwen2.5 never stated a confidence that "
          "lm-polygraph could parse, so it has no score to rank by.",
          halvar(S["tick"], italic=True), CW - TABLE_X0, GRAY85, pitch=2.1)
    out = save(fig, FIGDIR / "self_report", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_consistency():
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, PALE, z=0)
    x = 2.0
    header(ax, x, 41.0, "the same question, sampled again", DARK)
    # five sampled answers, as letters: A' is A in other words
    chips = ["A", "A", "A′", "B", "A"]
    cx = [x + 1 + k * 10.4 for k in range(5)]
    for c, lab in zip(cx, chips):
        hue = WRONG if lab == "B" else BLUE60
        slab(ax, c, 32.0, 8.4, 5.0, hue, z=1, radius=0.8)
        put(ax, c + 4.2, 34.5, lab, halvar(S["panel_header"], bold=True), "white", ha="center",
            va="center", zorder=2)

    rows = [
        ("WORDS", "do the samples share words?", "ROUGE-L", 26.5),
        ("MEANING", "does each entail the other? A′ joins A", "NLI model", 19.5),
        ("+ LIKELIHOOD", "weight each sample by its probability", "white-box", 12.5),
    ]
    for head, what, how, yy in rows:
        put(ax, x, yy, head, hermann(S["category"]), DARK)
        put(ax, x, yy - 2.6, what, halvar(S["tick"]), DARK)
        put(ax, LEFT_W - 2, yy, how, halvar(S["tick"], italic=True), BLUE, ha="right")
    put(ax, x, 6.0, r"$SE = -\sum_{c}\ p(c \mid x)\,\log p(c \mid x)$",
        halvar(S["value"] + 1.0), DARK)
    put(ax, x, 0.9, "c: a cluster of samples with one meaning", halvar(S["gloss"]),
        GRAY85)

    method_table(fig, ax, TABLE_X0, CW, 42.5, [
        ("Lexical similarity", "word overlap between samples", 3),
        ("MC sequence entropy", "mean −log P over samples; also length-normalised", 3),
        ("Semantic entropy", "entropy over meaning clusters", 4),
        ("NumSemSets", "how many meaning clusters", 4),
        ("DegMat · Ecc. · EigVal", "graph of pairwise entailment", 4),
        ("Kernel lang. entropy", "entropy of a semantic kernel", 4),
        ("SAR · Sem. density · CoCoA", "likelihood and meaning combined", None),
    ], pitch=3.7, name_w=23.0)
    out = save(fig, FIGDIR / "consistency", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


def fig_trained_probes():
    fig, ax = board()
    slab(ax, 0, 0, LEFT_W, CH, PALE, z=0)
    x = 2.0
    header(ax, x, 41.0, "read the model's insides, learn the pattern", DARK)
    # the LLM as a stack of layers
    for k in range(6):
        hue = BLUE if k == 3 else BLUE60
        slab(ax, x + 1, 9.5 + k * 4.6, 14, 3.4, hue, z=1, radius=0.5)
    put(ax, x + 8, 6.2, "the LLM", halvar(S["tick"]), GRAY85, ha="center")
    put(ax, x + 17.5, 28.6, "hidden state", halvar(S["tick"]), DARK)
    put(ax, x + 17.5, 26.5, "of one layer", halvar(S["tick"]), DARK)
    arrow(ax, x + 15.6, 25.0, x + 28.5, 22.6)
    slab(ax, x + 29, 18.0, 12.5, 8.4, DARK, z=1)
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
    ], pitch=4.2)
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
    stages = [
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
    for k, (name, body) in enumerate(stages):
        x = k * (w + gap)
        focus = name == "Stat calculators"
        slab(ax, x, bot, w, top - bot, HIGHLIGHT if focus else PALE, z=0)
        put(ax, x + 1.6, top - 4.0, name.upper(), hermann(S["panel_header"]), DARK)
        for j, ln in enumerate(body):
            put(ax, x + 1.6, top - 9.0 - j * 2.4, ln, halvar(S["tick"]), DARK)
        if k < n - 1:
            arrow(ax, x + w + 0.6, (top + bot) / 2, x + w + gap - 0.6, (top + bot) / 2)

    put(ax, 0, 14.0, "SHARED STATISTICS", hermann(S["category"]), BLUE)
    lines(fig, ax, 0, 10.8,
          "An estimator declares the statistics it needs, and each is computed "
          "once per answer. Every sampling estimator reads the same samples, and "
          "every NLI-based one the same entailment matrix, so estimators that "
          "share a statistic compute it once.",
          halvar(S["tick"]), 80.0, DARK, pitch=2.2)
    put(ax, CW, 14.0, "IN THIS TALK", hermann(S["category"]), GRAY85, ha="right")
    for j, ln in enumerate(["lm-polygraph 0.7.0", "Qwen2.5 3B and 7B, bf16",
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
    "LexicalSimilarity": "Lexical sim.",
    "KernelLanguageEntropy": "KLE",
    "EigValLaplacian": "EigVal",
    "MonteCarloSequenceEntropy": "MC seq. entropy",
    "MonteCarloNormalizedSequenceEntropy": "MC norm. seq. ent.",
    "MeanTokenEntropy": "Mean token entropy",
}


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
        wrapped = {t: wrap_measured(fig, ", ".join(v), font, col_w - 2.4)
                   for t, v in cells.items() if v}
        n_lines = max([len(v) for v in wrapped.values()] + [1])
        h = 2.0 + n_lines * 2.1
        put(ax, 0, y - 2.9, typ.upper(), hermann(S["category"]), DARK)
        if typ == "Trained probe":
            slab(ax, col_x[0], y - h, CW - col_x[0], h, "white", z=0)
            ax.add_patch(plt.Rectangle((col_x[0], y - h), CW - col_x[0], h, fc="none",
                                       ec=GRAY45, lw=WEIGHT["grid"], ls=(0, (4, 3))))
            put(ax, col_x[0] + 1.2, y - 2.9, "not run: needs labelled answers to "
                "train on", halvar(S["tick"], italic=True), GRAY85)
        for k, t in enumerate((1, 2, 3, 4)):
            if typ == "Trained probe":
                break
            filled = t in wrapped
            slab(ax, col_x[k], y - h, col_w, h, HIGHLIGHT if filled and t == 1 else
                 (PALE if filled else "white"), z=0)
            for j, ln in enumerate(wrapped.get(t, [])):
                put(ax, col_x[k] + 1.2, y - 2.9 - j * 2.1, ln, font, DARK)
        y -= h + gap
    out = save(fig, FIGDIR / "types_by_tier", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# the words that belong on the slide, not in the figure
# =====================================================================================
def slide_copy(acc, half):
    return {
        "confidently_wrong": (
            "01",
            "Confidently wrong",
            "A wrong answer arrives with the same certainty as a right one.",
            "Four of the 7B model's BioASQ answers. Each is wrong against the reference, "
            "and each got a probability above 0.98 from the model itself. Nothing in "
            "the text of an answer says which ones to check.",
        ),
        "selective_generation": (
            "02",
            "What an uncertainty score is for",
            "Rank the answers, flag the least certain, and what is kept gets better.",
            f"Accuracy of the answers kept as the most uncertain are flagged for "
            f"review, ranked by mean token entropy. Flagging half of them lifts the "
            f"kept accuracy from {acc:.0%} to {half:.0%}. A useless score stays on "
            f"the flat line; a perfect one follows the upper dashed line. AUROC, used "
            f"in the results, measures how well the ranking separates right from wrong.",
        ),
        "uq_types": (
            "03",
            "Five places to look for doubt",
            "Every estimator reads one of five signals, and each needs different "
            "access to the model.",
            "White-box needs the model's probabilities or internals; black-box needs "
            "only its text. Families and method names follow Vazhentsev et al. "
            "(arXiv:2605.14570); self-report is added here because p(True) is in the "
            "run. T1–T4 are the tiers the results are grouped by.",
        ),
        "information_based": (
            "04",
            "Reading the probabilities",
            "One forward pass, and the model has already said how sure it was.",
            "All three come out of the pass that produced the answer, so they need "
            "nothing beyond generation. They need the token probabilities, which "
            "many hosted APIs do not return.",
        ),
        "self_report": (
            "05",
            "Asking the model",
            "A second prompt asks the model to judge its own answer.",
            "p(True) reads the probability the model puts on “True”; the empirical "
            "variant samples the verdict ten times and counts. Both need one more "
            "prompt per answer, which is tier 2.",
        ),
        "consistency": (
            "06",
            "Asking again",
            "If repeated samples disagree, the first answer was a guess.",
            "Sample several answers to the same question and measure how far they "
            "agree: by shared words, or by meaning, with an NLI model testing whether "
            "two answers entail each other. Sampling is tier 3; the NLI model on top "
            "is tier 4. White-box variants also weight each sample by its probability.",
        ),
        "trained_probes": (
            "07",
            "Training a detector",
            "A classifier on the model's hidden states can learn what a wrong "
            "answer looks like — given labels.",
            "Probes need graded answers from the task they will be used on, and a new "
            "set for every new task or model. They are named here for completeness "
            "and were not run.",
        ),
        "lm_polygraph": (
            "08",
            "LM-Polygraph",
            "One library runs every one of these estimators behind one interface.",
            "Open source, pip-installable. An estimator returns one uncertainty score per "
            "answer, higher meaning less sure; grading the answers and scoring the "
            "estimators against them is a separate step.",
        ),
        "types_by_tier": (
            "09",
            "Every estimator, by type and tier",
            "Types say what signal a method reads; tiers, what it needs to run.",
            "Every estimator in the results, placed by signal type and tier. "
            "Tiers are cumulative: each adds one requirement to the one before. Sampling "
            "methods split across tiers 3 and 4 by whether they also need the NLI "
            "model.",
        ),
    }


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
    outs += [sel, fig_uq_types(), fig_information(), fig_self_report(),
             fig_consistency(), fig_trained_probes(), fig_lm_polygraph(),
             fig_types_by_tier()]

    copy = slide_copy(acc, half)
    for out in outs:
        md, wrote = write_markdown(out.stem, *copy[out.stem], force=args.rewrite_copy)
        print(f"wrote {out.relative_to(ROOT)}  +  {md.relative_to(ROOT)}"
              + ("" if wrote else "  (kept: edited copy)"))


if __name__ == "__main__":
    main()
