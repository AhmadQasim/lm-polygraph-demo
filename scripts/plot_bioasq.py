"""Figures for the LM-Polygraph cost-vs-performance journal club talk.

Drawn in the Helmholtz house style, against `scripts/hgf_style.py` (vendored from
~/projects/slides). Read `STYLE.md` in that repo for the full spec; the parts
that bind here are repeated below.

  1. auroc_by_method.png   - all estimators ranked, 95% bootstrap CI per method.
  2. auroc_by_tier.png     - AUROC spread within each cost tier.
  3. uncertainty_by_correctness.png - raw scores split by correctness, cheapest
     good method against the most expensive one.
  4. model_size_comparison.png - the same methods on two model sizes (--compare).

Run:  .venv/bin/python scripts/plot_bioasq.py outputs/<cache>.json

THE COLOUR KEY. Colour is semantic here and holds across all four figures - a
reader decodes a hue once. Do not use these decoratively.

    tier 1, one forward pass - the cheap tier, and the story   blue      #005AA0
    tiers 2-4, the paid-for tiers - context                    blue 60   #669CC6
    the model answered correctly                               darkblue  #002864
    the model answered incorrectly - what we detect            orange    #ED7D31
    chance, axes, scope caveats                                gray 85   #738084

Orange is the deck's "the model's output" hue, so a wrong answer wearing it is
consistent rather than arbitrary. It is too light to set type in (2.49:1 on
white), so orange *text* is ORANGE_DARK. Red and green stay reserved for
true/false-positive semantics and are unused.

TIER IS NOT A HUE. It is encoded by position and by a T1-T4 label, and the only
colour distinction is the binary tier-1-vs-rest emphasis above. Four categorical
hues cannot clear the all-pairs CVD contrast floors when every mark is visible
at once, which these charts all are. This was checked with the palette
validator; do not "improve" it into a four-colour scale.

CANVAS. Every figure is 124 x 44.5 figure units (12.4 x 4.45 in at 10 u/in),
the aspect that exactly fills the slide's figure band at 11.50 x 4.13 in. That
band is 4.13 in tall and nothing gets more, so a figure needing more than ~20
rows of 11 pt text has to use two columns - which is why auroc_by_method does.

WORDS. The claim headline, the blue lead line and the grey caption are NOT drawn
into the PNG: on a slide `deck_common.figure_slide()` places them itself, and a
figure carrying its own title would double up. They are written beside each
figure as `markdowns/{nn}_slide_{name}.md` in the slides repo's format. Method
names (PRETTY) and tier labels stay in Python because they double as lookup keys.
"""

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bioasq  # noqa: E402
from analyze_bioasq import bootstrap_auc, fast_auc  # noqa: E402
from hgf_style import (  # noqa: E402
    BLUE, DARK, GRAY22, GRAY85, BLUE60, ORANGE, ORANGE_DARK, PALE, HIGHLIGHT,
    TYPE, WEIGHT, GEOM, REFERENCE_DASH, CHART_MIN_SIZE,
    halvar, hermann, save, slab, use_hgf,
)

ROOT = pathlib.Path(__file__).parent.parent
FIGDIR = ROOT / "figures"
MDDIR = ROOT / "markdowns"

# --- the colour key, by meaning (see the module docstring) ---------------------------
CHEAP = BLUE          # tier 1: one forward pass. The story.
PAID = BLUE60         # tiers 2-4: context.
CORRECT_C = DARK      # the model was right
INCORRECT_C = ORANGE  # the model was wrong - what an estimator has to catch
AXIS = GRAY85         # axis, ticks, reference lines, scope caveats

CANVAS = GEOM["canvas"]           # 124 x 44.5 u
MARGINS = GEOM["margins"]         # 9 / 4 / 6 / 7 u

TIER_NAME = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}
TIER_GLOSS = {
    1: "one forward pass",
    2: "+ self-query",
    3: "+ sampling",
    4: "+ NLI model",
}

PRETTY = {
    "MaximumSequenceProbability": "Max sequence prob.",
    "MeanTokenEntropy": "Mean token entropy",
    "MeanPointwiseMutualInformation": "Mean PMI",
    "MeanConditionalPointwiseMutualInformation": "Mean cond. PMI",
    "MonteCarloSequenceEntropy": "MC sequence entropy",
    "MonteCarloNormalizedSequenceEntropy": "MC norm. seq. entropy",
    "LexicalSimilarity_rougeL": "Lexical similarity",
    "DegMat_NLI_score_entail": "DegMat",
    "Eccentricity_NLI_score_entail": "Eccentricity",
    "EigValLaplacian_NLI_score_entail": "EigValLaplacian",
    "SemanticEntropy": "Semantic entropy",
    "PTrue": "p(True)",
    "PTrueEmpirical": "p(True) empirical",
    "BoostedProbSequence": "Boosted prob.",
    "SelfCertainty": "Self-certainty",
    "KernelLanguageEntropy": "Kernel language entropy",
    "NumSemSets": "NumSemSets",
    "Perplexity": "Perplexity",
    "FisherRao": "Fisher-Rao",
    "RenyiNeg": "Renyi-Neg",
}


# =====================================================================================
# canvas helpers
# =====================================================================================
def canvas(cw=None, ch=None):
    """A figure at the deck's canvas, in figure units. Transparent, as the deck wants."""
    cw = CANVAS[0] if cw is None else cw
    ch = CANVAS[1] if ch is None else ch
    fig = plt.figure(figsize=(cw / 10, ch / 10))
    fig.patch.set_alpha(0.0)
    return fig, (cw, ch)


def axes_at(fig, size, x, y, w, h):
    """An axes placed by figure units (x,y from the bottom-left of the canvas).

    Spines, ticks and grid are stripped here, as `hgf_style.chart_axes` does -
    anything an axis would say is said by a direct label instead, and what is
    left is added back explicitly by `y_ticks` / `x_ticks`.
    """
    cw, ch = size
    ax = fig.add_axes((x / cw, y / ch, w / cw, h / ch))
    ax.set_facecolor("none")
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False,
                   length=0)
    ax.grid(False)
    ax.set_axisbelow(True)
    return ax


def y_ticks(ax, values, fmt="{:.2f}", grid=True, size=None):
    """Y tick labels in Halvar, with the one permitted gridline per major tick.

    The spec allows a 1 pt gray-22 line per tick where a chart has more than six
    categories and the values cannot all be direct-labelled - which is the case
    for the tier strip and the score distributions.
    """
    size = size or TYPE["tick"]
    y0, y1 = ax.get_ylim()
    for v in values:
        # The label has clip_on=False so it survives a gridline that is clipped
        # away, which leaves a number floating above the plot. Filter here.
        if not (y0 <= v <= y1):
            continue
        if grid:
            ax.axhline(v, color=GRAY22, lw=WEIGHT["grid"], zorder=0)
        ax.annotate(fmt.format(v), (0.0, v), xycoords=_blend(ax),
                    textcoords="offset points", xytext=(-6, 0), ha="right",
                    va="center", color=AXIS, fontproperties=halvar(size),
                    clip_on=False, zorder=7)


def x_ticks(ax, values, fmt="{:.2f}", size=None):
    size = size or TYPE["tick"]
    x0, x1 = ax.get_xlim()
    for v in values:
        if not (x0 <= v <= x1):
            continue
        ax.annotate(fmt.format(v), (v, 0.0), xycoords=_blend_y(ax),
                    textcoords="offset points", xytext=(0, -7), ha="center",
                    va="top", color=AXIS, fontproperties=halvar(size),
                    clip_on=False, zorder=7)


def _blend(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transAxes, ax.transData)


def _blend_y(ax):
    from matplotlib.transforms import blended_transform_factory
    return blended_transform_factory(ax.transData, ax.transAxes)


def heading(ax, text, x=0.0, y=1.0, dy=8, color=None, size=None, ha="left"):
    """One line of Hermann caps, unboxed - the deck's panel header / axis title."""
    return ax.annotate(text.upper(), (x, y), xycoords="axes fraction",
                       textcoords="offset points", xytext=(0, dy), ha=ha,
                       va="bottom", color=color or DARK,
                       fontproperties=hermann(size or TYPE["axis_title"]),
                       clip_on=False, zorder=7)


def caveat(fig, size, text, x=None, y=2.0):
    """A scope caveat in Halvar, bottom left. Every fragile number carries its scope."""
    cw, ch = size
    x = MARGINS[0] if x is None else x
    fig.text(x / cw, y / ch, text, ha="left", va="bottom", color=AXIS,
             fontproperties=halvar(TYPE["gloss"]))


def chance_line(ax, vertical=False, at=0.5, label="CHANCE", side="start"):
    """The 0.5 reference, dashed and gray - context, never the claim."""
    if vertical:
        ax.axvline(at, color=AXIS, lw=WEIGHT["reference"], ls=(0, REFERENCE_DASH),
                   zorder=2)
        y = 1.0 if side == "start" else 0.0
        ax.annotate(label, (at, y), xycoords=_blend_y(ax),
                    textcoords="offset points", xytext=(0, 5), ha="center",
                    va="bottom", color=AXIS, fontproperties=hermann(TYPE["tick"]),
                    clip_on=False, zorder=7)
    else:
        ax.axhline(at, color=AXIS, lw=WEIGHT["reference"], ls=(0, REFERENCE_DASH),
                   zorder=2)
        ax.annotate(label, (1.0, at), xycoords=_blend(ax),
                    textcoords="offset points", xytext=(0, 5), ha="right",
                    va="bottom", color=AXIS, fontproperties=hermann(TYPE["tick"]),
                    clip_on=False, zorder=7)


# =====================================================================================
# data
# =====================================================================================
def load(cache_path, manual_path):
    d = json.loads(pathlib.Path(cache_path).read_text())
    graded = [bioasq.grade(g, gg) for g, gg in zip(d["generations"], d["golds"])]
    if manual_path:
        applied, stale = bioasq.apply_manual_labels(
            graded, d["questions"], d["generations"], manual_path)
        if stale:
            print(f"note: {stale} manual labels skipped (prediction changed)")
    y = np.array([0 if g["correct"] else 1 for g in graded])
    return d, y


def scope_of(d):
    """The scope line every figure carries: run, model, n."""
    model = d.get("model", "?").split("/")[-1]
    return f"{model} · {d.get('n', '?')} BioASQ factoids · seed {d.get('seed', 0)}"


# =====================================================================================
# 1. every method, ranked
# =====================================================================================
def fig_by_method(d, y, stats):
    """Twenty methods ranked, in two columns of ten.

    One column would need ~2 u per row against the 11 u type floor, and the
    slide's figure band is only 44.5 u tall, so the rows would collide. The rank
    runs down the left column and continues down the right; the column header
    says which range each holds.
    """
    order = sorted(stats, key=lambda n: -stats[n][0])
    fig, size = canvas()
    cw, ch = size

    lo_all = min(s[1] for s in stats.values())
    hi_all = max(s[2] for s in stats.values())
    pad = 0.03
    xlim = (max(0.0, lo_all - pad), min(1.0, hi_all + pad))

    per_col = (len(order) + 1) // 2
    col_w, gutter = 53.0, 5.0
    name_w, value_w = 19.0, 9.0
    plot_w = col_w - name_w - value_w
    y0, col_h = 8.0, 30.0

    for col in (0, 1):
        names = order[col * per_col:(col + 1) * per_col]
        if not names:
            continue
        x0 = MARGINS[0] + col * (col_w + gutter)
        ax = axes_at(fig, size, x0 + name_w, y0, plot_w, col_h)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.5, per_col - 0.5)
        ax.invert_yaxis()

        chance_line(ax, vertical=True, label="CHANCE" if col == 0 else "")
        # Right-aligned: the CHANCE label sits over the 0.5 rule about a third
        # across, and a left-aligned rank header runs straight into it.
        heading(ax, f"rank {col * per_col + 1}–{col * per_col + len(names)}",
                x=1.0, ha="right", dy=11, color=AXIS, size=TYPE["tick"])

        for i, name in enumerate(names):
            auc, lo, hi = stats[name]
            tier = bioasq.tier_of(name)
            cheap = tier == 1
            colour = CHEAP if cheap else PAID
            inverted = hi < 0.5

            ax.plot([lo, hi], [i, i], color=colour, lw=WEIGHT["data"],
                    solid_capstyle="round", zorder=4)
            ax.plot([auc], [i], "o", ms=8 if cheap else 7,
                    markerfacecolor="none" if inverted else colour,
                    markeredgecolor=colour,
                    markeredgewidth=WEIGHT["context"] if inverted else 0.0,
                    zorder=5)

            # the method name, in its own gutter to the left of the plot
            ax.annotate(PRETTY.get(name, name), (0.0, i), xycoords=_blend(ax),
                        textcoords="offset points", xytext=(-7, 0), ha="right",
                        va="center", color=DARK if cheap else AXIS,
                        fontproperties=halvar(TYPE["tick"], bold=cheap),
                        clip_on=False, zorder=7)
            # the tier tag - the only thing that says which tier this is
            ax.annotate(TIER_NAME[tier], (1.0, i), xycoords=_blend(ax),
                        textcoords="offset points",
                        xytext=(8, 0), ha="left", va="center", color=AXIS,
                        fontproperties=hermann(TYPE["tick"]), clip_on=False,
                        zorder=7)
            # the value. The top method of the whole ranking is the claim.
            is_claim = col == 0 and i == 0
            ax.annotate(f"{auc:.3f}", (1.0, i), xycoords=_blend(ax),
                        textcoords="offset points", xytext=(26, 0), ha="left",
                        va="center",
                        color=DARK if is_claim else (DARK if cheap else AXIS),
                        fontproperties=(hermann(TYPE["panel_header"]) if is_claim
                                        else halvar(TYPE["value"], bold=True)),
                        clip_on=False, zorder=7)

        x_ticks(ax, [t for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
                     if xlim[0] <= t <= xlim[1]])

    fig.text(MARGINS[0] / cw, (y0 + col_h + 4.4) / ch,
             "AUROC FOR DETECTING A WRONG ANSWER", ha="left", va="bottom",
             color=DARK, fontproperties=hermann(TYPE["axis_title"]))

    caveat(fig, size,
           f"{scope_of(d)}  ·  bars are 95% bootstrap CIs  ·  "
           "hollow = sign-inverted, read as 1−AUROC")
    out = save(fig, FIGDIR / "auroc_by_method", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 2. spread within each cost tier
# =====================================================================================
def fig_by_tier(d, y, stats):
    """One column per cost tier, every estimator a dot, the median a bar."""
    rng = np.random.default_rng(0)
    fig, size = canvas()
    ml, mr, mt, mb = MARGINS
    ax = axes_at(fig, size, ml + 4, mb + 3, CANVAS[0] - ml - mr - 4,
                 CANVAS[1] - mt - mb - 6)

    vals_all = [s[0] for s in stats.values()]
    ax.set_xlim(0.4, 4.6)
    ax.set_ylim(min(vals_all) - 0.05, max(vals_all) + 0.07)

    y_ticks(ax, [round(v, 1) for v in np.arange(0.3, 0.91, 0.1)], fmt="{:.1f}")
    chance_line(ax)

    for tier in sorted(TIER_NAME):
        names = [n for n in stats if bioasq.tier_of(n) == tier]
        if not names:
            continue
        v = np.asarray([stats[n][0] for n in names])
        inv = np.asarray([stats[n][2] < 0.5 for n in names])
        x = tier + rng.uniform(-0.13, 0.13, len(v))
        cheap = tier == 1
        colour = CHEAP if cheap else PAID

        if (~inv).any():
            ax.plot(x[~inv], v[~inv], "o", ms=8 if cheap else 7, color=colour,
                    markeredgewidth=0.0, zorder=5)
        if inv.any():
            ax.plot(x[inv], v[inv], "o", ms=7, markerfacecolor="none",
                    markeredgecolor=colour, markeredgewidth=WEIGHT["context"],
                    zorder=5)
        ax.plot([tier - 0.26, tier + 0.26], [np.median(v)] * 2, color=colour,
                lw=WEIGHT["focus"], solid_capstyle="round", zorder=6)

        # category + gloss, which is what removes the legend
        ax.annotate(TIER_NAME[tier], (tier, 0.0), xycoords=_blend_y(ax),
                    textcoords="offset points", xytext=(0, -8), ha="center",
                    va="top", color=DARK if cheap else AXIS,
                    fontproperties=hermann(TYPE["category"]), clip_on=False,
                    zorder=7)
        ax.annotate(TIER_GLOSS[tier], (tier, 0.0), xycoords=_blend_y(ax),
                    textcoords="offset points", xytext=(0, -22), ha="center",
                    va="top", color=AXIS, fontproperties=halvar(TYPE["gloss"]),
                    clip_on=False, zorder=7)
        ax.annotate(f"{np.median(v):.3f}", (tier + 0.28, np.median(v)),
                    textcoords="offset points", xytext=(4, 0), ha="left",
                    va="center", color=DARK if cheap else AXIS,
                    fontproperties=(hermann(TYPE["panel_header"]) if cheap
                                    else halvar(TYPE["value"], bold=True)),
                    clip_on=False, zorder=7)

    heading(ax, "AUROC", dy=12)
    caveat(fig, size,
           f"{scope_of(d)}  ·  one dot per estimator  ·  bar is the tier "
           "median  ·  hollow = sign-inverted, read as 1−AUROC")
    out = save(fig, FIGDIR / "auroc_by_tier", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 3. the scores themselves
# =====================================================================================
def fig_scores(d, y, methods=("MeanTokenEntropy", "SemanticEntropy")):
    """Raw uncertainty scores split by correctness, cheapest against dearest."""
    rng = np.random.default_rng(0)
    fig, size = canvas()
    ml, mr, mt, mb = MARGINS
    gutter = GEOM["gutter"] + 8
    panel_w = (CANVAS[0] - ml - mr - gutter) / 2
    panel_h = CANVAS[1] - mt - mb - 7

    for col, name in enumerate(methods):
        x0 = ml + 5 + col * (panel_w + gutter)
        ax = axes_at(fig, size, x0, mb + 4, panel_w - 5, panel_h)
        s = np.asarray(d["scores"][name], dtype=float)

        # A long right tail (semantic entropy runs to ~300 against a median near
        # zero) flattens everything if plotted raw. Clip the view to a robust
        # range and say how many points are off-scale rather than dropping them.
        lo, hi = np.percentile(s, [0.5, 99.0])
        pad = 0.08 * (hi - lo) if hi > lo else 1.0
        n_off = int((s > hi + pad).sum())
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(lo - pad, hi + pad)

        from matplotlib.ticker import MaxNLocator
        ticks = MaxNLocator(4).tick_values(lo, hi)
        y_ticks(ax, ticks, fmt="{:.2f}" if hi - lo < 20 else "{:.0f}")

        for cls, colour in ((0, CORRECT_C), (1, INCORRECT_C)):
            vals = s[y == cls]
            jitter = cls + rng.uniform(-0.18, 0.18, len(vals))
            ax.plot(jitter, vals, "o", ms=3.4, color=colour, alpha=0.38,
                    markeredgewidth=0.0, zorder=4)
            med = float(np.median(vals))
            ax.plot([cls - 0.3, cls + 0.3], [med] * 2, color=colour,
                    lw=WEIGHT["focus"], solid_capstyle="round", zorder=6)
            ax.annotate(f"{med:.2f}",
                        (cls, med), textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom",
                        color=ORANGE_DARK if cls else colour,
                        fontproperties=halvar(TYPE["value"], bold=True),
                        clip_on=False, zorder=7)

        for cls, text in ((0, "correct"), (1, "incorrect")):
            ax.annotate(text.upper(), (cls, 0.0), xycoords=_blend_y(ax),
                        textcoords="offset points", xytext=(0, -8), ha="center",
                        va="top", color=ORANGE_DARK if cls else CORRECT_C,
                        fontproperties=hermann(TYPE["category"]), clip_on=False,
                        zorder=7)

        tier = bioasq.tier_of(name)
        heading(ax, f"{PRETTY.get(name, name)}   {TIER_NAME[tier]}", dy=10)
        if n_off:
            ax.annotate(f"{n_off} points above the axis", (1.0, 1.0),
                        xycoords="axes fraction", textcoords="offset points",
                        xytext=(0, -4), ha="right", va="top", color=AXIS,
                        fontproperties=halvar(TYPE["gloss"]), zorder=7)

    caveat(fig, size,
           f"{scope_of(d)}  ·  one dot per question  ·  bar is the median "
           "·  view clipped to the 0.5–99th percentile")
    out = save(fig, FIGDIR / "uncertainty_by_correctness", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# 4. two model sizes
# =====================================================================================
def fig_model_compare(caches, manual, labels_for=None):
    """Same methods, same questions, two model sizes, against the y=x diagonal.

    The diagonal shows both things at once: how stable the ranking is (scatter
    along the line) and that every point sits below it (detection is harder on
    the stronger model). The reading goes in a panel on the right rather than
    into the plot, which keeps the square data box square and fills the band.
    """
    from scipy.stats import spearmanr

    (small_path, small_tag), (big_path, big_tag) = caches
    aucs, accs, ds = [], [], []
    for path in (small_path, big_path):
        d, y = load(path, manual)
        aucs.append({n: fast_auc(y, np.asarray(v, float)) for n, v in d["scores"].items()})
        accs.append(int((y == 0).sum()) / len(y))
        ds.append(d)
    a_small, a_big = aucs
    names = [n for n in a_small if n in a_big]
    rho, pval = spearmanr([a_small[n] for n in names], [a_big[n] for n in names])

    # The sign-inverted estimators sit near 0.3 and would squash the informative
    # 0.70-0.81 cluster into a corner. They are in auroc_by_method.png; here the
    # range is cut to where the comparison actually lives, and the cut is stated.
    shown = [n for n in names if max(a_small[n], a_big[n]) >= 0.45]
    n_omitted = len(names) - len(shown)
    delta = float(np.mean([a_big[n] - a_small[n] for n in shown]))

    fig, size = canvas()
    ml, mr, mt, mb = MARGINS
    # The data box stays square (both axes are the same AUROC scale), so it can
    # only be as wide as the band is tall. The space that leaves on the right is
    # the reading panel, not padding.
    box = 30.0
    ax = axes_at(fig, size, ml + 6, mb + 2, box, box)

    lo = min(min(a_small[n] for n in shown), min(a_big[n] for n in shown)) - 0.03
    hi = max(max(a_small[n] for n in shown), max(a_big[n] for n in shown)) + 0.03
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    ax.plot([lo, hi], [lo, hi], color=AXIS, lw=WEIGHT["reference"],
            ls=(0, REFERENCE_DASH), zorder=2)
    ax.annotate("EQUAL", (hi, hi), textcoords="offset points", xytext=(-6, 4),
                ha="right", va="bottom", rotation=45, color=AXIS,
                fontproperties=hermann(TYPE["tick"]), zorder=7)

    ticks = [round(v, 2) for v in np.arange(np.ceil(lo * 20) / 20, hi, 0.05)]
    y_ticks(ax, ticks, grid=False)
    x_ticks(ax, ticks)

    OFFSETS = {
        "MeanTokenEntropy": (-9, 8),
        "SemanticEntropy": (9, -10),
        "PTrue": (9, -2),
        "PTrueEmpirical": (9, -9),
    }
    labels_for = labels_for or set()
    for n in shown:
        xv, yv = a_small[n], a_big[n]
        tier = bioasq.tier_of(n)
        cheap = tier == 1
        colour = CHEAP if cheap else PAID
        ax.plot([xv], [yv], "o", ms=8 if cheap else 7, color=colour,
                markeredgewidth=0.0, zorder=5)
        if n in labels_for:
            dx, dy = OFFSETS.get(n, (9, -2))
            ax.annotate(PRETTY.get(n, n), (xv, yv), textcoords="offset points",
                        xytext=(dx, dy), ha="right" if dx < 0 else "left",
                        va="center", color=DARK if cheap else AXIS,
                        fontproperties=halvar(TYPE["tick"], bold=cheap),
                        clip_on=False, zorder=7)

    heading(ax, f"AUROC on {big_tag}", dy=12)
    # (0.5, 0) in AXES FRACTION. _blend_y reads the 0.5 as an AUROC value and
    # drops the title off to the left of the plot.
    ax.annotate(f"AUROC ON {small_tag.upper()}", (0.5, 0.0),
                xycoords="axes fraction",
                textcoords="offset points", xytext=(0, -24), ha="center", va="top",
                color=DARK, fontproperties=hermann(TYPE["axis_title"]),
                clip_on=False, zorder=7)

    # --- the reading, as a flat panel on a pale ground --------------------------------
    # Hermann carries no Greek and no arrows (checked against the OTF cmap), so
    # the big numeral is Hermann and anything symbolic lives in the Halvar gloss.
    px = ml + 6 + box + 13
    pw = CANVAS[0] - mr - px
    ph = CANVAS[1] - mt - mb
    pax = axes_at(fig, size, px, mb, pw, ph)
    pax.set_xlim(0, pw)
    pax.set_ylim(0, ph)
    slab(pax, 0, 0, pw, ph, PALE, z=1)

    rows = [
        ("THE RANKING TRANSFERS", f"{rho:.3f}",
         f"Spearman \u03c1 over all {len(names)} methods, p = {pval:.0e}"),
        ("THE TASK GETS HARDER", f"{delta:+.3f}",
         f"mean AUROC change over the {len(shown)} methods above chance"),
        ("ACCURACY", f"{accs[0]:.1%} / {accs[1]:.1%}",
         f"{small_tag} / {big_tag}, the same questions"),
    ]
    inset, pad_x = 2.2, 3.4
    pitch = (ph - 2 * inset) / len(rows)
    gloss_font = halvar(TYPE["gloss"])
    room = int((pitch - 6.8) / 1.75)
    for i, (head, value, gloss) in enumerate(rows):
        top = ph - inset - i * pitch
        pax.text(pad_x, top, head, ha="left", va="top", color=DARK,
                 fontproperties=hermann(TYPE["panel_header"]))
        pax.text(pad_x, top - 2.9, value, ha="left", va="top", color=DARK,
                 fontproperties=hermann(TYPE["claim"]))
        gloss_lines = wrap_measured(fig, gloss, gloss_font, pw - 2 * pad_x)
        if len(gloss_lines) > room:
            raise AssertionError(
                f"gloss for {head!r} wraps to {len(gloss_lines)} lines and the "
                f"row has room for {room} - shorten it or widen the panel")
        for j, line in enumerate(gloss_lines):
            pax.text(pad_x, top - 6.8 - j * 1.75, line, ha="left", va="top",
                     color=AXIS, fontproperties=gloss_font)

    caveat(fig, size,
           f"{ds[0].get('n', '?')} BioASQ factoids \u00b7 seed "
           f"{ds[0].get('seed', 0)}  \u00b7  blue = tier 1  \u00b7  "
           f"{n_omitted} sign-inverted methods off-range")
    out = save(fig, FIGDIR / "model_size_comparison", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# the words that belong on the slide, not in the figure
# =====================================================================================
SLIDE_COPY = {
    "auroc_by_method": (
        "10",
        "Every estimator, ranked",
        "The cheapest tier holds both the winner and the traps.",
        "AUROC for separating wrong answers from right ones, with 95% bootstrap "
        "confidence intervals. Blue is tier 1 — one forward pass, no sampling "
        "and no second model. Hollow markers are sign-inverted: they return a "
        "confidence rather than an uncertainty, and read as 1−AUROC.",
    ),
    "auroc_by_tier": (
        "11",
        "What each cost tier buys",
        "Paying more buys consistency, not a higher ceiling.",
        "Each dot is one estimator, placed in its cost tier; the bar is the tier "
        "median. Tiers are cumulative, so the step from one column to the next is "
        "the marginal cost of that tier — a second forward pass, then sampling, "
        "then a DeBERTa NLI model over the sample set.",
    ),
    "uncertainty_by_correctness": (
        "12",
        "The scores behind the AUROC",
        "Cheapest against dearest — the overlap looks much the same.",
        "Raw uncertainty scores for every question, split by whether the answer was "
        "graded correct. An AUROC in the high 0.7s still means two heavily "
        "overlapping distributions: this is a triage signal, not a verdict.",
    ),
    "model_size_comparison": (
        "13",
        "The same methods on a bigger model",
        "The ranking survives scale; the task gets harder.",
        "Every method's AUROC on the smaller model against the larger one. Points "
        "cluster along the diagonal — the ranking transfers — but all of "
        "them sit below it, so the stronger model's residual errors are harder to "
        "catch even though it is right more often.",
    ),
}


def write_markdown(stem):
    """The slide's three lines, in the slides repo's `{slide}_slide_*.md` format.

    They are deliberately not drawn into the PNG: `deck_common.figure_slide()`
    places the title, the blue lead line and the grey caption itself, and a
    figure carrying its own headline would double up on the slide.
    """
    number, title, lead, caption = SLIDE_COPY[stem]
    MDDIR.mkdir(exist_ok=True)
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
        f"  > {lead}\n\n"
        "- **caption** — the grey paragraph under the figure — it wraps on its own\n"
        + "".join(f"  > {line}\n" for line in _wrap(caption, 84))
    )
    return path


def _wrap(text, width):
    import textwrap
    return textwrap.wrap(text, width)


def wrap_measured(fig, text, font, max_u):
    """Greedy wrap at a measured width, in figure units.

    The slides repo lays its figures out by hand and has `copy_check.py` because
    a guessed characters-per-line estimate lets a one-word edit run off a panel
    with nothing failing. Measuring is the cheap version of that: the string
    that gets drawn is the string that gets measured.
    """
    renderer = fig.canvas.get_renderer()

    def width_u(t):
        art = fig.text(0, 0, t, fontproperties=font)
        w = art.get_window_extent(renderer).width / fig.dpi * 10.0
        art.remove()
        return w

    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and width_u(trial) > max_u:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


# =====================================================================================
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--manual", default=str(ROOT / "manual_labels.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--compare", help="second cache: plot model-size comparison")
    ap.add_argument("--labels", default="small,big", help="axis labels for --compare")
    args = ap.parse_args()

    use_hgf()
    FIGDIR.mkdir(exist_ok=True)
    d, y = load(args.cache, args.manual)

    stats = {}
    for name, vals in d["scores"].items():
        v = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_auc(y, v, n_boot=args.n_boot)
        stats[name] = (fast_auc(y, v), lo, hi)

    # The table view. Tier is not a hue, so the numbers have to be legible
    # somewhere that is not the chart.
    print(f"n={len(y)}  correct={(y==0).sum()}  wrong={(y==1).sum()}\n")
    print(f"{'method':<30}{'tier':>5}{'AUROC':>8}{'95% CI':>18}")
    for n in sorted(stats, key=lambda n: -stats[n][0]):
        a, lo, hi = stats[n]
        flag = "  inverted" if hi < 0.5 else ""
        print(f"{PRETTY.get(n,n):<30}{'T'+str(bioasq.tier_of(n)):>5}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]{flag}")

    outs = [fig_by_method(d, y, stats), fig_by_tier(d, y, stats), fig_scores(d, y)]
    if args.compare:
        small, big = args.labels.split(",")
        outs.append(fig_model_compare(
            [(args.cache, small), (args.compare, big)], args.manual,
            labels_for={"MeanTokenEntropy", "SemanticEntropy", "PTrue",
                        "PTrueEmpirical"}))
    print()
    for out in outs:
        md = write_markdown(out.stem)
        print(f"wrote {out}\n      {out.with_suffix('.svg')}\n      {md}")


if __name__ == "__main__":
    main()
