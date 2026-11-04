"""Figures for the LM-Polygraph cost-vs-performance journal club demo.

  1. auroc_by_method.png - all estimators ranked, with 95% bootstrap CIs and a
     tier tag. The point: the cheapest tier holds both the winner and the traps.
  2. auroc_by_tier.png   - AUROC spread within each cost tier. The point:
     paying more buys consistency, not a higher ceiling.
  3. uncertainty_by_correctness.png - raw scores split by correctness, for the
     cheapest good method against the expensive one.

Tier is encoded by POSITION and LABEL, never by colour: a four-hue categorical
set cannot clear the all-pairs CVD floors (checked with the palette validator),
and every mark in these charts is visible simultaneously.

Run:  .venv/bin/python scripts/plot_bioasq.py outputs/<cache>.json
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

ROOT = pathlib.Path(__file__).parent.parent
FIGDIR = ROOT / "figures"

ACCENT = "#2a78d6"     # the cheap tier - the story
MUTED = "#9a9992"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
CORRECT_C, INCORRECT_C = "#6f6e68", "#e34948"

TIER_LABEL = {
    1: "T1  one forward pass",
    2: "T2  + self-query",
    3: "T3  + sampling",
    4: "T4  + NLI model",
}

PRETTY = {
    "MaximumSequenceProbability": "Max sequence prob.",
    "MeanTokenEntropy": "Mean token entropy",
    "MeanPointwiseMutualInformation": "Mean PMI",
    "MeanConditionalPointwiseMutualInformation": "Mean conditional PMI",
    "MonteCarloSequenceEntropy": "MC sequence entropy",
    "MonteCarloNormalizedSequenceEntropy": "MC norm. seq. entropy",
    "LexicalSimilarity_rougeL": "Lexical similarity (RougeL)",
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


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


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


def fig_by_method(d, y, stats):
    order = sorted(stats, key=lambda n: -stats[n][0])
    fig, ax = plt.subplots(figsize=(8.6, 7.2), facecolor=SURFACE)
    style(ax)
    ypos = np.arange(len(order))[::-1]
    for yi, name in zip(ypos, order):
        auc, lo, hi = stats[name]
        tier = bioasq.tier_of(name)
        cheap = tier == 1
        color = ACCENT if cheap else MUTED
        ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round",
                alpha=1.0 if cheap else 0.8)
        ax.plot([auc], [yi], "o", color=color, ms=8,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        label = f"{auc:.3f}" + ("  inverted" if hi < 0.5 else "")
        ax.text(hi + 0.012, yi, label, va="center", fontsize=8.5,
                color=INK if cheap else INK_SOFT)
    ax.axvline(0.5, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(0.506, (len(order) - 1) / 2, "chance", fontsize=8, color=MUTED,
            rotation=90, ha="left", va="center")
    ax.set_yticks(ypos,
                  [f"{PRETTY.get(n, n)}   T{bioasq.tier_of(n)}" for n in order],
                  fontsize=9, color=INK)
    ax.set_xlabel("AUROC for detecting an incorrect answer  (95% bootstrap CI)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_xlim(0.22, 1.0)
    ax.grid(axis="y", visible=False)
    fig.suptitle("The cheapest tier holds both the winner and the traps",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title("blue = one forward pass, no sampling, no extra model",
                 loc="left", fontsize=9.5, color=INK_SOFT, pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    out = FIGDIR / "auroc_by_method.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_by_tier(d, y, stats):
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor=SURFACE)
    style(ax)
    for tier in sorted(TIER_LABEL):
        names = [n for n in stats if bioasq.tier_of(n) == tier]
        vals = [stats[n][0] for n in names]
        # Sign-inverted estimators get a hollow marker. Without that secondary
        # encoding they read as failures, when in fact they return a confidence
        # and are mid-table once flipped.
        inv = np.array([stats[n][2] < 0.5 for n in names])
        x = tier + rng.uniform(-0.13, 0.13, len(vals))
        cheap = tier == 1
        col = ACCENT if cheap else MUTED
        v = np.asarray(vals)
        if (~inv).any():
            ax.plot(x[~inv], v[~inv], "o", ms=7, color=col, alpha=0.8,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        if inv.any():
            ax.plot(x[inv], v[inv], "o", ms=7, markerfacecolor="none",
                    markeredgecolor=col, markeredgewidth=1.8, alpha=0.9)
        ax.plot([tier - 0.26, tier + 0.26], [np.median(vals)] * 2,
                color=col, lw=2.4, solid_capstyle="round", zorder=4)
    ax.axhline(0.5, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(4.46, 0.5, "chance", fontsize=8, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(list(TIER_LABEL), [TIER_LABEL[t] for t in sorted(TIER_LABEL)],
                  fontsize=9, color=INK)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylabel("AUROC", fontsize=9.5, color=INK_SOFT)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Paying more buys consistency, not a higher ceiling",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    ax.set_title("each dot is one estimator; bar = tier median; "
                 "hollow = sign-inverted (read as 1-AUROC)",
                 loc="left", fontsize=9.5, color=INK_SOFT, pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = FIGDIR / "auroc_by_tier.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_scores(d, y, methods=("MeanTokenEntropy", "SemanticEntropy")):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(methods), figsize=(7.6, 3.8), facecolor=SURFACE)
    for ax, name in zip(np.atleast_1d(axes), methods):
        style(ax)
        s = np.asarray(d["scores"][name], dtype=float)
        for cls, (label, color) in enumerate(
            [("Correct", CORRECT_C), ("Incorrect", INCORRECT_C)]
        ):
            vals = s[y == cls]
            x = cls + rng.uniform(-0.17, 0.17, len(vals))
            ax.plot(x, vals, "o", ms=3, color=color, alpha=0.4,
                    markeredgecolor="none")
            ax.plot([cls - 0.3, cls + 0.3], [np.median(vals)] * 2,
                    color=color, lw=2.4, solid_capstyle="round", zorder=4)
        # Semantic entropy has a long right tail (values to ~300 against a
        # median near 0), which flattens the whole distribution if plotted raw.
        # Clip the view to a robust range and say how many points are off-scale.
        lo, hi = np.percentile(s, [0.5, 99.0])
        pad = 0.06 * (hi - lo) if hi > lo else 1.0
        n_off = int((s > hi + pad).sum())
        ax.set_ylim(lo - pad, hi + pad)
        if n_off:
            ax.text(0.015, 0.985, f"{n_off} points above axis", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=MUTED)
        ax.set_xticks([0, 1], ["Correct", "Incorrect"], fontsize=9.5, color=INK)
        ax.set_xlim(-0.55, 1.55)
        tier = bioasq.tier_of(name)
        ax.set_title(f"{PRETTY.get(name, name)}  (T{tier})", fontsize=10.5,
                     color=INK, pad=8)
        ax.set_ylabel("uncertainty score", fontsize=9, color=INK_SOFT)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Cheapest vs most expensive \u2014 the overlap looks the same",
                 x=0.012, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGDIR / "uncertainty_by_correctness.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--manual", default=str(ROOT / "manual_labels.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    d, y = load(args.cache, args.manual)

    stats = {}
    for name, vals in d["scores"].items():
        v = np.asarray(vals, dtype=float)
        lo, hi = bootstrap_auc(y, v, n_boot=args.n_boot)
        stats[name] = (fast_auc(y, v), lo, hi)

    # Table view: three palette hues sit under 3:1 on this surface, so the
    # numbers must be legible somewhere that is not colour.
    print(f"n={len(y)}  correct={(y==0).sum()}  wrong={(y==1).sum()}\n")
    print(f"{'method':<30}{'tier':>5}{'AUROC':>8}{'95% CI':>18}")
    for n in sorted(stats, key=lambda n: -stats[n][0]):
        a, lo, hi = stats[n]
        flag = "  inverted" if hi < 0.5 else ""
        print(f"{PRETTY.get(n,n):<30}{'T'+str(bioasq.tier_of(n)):>5}{a:>8.3f}   [{lo:.3f}, {hi:.3f}]{flag}")

    for out in (fig_by_method(d, y, stats),
                fig_by_tier(d, y, stats),
                fig_scores(d, y)):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
