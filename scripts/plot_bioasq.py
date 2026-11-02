"""Figures for the semantic-entropy journal club demo.

Three figures, in order of what they are for:
  1. auroc_ci.png   - AUROC per method with 95% bootstrap CIs. The headline:
                      the intervals overlap, so the methods are not separated.
  2. roc_curves.png - the same result as ROC curves.
  3. uncertainty_by_correctness.png - raw score distributions split by whether
                      the answer was right, for semantic entropy vs token entropy.

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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bioasq  # noqa: E402
from analyze_bioasq import bootstrap_auc  # noqa: E402

ROOT = pathlib.Path(__file__).parent.parent
FIGDIR = ROOT / "figures"

# Validated categorical palette, fixed order (see dataviz skill palette.md).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d7d2"
MUTED = "#9a9992"

METHOD_COLOR = {
    "SemanticEntropy": SERIES[0],
    "MeanTokenEntropy": SERIES[1],
    "MaximumSequenceProbability": SERIES[2],
    "PTrue": SERIES[3],
    "EigValLaplacian_NLI_score_entail": SERIES[4],
    "NumSemSets": SERIES[5],
}

# Correctness is a different axis from method identity, so it gets colours
# outside the method palette: neutral for correct, one accent for the class
# we are trying to detect.
CORRECT_C, INCORRECT_C = "#6f6e68", "#e34948"

SHORT = {
    "SemanticEntropy": "Semantic entropy",
    "MeanTokenEntropy": "Mean token entropy",
    "MaximumSequenceProbability": "Max sequence prob.",
    "EigValLaplacian_NLI_score_entail": "EigValLaplacian (BB)",
    "NumSemSets": "NumSemSets (BB)",
    "PTrue": "p(True)",
}


def style(ax):
    """Recessive axes and grid; data should be the only assertive thing."""
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
    if manual_path and pathlib.Path(manual_path).exists():
        manual = json.loads(pathlib.Path(manual_path).read_text())["labels"]
        for i, q in enumerate(d["questions"]):
            if q in manual:
                graded[i]["correct"] = manual[q]["correct"]
    y = np.array([0 if g["correct"] else 1 for g in graded])  # 1 = model was wrong
    return d, y


def fig_auroc_ci(d, y, order, aucs):
    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    style(ax)
    ypos = np.arange(len(order))[::-1]
    for yi, name in zip(ypos, order):
        lo, hi = bootstrap_auc(y, d["scores"][name])
        focus = name == "SemanticEntropy"
        color = SERIES[0] if focus else MUTED
        ax.plot([lo, hi], [yi, yi], color=color, lw=2, solid_capstyle="round",
                alpha=1.0 if focus else 0.75)
        ax.plot([aucs[name]], [yi], "o", color=color, ms=9,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.text(hi + 0.012, yi, f"{aucs[name]:.3f}", va="center", ha="left",
                fontsize=9, color=INK if focus else INK_SOFT,
                fontweight="bold" if focus else "normal")
    ax.axvline(0.5, color=GRID, lw=1.2, ls="--", zorder=0)
    ax.text(0.506, (len(order) - 1) / 2, "chance", fontsize=8, color=MUTED,
            rotation=90, ha="left", va="center")
    ax.set_yticks(ypos, [SHORT.get(n, n) for n in order], fontsize=9.5, color=INK)
    ax.set_xlabel("AUROC for detecting an incorrect answer  (95% bootstrap CI)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="y", visible=False)
    fig.suptitle("No method is cleanly separated from the others",
                 x=0.015, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGDIR / "auroc_ci.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_roc(d, y, order, aucs):
    fig, ax = plt.subplots(figsize=(5.6, 5.0), facecolor=SURFACE)
    style(ax)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", zorder=0)
    for name in order:
        fpr, tpr, _ = roc_curve(y, d["scores"][name])
        focus = name == "SemanticEntropy"
        ax.plot(fpr, tpr, color=METHOD_COLOR[name],
                lw=2.4 if focus else 1.6, alpha=1.0 if focus else 0.8,
                label=f"{SHORT.get(name, name)}  {aucs[name]:.3f}",
                zorder=3 if focus else 2)
    ax.set_xlabel("False positive rate", fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel("True positive rate", fontsize=9.5, color=INK_SOFT)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    leg = ax.legend(loc="lower right", frameon=True, fontsize=8.5,
                    facecolor=SURFACE, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle("Detecting incorrect answers", x=0.015, ha="left",
                 fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = FIGDIR / "roc_curves.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_scores(d, y, methods=("SemanticEntropy", "MeanTokenEntropy")):
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
            ax.plot(x, vals, "o", ms=5, color=color, alpha=0.65,
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
            ax.plot([cls - 0.3, cls + 0.3], [np.median(vals)] * 2,
                    color=color, lw=2.4, solid_capstyle="round", zorder=4)
        ax.set_xticks([0, 1], ["Correct", "Incorrect"], fontsize=9.5, color=INK)
        ax.set_xlim(-0.55, 1.55)
        ax.set_title(SHORT.get(name, name), fontsize=10.5, color=INK, pad=8)
        ax.set_ylabel("uncertainty score", fontsize=9, color=INK_SOFT)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Scores overlap heavily — separation is real but far from clean",
                 x=0.015, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = FIGDIR / "uncertainty_by_correctness.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--manual", default=str(ROOT / "manual_labels.json"))
    args = ap.parse_args()

    FIGDIR.mkdir(exist_ok=True)
    d, y = load(args.cache, args.manual)
    aucs = {n: roc_auc_score(y, v) for n, v in d["scores"].items()}
    order = sorted(aucs, key=lambda n: -aucs[n])

    # Table view - the contrast WARN on three palette hues obliges visible values.
    print(f"n={len(y)}  correct={(y==0).sum()}  wrong={(y==1).sum()}\n")
    print(f"{'method':<24}{'AUROC':>8}{'95% CI':>20}")
    for n in order:
        lo, hi = bootstrap_auc(y, d["scores"][n])
        print(f"{SHORT.get(n,n):<24}{aucs[n]:>8.3f}   [{lo:.3f}, {hi:.3f}]")

    for out in (fig_auroc_ci(d, y, order, aucs),
                fig_roc(d, y, order, aucs),
                fig_scores(d, y)):
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
