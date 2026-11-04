"""Bootstrap CIs, paired comparisons and grading sensitivity for a cached run.

With n=100 and only ~25 correct answers, raw AUROC differences are easy to
over-read. This reports interval estimates and shows how much the automated
grader's borderline calls could move the result.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bioasq  # noqa: E402


def fast_auc(y, s):
    """AUROC via the Mann-Whitney U identity, tie-corrected.

    Identical to sklearn.metrics.roc_auc_score but without its per-call
    validation, which dominates when bootstrapping hundreds of thousands of
    times. At n=1000 x 5000 resamples x 20 estimators the sklearn version is
    slow enough to get the process killed.
    """
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def bootstrap_auc(y, s, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    idx = np.arange(len(y))
    out = []
    for _ in range(n_boot):
        b = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[b])) < 2:
            continue
        out.append(fast_auc(y[b], s[b]))
    return np.percentile(out, [2.5, 97.5])


def paired_delta(y, a, b, n_boot=5000, seed=0):
    """Bootstrap distribution of AUROC(a) - AUROC(b) on the same resamples."""
    rng = np.random.default_rng(seed)
    y, a, b = np.asarray(y), np.asarray(a), np.asarray(b)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) < 2:
            continue
        deltas.append(fast_auc(y[s], a[s]) - fast_auc(y[s], b[s]))
    deltas = np.array(deltas)
    return deltas.mean(), np.percentile(deltas, [2.5, 97.5]), (deltas > 0).mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("--show-borderline", action="store_true")
    ap.add_argument("--manual", help="path to manual_labels.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.cache).read_text())
    graded = [bioasq.grade(g, gg) for g, gg in zip(d["generations"], d["golds"])]

    applied, stale = bioasq.apply_manual_labels(
        graded, d["questions"], d["generations"], args.manual) if args.manual else (0, 0)
    n_over = applied

    y = np.array([0 if g["correct"] else 1 for g in graded])
    print(f"n={len(y)}  correct={int((y==0).sum())}  wrong={int((y==1).sum())}"
          + (f"  (manual overrides: {n_over}" if n_over else "")
          + (f", {stale} labelled questions with no label for this answer" if stale else "")
          + (")" if n_over else ""))

    names = list(d["scores"].keys())
    aucs = {n: roc_auc_score(y, d["scores"][n]) for n in names}
    order = sorted(names, key=lambda n: -aucs[n])

    print(f"\n{'estimator':<34}{'AUROC':>7}{'95% CI':>18}")
    print("-" * 59)
    inverted = []
    for n in order:
        lo, hi = bootstrap_auc(y, d["scores"][n], n_boot=args.n_boot)
        # An AUROC whose whole interval sits below 0.5 does not mean the method
        # is useless - it means the score is ordered as CONFIDENCE rather than
        # uncertainty, so it ranks correct answers high. Flag it; do not quietly
        # flip it, because which convention a given estimator uses is a real
        # property of the library that a user has to know about.
        if hi < 0.5:
            inverted.append((n, aucs[n]))
        mark = "   <- sign-inverted" if hi < 0.5 else ""
        print(f"{n:<34}{aucs[n]:>7.3f}   [{lo:.3f}, {hi:.3f}]{mark}")
    if inverted:
        print("\n  Sign-inverted: the entire CI lies below chance, i.e. the estimator")
        print("  returns a confidence, not an uncertainty. Read as 1-AUROC:")
        for n, a in inverted:
            print(f"    {n:<40} {a:.3f}  ->  {1 - a:.3f}")

    best = order[0]
    print(f"\n--- paired bootstrap vs SemanticEntropy ---")
    for n in order:
        if n == "SemanticEntropy":
            continue
        m, (lo, hi), p = paired_delta(y, d["scores"][n], d["scores"]["SemanticEntropy"], n_boot=args.n_boot)
        sig = "" if lo < 0 < hi else "  *"
        print(f"  {n:<34} delta={m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P(>SE)={p:.2f}{sig}")
    print("  (* = 95% interval excludes zero)")

    # Sensitivity: how much can the grader's soft calls move things?
    border = [i for i, g in enumerate(graded) if g["borderline"]]
    print(f"\n--- grading sensitivity ({len(border)} borderline of {len(y)}) ---")
    y_flip = y.copy()
    for i in border:
        y_flip[i] = 1 - y_flip[i]
    print(f"{'estimator':<34}{'as-graded':>11}{'all flipped':>13}")
    for n in order:
        a2 = roc_auc_score(y_flip, d["scores"][n]) if len(np.unique(y_flip)) > 1 else float("nan")
        print(f"{n:<34}{aucs[n]:>11.3f}{a2:>13.3f}")

    if args.show_borderline:
        print(f"\n--- borderline gradings (review these) ---")
        for i in border:
            g = graded[i]
            print(f"  [{'OK ' if g['correct'] else 'BAD'}] f1={g['f1']:.2f}")
            print(f"      Q    {d['questions'][i][:76]}")
            print(f"      pred {d['generations'][i].strip()[:70]!r}")
            print(f"      gold {g['gold'][:70]!r}")


if __name__ == "__main__":
    main()
