"""BioASQ factoid loading, prompting and correctness grading.

The `answer` field in jmhb/BioASQ is a stringified Python list whose nesting is
inconsistent (`"['x']"` vs `"[['x']]"`); the inner lists are synonym sets, so a
prediction matching any flattened entry counts as correct.
"""

import ast
import random
import re
import string
from typing import Iterable

HF_DATASET = "jmhb/BioASQ"

# Estimators grouped by what they cost, which is the axis the talk is about.
# Tiers are cumulative: running "4" also runs 1-3, so the wall-clock delta
# between tier runs is the marginal cost of that tier. Attention-based methods
# (RAUQ, CSL, AttentionScore) are excluded - they need output_attentions=True,
# which forces the slow eager path on transformers 5.x.
TIERS = {
    # One greedy forward pass. Logits only, no extra model, no sampling.
    1: [
        "MaximumSequenceProbability",
        "Perplexity",
        "MeanTokenEntropy",
        "FisherRao",
        "RenyiNeg",
        "SelfCertainty",
        "BoostedProbSequence",
        # These two need a second, context-free LM pass (~2x tier-1 cost).
        "MeanPointwiseMutualInformation",
        "MeanConditionalPointwiseMutualInformation",
    ],
    # Ask the model about its own answer: one or two extra prompts.
    # Verbalized1S is deliberately absent: it regex-parses a confidence out of
    # the model's own text and raises IndexError on Qwen2.5-3B, which does not
    # emit a parseable confidence. That is a finding, not a bug to fix here.
    2: ["PTrue", "PTrueEmpirical"],
    # Sample N continuations (~10x), but no entailment model.
    3: ["MonteCarloSequenceEntropy", "MonteCarloNormalizedSequenceEntropy",
        "LexicalSimilarity"],
    # Sampling plus a DeBERTa NLI model over the sample set: the expensive tier.
    4: ["SemanticEntropy", "DegMat", "Eccentricity", "EigValLaplacian",
        "NumSemSets", "KernelLanguageEntropy"],
}


# Hand-written, not drawn from the eval set, so nothing leaks.
FEWSHOT = [
    ("Which enzyme is inhibited by statins?", "HMG-CoA reductase"),
    ("What is the inheritance pattern of cystic fibrosis?", "Autosomal recessive"),
    ("Which organ produces insulin?", "The pancreas"),
]

INSTRUCTION = (
    "Answer each biomedical question with a short noun phrase. "
    "Give only the answer, with no explanation and no full sentence."
)


def parse_golds(raw: str) -> list[str]:
    """Flatten the nested synonym structure into a flat list of gold strings."""
    out: list[str] = []

    def flat(x) -> None:
        if isinstance(x, str):
            if x.strip():
                out.append(x.strip())
        else:
            for item in x:
                flat(item)

    flat(ast.literal_eval(raw))
    return out


def load_factoid(n: int | None = 100, max_gold_words: int = 5, seed: int = 0) -> list[dict]:
    """Short-answer factoid subset, deterministically sampled."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="factoid")
    items = []
    for row in ds:
        golds = parse_golds(row["answer"])
        if not golds:
            continue
        if min(len(g.split()) for g in golds) > max_gold_words:
            continue
        items.append({"id": row["id"], "question": row["question"], "golds": golds})
    random.Random(seed).shuffle(items)
    return items[:n] if n else items


def build_prompt(question: str) -> str:
    shots = "\n\n".join(f"Question: {q}\nAnswer: {a}" for q, a in FEWSHOT)
    return f"{INSTRUCTION}\n\n{shots}\n\nQuestion: {question}\nAnswer:"


# --- grading -------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    # Crude depluralisation so "cell membranes" and "plasma membrane" share a
    # token. Without it such pairs score f1=0 and are silently graded wrong
    # rather than surfacing as borderline for review.
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in s.split()]
    return " ".join(toks)


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return 0.0
    common = 0
    g_pool = list(g)
    for tok in p:
        if tok in g_pool:
            g_pool.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def grade(pred: str, golds: Iterable[str], f1_threshold: float = 0.6) -> dict:
    """Best match over the synonym set. `correct` is the label used for AUROC."""
    best = {"em": False, "contains": False, "f1": 0.0, "gold": ""}
    npred = normalize(pred)
    for gold in golds:
        ngold = normalize(gold)
        em = npred == ngold
        contains = bool(ngold) and ngold in npred
        f1 = token_f1(pred, gold)
        if (em, contains, f1) > (best["em"], best["contains"], best["f1"]):
            best = {"em": em, "contains": contains, "f1": f1, "gold": gold}
    best["correct"] = best["em"] or best["contains"] or best["f1"] >= f1_threshold
    # Flag anything decided by a soft rule near the threshold for human review.
    best["borderline"] = (not best["em"]) and 0.3 <= best["f1"] < 0.8
    return best


def tier_of(display_name: str) -> int | None:
    """Map an LM-Polygraph display name back to its cost tier.

    Display names carry suffixes the TIERS keys do not ("DegMat" ->
    "DegMat_NLI_score_entail"), so match on the longest class-name prefix.
    """
    best, best_len = None, 0
    for tier, names in TIERS.items():
        for n in names:
            if display_name.startswith(n) and len(n) > best_len:
                best, best_len = tier, len(n)
    return best


def apply_manual_labels(graded, questions, generations, path) -> tuple[int, int]:
    """Apply human adjudications, but only where the prediction still matches.

    Labels record the model output they were judged against. A different model
    (or decoding config) answers differently, so applying the old verdict would
    silently mislabel. Returns (applied, skipped_stale).
    """
    import json as _json
    import pathlib as _pathlib

    p = _pathlib.Path(path)
    if not p.exists():
        return 0, 0
    labels = _json.loads(p.read_text())["labels"]
    applied = stale = 0
    for i, q in enumerate(questions):
        lab = labels.get(q)
        if lab is None:
            continue
        judged_against = lab.get("pred")
        if judged_against is not None and judged_against != generations[i].strip():
            stale += 1
            continue
        if graded[i]["correct"] != lab["correct"]:
            graded[i]["correct"] = lab["correct"]
            applied += 1
    return applied, stale
