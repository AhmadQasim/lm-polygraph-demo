"""BioASQ factoid loading, prompting and correctness grading.

The `answer` field in jmhb/BioASQ is a stringified Python list whose nesting is
inconsistent (`"['x']"` vs `"[['x']]"`); the inner lists are synonym sets, so a
prediction matching any flattened entry counts as correct.
"""

import ast
import random
import re
import string
import unicodedata
from typing import Iterable

HF_DATASET = "jmhb/BioASQ"

# Estimators grouped by what they need to run. Tiers are cumulative: each adds
# one requirement to the one before, and running "4" also runs 1-3. Attention-based methods
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
        # These two need a second, context-free LM pass.
        "MeanPointwiseMutualInformation",
        "MeanConditionalPointwiseMutualInformation",
    ],
    # Ask the model about its own answer: one or two extra prompts.
    # Verbalized1S is deliberately absent: it regex-parses a confidence out of
    # the model's own text and raises IndexError on Qwen2.5-3B, which does not
    # emit a parseable confidence. That is a finding, not a bug to fix here.
    2: ["PTrue", "PTrueEmpirical"],
    # Sample several continuations (10 by default), but no entailment model.
    3: ["MonteCarloSequenceEntropy", "MonteCarloNormalizedSequenceEntropy",
        "LexicalSimilarity"],
    # Sampling plus a DeBERTa NLI model over the sample set.
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
# Punctuation becomes a SPACE, not nothing. Deleting it made "X-linked" into
# "xlinked" while the gold's "X- linked" became "x linked", so the two never
# matched; with spaces they agree, and the compact comparison in grade() covers
# the opposite case ("Na+/Ca2+" against "NaCa2").
_PUNCT = str.maketrans({c: " " for c in string.punctuation})
_PAREN = re.compile(r"\(([^()]*)\)")
_SPECIAL = re.compile(r"<\|[^|>]*\|>")      # "<|endoftext|>" leaking into answers
_POSSESSIVE = re.compile(r"['\u2019]s\b")   # "Gaucher's" -> "Gaucher"
_WORDY = re.compile(r"[A-Za-z]{2,}")           # a parenthetical that holds a word
_GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
          "κ": "kappa", "λ": "lambda", "μ": "mu", "σ": "sigma", "τ": "tau",
          "ω": "omega"}
# Token rewrites applied after tokenising. Only spellings that mean the same
# thing in every BioASQ context: British/US variants, and the cytokine prefixes
# that were the largest single source of false "wrong" grades ("IL-6" against
# "interleukin-6"). Anything less mechanical is a manual label, not a rule here.
_SYNONYM = {
    "il": "interleukin", "ifn": "interferon",
    "tumour": "tumor", "oestrogen": "estrogen", "haemoglobin": "hemoglobin",
    "leukaemia": "leukemia", "anaemia": "anemia", "oedema": "edema",
    "haemophilia": "hemophilia", "behaviour": "behavior", "colour": "color",
}


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", _SPECIAL.sub(" ", s))
    for g, name in _GREEK.items():
        s = s.replace(g, f" {name} ")
    # The models sometimes drift into another script mid-answer ("Migraine头痛",
    # "B细胞表面CD19"). Those characters are not part of any gold, and left in
    # place they glue onto the Latin token and hide it from whole-token matching.
    s = "".join(c if ord(c) < 0x250 else " " for c in s)
    s = _POSSESSIVE.sub("", s)
    s = s.lower().translate(_PUNCT)
    s = _ARTICLES.sub(" ", s)
    toks = [_SYNONYM.get(t, t) for t in s.split()]
    # Crude depluralisation so "cell membranes" and "plasma membrane" share a
    # token. Without it such pairs score f1=0 and are silently graded wrong
    # rather than surfacing as borderline for review.
    toks = [t[:-2] if len(t) > 4 and t.endswith("oes") else
            t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks]
    return " ".join(toks)


def variants(s: str) -> list[str]:
    """The string, plus each side of a parenthetical: "Platelet derived growth
    factor-alpha (PDGFRA)" is also "Platelet derived growth factor-alpha" and
    "PDGFRA". The gold set uses this form for name-plus-abbreviation."""
    out = [s]
    # Only a parenthetical holding a word is an alternative name. "t(11;22)",
    # "Na(+)" and "mg/m(2)" are notation, and splitting them made "t(11;14)"
    # match "t(11;22)" on the leftover "t".
    inner = [i for i in _PAREN.findall(s) if _WORDY.search(i)]
    if inner:
        outer = _PAREN.sub(lambda m: " " if _WORDY.search(m.group(1)) else m.group(0),
                           s).strip()
        out += [outer] + [i.strip() for i in inner]
    return [v for v in out if v.strip()]


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


def _numbers(norm: str) -> list[str]:
    return sorted(t for t in norm.split() if any(c.isdigit() for c in t))


def _contains(hay: str, needle: str) -> bool:
    """Whole-token containment, so "ra" is not found inside "ataxia"."""
    return bool(needle) and f" {needle} " in f" {hay} "


def answer_text(pred: str) -> str:
    """The answer itself: its first non-empty line, with chat tags removed.

    Generations sometimes run on past the answer into "<|endoftext|>" padding,
    a "<tool_call>" tag or a second line of chatter. None of that is part of
    the answer, and left in it can only add tokens that dilute the match.
    """
    text = re.sub(r"<\|[^|>]*\|>|</?[a-z_]+>", "\n", pred)
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def grade(pred: str, golds: Iterable[str], f1_threshold: float = 0.6) -> dict:
    """Best match over the synonym set. `correct` is the label used for AUROC."""
    best = {"em": False, "contains": False, "f1": 0.0, "gold": ""}
    preds = variants(answer_text(pred))
    for gold in golds:
        for gv in variants(gold):
            ngold = normalize(gv)
            for pv in preds:
                npred = normalize(pv)
                # compact form: spacing and punctuation differences only
                em = npred == ngold or (
                    bool(ngold) and npred.replace(" ", "") == ngold.replace(" ", ""))
                contains = _contains(npred, ngold)
                f1 = token_f1(pv, gv)
                # Token overlap cannot settle a number: "50" shares a token with
                # the range "45-50", and "1625 mg/m2" most of its tokens with
                # "825 mg/m2". When both sides carry numbers they must agree.
                np_, ng_ = _numbers(npred), _numbers(ngold)
                if np_ and ng_ and np_ != ng_:
                    f1 = 0.0
                if (em, contains, f1) > (best["em"], best["contains"], best["f1"]):
                    best = {"em": em, "contains": contains, "f1": f1, "gold": gold}
    best["correct"] = best["em"] or best["contains"] or best["f1"] >= f1_threshold
    # Flag anything decided by a soft rule near the threshold for human review.
    best["borderline"] = (not best["em"]) and 0.3 <= best["f1"] < 0.8
    return best


def tier_of(display_name: str) -> int | None:
    """Map an LM-Polygraph display name back to its tier.

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
        entry = labels.get(q)
        if entry is None:
            continue
        # One label per distinct prediction: the 3B and 7B models answer the
        # same question differently, and each answer needs its own verdict.
        cands = entry if isinstance(entry, list) else [entry]
        pred = generations[i].strip()
        lab = next((c for c in cands if c.get("pred") in (None, pred)), None)
        if lab is None:
            stale += 1
            continue
        if graded[i]["correct"] != lab["correct"]:
            graded[i]["correct"] = lab["correct"]
            applied += 1
    return applied, stale
