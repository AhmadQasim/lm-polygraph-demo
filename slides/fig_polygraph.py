#!/usr/bin/env python3
"""Figures for the LM-Polygraph chapters: what is in the library, and how to use it.

  polygraph_catalogue  - the 38 methods in the README table, by LM-Polygraph's own
                         categories, and what else the library brings
  wrap_model           - wrapping a Hugging Face model or an API, scoring one answer
  ue_manager           - UEManager: a whole dataset, quality metrics, PRR
  custom_estimator     - writing your own estimator

Run:  .venv/bin/python slides/fig_polygraph.py            # draw the figures
      .venv/bin/python slides/fig_polygraph.py --test     # run every code block

THE CODE ON THE SLIDES IS RUN, NOT TRANSCRIBED. Each block below is one string that
the figure draws and `--test` executes, in slide order, against
Qwen/Qwen2.5-0.5B-Instruct from the local Hugging Face cache (lm-polygraph 0.7.0,
transformers 5.x). The API block is the exception: it needs a key, so `--test` only
checks that its names import and its constructor accepts the arguments shown.

Facts on the catalogue slide come from the package's own README (the method table in
lm_polygraph-0.7.0.dist-info/METADATA) and its module listing, not from memory.
"""

import argparse
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SLIDES = pathlib.Path(__file__).resolve().parent
ROOT = SLIDES.parent
sys.path.insert(0, str(SLIDES))
sys.path.insert(0, str(ROOT / "scripts"))

from fig_methods import (  # noqa: E402
    CW, CH, S, FIGDIR, MDDIR, board, put, lines, header, rule, arrow,
    write_markdown, LIB, FAMILY, fam_fill, fam_ink, fam_ground,
)
from hgf_style import (  # noqa: E402
    BLUE, DARK, LIGHT, GRAY22, GRAY45, GRAY85, PALE, HIGHLIGHT,
    halvar, hermann, mono, shade, slab, use_hgf,
)
from hgf_layers import layered_save as save  # noqa: E402  # writes the click layers too

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# =====================================================================================
# the code blocks: drawn on the slides, executed by --test
# =====================================================================================
CODE_WRAP = '''\
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_polygraph import WhiteboxModel, estimate_uncertainty
from lm_polygraph.estimators import MeanTokenEntropy

path = "Qwen/Qwen2.5-0.5B-Instruct"
llm = AutoModelForCausalLM.from_pretrained(
    path, dtype="bfloat16", attn_implementation="eager")
tok = AutoTokenizer.from_pretrained(path, padding_side="left")
model = WhiteboxModel(llm, tok, model_path=path)

q = "Which organ produces insulin?"
out = estimate_uncertainty(model, MeanTokenEntropy(), q)
out.uncertainty, out.generation_text'''

CODE_API = '''\
from lm_polygraph import BlackboxModel, estimate_uncertainty
from lm_polygraph.estimators import EigValLaplacian

model = BlackboxModel.from_openai(
    openai_api_key="...",       # any OpenAI-compatible
    model_path="gpt-4o-mini",   # endpoint
    supports_logprobs=False)    # True unlocks Perplexity etc.

out = estimate_uncertainty(model, EigValLaplacian(), q)'''

#: Run before CODE_MANAGER by --test, but not drawn: imports are noise on a slide.
CODE_MANAGER_IMPORTS = '''\
from lm_polygraph import UEManager, Dataset
from lm_polygraph.estimators import MeanTokenEntropy, LexicalSimilarity
from lm_polygraph.generation_metrics import RougeMetric
from lm_polygraph.ue_metrics import PredictionRejectionArea
from lm_polygraph.defaults.register_default_stat_calculators import (
    register_default_stat_calculators)
from lm_polygraph.utils.builder_enviroment_stat_calculator import (
    BuilderEnvironmentStatCalculator)
'''

CODE_MANAGER = '''\
calcs = register_default_stat_calculators("Whitebox")
env = BuilderEnvironmentStatCalculator(model)

man = UEManager(
    Dataset(questions, answers, batch_size=8),
    model,
    estimators=[MeanTokenEntropy(), LexicalSimilarity()],
    builder_env_stat_calc=env,
    available_stat_calculators=calcs,
    generation_metrics=[RougeMetric("rougeL")],
    ue_metrics=[PredictionRejectionArea()],
    processors=[],
)
metrics = man()
scores = man.estimations[("sequence", "MeanTokenEntropy")]'''

CODE_CUSTOM = '''\
import numpy as np
from lm_polygraph.estimators import Estimator

class MaxTokenSurprisal(Estimator):
    """The least likely token in the answer."""

    def __init__(self):
        super().__init__(["greedy_log_likelihoods"], "sequence")

    def __str__(self):
        return "MaxTokenSurprisal"

    def __call__(self, stats):
        return np.array([-np.min(ll)
                         for ll in stats["greedy_log_likelihoods"]])

# then: UEManager(..., estimators=[MaxTokenSurprisal(), ...])'''


# =====================================================================================
# drawing code
# =====================================================================================
#: Library names drawn bold, in the colour of the part they belong to - the same
#: colours as the pipeline on the LM-Polygraph slide (fig_methods.LIB).
API_KIND = {
    "model": ["WhiteboxModel", "BlackboxModel", "from_openai"],
    "stats": ["register_default_stat_calculators", "BuilderEnvironmentStatCalculator",
              "calcs", "env"],
    "estimator": ["Estimator", "MeanTokenEntropy", "LexicalSimilarity",
                  "EigValLaplacian", "MaxTokenSurprisal", "estimations"],
    "manager": ["UEManager", "Dataset", "estimate_uncertainty"],
    "metric": ["RougeMetric", "PredictionRejectionArea"],
}
KIND_OF = {name: kind for kind, names in API_KIND.items() for name in names}
API = r"\b(" + "|".join(sorted(KIND_OF, key=len, reverse=True)) + r")\b"


def char_width(fig, font):
    """Width of one monospace character in figure units, measured."""
    renderer = fig.canvas.get_renderer()
    probe = fig.text(0, 0, "M" * 40, fontproperties=font)
    w = probe.get_window_extent(renderer).width / fig.dpi * 10.0 / 40
    probe.remove()
    return w


def code_block(fig, ax, x, top, width, code, pitch=2.0, size=None):
    """Draw `code` line by line in TeX Gyre Cursor: comments grey, API names blue.

    Raises if any line is wider than `width` - code does not wrap.
    """
    reg = mono(size or S["tick"])
    bold = mono(size or S["tick"], bold=True)
    cw = char_width(fig, reg)
    for i, line in enumerate(code.splitlines()):
        if len(line) * cw > width:
            raise ValueError(f"code line {len(line)} chars, {len(line) * cw:.1f} u "
                             f"> {width} u: {line!r}")
        y = top - i * pitch
        code_part, _, comment = line.partition("#")
        if comment and code_part.count('"') % 2 == 0:
            put(ax, x + len(code_part) * cw, y, "#" + comment, reg, GRAY85)
        else:
            code_part = line
        pos = 0
        # on an import line only the imported names are highlighted, not the module path
        start = code_part.index(" import ") + 8 if code_part.lstrip().startswith("from ") \
            and " import " in code_part else 0
        for m in re.finditer(API, code_part):
            if m.start() < start:
                continue
            if m.start() > pos:
                put(ax, x + pos * cw, y, code_part[pos:m.start()], reg, DARK)
            put(ax, x + m.start() * cw, y, m.group(0), bold, LIB[KIND_OF[m.group(0)]][1])
            pos = m.end()
        if pos < len(code_part):
            put(ax, x + pos * cw, y, code_part[pos:], reg, DARK)
    return top - len(code.splitlines()) * pitch


def code_panel(fig, ax, x, y, w, h, title, code, pitch=2.0, accent=DARK, size=None):
    """A code panel whose title and top bar wear the colour of what it is about."""
    slab(ax, x, y, w, h, PALE, z=0)
    slab(ax, x, y + h - 1.2, w, 1.2, accent, z=1, radius=0.5)
    put(ax, x + 1.5, y + h - 4.0, title.upper(), hermann(S["category"]), accent)
    return code_block(fig, ax, x + 1.5, y + h - 7.6, w - 3.0, code, pitch=pitch,
                      size=size)


# =====================================================================================
# the catalogue
# =====================================================================================
# From the method table in the lm-polygraph 0.7.0 README (38 rows): category, how
# many are white-box / black-box, whether training data is needed, and the closest of
# the five families on the methods slides.
CATALOGUE = [  # (category, white-box, black-box, needs training data, family, colour)
    ("Information-based", 13, 0, False, "Information", "Information"),
    ("Meaning diversity", 9, 7, False, "Consistency, Sampling + likelihood", "Consistency"),
    ("Reflexive", 1, 2, False, "Self-report", "Self-report"),
    ("Density-based", 4, 0, True, "Trained probe", "Trained probe"),
    ("Ensembling", 2, 0, True, "several models in an ensemble", None),
]
CLAIM_LEVEL = 11   # of the 38, those listed for claim level as well as sequence


def fig_catalogue():
    fig, ax = board()
    total = sum(c[1] + c[2] for c in CATALOGUE)
    assert total == 38, total

    # --- left: the catalogue as bars ----------------------------------------------
    header(ax, 0, 42.2, f"the {total} methods in the README, by its own categories")
    label_w, bar_x, unit = 21.0, 21.5, 1.45
    y = 36.0
    for name, white, black, trained, family, hue in CATALOGUE:
        fill = fam_fill(hue) if hue else GRAY45
        ink = fam_ink(hue) if hue else GRAY85
        put(ax, label_w, y, name, halvar(S["tick"], bold=True), ink, ha="right")
        slab(ax, bar_x, y - 0.9, white * unit, 3.0, fill, z=1, radius=0.4)
        if black:
            slab(ax, bar_x + white * unit + 0.3, y - 0.9, black * unit, 3.0,
                 shade(fill, 40), z=1, radius=0.4)
        end = bar_x + (white + black) * unit + 1.2
        put(ax, end, y, f"{white + black}", halvar(S["value"], bold=True), ink)
        put(ax, end + 3.2, y, ("needs training data  ·  " if trained else "")
            + "≈ " + family, halvar(S["tick"]), GRAY85)
        y -= 6.0
    # key
    slab(ax, bar_x, 4.2, 2.6, 2.4, GRAY85, z=1, radius=0.4)
    put(ax, bar_x + 3.4, 4.6, "solid: white-box", halvar(S["tick"]), DARK)
    slab(ax, bar_x + 20, 4.2, 2.6, 2.4, shade(GRAY85, 40), z=1, radius=0.4)
    put(ax, bar_x + 23.4, 4.6, "pale: black-box, text only", halvar(S["tick"]), DARK)
    put(ax, 0, 1.0, f"{CLAIM_LEVEL} of the {total} also score claim by claim, "
        "for long answers", halvar(S["gloss"]), GRAY85)

    # --- right: what comes with them -----------------------------------------------
    x0 = 86.0
    slab(ax, x0 - 1.5, 0, CW - x0 + 1.5, CH, PALE, z=0)
    header(ax, x0, 42.2, "and around them", DARK)
    items = [
        ("Models", "Hugging Face, any OpenAI-compatible API, vLLM, visual LLMs", "model"),
        ("Quality metrics", "ROUGE, accuracy, AlignScore, COMET, BERTScore, ...",
         "metric"),
        ("UE metrics", "PRR, ROC-AUC, PR-AUC, ECE, rank correlations", "metric"),
        ("Benchmark", "polygraph_eval with YAML configs; datasets on "
                      "huggingface.co/LM-Polygraph", "manager"),
    ]
    yy = 37.0
    for head, body, kind in items:
        put(ax, x0, yy, head.upper(), hermann(S["category"]), LIB[kind][1])
        yy = lines(fig, ax, x0, yy - 2.7, body, halvar(S["tick"]), CW - x0 - 1.0, DARK,
                   pitch=2.1) - 2.4
    out = save(fig, FIGDIR / "polygraph_catalogue", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# wrapping a model
# =====================================================================================
def fig_wrap_model(result=None):
    fig, ax = board()
    gap = 2.4
    lw = 60.0
    code_panel(fig, ax, 0, 6.0, lw, CH - 6.0, "your Hugging Face model", CODE_WRAP,
               pitch=2.15, accent=LIB["model"][1])
    rx = lw + gap
    code_panel(fig, ax, rx, 6.0, CW - rx, CH - 6.0, "or an API", CODE_API, pitch=2.15,
               accent=LIB["model"][1])
    if result:
        score, text = result
        put(ax, 0, 2.4, f"→  {score:.2f}   “{text}”", mono(S["tick"]), DARK)
    put(ax, rx, 2.4, "not run here: needs a key", halvar(S["tick"], italic=True), GRAY85)
    out = save(fig, FIGDIR / "wrap_model", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# UEManager
# =====================================================================================
#: (line index in CODE_MANAGER, part, note). Each note sits level with its line.
MANAGER_NOTES = [
    (0, "stats", "the default statistics, each computed once"),
    (1, "stats", "where those statistics are built"),
    (4, "manager", "questions and reference answers, in batches"),
    (5, "model", "any wrapped model, as on the last slide"),
    (6, "estimator", "any mix of the 38 estimators"),
    (9, "metric", "how good each answer is"),
    (10, "metric", "how well the scores rank that quality"),
    (13, "manager", "runs it all: one value per estimator"),
    (14, "estimator", "one score per answer"),
]


def fig_ue_manager():
    fig, ax = board()
    lw, pitch, size = 72.0, 2.45, S["value"]
    code_panel(fig, ax, 0, 0, lw, CH, "a dataset, many estimators, one call",
               CODE_MANAGER, pitch=pitch, accent=LIB["manager"][1], size=size)
    top = CH - 7.6
    x0 = lw + 2.0
    for i, kind, note in MANAGER_NOTES:
        y = top - i * pitch
        fill, ink = LIB[kind]
        slab(ax, x0, y - 0.55, 1.0, 2.2, fill, z=1, radius=0.3)
        put(ax, x0 + 2.0, y, note, halvar(S["tick"]), ink)
    put(ax, x0, 42.0, "imports left out; the full script runs in --test",
        halvar(S["gloss"]), GRAY85)
    out = save(fig, FIGDIR / "ue_manager", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# your own estimator
# =====================================================================================
CUSTOM_NOTES = [
    ("1  Declare what it needs",
     "Name statistics such as greedy_log_likelihoods, sample_texts or "
     "semantic_matrix_entail. They are computed for you, once, and shared."),
    ("2  Name it",
     "__str__ is the key it is reported under."),
    ("3  Return one score per answer",
     "Higher means less sure. The level, “sequence” here, can also be "
     "“claim” or “token”."),
]


def fig_custom_estimator():
    fig, ax = board()
    lw = 70.0
    code_panel(fig, ax, 0, 0, lw, CH, "a complete new estimator", CODE_CUSTOM,
               pitch=2.15, accent=LIB["estimator"][1])
    x0 = lw + 2.6
    yy = 41.6
    for head, body in CUSTOM_NOTES:
        put(ax, x0, yy, head.upper(), hermann(S["category"]), LIB["estimator"][1])
        yy = lines(fig, ax, x0, yy - 2.9, body, halvar(S["tick"]), CW - x0, DARK,
                   pitch=2.1) - 3.0
    out = save(fig, FIGDIR / "custom_estimator", formats=("png", "svg"))
    plt.close(fig)
    return out[0]


# =====================================================================================
# slide copy
# =====================================================================================
SLIDE_COPY = {
    "polygraph_catalogue": (
        "08b",
        "What is in the library",
        "Thirty-eight methods, filed under five categories, with the plumbing to "
        "compare them.",
        "Categories and counts are LM-Polygraph's own, from its README; the grey note "
        "on each bar is the closest family on the methods slides. Density-based and "
        "ensembling methods need training data or several models, which is why the "
        "talk does not run them.",
    ),
    "wrap_model": (
        "08c",
        "Wrapping a model",
        "Wrap the model once, then score any answer with any estimator.",
        "WhiteboxModel takes a Hugging Face model and tokenizer; BlackboxModel takes "
        "an API, which limits you to black-box estimators unless it returns log-"
        "probabilities. Eager attention is required on transformers 5.x. The left "
        "block was run as shown; its output is underneath.",
    ),
    "ue_manager": (
        "08d",
        "UEManager: a whole dataset at once",
        "One object runs every estimator over a dataset and scores how well each "
        "ranks the answers.",
        "Each answer gets a quality from the generation metric and a score from every "
        "estimator; the UE metric then asks how well each estimator's ranking follows "
        "the quality. The BioASQ study in this talk is this call with 20 estimators "
        "and its own grader.",
    ),
    "custom_estimator": (
        "08e",
        "Adding your own estimator",
        "A new method is a class with three parts, and it plugs into everything "
        "above.",
        "This one scores an answer by its least likely token. It ran inside UEManager "
        "alongside the built-in estimators, sharing their statistics.",
    ),
}


# =====================================================================================
# --test: run the code blocks
# =====================================================================================
def test_snippets():
    """Execute the slide code, in slide order, and return the one-answer result."""
    import os
    import warnings

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    warnings.filterwarnings("ignore")
    ns = {}
    exec(CODE_WRAP, ns)
    print("wrap_model:", ns["out"].uncertainty, repr(ns["out"].generation_text[:60]))

    # the API block: names and the constructor's arguments, without a network call
    import inspect
    from lm_polygraph import BlackboxModel
    sig = inspect.signature(BlackboxModel.from_openai)
    assert {"openai_api_key", "model_path", "supports_logprobs"} <= set(sig.parameters)
    exec(CODE_API.split("model =")[0], dict(ns))
    print("api block: imports and from_openai signature ok")

    exec(CODE_CUSTOM.split("# then:")[0], ns)
    ns["questions"] = ["Which organ produces insulin?", "What is the capital of France?",
                       "Which enzyme is inhibited by statins?",
                       "What gene is mutated in cystic fibrosis?"]
    ns["answers"] = ["The pancreas", "Paris", "HMG-CoA reductase", "CFTR"]
    code = CODE_MANAGER_IMPORTS + CODE_MANAGER.replace(
        "estimators=[MeanTokenEntropy(), LexicalSimilarity()],",
        "estimators=[MeanTokenEntropy(), LexicalSimilarity(), MaxTokenSurprisal()],")
    assert code.count("MaxTokenSurprisal()") == 1, "the estimators line moved"
    code = code.replace("    processors=[],\n", "    processors=[],\n    max_new_tokens=24,\n")
    exec(code, ns)
    assert len(ns["scores"]) == len(ns["questions"])
    custom = ns["man"].estimations[("sequence", "MaxTokenSurprisal")]
    assert len(custom) == len(ns["questions"])
    print("ue_manager: metrics", {k[1]: round(v, 3) for k, v in ns["metrics"].items()
                                  if k[3] == "prr"})
    print("custom_estimator: MaxTokenSurprisal", [round(float(v), 2) for v in custom])
    return float(ns["out"].uncertainty), ns["out"].generation_text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--test", action="store_true",
                    help="run the slides' code blocks against the cached small model")
    ap.add_argument("--rewrite-copy", action="store_true",
                    help="regenerate the slide markdowns, overwriting edits made there")
    args = ap.parse_args()

    result_file = FIGDIR / "wrap_model_result.txt"
    if args.test:
        score, text = test_snippets()
        FIGDIR.mkdir(exist_ok=True)
        result_file.write_text(f"{score}\n{text}\n")
    result = None
    if result_file.exists():
        score, text = result_file.read_text().split("\n", 1)
        text = text.strip()
        result = (float(score), text if len(text) <= 46 else text[:45] + "…")

    use_hgf()
    FIGDIR.mkdir(exist_ok=True)
    outs = [fig_catalogue(), fig_wrap_model(result), fig_ue_manager(),
            fig_custom_estimator()]
    for out in outs:
        md, wrote = write_markdown(out.stem, *SLIDE_COPY[out.stem],
                                   force=args.rewrite_copy)
        print(f"wrote {out.relative_to(ROOT)}  +  {md.relative_to(ROOT)}"
              + ("" if wrote else "  (kept: edited copy)"))


if __name__ == "__main__":
    main()
