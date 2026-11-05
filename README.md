# Knowing when an LLM is wrong

Material for a journal club talk to the Helmholtz AI consultants group (Munich) on
**uncertainty estimation for LLMs and the [LM-Polygraph](https://github.com/IINemo/lm-polygraph)
library**. The talk surveys the method families, shows how to use the library, and
includes a small worked example on BioASQ.

The repo holds three things:

- **The experiment.** 20 LM-Polygraph estimators are run on 1,000 BioASQ factoid
  questions. Each one is scored by AUROC at telling wrong answers from right ones, and
  the estimators are grouped into four tiers by what they need to run.
- **The slide deck.** It is built from scripts, so every number and code block on a
  slide comes from the data or was executed.
- **A local web UI.** You pick a model, ask a question, and see the answer next to every
  estimator.

[CLAUDE.md](CLAUDE.md) is the detailed working record: design decisions, gotchas, the
grader's history and the full results.

## Setup

lm-polygraph 0.7.0 supports Python 3.10–3.12 only.

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt        # requirements-lock.txt: the exact set verified on MPS
```

Pinned constraints, explained in CLAUDE.md §3:
- **bfloat16**, not float16.
- **sdpa attention with `output_attentions=False`**. transformers 5.x breaks
  lm-polygraph's attention path, so the attention-based estimators are excluded.
- `register_default_stat_calculators(model_type="Whitebox")`.

## The experiment

```bash
.venv/bin/python scripts/run_bioasq.py --n 1000 --max-tier 4 --deberta-batch-size 64   # a CUDA box for 7B+
.venv/bin/python scripts/analyze_bioasq.py outputs/<run>.json --manual manual_labels.json
```

**Data.** `jmhb/BioASQ`, factoid split. The study keeps questions whose reference answer
is at most 5 words and samples 1,000 of them with seed 0. Questions are asked
closed-book, with a three-example few-shot prompt and greedy decoding.

**Tiers.** Each tier adds one requirement to the one before:

| tier | needs | estimators |
|---|---|---|
| T1 | no sampling | MSP, perplexity, mean token entropy, Rényi-Neg, Fisher-Rao, self-certainty, boosted prob., two PMIs |
| T2 | + a second prompt | p(True), p(True) empirical |
| T3 | + sampling (10 answers) | MC sequence entropy (plain and normalised), lexical similarity |
| T4 | + an NLI model (DeBERTa-large-MNLI) | semantic entropy, DegMat, eccentricity, EigVal, NumSemSets, KLE |

**Grading** is the fragile part. `bioasq.grade` normalises answers and matches them by
tokens. `manual_labels.json` then adds 534 human-reviewed verdicts covering every answer
that isn't an exact match, across all six models. See CLAUDE.md §5.

**Runs.** Each model has a cached n=1000 run in `outputs/`, which git ignores: Qwen2.5 3B
and 7B, Llama 3.2 3B, Llama 3.1 8B, Mistral 7B v0.3 and OLMo 2 7B.

**Headline results.** Qwen2.5-3B is the model in the main deck:
- **Mean token entropy**, a tier-1 method, ranks first on both Qwen models: AUROC 0.810
  on 3B and 0.786 on 7B. It beats semantic entropy (0.768 and 0.765) by a significant
  margin.
- Across all six models, one forward pass is statistically indistinguishable from the
  best sampling + NLI method on four. The ranking of methods mostly holds (mean pairwise
  Spearman ρ 0.87).
- p(True) is unreliable: it ranks 16th of 20 on the 3B model and is at chance on the 7B.
- Three estimators are sign-inverted (Fisher-Rao, Rényi-Neg, conditional PMI): they
  return a confidence, not an uncertainty.

## The deck

```bash
.venv/bin/python slides/build_deck.py --figures --pdf   # regenerate figures, build, export PDF
.venv/bin/python slides/build_deck.py                   # rebuild from the current figures (~3 s)
```

The deck builds into `slides/build/journal_club_deck.pptx`, 29 slides in Helmholtz AI
house style.
- Each slide's text is in `slides/markdowns/`; edit there and rebuild.
- `slides/fig_polygraph.py --test` executes the code shown on the library slides.
- `scripts/method_examples.py` regenerates the worked examples on slides 8–10.

The template (`slides/journal_club_template.pptx`, 187 MB) is not in git. See
[slides/README.md](slides/README.md).

## The web UI

```bash
.venv/bin/python ui/server.py          # http://127.0.0.1:8765
```

1. Load one of the project's models.
2. Ask a question, or click "Random BioASQ question".
3. Read the results:
   - the answer, with per-token entropy;
   - the ten samples, at tier 3 and above;
   - every estimator up to the chosen tier;
   - in BioASQ mode, each score's place in that model's n=1000 run.

   BioASQ questions are graded right or wrong against the reference. See
   [ui/README.md](ui/README.md).

## Layout

```text
scripts/            the experiment: data and grader (bioasq.py), runs, analysis, worked examples
slides/             the deck: figure scripts, slide text, build
ui/                 the local web UI
manual_labels.json  human adjudications of the grader
outputs/            cached runs (gitignored)
```
