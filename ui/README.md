# Uncertainty explorer

A local web UI on top of this repo: pick one of the project's models, ask a question, and
see the answer next to every LM-Polygraph estimator the BioASQ study runs.

```bash
.venv/bin/python ui/server.py          # then open http://127.0.0.1:8765
```

It listens on localhost only, uses no dependencies beyond the venv's (FastAPI and uvicorn
come with lm-polygraph), and holds one model at a time.

## What it shows

- **Models**: the six n=1000 runs (Qwen2.5 3B/7B, Llama 3.2 3B, Llama 3.1 8B, Mistral 7B,
  OLMo 2 7B) plus Qwen2.5 0.5B, the slide-code test model. Each card says whether the
  weights are in the local Hugging Face cache; others are downloaded on load (Llama is gated
  and needs a Hub token). A 7-8B model plus the tier-4 NLI model may not fit in 24 GB.
- **The answer**, token by token: each chip is shaded by the entropy of the distribution
  it was drawn from, and hovering shows the top five candidates.
- **Every estimator up to the chosen tier** (T1 no sampling, T2 a second prompt, T3
  sampling, T4 the DeBERTa NLI model), with the ten samples for T3 and above.
- **BioASQ context**, when the model has a run in `outputs/` and the prompt is in BioASQ
  mode:
  - where the score falls among that run's 1,000 answers;
  - the share of wrong answers among the 100 BioASQ answers with the closest score, a
    rough empirical error rate and not a calibrated probability;
  - the estimator's AUROC on that run, with sign-inverted estimators flagged.
- **A right/wrong verdict** when the question is one of the 1,000 BioASQ questions, using
  `bioasq.grade` and `manual_labels.json`. The "Random BioASQ question" button picks one.

## Prompt modes

- **BioASQ few-shot** (default): the question is wrapped in `bioasq.build_prompt`, and
  generation stops at the first newline with 24 new tokens. These are the run settings, so
  scores are comparable with the cached runs.
- **Free chat**: the text is sent as one chat message, with no stop string and 128 tokens
  by default. Scores are computed the same way, but the BioASQ comparison is off.

## Files

- `engine.py`: model loading and the UEManager call (settings mirror
  `scripts/run_bioasq.py`), plus the reference statistics.
- `server.py`: the FastAPI app, serving `/api/models`, `/api/load`, `/api/status`,
  `/api/run` and `/api/random_question`.
- `static/index.html`: the page, plain HTML and JS.

Measured on the M4 Air with Qwen2.5 3B, one BioASQ question: about 2 s up to tier 1,
5 s up to tier 2 (p(True) empirical samples ten verdicts), 14 s up to tier 3 (ten
samples), and 18-20 s up to tier 4 (the NLI pairs, plus loading DeBERTa the first time).
