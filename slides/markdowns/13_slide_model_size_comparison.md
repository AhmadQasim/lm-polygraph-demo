# slide_model_size_comparison

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > The same methods on six models

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > Among single-pass estimators, only mean token entropy transfers.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > Each line traces one estimator's AUROC across six instruction-tuned models from four
  > families; a flat line indicates stability. Mean pairwise Spearman ρ = 0.87, so the
  > ranking is largely preserved. Mean token entropy is the only tier-1 estimator above
  > 0.75 on every model (0.76–0.83); maximum sequence probability and perplexity,
  > computed from the same forward pass, fall to approximately 0.60 on Mistral. The best
  > tier-4 estimator exceeds mean token entropy by a significant margin on two of the
  > six (paired bootstrap). All runs use the same 1,000 questions and the full manual
  > label set.
