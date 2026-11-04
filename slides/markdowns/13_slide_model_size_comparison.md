# slide_model_size_comparison

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > The same methods on other models

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > Of the one-pass methods, only mean token entropy holds everywhere.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > Each line is one estimator's AUROC across six models from four families: flat means
  > it holds. The ranking mostly transfers (mean pairwise Spearman ρ 0.87). Mean token
  > entropy is the only tier-1 method that stays above 0.75 on all six, between 0.76 and
  > 0.83; max sequence probability and perplexity, read from the same forward pass, fall
  > to around 0.60 on Mistral. A tier-4 method beats mean token entropy significantly on
  > two of the six. All six runs were graded with the full manual label set.
