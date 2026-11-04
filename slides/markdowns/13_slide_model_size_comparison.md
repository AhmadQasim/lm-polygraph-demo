# slide_model_size_comparison

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > The same methods on other models

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > Mean token entropy is the steadiest method that is also any good.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > Each line is one estimator's AUROC across five models from four families: flat means
  > it holds. The ranking mostly transfers (mean pairwise Spearman ρ 0.86). Mean token
  > entropy stays between 0.77 and 0.83, while max sequence probability, read from the
  > same forward pass, drops to 0.62 on Mistral. A tier-4 method beats mean token
  > entropy significantly on Mistral only. The three non-Qwen models were graded mostly
  > by the matcher alone: the manual labels were written against the Qwen answers.
