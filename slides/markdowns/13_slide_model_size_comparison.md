# slide_model_size_comparison

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > The same methods on other models

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > The ranking mostly holds; one cheap method collapses.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > Each line is one estimator's AUROC across models — flat means robust, a dive means
  > model-dependent. Most run roughly parallel, so the ranking transfers across a change
  > of size and of family. The exception carries the lesson: max sequence probability
  > collapses on Mistral while mean token entropy, which costs exactly the same single
  > forward pass, barely moves. The panel gives the figures.
