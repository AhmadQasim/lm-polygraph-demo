# slide_information_based

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > Information-based: mean token entropy

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > A model unsure of the next token spreads its probability; mean token entropy
  > averages that spread over the answer.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > Two answers from Qwen2.5-3B. Each chip is one generated token, with the entropy of
  > the distribution it was chosen from. The right answer was near-certain at every
  > step; the wrong one was a guess from its first token. Of the 20 estimators, mean
  > token entropy ranked first on both Qwen models.
