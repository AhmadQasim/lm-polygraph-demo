# slide_types_by_tier

The three pieces of text the *slide* carries, around the figure. Edit them
here. The lead and the caption flow — they are wrapped by PowerPoint, so
write them over as many `>` lines as reads well and they are joined back into
one paragraph. Titles do not flow.

## the slide

- **title** — the slide's heading, top left
  > Every estimator, by type and tier

- **lead** — the blue line under the title — one sentence, it wraps on its own
  > Type denotes the signal an estimator reads; tier, what it requires to run.

- **caption** — the grey paragraph under the figure — it wraps on its own
  > The twenty estimators compared in this talk, placed by signal type and tier. Tiers
  > are cumulative: T1 needs no sampling (the answering pass, plus a question-free pass
  > for the two PMIs), T2 adds a second prompt, T3 adds sampling, and T4 adds a DeBERTa
  > NLI model over the samples. Consistency and sampling methods divide between T3 and
  > T4 according to whether they require the NLI model. The later slides use these tiers
  > to group the results.
