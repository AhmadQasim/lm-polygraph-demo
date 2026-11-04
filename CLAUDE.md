# CLAUDE.md — journal club: uncertainty estimation for LLMs, and LM-Polygraph

Working notes for this repo. Read this first in a fresh session. Numbers here were
current on 2026-11-04; the code and `outputs/` are the source of truth if they differ.

## 1. What this is

A journal club talk for the Helmholtz AI consultants group (Munich), who build LLM
agentic workflows for labs (genomics, imaging, EHR). Working title (in
`slides/chapters.json`): *"Knowing when an LLM is wrong: uncertainty estimation
methods, and the LM-Polygraph library"*. Speaker: Ahmad Bin Qasim. The
talk date is not recorded anywhere; the title slide's date line is deliberately blank.

**Focus.** The talk was originally about semantic entropy (Farquhar et al., Nature
2024). On 2026-11-02 it became an LM-Polygraph introduction plus a survey of UQ
methods. **On 2026-11-04 it was re-centred: the talk introduces uncertainty estimation
methods for LLMs (the families, as a whole and one by one) and the LM-Polygraph
library (what is in it, wrapping a model, UEManager, writing your own estimator).** The
BioASQ study is a short worked example, not the point. Two results slides stay in the
main deck and two moved to Backup. Semantic entropy is one method among many.
Farquhar et al. and Kuhn et al. (ICLR 2023) stay as background.

**No cost angle** (decided 2026-11-04). Measuring what each method costs would need
many more runs, and the tiers already say what each method needs. So the talk
presents tiers as *requirements* (one pass, + a prompt, + sampling, + an NLI model),
never as prices: no wall-clock comparisons, and no "cheap", "expensive" or "paying more"
in slide text.

**The worked example.** 20 LM-Polygraph estimators on 1,000 BioASQ factoid questions,
answered by Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct. Each estimator is scored by
AUROC at separating wrong answers from right ones, and grouped into four tiers by what
it needs to run.

## 2. Repo layout

```text
scripts/            the experiment: data, grading, running, scoring
  bioasq.py           dataset loading, prompt, TIERS, the grader, manual-label lookup
  run_bioasq.py       runs the estimators through LM-Polygraph, caches to outputs/
  analyze_bioasq.py   AUROC + bootstrap CIs, paired deltas vs semantic entropy, sensitivity
  time_model.py       per-estimator timing on MPS (feasibility probe)
  smoke_test_mps.py   MPS sanity check
  sync_gpu.sh         push source to / pull outputs from the GPU box
slides/             everything that makes the deck (see slides/README.md)
  build_deck.py       template + figures + markdowns -> slides/build/journal_club_deck.pptx
  fig_methods.py      figures for the problem, method, example-setup and limitations slides
  fig_polygraph.py    the library slides; its code blocks are run by `--test`
  plot_bioasq.py      figures for the results slides, from outputs/
  markdowns/          NN_slide_<name>.md: each slide's title, lead and caption
  chapters.json       chapter divider titles + nav pills; set_chapters.py writes them
  strip_template.py   empties the vitiligo deck into a fresh template
  hgf_style.py, deck_common.py, nav_bar.py, hgf_copy.py   vendored from ~/projects/slides
  fonts/              Helmholtz brand OTFs (Hermann, HalvarMittel, TeX Gyre Cursor)
  deck_outline.json   text of every built slide, for review in git
manual_labels.json  human adjudications of the grader (section 5)
outputs/            cached runs (gitignored)
```

Not in git: `outputs/`, `slides/figures/`, `slides/build/`, `.venv/` and
`slides/journal_club_template.pptx` (187 MB).

## 3. Environment and hard-won constraints

- **Python 3.10** venv at `.venv/` (3.10.20). lm-polygraph 0.7.0 supports 3.10–3.12
  only. `requirements.txt` pins lm-polygraph 0.7.0 and torch; `requirements-lock.txt` has
  the resolved set verified on MPS.
- **transformers 5.x breaks attention.** pip resolves transformers 5.x, which
  lm-polygraph predates. With `output_attentions=True`, the default sdpa backend returns
  no attention tensors and `greedy_probs.py` raises `IndexError`. `run_bioasq.py` keeps
  sdpa and sets `output_attentions=False` (and `output_hidden_states=False`). Hence the
  attention-based estimators (RAUQ, CSL, AttentionScore) are **excluded**; re-enabling
  them needs `--attn eager`, the slow path.
- **bfloat16, not float16.** An fp16 7B run spliced token id 0 ("!") into answers.
- `register_default_stat_calculators(model_type="Whitebox")`; `"CausalLM"` raises.
- **Memory is the binding constraint locally** (24 GB MacBook Air M4). The DeBERTa NLI
  model (~1.6 GB) loads alongside the LLM: 7B bf16 + NLI thrashes swap (per-question time
  went bimodal, 54 s median, 1,523 s worst); 3B bf16 runs clean. So **the n=1000 runs were
  done on a CUDA GPU box** via `scripts/sync_gpu.sh` (push source, run, pull `outputs/`).
- Run settings recorded in the 7B cache: cuda, bf16, sdpa, batch 16, `max_new_tokens`
  24, DeBERTa batch 64, torch 2.12.1+cu130, 2,030 s total. The 3B n=1000 run predates
  config recording, so its batch size is unknown and its wall clock is not comparable.

## 4. The experiment

**Data.** `jmhb/BioASQ`, `factoid` split, via `bioasq.load_factoid`: golds are a
stringified nested list of synonym sets, flattened; questions whose shortest gold is over
5 words are dropped; shuffled with `random.Random(seed=0)`; first n kept.

**Prompt.** An instruction ("Answer each biomedical question with a short noun
phrase...") plus three hand-written few-shot pairs (statins, cystic fibrosis, insulin:
deliberately not from the eval set), then `Question: ...\nAnswer:`. Greedy generation,
stop at newline, up to 24 new tokens.

**Estimators and tiers** (`bioasq.TIERS`). Tiers group the estimators by what they need
to run. They are cumulative: each adds one requirement to the one before, and running
tier 4 also runs 1–3.

| tier | what it needs | estimators |
|---|---|---|
| T1 | one greedy forward pass, logits only | MaximumSequenceProbability, Perplexity, MeanTokenEntropy, FisherRao, RenyiNeg, SelfCertainty, BoostedProbSequence, MeanPointwiseMutualInformation, MeanConditionalPointwiseMutualInformation (the two PMIs need a second context-free pass, ~2× T1) |
| T2 | asking the model about its answer | PTrue, PTrueEmpirical |
| T3 | sampling (~10×), no NLI | MonteCarloSequenceEntropy, MonteCarloNormalizedSequenceEntropy, LexicalSimilarity |
| T4 | sampling + DeBERTa NLI over the samples | SemanticEntropy, DegMat, Eccentricity, EigValLaplacian, NumSemSets, KernelLanguageEntropy |

LM-Polygraph defaults in force: 10 samples per question (`SamplingGenerationCalculator`),
NLI model `microsoft/deberta-large-mnli`. **Verbalized1S is deliberately absent**: it
regex-parses a confidence the model never emits and crashes on Qwen2.5. That is a
finding, not a bug to fix.

Definitions checked against the lm-polygraph 0.7.0 source: MSP = −Σ log p(y_l|y<l,x);
Perplexity = the per-token mean of that; MeanTokenEntropy = mean per-step entropy; PTrue =
−log p("True") after the prompt `Question: {q}\n Possible answer:{a}\n Is the possible
answer True or False? The possible answer is: `; PTrueEmpirical = 1 − share of 10 sampled
one-token verdicts containing "True". Higher score = less certain, except the
sign-inverted ones below.

**Runs** (`outputs/`, gitignored). The two the talk uses:
`bioasq_Qwen2.5-3B-Instruct_n1000_seed0_t4.json` and
`bioasq_Qwen2.5-7B-Instruct_n1000_seed0_t4.json`. Each holds questions, golds,
generations and every estimator's scores; there are also n=5 and n=100 smoke runs.

```bash
.venv/bin/python scripts/run_bioasq.py --n 1000 --max-tier 4 --deberta-batch-size 64   # on the GPU box
.venv/bin/python scripts/analyze_bioasq.py outputs/<file>.json --manual manual_labels.json
```

## 5. The grader and manual labels

The label every AUROC is computed against is "was this answer correct", so grading
errors bias every result. It had a lot of them; this is the most fragile part of the
repo.

**`bioasq.grade(pred, golds)`**. It grades only the answer itself (`answer_text`:
the first non-empty line, with `<|endoftext|>`, `<tool_call>` and similar tags removed).
Then, against every gold:
- **Normalisation** (`normalize`): NFKC; Greek letters spelled out (α → alpha);
  non-Latin scripts dropped (the models sometimes drift into Chinese/Arabic mid-answer);
  possessive 's dropped; lowercase; **punctuation becomes a space, not nothing** (so
  "X-linked" and "X- linked" agree); articles removed; a small synonym map (`il` →
  interleukin, `ifn` → interferon, British → US spellings); crude depluralisation.
- **Variants** (`variants`): a parenthetical that contains a word is an alternative name,
  so "Platelet derived growth factor-alpha (PDGFRA)" also matches "PDGFRA". Notation such
  as "t(11;22)", "Na(+)" or "mg/m(2)" is **not** split.
- **Correct** if any of: exact match, or exact after removing spaces ("Na+/Ca2+" vs
  "Na(+)/Ca(2+)"); the gold appears **as whole tokens** inside the prediction; token F1
  ≥ 0.6. **When both sides contain numbers they must agree before F1 can count**: this
  stops "50" matching the range "45-50" and "1625 mg/m²" matching "825 mg/m²".
- `borderline` flags soft calls (not exact, 0.3 ≤ F1 < 0.8) for review;
  `analyze_bioasq.py` reports how far AUROC moves if all of them flip.

**History.** The old grader deleted punctuation instead of spacing it, matched
substrings rather than whole tokens, and never split parentheticals. That made "IL-6"
vs "interleukin-6" and "X-linked recessive" vs "X- linked recessive" wrong, and "Phase III"
vs "phase II" and "250" vs "5" right. Most of the 7B model's *most confident* "errors"
were grader misses. Fixed and committed 2026-11-04.

**`manual_labels.json`**. Human adjudications of what the matcher cannot settle (no rule
makes "PD-1" = "programmed death receptor-1" or "Onpattro" = "patisiran" safe).
- Keyed by question text. A value is one label, or a **list** of labels when the 3B and
  7B models answered differently. Each label holds `correct`, `note`, `pred` (the exact
  answer it was judged against) and `models`.
- `apply_manual_labels` applies a label only when the run produced exactly that `pred`.
  The "N labelled questions have no label for this exact answer" note is normal: it
  counts the other model's answers.
- Built 2026-11-04. Every non-exact-match answer in both n=1000 runs (1,508 unique
  question/answer pairs) was reviewed by five Claude subagents against a shared rubric.
  They proposed 204 changes. Claude reviewed all 48 marked debatable and overruled 6,
  where a true but less specific answer misses the detail the question asks for. One
  inconsistency was fixed. Result: 199 labels, which replaced or joined the 8 earlier ones. Only verdicts that
  **differ** from `grade()` are stored.
- **The rubric.** Correct if it names the same entity as any gold: synonym, abbreviation,
  brand/generic, renamed organism, a gene for its protein, or a typo in the gold. A *less*
  specific answer is correct only if the gold's extra detail is not what the question
  asks for: "Ebola" for ZMapp is fine; "Mitochondria" for "outer mitochondrial membrane",
  "Tick" for *Ixodes ricinus* and "Fusarium oxysporum" for f. sp. *cubense* are wrong. A
  number must match or fall inside a gold range. A hedged answer with a wrong alternative
  is wrong. Factor X ≠ Factor Xa.
- **Judgement calls.** 44 labels have a note starting `JUDGEMENT CALL, re-checked
  2026-11-04`. These are not simple synonym matches. Either the answer is less specific
  than the gold, or it is phrased indirectly, or it is right where the gold is wrong or
  outdated: palbociclib → CDK4/6 (gold says EGFR), Flaviviridae has four genera (gold
  says three), H2AX is phosphorylated, lorlatinib also inhibits ROS1, bimekizumab is also
  approved for psoriatic arthritis. All 44 were re-checked against the rubric. Three were
  flipped to wrong on that pass: PRDX2 "Thermostable peroxidase" (adds a false property),
  MLL1 "K4" (no histone named), and Askin tumour "Ewing sarcoma translocation" (names the
  family, not the translocation). The other 41 stand. They were not checked by a
  biomedical expert; one could still disagree with individual calls.

Effect on accuracy: 3B 23.3% → 31.5%, 7B 31.0% → 44.7%.

## 6. Results (post-fix labels, n=1000, seed 0)

AUROC for detecting a wrong answer, 95% bootstrap CIs (`analyze_bioasq.py`).

| | Qwen2.5-3B | Qwen2.5-7B |
|---|---|---|
| accuracy | 31.5% | 44.7% |
| best | **Mean token entropy 0.808** | **Mean token entropy 0.786** |
| next | DegMat 0.799, Lexical sim. 0.799, Perplexity 0.796 | Boosted prob. 0.783, EigVal 0.776, KLE 0.774 |
| Semantic entropy | 0.769 | 0.765 |
| p(True) | 0.669 | 0.504 (chance) |
| p(True) empirical | 0.738 | 0.708 |

Headline findings, all of which survived the grader fix:
- **A tier-1 method wins on both models.** Mean token entropy (one forward pass, no
  sampling, no NLI) beats semantic entropy significantly on both: 3B +0.039 [+0.016,
  +0.064], 7B +0.020 [+0.004, +0.037] (paired bootstrap).
- **Sampling and an NLI model add consistency, not a higher ceiling**: the T3/T4 methods
  cluster just under the best T1 methods, while T1 also holds the worst ones.
- **The ranking transfers across scale.** Spearman ρ = 0.902 over all 20 methods (p ≈
  5e-8, 3B vs 7B).
- **Other families (backup slide 13).** n=1000 runs also exist for Mistral-7B-Instruct-v0.3
  (38.5% correct), OLMo-2-1124-7B-Instruct (27.6%) and Llama-3.1-8B-Instruct (40.5%). All
  five are on slide 13. The three non-Qwen runs are graded by `grade()` only (the manual
  labels were written against the Qwen answers: 129 apply to Qwen-7B but only 19–35 to the
  others), so their accuracies are under-corrected and their AUROCs carry more label
  noise. Mean pairwise ρ over the five is 0.863.
- **One forward pass is not beaten, on four models of five.** Paired bootstrap, the best
  tier-4 method minus MeanTokenEntropy: Qwen-3B −0.009 [−0.032, +0.014], Qwen-7B −0.009
  [−0.031, +0.013], **Mistral +0.031 [+0.006, +0.055]**, OLMo-2 +0.003 [−0.023, +0.028],
  Llama +0.016 [−0.005, +0.037]. Only Mistral is significant. Tier-4 methods take the top
  *ranks* on Mistral, OLMo and Llama, but the gap is inside the noise except on Mistral,
  which is also the model whose token probabilities are broken. So the claim is "a single
  forward pass is statistically indistinguishable from the full sampling+NLI pipeline on
  four of five models", not "tier 1 wins".
- **Steadiness across models.** Over the five runs, swing (best minus worst model) ranks:
  Self-certainty 0.053 but it never clears 0.76; MeanTokenEntropy 0.063 (0.766–0.829);
  KLE 0.065; EigVal/DegMat/NumSemSets 0.069; SemanticEntropy 0.115; and the collapsing
  tier-1 trio Boosted prob. 0.173, Perplexity 0.190, MSP 0.197, all of them on Mistral.
  `fig_model_compare` picks its "steadiest" row with a 0.75 floor for exactly this reason
  — at 0.70 self-certainty wins the row on being flat and mediocre.
- **Llama is the only model that abstains**: 22 of 1000 answers are "I couldn't find any
  information on …". Qwen, Mistral and OLMo abstain zero times. It is also the only model
  where p(True) sits significantly *below* chance (0.453 [0.418, 0.489]), i.e. it ranks
  correct answers as less true. p(True) swings 0.215 across the five, the widest of any
  method.
- **Detection gets somewhat harder on the bigger model**: mean AUROC change −0.021 over
  the 17 methods above chance, with 15 of 20 below the diagonal. Before the grader fix
  this looked twice as large; part of it was label noise.
- **Self-report is unreliable**: p(True) collapses to chance at 7B.
- **Three estimators are sign-inverted** on both models: FisherRao, RenyiNeg,
  MeanConditionalPointwiseMutualInformation (AUROC ≈ 0.28–0.32). They return a
  confidence, not an uncertainty; read them as 1 − AUROC. This is a library convention a
  user has to know, so it is flagged, not silently flipped.
- **Triage, not verdict**: even AUROC ≈ 0.8 leaves heavily overlapping score
  distributions. On 7B, flagging the most uncertain half by mean token entropy lifts the
  accuracy of what is kept from 45% to 67% (slide 02).

## 7. Method taxonomy (the methods chapter)

The five signal types on slide 03 take their **categories and method names** from
Vazhentsev, Smirnov, Li, Panov, Baldwin, Shelmanov, *Uncertainty Quantification for Large
Language Diffusion Models*, arXiv:2605.14570 (May 2026; Sec. 2 and App. D.2). It is by
the LM-Polygraph group. **Use it only for categories and names, never its results**:
the paper is about diffusion LMs, where logit-based methods are near random and KLE
wins, the opposite of this AR setting. The user asked for the diffusion angle to be
dropped.

| type | reads | access | names on the slide | our tier |
|---|---|---|---|---|
| Information | token probabilities from the answering pass | white-box | MSP, Perplexity, MTE, CCP, Attention score, RAUQ | T1 |
| Self-report *(added, not in the paper)* | the model's verdict on its own answer | white- or black-box | p(True), Verbalized confidence | T2 |
| Consistency | agreement between re-sampled answers, by words or meaning | black-box | Lexical similarity, LUQ, DegMat, Eccentricity, EigVal, KLE | T3/T4 |
| Sampling + likelihood | samples weighted by their probabilities | white-box | MC sequence entropy, Semantic entropy, SAR, Semantic density, EigenScore, CoCoA | T3/T4 |
| Trained probe | hidden states/attention through a trained classifier | white-box + labels | SAPLMA, SATRMD, Lookback Lens, TAD | not run |

Probes were left out on purpose: they need graded in-domain answers, and this comparison
is between label-free methods. `fig_methods.TYPE_OF` maps each of our 20 estimators to a
type and asserts the mapping is complete. LM-Polygraph's own README files
its 38 methods under five different categories: information-based, meaning diversity,
reflexive, density-based and ensembling. That is now verified from the 0.7.0 package
metadata. The polygraph_catalogue slide shows them and maps each to the nearest family
here. Information-based maps to Information. Meaning diversity covers Consistency plus
Sampling + likelihood. Reflexive maps to Self-report. Density-based maps to Trained
probe. Ensembling (several models) has no counterpart.

## 8. The deck

```bash
.venv/bin/python slides/build_deck.py --figures --pdf   # regenerate figures, build, export PDF
.venv/bin/python slides/build_deck.py                   # rebuild from current figures (~3 s)
.venv/bin/python slides/build_deck.py --list            # running order
.venv/bin/python slides/build_deck.py --only uq_types   # partial build
```

**Template.** `slides/journal_club_template.pptx` is a copy of
`~/projects/slides/vitiligo_voucher.pptx` (the Helmholtz AI PowerPoint template, 16:9, 25
layouts), stripped by `strip_template.py` and given chapter titles by `set_chapters.py`.
It is **read-only source**: every build copies it and the copy is written to
`slides/build/`. It holds the title slide plus eight chapter dividers (Earth_Custom
layout):

| template slide | chapter | nav pill | content slides |
|---|---|---|---|
| 2 | Problem: Confidently wrong | Problem | confidently_wrong, selective_generation |
| 3 | Methods: Where the uncertainty is | Methods | uq_map, uq_types, information_based, self_report, consistency, trained_probes |
| 4 | Tooling: LM-Polygraph | Polygraph | lm_polygraph, polygraph_catalogue |
| 5 | Tooling: Your model, your data | Your model | wrap_model, ue_manager, custom_estimator |
| 6 | Example: A small BioASQ study | Example | dataset, types_by_tier |
| 7 | Example: What it showed | Findings | auroc_by_method, auroc_by_tier |
| 8 | Conclusions and outlook | Conclusions | limitations, takeaways, references |
| 9 | Backup | Backup | uncertainty_by_correctness, model_size_comparison |

The built deck has 31 slides and 34 clicks. Markdown numbers (`02b_`, `08c_`, ...) are
decoration only: `build_deck.SLIDES` sets the order.

**The library slides** (`slides/fig_polygraph.py`):
- **The code on them is executed, not transcribed.** Each block (`CODE_WRAP`,
  `CODE_API`, `CODE_MANAGER`, `CODE_CUSTOM`) is one string. The UEManager slide leaves
  out its imports (`CODE_MANAGER_IMPORTS`, prepended by `--test`) and names the two
  helper objects (`calcs`, `env`), so the call fits at a larger size. Its notes sit level
  with the lines they explain (`MANAGER_NOTES` holds line indices: renumber them if
  the block changes). The figure draws it, and
  `fig_polygraph.py --test` runs it in slide order against the cached
  Qwen/Qwen2.5-0.5B-Instruct (about 40 s on the Mac).
- `--test` also writes the real one-answer result to `figures/wrap_model_result.txt`,
  and the wrap_model slide prints that under the code (1.14, "The pancreas produces
  insulin…"). The API block needs a key, so `--test` checks only its imports and the
  `BlackboxModel.from_openai` signature. Re-run `--test` after changing any block.
- The code draws in TeX Gyre Cursor, with API names in bold blue and comments in grey.
  `code_block` raises if a line is wider than its panel.
- **API facts verified on lm-polygraph 0.7.0 + transformers 5.20.2:**
  - `estimate_uncertainty(model, estimator, input_text)` needs
    `attn_implementation="eager"`; with sdpa it raises `IndexError`.
  - Load the tokenizer with `padding_side="left"` for batched runs.
  - `ROCAUC` needs a binary quality metric, so on ROUGE use `PredictionRejectionArea`.
  - `UEManager(...)()` returns `{(level, estimator, quality, ue_metric): value}`;
    per-answer scores live in `man.estimations[(level, name)]`.
  - A custom estimator is an `Estimator` subclass: `super().__init__([stat names],
    level)`, plus `__str__` and `__call__(stats) -> np.ndarray`, higher = less sure.
- **Catalogue numbers** come from the README method table in
  `lm_polygraph-0.7.0.dist-info/METADATA`: 38 methods. Information-based 13,
  meaning diversity 16 (9 white-box, 7 black-box), reflexive 3, density-based 4 and
  ensembling 2 (both need training data). 11 of the 38 also work at claim level.
- The **uq_map** slide opens the methods chapter. It places the five families on two
  axes: what the method sees (text, token probabilities, hidden states) and how many
  answers it draws (one, several). Retitle chapters in `chapters.json`, then run
`set_chapters.py`. `build_deck.SLIDES` assigns slides to chapters by title prefix.

**What each content slide is.** One figure plus three pieces of text placed by
`deck_common.figure_slide` on the "Title Only" layout:
- the **title**;
- a blue **lead**: one sentence, a claim rather than a topic ("Sampling and an NLI
  model add consistency, not a higher ceiling.");
- a grey **caption** under the figure, which says how to read it.

The text lives in `slides/markdowns/NN_slide_<name>.md`. Edit there and rebuild. The
format is a `- **key** — note` line followed by `>` lines. Lead and caption lines are
joined into one paragraph, so never break a line inside a hyphenated word. The `NN` is
decoration: files are found by stem. The build also numbers slides with a real slidenum
field, and stamps the chapter nav bar and the side marker.

**Figure scripts never overwrite an existing markdown**, so wording edits survive
`--figures`. Pass `--rewrite-copy` to `fig_methods.py` or `plot_bioasq.py` to regenerate
the text from their `slide_copy` / `SLIDE_COPY`. That is also the only way a caption
picks up a changed number: slide 02 quotes 45% → 67%, slide 01 says "above 0.98".

**Which data each figure uses.** Slides 01–02 come from the 7B run, and 09b from both. Slides 10–12 come
from the 3B run. Slide 13 (backup) is a slopegraph over
Qwen 3B, Qwen 7B, Mistral 7B, OLMo-2 7B and Llama-3.1 8B (`build_deck.CACHE_OTHERS`),
with names from `--labels`. `build_deck.FIGURES`
passes all of this. Running `plot_bioasq.py` on 7B alone **overwrites** the 3B results
figures.

**Slide 09b (dataset)** is all measured. The funnel comes from the cached `jmhb/BioASQ`
factoid split (it loads with `HF_HUB_OFFLINE=1`): 1,609 questions → 1,365 whose
reference is at most 5 words → 1,000 sampled with seed 0, the same for both models.
Questions have a median of 9 words, references a median of 2, and 19% list synonyms;
they come from BioASQ challenges 5–13. The setup is closed-book: no BioASQ snippets are
given. The grading table shows exact matches, then the net gain from token rules, then
the net gain from manual labels. It is net so the rows sum to the final count (3B
156+87+72 = 315, 7B 240+90+117 = 447), and the script asserts that. The example question
(`DATASET_EXAMPLE`, upadacitinib) has 3B answering "JAK2" (wrong) and 7B "JAK1" (right,
decided by a manual label); the script asserts both verdicts.

**Slide 14 (limitations)** is in the Conclusions chapter. Its lead is "An introduction,
not a benchmark: your mileage may vary." Six cards: one dataset, two small models, one
setting, imperfect labels (the judgement-call count is read from `manual_labels.json`),
a subset of methods, and one metric (AUROC says nothing about thresholds or
calibration).

**Slide 01 examples** (`fig_methods.WRONG_EXAMPLES`) were picked by hand from the 7B
run. Each is wrong on any reading and carries model probability exp(−MSP) > 0.98: PDQ39
→ "Cancer", kallikrein genes 13 vs 15, Molgramostim → "Neutropenia", Bulevirtide →
"Hepatitis B". The script asserts each is still graded wrong.

**Click-by-click builds** (see `slides/README.md`). `slides/clicks.json` declares bands
in figure units. `hgf_layers.py` writes one transparent PNG per click, and the slide
stacks them with an Appear effect per layer. Rules that keep the stack honest:
- **Layer 0 stays on screen** and carries the furniture (anything spanning several bands,
  or outside all of them). Later layers carry only their own band. Furniture repeated in
  a later layer was painted over earlier clicks: gridlines across the tier-1 dots, and an
  opaque row ground hiding the "not run" note on slide 14.
- **A wide artist belongs to a band only if** at least 60% of it lies inside that band
  and it spills out by no more than 25% of the band's width. Otherwise full-width
  gridlines join the wider band.
- **Extents are clipped to the axes**, so a dot strip with outliers above the view is not
  measured as reaching off the canvas, which made it furniture and showed it before its
  click.
- **Avoid opaque fills that span bands.** Leave empty cells transparent rather than
  white.
- `hgf_layers.py --verify`, which `build_deck.py --figures` runs, fails a figure if any
  ink is missing from the stack, or if the stack flattened onto white differs from the
  whole figure ("PAINTED OVER").

## 9. Figure styling (house style)

Defined in `slides/hgf_style.py`, which is vendored from `~/projects/slides`. Keep it a
copy; only `FONT_DIR` differs. The full spec is `~/projects/slides/STYLE.md`.

- **Canvas** 124 × 44.5 figure units (10 u = 1 in), the aspect that exactly fills the
  slide's figure band (11.50 × 4.13 in). Every figure uses it. `fig_methods.board()` puts
  one axes over the whole canvas so everything is placed in figure units.
- **Type.** Headings in Hermann Bold, body in HalvarMittel, mono in TeX Gyre Cursor. The
  fonts are rasterised into the PNG, so the slide machine needs none of them. Sizes come
  from `TYPE`: tick/gloss 11, category/axis title 13, value 13.5, panel header 14, claim
  20, hero 28. **11 is the floor**; smaller dies on a projector. Wrap text with
  `wrap_measured` (measured, not by character count): figures do not reflow, so a longer
  line overlaps its neighbour.
- **Colour is semantic** and holds across every figure:

  | meaning | colour |
  |---|---|
  | tier 1, the story | blue `#005AA0` |
  | tiers 2–4, context | blue 60 `#669CC6` |
  | answer correct | dark blue `#002864` |
  | answer wrong, what we detect | orange `#ED7D31` (text in ORANGE_DARK `#A34A12`; orange is 2.49:1 on white) |
  | chance, axes, caveats | gray 85 `#738084` |

  **Tier is not a hue.** It is shown by position and a T1–T4 pill. Four categorical hues
  fail the colour-vision-deficiency contrast floors, so don't "improve" it. Red and green
  are reserved and unused. The accent palette is settled; stop revisiting it.
- **Concept colours** (2026-11-04), defined once in `fig_methods.py` (`FAMILY`, `LIB`
  and the accent constants) and used on the explanatory slides. Each idea gets one
  theme colour everywhere it appears. A light colour that sets type uses a darkened
  "ink" version.

  | idea | colour | where |
  |---|---|---|
  | Information family | blue `#005AA0` | map, cards, family slide, type-by-tier rows, catalogue bar |
  | Self-report family | purple `#7030A0` | same |
  | Consistency family | aerospace `#50C8AA`, ink `#307866` | same |
  | Sampling + likelihood family | information wine `#A0235A` | same |
  | Trained probe family | green `#8CB423`, ink `#57701A` | same |
  | the wrapped model | earth `#326469` | pipeline stage, code highlighting, wrap-model panels |
  | stat calculators | mint `#05E5BA`, ink `#037E66` | pipeline, code |
  | estimators | blue `#005AA0` | pipeline, code, custom-estimator slide |
  | UEManager, Dataset | dark blue `#002864` | pipeline, code |
  | quality and UE metrics | energy `#FFD228`, ink `#8C7416` | code, catalogue side panel |
  | caution | energy | limitations cards |
  | take-home | green ink | take-home numbers |
  | a perfect-ranking line | green ink | selective-generation chart |

  The pattern: a coloured top bar on a panel, the panel ground at about 12–14% of the
  colour, and the heading in the ink. Code is highlighted by kind, not by syntax: a
  library name is bold in the colour of the part it belongs to, and comments are grey.
  **The data-chart key above still holds on the result charts** (`plot_bioasq.py`):
  tier 1 blue, tiers 2–4 blue 60, wrong answers orange. The tier pills keep that key
  everywhere.
- **Marks.** No spines and no gridlines by default: values are labelled directly. One
  1 pt gray-22 line per tick is allowed only where a chart can't be direct-labelled.
  Rounded slabs (`RADIUS` 1.1), flat fills, panel grounds in PALE/HIGHLIGHT. Stroke
  weights come from `WEIGHT`; reference lines are dashed `(6, 4)`.
- **No title, lead or caption inside a PNG**: the slide places them, and a figure with
  its own headline doubles up. A small scope line at the figure's foot is fine.
- `save()` writes a 300 dpi PNG (what the build places) plus an SVG with real text (for
  hand-tuning in PowerPoint via Convert to Shape). It is deterministic, so a rebuild with
  no changes is a no-op.
- In the .pptx, text is Aptos (the template's font). A LibreOffice PDF shows a serif
  substitute; that is expected.

## 10. Decisions, gotchas and open items

- **Take-home slide** (`fig_methods.TAKEAWAYS`) closes the talk with five points:
  fluent is not reliable; let your access choose the family; try one forward pass first
  (the only point resting on the BioASQ result, and the caption says so); use scores to
  triage, not to judge; compare on your own task with LM-Polygraph.
- **Not built yet**: the demo notebook. Possible extra library material:
  `polygraph_eval` benchmarking, claim-level UE for long answers, and the
  `CausalLMWithUncertainty` / `APIWithUncertainty` wrappers (exported in 0.7.0, not yet
  tested here).
- **Judgement-call labels** (section 5) are re-checked, but not by a biomedical expert.
- **Verify a slide's claim against the data before keeping it.** The model-size caption
  once said "all" methods sit below the diagonal; it was 17/20, and is now 15/20.
- When regenerating results figures, always run `plot_bioasq.py` on the 3B cache with
  `--compare` on the 7B cache, or use `build_deck.py --figures`.
