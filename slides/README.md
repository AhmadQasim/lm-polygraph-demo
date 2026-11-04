# slides/

Everything that makes the journal club deck. The experiment itself (running,
grading and scoring the estimators) stays in `../scripts/`.

```bash
.venv/bin/python slides/build_deck.py --figures --pdf   # full rebuild -> slides/build/
.venv/bin/python slides/build_deck.py                   # rebuild from the current figures
.venv/bin/python slides/build_deck.py --list            # running order
.venv/bin/python slides/hgf_layers.py                   # clicks per figure
.venv/bin/python slides/hgf_layers.py --verify          # layers composite back?
```

| file | what |
|---|---|
| `build_deck.py` | template + figures + markdowns → `build/journal_club_deck.pptx`. `SLIDES` is the running order |
| `markdowns/NN_slide_<name>.md` | each slide's title, lead and caption. **Edit the words here** |
| `fig_methods.py` | figures for the opening and method slides |
| `plot_bioasq.py` | figures for the results slides, from `../outputs/` |
| `clicks.json` | which figures build up click by click, and what arrives on each click |
| `hgf_layers.py` | writes a figure's click layers; `--verify` composites them back |
| `hgf_anim.py` | the `p:timing` XML that reveals a layer per click. Vendored, unchanged |
| `chapters.json`, `set_chapters.py` | chapter divider titles, written into the template |
| `strip_template.py` | empties the vitiligo deck into a fresh template |
| `hgf_style.py`, `deck_common.py`, `nav_bar.py`, `hgf_copy.py` | vendored from `~/projects/slides/scripts/` |
| `fonts/` | Helmholtz brand fonts, rasterised into the figures |
| `deck_outline.json` | every slide's text after the last full build, for review in git |

Not in git: `journal_club_template.pptx` (187 MB), `figures/` and `build/`, all
regenerable.

## Click-by-click builds

Twelve of the thirteen slides reveal their panels one click at a time - 22 clicks
in all. `python3 slides/hgf_layers.py` prints the count per figure.

A PNG cannot be revealed in parts, so each of those figures is also written as one
transparent layer per click (`figures/<name>_l0.png`, `_l1.png`, ...), all the same
size, and `deck_common.figure_slide` stacks them at one box and asks `hgf_anim` for
an Appear effect per layer. Only the layers are animated: the title, the blue lead,
the caption, the navigation bar and the slide number are on screen from the start.

**What arrives on each click is declared in `clicks.json`**, as bands in figure
units - the same 0..124 x 0..44.5 the figure scripts lay out in. Anything outside
every band is furniture and stays visible throughout, which is how a figure's
heading and its grey scope line survive the build. A figure with no entry gets no
layers; `selective_generation` is one plate with three overlapping curves and is
left alone deliberately.

The risk with declared bands is a band that misses part of its panel: that artist
lands in no layer and vanishes from the slide without anything failing. So
`build_deck.py --figures` composites every figure's layers back together and
refuses to build if a pixel the whole figure inks is missing from the stack. Run it
alone with `python3 slides/hgf_layers.py --verify`.

`--no-clicks` places every figure whole. The PDF export is unaffected either way:
LibreOffice ignores `p:timing` and flattens the layers, so the handout is 22 pages,
one per slide, each one finished.

Figure scripts write a slide's markdown only if it does not exist yet, so a
rebuild never overwrites wording you edited. Pass `--rewrite-copy` to either
figure script to regenerate the markdown from the Python, which is also how a
caption picks up a changed number (slide 02 quotes the kept accuracy).
