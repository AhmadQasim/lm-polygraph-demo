# slides/

Everything that makes the journal club deck. The experiment itself (running,
grading and scoring the estimators) stays in `../scripts/`.

```bash
.venv/bin/python slides/build_deck.py --figures --pdf   # full rebuild -> slides/build/
.venv/bin/python slides/build_deck.py                   # rebuild from the current figures
.venv/bin/python slides/build_deck.py --list            # running order
```

| file | what |
|---|---|
| `build_deck.py` | template + figures + markdowns → `build/journal_club_deck.pptx`. `SLIDES` is the running order |
| `markdowns/NN_slide_<name>.md` | each slide's title, lead and caption. **Edit the words here** |
| `fig_methods.py` | figures for the opening and method slides |
| `plot_bioasq.py` | figures for the results slides, from `../outputs/` |
| `chapters.json`, `set_chapters.py` | chapter divider titles, written into the template |
| `strip_template.py` | empties the vitiligo deck into a fresh template |
| `hgf_style.py`, `deck_common.py`, `nav_bar.py`, `hgf_copy.py` | vendored from `~/projects/slides/scripts/` |
| `fonts/` | Helmholtz brand fonts, rasterised into the figures |
| `deck_outline.json` | every slide's text after the last full build, for review in git |

Not in git: `journal_club_template.pptx` (187 MB), `figures/` and `build/`, all
regenerable.

Figure scripts write a slide's markdown only if it does not exist yet, so a
rebuild never overwrites wording you edited. Pass `--rewrite-copy` to either
figure script to regenerate the markdown from the Python, which is also how a
caption picks up a changed number (slide 02 quotes the kept accuracy).
