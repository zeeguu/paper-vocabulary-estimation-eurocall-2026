# Vocabulary estimation from reading behaviour

Code and data for *"Estimating receptive vocabulary from incidental reading
behavior"* (EuroCALL 2026).

**Idea.** On a reading platform with one-click translation, learners reveal not
only the words they *don't* know (by looking them up) but also, over time, the
ones they *do*: a word met repeatedly across articles without ever being looked
up is evidence that it is known. This repository validates that signal and
shows how per-word estimates aggregate into a continuously updated vocabulary
estimate. Data comes
from [Zeeguu](https://github.com/zeeguu/api), an open-source reading platform.

The camera-ready paper is in [`paper/`](paper/eurocall-2026-vocabulary-estimation.pdf).

## Layout

```
extract/   _common.py                     shared queries + the "read article" definition
           extract_word_thresholds.py     DB -> translation_timing.csv, threshold_confidence.csv  (§5.1, §6.1)
           extract_progression.py         DB -> monthly_progression.csv                           (§4.5, §5.2, §5.3, Fig 1)
           check_retranslation.py         DB -> retranslation.csv                                  (§6.2)
           check_survivorship.py          robustness of the §6.1 curve to survivorship inflation
analysis/  reproduce_paper_numbers.py     CSVs -> every §4.5-§6 table and number          (no DB)
           generate_progression_charts.py monthly_progression.csv -> figures/ (Figure 1)  (no DB)
           decompose_growth.py            known words split by lookup history (§5.3)     (no DB)
data/      the four CSVs + data/README.md (column-by-column data dictionary)
paper/     the camera-ready paper (PDF)
```

## Reproducing

**1. From the shipped CSVs — no database.** Everything in §4.5–§6 regenerates from
the anonymised CSVs:

```bash
python analysis/reproduce_paper_numbers.py          # stdlib only; prints every §4.5-§6 table/number
python -m venv .venv && source .venv/bin/activate   # for the chart (pandas + matplotlib)
pip install -r requirements.txt
python analysis/generate_progression_charts.py      # writes Figure 1 to figures/
```

`reproduce_paper_numbers.py` prints each value next to the paper's stated figure,
so the whole results/validation section is checkable against the data alone. See
`data/README.md` for what every column means.

**2. Regenerating the CSVs from the database.** The only step that needs a DB.
Run it in a zeeguu-api virtualenv, with an api checkout as a sibling of this repo:

```bash
python extract/extract_word_thresholds.py   # -> translation_timing.csv, threshold_confidence.csv
python extract/extract_progression.py        # -> monthly_progression.csv
python extract/check_retranslation.py         # -> retranslation.csv
```

Only anonymised aggregates leave the database: an opaque learner id (`L01`, …),
target-language code, and per-word / per-month counts. **No learner names and no
article text are exported.**

## What the analysis shows

Over 322 learners (324 learner–language profiles):

- **Lookups come early (§5.1).** 74% of looked-up words are looked up in the very
  first article containing them, 92% by the second and 98% by the fifth. So
  sustained non-lookup is a meaningful signal.
- **Clean encounters predict no later lookup (§6.1).** Of words met once without a
  lookup and then met again, 92% are never looked up afterward; the rate rises to
  95% after two clean encounters, 96% after three and 98% after twenty. This
  *non-lookup rate* measures behavioral consistency, not knowledge tested
  independently. A stem counts as known at three clean encounters.
- **A single lookup is noisy (§6.2).** Of looked-up words that recur, 23% are
  looked up a second time.
- **Growth is not the same as learning (§5.3).** Only about 4% of known words were
  ever looked up; for the rest, the model cannot tell prior knowledge from words
  learned without help.

The lookup definition was corrected on 2026-09-09 (multi-word selections are no
longer split into component lookups, and lookups made outside reading are
excluded); the data and numbers here follow the corrected definition, as does
the camera-ready paper.

## License

See [LICENSE](LICENSE).
