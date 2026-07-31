# Vocabulary estimation from reading behaviour

Code and data for *"Estimating receptive vocabulary from incidental reading
behavior"* (EuroCALL 2026).

**Idea.** On a reading platform with one-click translation, learners reveal not
only the words they *don't* know (by looking them up) but also, over time, the
ones they *do*: a word met repeatedly across articles without ever being looked
up is evidence of prior knowledge. This repository validates that signal and
shows how per-word estimates aggregate into a vocabulary-growth curve. Data comes
from [Zeeguu](https://github.com/zeeguu/api), an open-source reading platform.

## Layout

```
extract/   _common.py                     shared queries + the "read article" definition
           extract_word_thresholds.py     DB -> translation_timing.csv, threshold_confidence.csv  (§5.2, §6.1)
           extract_progression.py         DB -> monthly_progression.csv                           (§5.1, §5.3, §5.4, Fig 1)
           check_retranslation.py         DB -> retranslation.csv                                  (§6.2)
           check_survivorship.py          robustness of the §6.1 curve to survivorship inflation
analysis/  reproduce_paper_numbers.py     CSVs -> every §5-§6 table and number   (no DB)
           generate_progression_charts.py monthly_progression.csv -> figures/    (no DB)
data/      the four CSVs + data/README.md (column-by-column data dictionary)
```

## Reproducing

**1. From the shipped CSVs — no database.** Everything in §5–§6 regenerates from
the anonymised CSVs:

```bash
python analysis/reproduce_paper_numbers.py          # stdlib only; prints every §5-§6 table/number
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

- **Lookups come early (§5.2).** 71% of translated words are looked up on the very
  first article containing them, and 90% by the second. So sustained
  non-translation is a meaningful signal.
- **Clean encounters signal knowledge (§6.1).** Of words seen once without a lookup
  and then met again, ~87% are never translated afterward; a second clean
  encounter lifts this to ~90%, essentially flat out to twenty. We count a stem
  known at three clean encounters, a conservative margin.
- **Single events are noisy (§6.2).** Of looked-up words that recur, 36% are looked
  up a *second* time — so one lookup "resolves" a word only ~64% of the time, and
  neither a lone lookup nor a lone clean pass is reliable on its own.

## License

See [LICENSE](LICENSE).
