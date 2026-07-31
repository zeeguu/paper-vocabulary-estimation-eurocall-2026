# Data dictionary

Anonymised extracts behind the paper's §5 (Results) and §6 (Validation). Every
figure and table in those sections is reproducible from these CSVs alone, with
no database:

```bash
python analysis/reproduce_paper_numbers.py     # prints all §5-§6 tables/numbers
python analysis/generate_progression_charts.py # writes Figure 1
```

The `extract/*.py` scripts regenerate these CSVs from the Zeeguu database (the
only step that needs a DB). Learners appear only as opaque codes (`L01`, `L02`,
…); no names, emails, or article text are ever exported.

**Source snapshot.** These CSVs were generated from the Zeeguu production
database dump of **2026-07-29** (`vocab-estimation-paper-dataset_2026-07-29.sql`,
SHA-256 `d8c0abaa577be62f9d6f4dbd90a87a700860efce050865dfd4c234f67b4c19f5`), held
privately (it contains personal data). Default `MIN_LANG_LEARNERS`=10 → 324
learner-language profiles across fr/nl/en/de/da.

Unit conventions: a "word" is a Snowball stem (near-lemma); an "encounter" is a
read article (≥ 30 s and ≤ 300 wpm) containing that stem, counted once per
article; a "lookup"/"translation" is an explicit click, with multi-word
selections split into their component clicks. See `extract/_common.py` and
`extract/extract_progression.py` for the exact definitions.

---

## `translation_timing.csv` — §5.2 "When a Word Is Translated, How Early?"

For each translated word, which article containing it holds its *first*
translation. One row per language plus an `ALL` row. Produced by
`extract/extract_word_thresholds.py`.

| column | meaning |
|---|---|
| `language` | target-language code, or `ALL` for the pooled row |
| `n_translated` | number of (learner, word) pairs that were ever translated |
| `by_1st_pct` | % whose first translation fell on the **1st** article containing the word |
| `by_2nd_pct` | cumulative % translated **by the 2nd** article (≤ 1 clean article before) |
| `by_3rd_pct` | cumulative % translated **by the 3rd** article |
| `seen_5plus_pct` | % seen in **5+** articles before their first translation |

---

## `threshold_confidence.csv` — §6.1 "How Many Clean Encounters Make a Word Known?"

The confidence curve: of words seen in *N*+ articles without translation **and
then encountered again**, what share is never translated afterward. One row per
(language, N); `language = ALL` is the pooled curve the paper reports. Produced
by `extract/extract_word_thresholds.py`. Conditioned on a later encounter to
remove survivorship inflation (see `extract/check_survivorship.py`).

| column | meaning |
|---|---|
| `language` | target-language code, or `ALL` |
| `threshold_N` | number of clean (untranslated) encounters, *N* |
| `reached` | (learner, word) pairs reaching *N* clean encounters and recurring afterward |
| `failures` | of those, how many were translated later (the rule was wrong) |
| `confidence_pct` | `(reached − failures) / reached × 100` — confidence the word is known |

---

## `monthly_progression.csv` — §5.1 sample, §5.3 coverage, §5.4 progression, Figure 1

One row per (learner, month): the P(known) model (`extract/extract_progression.py`,
paper §4.2) replayed over each learner's history, snapshotted monthly. §5.1's
sample counts, §5.3's coverage table, §5.4's figures, and Figure 1 all derive
from this file.

| column | meaning |
|---|---|
| `learner_id` | opaque learner code (`L01` … ), ordered by articles read, descending |
| `language` | target-language code |
| `month` | snapshot month, `YYYY-MM` |
| `articles_cumulative` | read articles up to and including this month |
| `distinct_words_encountered` | distinct stems encountered so far |
| `words_known_est` | estimated known stems among the 5,000 most frequent (extrapolated; a floor) |
| `top100_cov` | % of the 100 most-frequent stems estimated known |
| `top500_cov` | % of the 500 most-frequent stems estimated known |
| `top1000_cov` | % of the 1,000 most-frequent stems estimated known |
| `cefr` | indicative CEFR level mapped from `words_known_est` (Milton & Alexiou 2009 bands; a floor) |

---

## `retranslation.csv` — §6.2 "How Reliable Is a Single Lookup or Clean Pass?"

A single summary row: how often a looked-up word, once met again, is looked up a
*second* time (the conditioned re-translation rate behind §6.2's "35%"). Produced
by `extract/check_retranslation.py`.

| column | meaning |
|---|---|
| `translated_words` | total (learner, word) pairs ever translated |
| `recurred_after_first_lookup` | of those, how many were met again after the first lookup |
| `re_translated` | of the recurred, how many were looked up again |
| `conditioned_retranslation_pct` | `re_translated / recurred × 100` (paper's "35%") |
| `resolved_pct` | `100 − conditioned_retranslation_pct` — one lookup "resolves" the word |
