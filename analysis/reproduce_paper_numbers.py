#!/usr/bin/env python
"""
Reproduce every §5-§6 number in the paper from the committed CSVs alone -- no
database, no dependencies beyond the standard library. Run:

    python analysis/reproduce_paper_numbers.py

It reads the four CSVs the extract/ scripts produce against the study database --
  data/translation_timing.csv     (§5.2)
  data/threshold_confidence.csv    (§6.1)
  data/monthly_progression.csv     (§5.1, §5.3, §5.4)
  data/retranslation.csv           (§6.2)
-- and prints the paper's sample counts, timing table, coverage examples,
progression figures, confidence curve, and re-lookup rate, each next to the
value stated in the paper. A reader with only the published data can thus check
every reported number; the extract/ scripts (which need the DB) only regenerate
these CSVs. See analysis/generate_progression_charts.py for Figure 1, also
DB-free from monthly_progression.csv.
"""
import csv
import os
import statistics
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]


def rows(name):
    with open(os.path.join(DATA, name)) as f:
        return list(csv.DictReader(f))


def rule(title):
    print("\n" + "=" * 68 + f"\n{title}")


def main():
    prog = rows("monthly_progression.csv")
    by_learner = defaultdict(list)
    for r in prog:
        by_learner[r["learner_id"]].append(r)
    final = {lid: sorted(rs, key=lambda x: x["month"])[-1]
             for lid, rs in by_learner.items()}

    # ---- §5.1 User Sample ----
    by_lang = Counter(r["language"] for r in final.values())
    rule("§5.1  User Sample")
    print(f"  profiles: {len(final)}   (paper: 324)")
    print("  " + "  ".join(f"{k}={by_lang.get(k, 0)}" for k in ("fr", "nl", "en", "de", "da"))
          + "   (paper: fr=264 nl=17 en=11 de=14 da=18)")

    # ---- §5.2 When a Word Is Translated, How Early? ----
    rule("§5.2  When a Word Is Translated, How Early?   [translation_timing.csv]")
    print(f"  {'lang':>5} {'n':>8} {'by1st':>6} {'by2nd':>6} {'by3rd':>6} {'5+':>5}")
    for r in rows("translation_timing.csv"):
        print(f"  {r['language']:>5} {int(r['n_translated']):>8,} "
              f"{r['by_1st_pct']:>6} {r['by_2nd_pct']:>6} {r['by_3rd_pct']:>6} {r['seen_5plus_pct']:>5}")

    # ---- §5.3 Vocabulary Coverage Patterns (final snapshot per learner) ----
    rule("§5.3  Vocabulary Coverage Patterns   [final snapshot per learner]")
    print(f"  {'learner':>7} {'lang':>4} {'arts':>5} {'t100':>6} {'t500':>6} "
          f"{'t1000':>6} {'known':>6} {'cefr':>4}")

    def show(r, tag):
        print(f"  {r['learner_id']:>7} {r['language']:>4} {r['articles_cumulative']:>5} "
              f"{r['top100_cov']:>6} {r['top500_cov']:>6} {r['top1000_cov']:>6} "
              f"{r['words_known_est']:>6} {r['cefr']:>4}   {tag}")

    # §5.3 table, sorted ascending by reading volume: A = lightest Danish,
    # B = English 79, C = French 88, D = heaviest Dutch. C and D are the two
    # profiles plotted in Figure 1.
    nl = max((r for r in final.values() if r["language"] == "nl"),
             key=lambda x: int(x["articles_cumulative"]))
    da = min((r for r in final.values() if r["language"] == "da"),
             key=lambda x: int(x["articles_cumulative"]))

    def find(lang, arts):
        return next(r for r in final.values()
                    if r["language"] == lang and r["articles_cumulative"] == arts)

    show(da, "A")
    show(find("en", "79"), "B")
    show(find("fr", "88"), "C")
    show(nl, "D")

    # ---- §5.4 Monthly Progression ----
    crossed = multi = 0
    finals = []
    for rs in by_learner.values():
        idx = [LEVELS.index(x["cefr"]) for x in sorted(rs, key=lambda x: x["month"])]
        if max(idx) > min(idx):
            crossed += 1
        if max(idx) - min(idx) > 2:
            multi += 1
        finals.append(int(sorted(rs, key=lambda x: x["month"])[-1]["articles_cumulative"]))
    rule("§5.4  Monthly Progression")
    print(f"  profiles crossing >=1 CEFR level: {crossed}   (paper: 53)")
    print(f"  profiles crossing  >2 levels:     {multi}   (paper: 'just one')")
    print(f"  median final articles:            {statistics.median(finals):g}   (paper: 27)")

    # ---- §6.1 How Many Clean Encounters Make a Word Known? ----
    rule("§6.1  How Many Clean Encounters Make a Word Known?   [threshold_confidence.csv, ALL]")
    print(f"  {'N':>3} {'reached':>9} {'failures':>9} {'confidence':>11}")
    for r in rows("threshold_confidence.csv"):
        if r["language"] == "ALL":
            print(f"  {r['threshold_N']:>3} {int(r['reached']):>9,} "
                  f"{int(r['failures']):>9,} {r['confidence_pct']:>10}%")

    # ---- §6.2 How Reliable Is a Single Lookup or Clean Pass? ----
    rule("§6.2  How Reliable Is a Single Lookup or Clean Pass?   [retranslation.csv]")
    rt = rows("retranslation.csv")[0]
    print(f"  translated words:            {int(rt['translated_words']):>8,}")
    print(f"  recurred after first lookup: {int(rt['recurred_after_first_lookup']):>8,}")
    print(f"  looked up a second time:     {int(rt['re_translated']):>8,}")
    print(f"  conditioned re-translation:  {rt['conditioned_retranslation_pct']:>7}%   (paper: '36%')")
    print()


if __name__ == "__main__":
    main()
