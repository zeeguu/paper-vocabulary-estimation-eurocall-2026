"""Discovery vs acquisition: how much of the growth curve is learning?

Reviewer A (EuroCALL 2026) asked how increases in the estimate can be told
apart from increased observation of vocabulary the learner already knew. The
progression extractor splits the known stems it has actually observed (top
5,000, no extrapolation) into

  known_never_looked_up : known, never looked up so far -> discovered prior knowledge
  known_after_lookup    : known, looked up at some point -> plausible acquisition

This prints, from data/monthly_progression.csv:
  1. the split at each learner's final snapshot;
  2. where monthly gains come from, by reading stage (articles read so far);
  3. the same for the two learners in Figure 1 (C = L07, D = L01).

    python3 analysis/decompose_growth.py
"""
import os
import statistics as st
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "monthly_progression.csv")
STAGES = [(0, 20), (20, 60), (60, 150), (150, 10**9)]


def stage_of(articles):
    for lo, hi in STAGES:
        if lo <= articles < hi:
            return (lo, hi)


def main():
    df = pd.read_csv(CSV)
    df = df.sort_values(["learner_id", "month"])

    # 1. final snapshot
    last = df.groupby("learner_id").tail(1)
    share = (last.known_after_lookup / last.known_observed.where(last.known_observed > 0))
    print("1. FINAL SNAPSHOT, known stems observed (top 5,000)")
    print(f"   profiles: {len(last)}")
    print(f"   median known observed: {last.known_observed.median():.0f}")
    print(f"   median share known AFTER a lookup (acquisition): {100 * share.median():.1f}%"
          f"  (IQR {100 * share.quantile(.25):.1f}-{100 * share.quantile(.75):.1f}%)")
    pooled = last.known_after_lookup.sum() / last.known_observed.sum()
    print(f"   pooled: {100 * pooled:.1f}% acquisition, {100 * (1 - pooled):.1f}% discovery")

    # 2. monthly gains by stage (stage = articles read at the START of the month)
    gains = defaultdict(lambda: [0, 0, 0])   # stage -> [disc, acq, months]
    for _, g in df.groupby("learner_id"):
        prev = None
        for _, r in g.iterrows():
            if prev is not None:
                s = stage_of(prev.articles_cumulative)
                gains[s][0] += r.known_never_looked_up - prev.known_never_looked_up
                gains[s][1] += r.known_after_lookup - prev.known_after_lookup
                gains[s][2] += 1
            prev = r
    print("\n2. WHERE MONTHLY GAINS COME FROM, by articles read so far (pooled)")
    print("   stage        months   discovery   acquisition   acquisition share")
    for s in STAGES:
        d, a, m = gains[s]
        tot = d + a
        label = f"{s[0]}-{s[1]}" if s[1] < 10**9 else f"{s[0]}+"
        print(f"   {label:<10} {m:>8}   {d:>9}   {a:>11}   "
              f"{(100 * a / tot) if tot else float('nan'):>14.1f}%")

    # 3. Figure 1 learners
    print("\n3. FIGURE 1 LEARNERS")
    for lid, name in [("L07", "C"), ("L01", "D")]:
        g = df[df.learner_id == lid]
        if g.empty:
            continue
        print(f"   Learner {name} ({lid}):")
        for _, r in g.iterrows():
            print(f"     {r.month}  arts {r.articles_cumulative:>5}  known_obs {r.known_observed:>5}"
                  f"  discovered {r.known_never_looked_up:>5}  after-lookup {r.known_after_lookup:>4}")


if __name__ == "__main__":
    main()
