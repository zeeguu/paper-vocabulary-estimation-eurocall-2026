#!/usr/bin/env python
"""
Compute the confidence-vs-threshold curve for the "seen N times => known" rule
(paper §6.1), from real data.

For every (learner, word) pair over the learner's full reading history we derive:
  E (encounters)      = distinct read articles (duration>30s) containing the word form
  b (articles_before) = articles containing the word read BEFORE its first translation
                        (only defined if the word was ever translated)

A "translation" is any explicit click on the word, including a click made while
extending a multi-word selection: we split such selections into their component
clicks (see _common.clicked_words), so a word looked up only inside a phrase
counts as translated rather than as a clean survivor.

A word "reaches N untranslated encounters" if it was seen in N articles without
having been translated yet: (never translated and E >= N) or (translated and b >= N).
Of those, a translated word is a FAILURE of the rule (we would have called it
known at N, yet it was later looked up). The raw rule would be:

  confidence(N) = #{never translated, E>=N} / #{reached N untranslated}

but that is survivorship-inflated: a word seen exactly N times and never again
(E == N, never translated) counts as a correct "known" although the learner
never had another chance to translate it -- padding the numerator, most heavily
at N=1 (two-thirds of the N=1 "known" cases never recurred). We therefore
CONDITION ON A LATER ENCOUNTER, so translating was genuinely possible:

  never translated -> require E >= N+1 (seen again, still not translated)
  translated       -> b >= N already implies a later encounter (the lookup)
  confidence(N)    = #{never translated, E>=N+1} / #{reached* N}

The difference is large at low N and negligible by N>=5; see
extract/check_survivorship.py for both curves side by side.

This uses proper tokenisation of article text (not substring matching) and no
per-word cap, so it also removes two caveats of the earlier substring-matched analysis.

From the same tokenised `b` it also writes the paper's §5.2 "when do
translations occur" table (../data/translation_timing.csv): over translated
words, the share whose first translation had happened by the 1st / 2nd / 3rd
article containing them (b == 0 / b <= 1 / b <= 2) and the share seen in 5+
articles beforehand (b >= 5). This supersedes an earlier substring-matched
approach that over-counts prior encounters and so reads conservatively.

Runs in the zeeguu-api venv. Writes ../data/threshold_confidence.csv and
../data/translation_timing.csv, and prints both as readable tables.
"""
import csv
import os
import sys
from collections import defaultdict

try:
    from zeeguu.api.app import create_app
except ImportError:
    _sib = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "api"))
    if os.path.isdir(os.path.join(_sib, "zeeguu")):
        sys.path.insert(0, _sib)
    try:
        from zeeguu.api.app import create_app
    except ImportError as e:
        sys.exit(f"Could not import `zeeguu` (run in the api venv). {e}")

from zeeguu.core.model import db
from zeeguu.core.util.text import split_words_from_text
from sqlalchemy import text
from _common import get_active_users, read_articles, clicked_words

app = create_app()
app.app_context().push()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "threshold_confidence.csv")
TIMING_CSV = os.path.join(HERE, "..", "data", "translation_timing.csv")

MIN_ARTICLES = int(os.environ.get("MIN_ARTICLES", "20"))
USER_LIMIT = int(os.environ.get("USER_LIMIT", "100000"))
MIN_LEN = 3
THRESHOLDS = [1, 2, 3, 4, 5, 7, 10, 15, 20]


def first_translations(uid, lid):
    """word -> (earliest click time, the article it was clicked in), standalone
    or within a split multi-word selection (see _common.clicked_words). The
    article is returned so per_word_evidence can EXCLUDE it from the
    before-count: the word was looked up there, so that article is not a clean
    (untranslated) encounter."""
    first_trans = {}
    for word, click_time, article_id, _session in clicked_words(db, uid, lid, min_len=MIN_LEN):
        if word not in first_trans or click_time < first_trans[word][0]:
            first_trans[word] = (click_time, article_id)
    return first_trans


def per_word_evidence(user):
    """Yield (encounters, articles_before, translated) per word for one learner.

    `articles_before` counts read articles containing the word BEFORE its first
    lookup, excluding the article the lookup was made in: the learner started
    reading that article before clicking (read_time < click_time), but the word
    was translated there, so it is not a clean encounter. Counting it would
    inflate every count by one and push first-sight lookups into the "second
    article" bucket.
    """
    articles = read_articles(db, user.id, user.language_id)
    first_trans = first_translations(user.id, user.language_id)

    encounter_count = defaultdict(int)
    before_count = defaultdict(int)
    for article in articles:
        if not article.content or not article.read_time:
            continue
        forms = {w.lower() for w in split_words_from_text(article.content) if len(w) >= MIN_LEN}
        for word in forms:
            encounter_count[word] += 1
            lookup = first_trans.get(word)
            if lookup is not None and article.read_time < lookup[0] and article.id != lookup[1]:
                before_count[word] += 1

    for word in set(encounter_count) | set(first_trans):
        translated = word in first_trans
        yield encounter_count.get(word, 0), (before_count.get(word, 0) if translated else None), translated


def _pct(t, k):
    return 100.0 * t[k] / t["n"] if t["n"] else float("nan")


def write_confidence_csv(reached, failed):
    """§6.1 confidence: for each language and threshold N, how often a word seen
    in N+ articles WITHOUT translation stayed untranslated. lang '' = ALL."""
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["language", "threshold_N", "reached", "failures", "confidence_pct"])
        for lang in [""] + sorted(k for k in reached if k):
            for N in THRESHOLDS:
                d = reached[lang][N]
                fl = failed[lang][N]
                conf = 100.0 * (d - fl) / d if d else float("nan")
                wr.writerow([lang or "ALL", N, d, fl, f"{conf:.1f}"])


def write_timing_csv(timing):
    """§5.2 timing: over translated words, how soon the first translation
    happened (by the 1st/2nd/3rd article, or after 5+ prior encounters).
    Languages by descending word count; lang '' = ALL."""
    with open(TIMING_CSV, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["language", "n_translated", "by_1st_pct", "by_2nd_pct",
                     "by_3rd_pct", "seen_5plus_pct"])
        langs = sorted((k for k in timing if k), key=lambda k: -timing[k]["n"])
        for lang in [""] + langs:
            t = timing[lang]
            wr.writerow([lang or "ALL", t["n"], f"{_pct(t,'by1'):.1f}",
                         f"{_pct(t,'by2'):.1f}", f"{_pct(t,'by3'):.1f}",
                         f"{_pct(t,'seen5'):.1f}"])


def print_summary(reached, failed, timing):
    """Echo the two ALL-learner tables to the console (§6.1 confidence, §5.2 timing)."""
    print("\nConfidence that a word seen in N+ articles WITHOUT translation is "
          "never translated (ALL learners):")
    print(f"  {'N':>3}  {'reached':>9}  {'failures':>9}  {'confidence':>10}")
    for N in THRESHOLDS:
        d, fl = reached[""][N], failed[""][N]
        conf = 100.0 * (d - fl) / d if d else 0
        print(f"  {N:>3}  {d:>9,}  {fl:>9,}  {conf:>9.1f}%")

    print("\nWhen do translations occur? (tokenised; §5.2)")
    print(f"  {'lang':>5} {'n':>7} {'by1st':>6} {'by2nd':>6} {'by3rd':>6} {'5+':>5}")
    for lang in ["", *sorted((k for k in timing if k), key=lambda k: -timing[k]["n"])]:
        t = timing[lang]
        print(f"  {lang or 'ALL':>5} {t['n']:>7,} {_pct(t,'by1'):6.1f} "
              f"{_pct(t,'by2'):6.1f} {_pct(t,'by3'):6.1f} {_pct(t,'seen5'):5.1f}")


def main():
    users = get_active_users(db, MIN_ARTICLES, USER_LIMIT)
    print(f"Computing thresholds over {len(users)} learners\n")

    # counters[lang][N] = [reached_denom, failures]; lang '' = ALL
    reached = defaultdict(lambda: defaultdict(int))
    failed = defaultdict(lambda: defaultdict(int))
    # §5.2 timing: over translated words, the distribution of articles_before
    # (articles that contained the word before its first translation). lang '' = ALL.
    timing = defaultdict(lambda: {"n": 0, "by1": 0, "by2": 0, "by3": 0, "seen5": 0})

    for i, user in enumerate(users, 1):
        lang = user.language
        for encounters, articles_before, translated in per_word_evidence(user):
            if translated:
                for key in ("", lang):
                    t = timing[key]
                    t["n"] += 1
                    t["by1"] += articles_before == 0   # first translation by the 1st article
                    t["by2"] += articles_before <= 1   # ... by the 2nd
                    t["by3"] += articles_before <= 2   # ... by the 3rd
                    t["seen5"] += articles_before >= 5 # seen in 5+ articles beforehand
            for N in THRESHOLDS:
                # Condition on a later encounter (see docstring): never-translated
                # words need encounters >= N+1 so translating was genuinely still possible.
                hit = (translated and articles_before >= N) or ((not translated) and encounters >= N + 1)
                if hit:
                    reached[""][N] += 1
                    reached[lang][N] += 1
                    if translated:
                        failed[""][N] += 1
                        failed[lang][N] += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(users)} learners")

    write_confidence_csv(reached, failed)
    write_timing_csv(timing)
    print_summary(reached, failed, timing)
    print(f"\nWrote {OUT_CSV}\nWrote {TIMING_CSV}")


if __name__ == "__main__":
    main()
