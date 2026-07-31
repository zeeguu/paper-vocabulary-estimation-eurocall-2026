"""Does "one lookup resolves a word" survive conditioning on a later encounter?

The naive figure -- most translated words (~88%) are looked up exactly once --
reads as evidence that a single lookup resolves a word. But that rate is
survivorship-inflated: a word looked up once and never met again CANNOT be
looked up again, so it counts as "resolved" for free. §6.2 reports the honest
conditioned figure instead.

Honest test: of words the learner looked up AND then encountered again (a real
chance to look them up once more), how many were looked up again?

For each (learner, word) with a first lookup at ft and `cnt` distinct lookups:
  re-translated      := cnt > 1   (a second lookup is itself a later encounter)
  recurred, resolved := cnt == 1 AND the word appears in a read article after ft
  recurred           := re-translated OR (recurred, resolved)
  conditioned re-translation rate = #re-translated / #recurred

Run in the api venv:
    ~/code/zeeguu/api/.venv/bin/python extract/check_retranslation.py
"""
import csv
import os
import sys
from collections import Counter, defaultdict

try:
    from zeeguu.api.app import create_app
except ImportError:
    _sib = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "api"))
    if os.path.isdir(os.path.join(_sib, "zeeguu")):
        sys.path.insert(0, _sib)
    from zeeguu.api.app import create_app

from zeeguu.core.model import db
from zeeguu.core.util.text import split_words_from_text
from sqlalchemy import text
from _common import get_active_users, read_articles, clicked_words

app = create_app()
app.app_context().push()

MIN_ARTICLES = int(os.environ.get("MIN_ARTICLES", "20"))
USER_LIMIT = int(os.environ.get("USER_LIMIT", "100000"))
MIN_LEN = 3


def translations(user_id, language_id):
    """word -> (first_lookup_time, distinct-session lookup count).

    Each clicked word is a lookup; multi-word selections are split into their
    component clicks (see _common.clicked_words), so a word looked up only
    within a phrase still counts, with its own session/time."""
    first_time, reading_sessions = {}, defaultdict(set)
    for word, click_time, _article_id, reading_session in clicked_words(db, user_id, language_id, min_len=MIN_LEN):
        if word not in first_time or click_time < first_time[word]:
            first_time[word] = click_time
        reading_sessions[word].add(reading_session)
    return {word: (first_time[word], len(reading_sessions[word])) for word in first_time}


def reencountered_after_lookup(articles, word_lookups):
    """Words met in a read article AFTER their first lookup -- i.e. the learner
    saw them again and so had a real chance to look them up a second time. (A
    word never met again cannot be re-looked-up, so it must not count as
    "resolved" for free; this is the survivorship conditioning.)"""
    reencountered = set()
    for article in articles:
        if not article.content or not article.read_time:
            continue
        forms = {w.lower() for w in split_words_from_text(article.content) if len(w) >= MIN_LEN}
        for word in forms:
            lookup = word_lookups.get(word)
            if lookup is not None and article.read_time > lookup[0]:
                reencountered.add(word)
    return reencountered


def tally_recurrence(word_lookups, reencountered):
    """Classify one learner's translated words: how many recurred after their
    first lookup, and of those how many were looked up again. Returns a Counter
    so the per-learner tallies sum straight into the running totals."""
    counts = Counter()
    for word, (_first_time, lookup_count) in word_lookups.items():
        counts["total"] += 1
        re_looked_up = lookup_count > 1
        if lookup_count == 1:
            counts["once"] += 1
        # a word "recurred" only if it got a real second chance: looked up
        # again, or met again in a later read article
        if re_looked_up or word in reencountered:
            counts["recurred"] += 1
            if re_looked_up:
                counts["re_translated"] += 1
    return counts


def main():
    users = get_active_users(db, MIN_ARTICLES, USER_LIMIT)
    counts = Counter()

    for i, user in enumerate(users, 1):
        word_lookups = translations(user.id, user.language_id)
        if not word_lookups:
            continue
        articles = read_articles(db, user.id, user.language_id)
        reencountered = reencountered_after_lookup(articles, word_lookups)
        counts += tally_recurrence(word_lookups, reencountered)
        if i % 25 == 0:
            print(f"  ...{i}/{len(users)}", flush=True)

    total = counts["total"]
    once = counts["once"]
    recurred = counts["recurred"]
    re_translated = counts["re_translated"]

    print(f"\ntranslated words:                 {total:,}")
    print(f"  looked up exactly once:         {once:,}  ({100*once/total:.1f}%)   <- the survivorship-inflated figure")
    print(f"\nof translated words, RECURRED after first lookup: {recurred:,}  ({100*recurred/total:.1f}%)")
    print(f"  ...and were looked up AGAIN:    {re_translated:,}")
    print(f"\nCONDITIONED re-translation rate (recurred -> looked up again): "
          f"{100*re_translated/recurred:.1f}%")
    print(f"  => one lookup 'resolves' the word {100*(recurred-re_translated)/recurred:.1f}% of the "
          f"time it recurs")

    # Freeze §6.2's figure into the shipped dataset so it is reproducible from
    # the CSVs without the database (analysis/reproduce_paper_numbers.py reads it).
    here = os.path.dirname(os.path.abspath(__file__))
    out_csv = os.path.join(here, "..", "data", "retranslation.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["translated_words", "recurred_after_first_lookup", "re_translated",
                         "conditioned_retranslation_pct", "resolved_pct"])
        writer.writerow([total, recurred, re_translated,
                         f"{100*re_translated/recurred:.1f}", f"{100*(recurred-re_translated)/recurred:.1f}"])
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
