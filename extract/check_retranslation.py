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
from collections import defaultdict

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


def translations(uid, lid):
    """word -> (first_time, distinct-session click count).

    Each single-word click is a lookup; multi-word selections are skipped
    (see _common.clicked_words)."""
    ft, sessions = {}, defaultdict(set)
    for word, click_time, _article_id, session in clicked_words(db, uid, lid, min_len=MIN_LEN):
        if word not in ft or click_time < ft[word]:
            ft[word] = click_time
        sessions[word].add(session)
    return {word: (ft[word], len(sessions[word])) for word in ft}


def main():
    users = get_active_users(db, MIN_ARTICLES, USER_LIMIT)
    total = once = recurred = re_translated = 0

    for i, user in enumerate(users, 1):
        trans = translations(user.id, user.language_id)
        if not trans:
            continue
        arts = read_articles(db, user.id, user.language_id)
        recurred_after = set()
        for a in arts:
            if not a.content or not a.read_time:
                continue
            forms = {w.lower() for w in split_words_from_text(a.content) if len(w) >= MIN_LEN}
            for w in forms:
                tv = trans.get(w)
                if tv is not None and a.read_time > tv[0]:
                    recurred_after.add(w)
        for w, (ft, cnt) in trans.items():
            total += 1
            if cnt == 1:
                once += 1
            retr = cnt > 1
            rec = retr or (w in recurred_after)
            if rec:
                recurred += 1
                if retr:
                    re_translated += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(users)}", flush=True)

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
        wr = csv.writer(f)
        wr.writerow(["translated_words", "recurred_after_first_lookup", "re_translated",
                     "conditioned_retranslation_pct", "resolved_pct"])
        wr.writerow([total, recurred, re_translated,
                     f"{100*re_translated/recurred:.1f}", f"{100*(recurred-re_translated)/recurred:.1f}"])
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
