"""Does the §6.1 confidence curve survive conditioning on a later encounter?

The committed curve counts, at threshold N,
    confidence(N) = #{never translated, E >= N} / #{reached N}
where "reached N" = (never translated & E>=N) or (translated & b>=N), with
    E (encounters)      = distinct read articles containing the word,
    b (articles_before) = such articles read before the word's first translation.

Concern (survivorship): a word seen exactly N times and never again (E == N,
never translated) is counted as a correct "known" although the learner never had
another chance to translate it. That pads the numerator, most heavily at low N.

Corrected metric: require an encounter AFTER the Nth untranslated one, so
translating was genuinely possible.
    never translated -> needs E >= N+1 (seen again, still not translated)
    translated       -> b >= N already implies a later encounter (the lookup)
    confidence*(N)   = #{never translated, E >= N+1} / #{reached* N}

Prints both curves side by side. Run in the api venv:
    ~/code/zeeguu/api/.venv/bin/python extract/check_survivorship.py
"""
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
THRESHOLDS = [1, 2, 3, 4, 5, 7, 10, 15, 20]


def first_translations(uid, lid):
    """word -> (earliest click, article it was clicked in), from single-word
    look-ups only (see _common.clicked_words). The article is returned so the
    lookup's own article can be excluded from the before-count."""
    first_trans = {}
    for word, click_time, article_id, _session in clicked_words(db, uid, lid, min_len=MIN_LEN):
        if word not in first_trans or click_time < first_trans[word][0]:
            first_trans[word] = (click_time, article_id)
    return first_trans


def per_word_evidence(user):
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
            # exclude the article the lookup was made in (not a clean encounter)
            if lookup is not None and article.read_time < lookup[0] and article.id != lookup[1]:
                before_count[word] += 1
    for word in set(encounter_count) | set(first_trans):
        translated = word in first_trans
        yield encounter_count.get(word, 0), (before_count.get(word, 0) if translated else None), translated


def main():
    users = get_active_users(db, MIN_ARTICLES, USER_LIMIT)
    reached_cur = defaultdict(int); fail_cur = defaultdict(int)
    reached_cor = defaultdict(int); fail_cor = defaultdict(int)

    for i, user in enumerate(users, 1):
        for encounters, articles_before, translated in per_word_evidence(user):
            for N in THRESHOLDS:
                # current
                if (translated and articles_before >= N) or ((not translated) and encounters >= N):
                    reached_cur[N] += 1
                    if translated:
                        fail_cur[N] += 1
                # corrected: require a post-N encounter
                if (translated and articles_before >= N) or ((not translated) and encounters >= N + 1):
                    reached_cor[N] += 1
                    if translated:
                        fail_cor[N] += 1
        if i % 25 == 0:
            print(f"  ...{i}/{len(users)} learners", flush=True)

    print(f"\n{'N':>3} | {'current conf':>12} {'(reached)':>10} | {'conditioned conf':>16} {'(reached)':>10}")
    for N in THRESHOLDS:
        dc, fc = reached_cur[N], fail_cur[N]
        dk, fk = reached_cor[N], fail_cor[N]
        cc = 100.0 * (dc - fc) / dc if dc else 0
        ck = 100.0 * (dk - fk) / dk if dk else 0
        print(f"{N:>3} | {cc:>11.1f}% {dc:>10,} | {ck:>15.1f}% {dk:>10,}")


if __name__ == "__main__":
    main()
