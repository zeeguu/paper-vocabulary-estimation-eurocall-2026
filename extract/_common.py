"""Shared query helpers and the single "read article" definition.

An article counts as READ by a user when the total time spent on it is at least
30 seconds AND at least long enough to read every word at 300 wpm (200 ms per
word), which excludes skimming a long article in a few seconds:

    total_ms >= max(30000, word_count * 200)

This is a LOWER bound only. Time is wall-clock for almost all of our data (the
platform's session timer only began pausing on inactivity in 2024), so an
article left open in a background tab can accumulate "reading" time and pass
this test without having been read -- which makes every downstream estimate an
upper bound (see the paper's Limitations section). We tried an upper duration
bound to catch that case and did not adopt it: duration is a GATE, not a weight,
so an inflated duration can only let a borderline article through, never
over-weight anything downstream; and the small diffuse bias that leaves is not
worth discarding a learner's whole reading history when a single lost article
drops them under the >=20-article threshold.

The 30 s / 300 wpm bounds are hand-set, like the model weights in the paper's
Method section.
"""
import os
from collections import Counter

from sqlalchemy import text
from zeeguu.core.util.text import split_words_from_text

# Restrict the study to languages with a real sample: a target language is kept
# only if MORE THAN MIN_LANG_LEARNERS learners read in it. This drops one-off
# languages (a single learner is not a sample) data-drivenly, without hard-coding
# any language code, and stays correct as the data grows. Set to 0 to keep all.
MIN_LANG_LEARNERS = int(os.environ.get("MIN_LANG_LEARNERS", "10"))


# The read-article test used by every query below (the LOWER bound described above).
PLAUSIBLY_READ = "SUM(urs.duration) >= GREATEST(30000, COALESCE(a.word_count, 0) * 200)"


def get_active_users(db, min_articles, user_limit):
    """Users ranked by their number of read articles (>= min_articles)."""
    q = f"""
        WITH read_article AS (
            -- one row per article this learner has plausibly READ (§4.1): the
            -- read test is summed over all their sessions on the article
            SELECT urs.user_id, urs.article_id, a.language_id
            FROM user_reading_session urs
            JOIN article a ON urs.article_id = a.id
            GROUP BY urs.user_id, urs.article_id, a.language_id, a.word_count
            HAVING {PLAUSIBLY_READ}
        )
        SELECT r.user_id AS id, l.code AS language, r.language_id AS language_id,
               COUNT(*) AS articles_read
        FROM read_article r
        JOIN language l ON r.language_id = l.id
        JOIN user     u ON r.user_id     = u.id
        WHERE NOT COALESCE(u.is_dev, 0)          -- real learners only, not dev/test accounts
        GROUP BY r.user_id, l.code, r.language_id
        HAVING COUNT(*) >= :min_articles
        ORDER BY articles_read DESC
        LIMIT :user_limit
    """
    rows = db.session.execute(
        text(q), {"min_articles": min_articles, "user_limit": user_limit}
    ).fetchall()
    if MIN_LANG_LEARNERS:
        learners_per_language = Counter(row.language for row in rows)
        rows = [row for row in rows if learners_per_language[row.language] > MIN_LANG_LEARNERS]
    return rows


def read_articles(db, user_id, language_id):
    """Read articles for one user in one language, time-ordered.

    Returns rows of (id, content, read_time). `article.content` holds the full
    article body and mirrors the fragment representation the reader actually
    sees (article_fragment): across read articles that have fragments, content
    length matches the concatenated fragment text within a few percent, both
    directions, so tokenising content reflects what was read.
    """
    q = f"""
        SELECT a.id, a.content, MIN(urs.start_time) AS read_time
        FROM user_reading_session urs
        JOIN article a ON urs.article_id = a.id
        WHERE urs.user_id = :user_id AND a.language_id = :language_id AND a.content IS NOT NULL
        GROUP BY a.id, a.content, a.word_count
        HAVING {PLAUSIBLY_READ}
        ORDER BY read_time
    """
    return db.session.execute(text(q), {"user_id": user_id, "language_id": language_id}).fetchall()


def clicked_words(db, user_id, language_id, min_len=1):
    """Yield (word, time, article_id, session_key) for each explicitly clicked word.

    In Zeeguu a learner grows a selection by clicking each successive word, so a
    multi-word lookup is a run of individual clicks. We therefore SPLIT a
    multi-word selection into its component words and treat each as its own
    lookup -- each word was clicked, so each is evidence the learner did not
    know it. Automatically-extended MWE bookmarks (is_mwe = 1) are dropped
    instead: there the platform, not the learner, added the neighbour, so we
    cannot say which word was clicked and the unit of interest is the expression
    rather than a word (this is a handful of rows in the current data). Single
    word lookups pass through unchanged.

    `min_len` drops short component forms; callers that count word forms of
    length >= 3 elsewhere pass min_len=3 so the two sides stay comparable.
    """
    q = """
        WITH lookups AS (
            -- This learner's word lookups in one language. A "lookup" is one
            -- bookmark; getting from a bookmark to the word text it points at is
            -- a four-table walk, because Zeeguu stores words normalised:
            SELECT b.id AS bookmark_id, b.time AS click_time,
                   b.reading_session_id, b.is_mwe, p.content AS word
            FROM bookmark b
            JOIN user_word uw ON b.user_word_id = uw.id   -- this learner's vocab entry
            JOIN meaning   m  ON uw.meaning_id  = m.id      -- a source<->translation pair
            JOIN phrase    p  ON m.origin_id    = p.id       -- the source-language word text
            WHERE uw.user_id = :user_id AND p.language_id = :language_id
              -- reading lookups only: exercise and article-preview lookups are a
              -- different modality (practice / previewing, not reading)
              AND b.translation_source = 'reading'
        )
        SELECT LOWER(word) AS content, click_time,
               s.article_id AS article_id,
               COALESCE(reading_session_id, -bookmark_id) AS reading_session
               -- ^ the reading-session id, or -bookmark_id when the lookup has
               --   no session, so each session-less lookup is its own occasion
        FROM lookups
        LEFT JOIN user_reading_session s ON s.id = reading_session_id   -- the article it was read in
        WHERE click_time IS NOT NULL
          AND COALESCE(is_mwe, 0) = 0
    """
    for row in db.session.execute(text(q), {"user_id": user_id, "language_id": language_id}):
        for word in split_words_from_text(row.content or ""):
            word = word.lower()
            if len(word) >= min_len:
                yield word, row.click_time, row.article_id, row.reading_session
