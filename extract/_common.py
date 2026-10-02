"""Shared query helpers and the single "read article" definition.

An article counts as READ by a user when the total time spent on it is at least
30 seconds AND at least long enough to read every word at 300 wpm (200 ms per
word), which excludes skimming a long article in a few seconds:

    total_ms >= max(30000, word_count * 200)

THE TWO TERMS DO DIFFERENT JOBS. The 300 wpm ceiling is an empirical claim about
reading speed (Brysbaert, 2019: adult L1 silent reading averages 238 wpm, most
readers 175-300, and L2 is slower still). The 30s floor is an engagement floor
against accidental opens -- not a reading-rate claim, so no reading-rate evidence
bears on it. Keeping them straight is what makes the pair defensible.

The floor over-rejects short articles, measured 2026-08-03 and kept anyway:
demanding 30s on a 90-word text is a 180 wpm ceiling, well under the 300 we
claim. Cost: 155 (user, article) pairs, 0.50% of 30,716 read pairs, across 96
users, mean length 78 words (legacy articles below today's 90-word ingest
minimum). Dropping the floor was considered and rejected -- it would mean
re-running the pipeline and re-verifying every number in the paper to buy 0.5%
more data. §4.1 states the trade-off openly instead.

    SELECT COUNT(*), AVG(a.word_count) FROM (
      SELECT urs.user_id, urs.article_id, SUM(urs.duration) d
      FROM user_reading_session urs GROUP BY 1,2) s
    JOIN article a ON a.id = s.article_id
    WHERE a.word_count <= 150 AND s.d >= a.word_count*200 AND s.d < 30000;

If the floor is ever dropped, re-check the profile count: learners at 18-19 read
articles may cross the >=20 threshold and enter the sample.

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
        SELECT pr.user_id AS id, l.code AS language, pr.language_id AS language_id,
               COUNT(*) AS articles_read
        FROM (
            SELECT urs.user_id, urs.article_id, a.language_id
            FROM user_reading_session urs
            JOIN article a ON urs.article_id = a.id
            GROUP BY urs.user_id, urs.article_id, a.language_id, a.word_count
            HAVING {PLAUSIBLY_READ}
        ) pr
        JOIN language l ON pr.language_id = l.id
        JOIN user u ON pr.user_id = u.id
        WHERE COALESCE(u.is_dev, 0) = 0    -- exclude developers / test accounts
        GROUP BY pr.user_id, l.code, pr.language_id
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


# Evidence definition for a look-up. Both flags default to the CORRECTED behaviour
# (2026-09-09) and can be flipped back via the environment to reproduce the older
# numbers for comparison:
#
#   SPLIT_MWE=1     restore the old behaviour of splitting a multi-word selection
#                   into per-component look-ups (see clicked_words).
#   KEEP_NONREADING=1  restore the old behaviour of counting every bookmark,
#                   including exercise and article-preview look-ups and the
#                   ExampleSentence re-anchors the practice UI creates.
SPLIT_MWE = os.environ.get("SPLIT_MWE", "0") == "1"
KEEP_NONREADING = os.environ.get("KEEP_NONREADING", "0") == "1"

# context_type 9 = ExampleSentence: rows the PRACTICE UI creates when it re-anchors
# a word to a generated example. They carry translation_source = 'reading' but are
# not reading (0 of them have a reading session). Absent before 2025 and 39% of all
# 2026 bookmarks, so they contaminate recent learners hardest.
EXAMPLE_SENTENCE_CONTEXT = 9


def mwe_component_encounters(db, user_id, language_id, min_len=1):
    """(word, article_id) pairs that a multi-word selection makes AMBIGUOUS.

    A multi-word selection is evidence that the learner did not know something in
    that phrase, but not WHICH word — the client fuses the selection, so there is
    no per-word click to attribute it to. clicked_words() therefore drops these
    rows rather than splitting them (see its docstring). That alone would be
    optimistic: with no look-up recorded, every component word would silently
    become a CLEAN encounter in the very article where the learner asked for help.

    So callers exclude these (word, article) encounters from the clean-encounter
    side as well, leaving the pair contributing no evidence in either direction.
    The exclusion is per ENCOUNTER, not per word: the same word met cleanly in
    another article still counts normally.
    """
    q = """
        SELECT LOWER(p.content) AS content, s.article_id AS article_id
        FROM bookmark b
        JOIN user_word uw ON b.user_word_id = uw.id
        JOIN meaning m ON uw.meaning_id = m.id
        JOIN phrase p ON m.origin_id = p.id
        LEFT JOIN user_reading_session s ON b.reading_session_id = s.id
        LEFT JOIN bookmark_context bc ON bc.id = b.context_id
        WHERE uw.user_id = :user_id AND p.language_id = :language_id
              AND b.time IS NOT NULL
              AND COALESCE(b.translation_source, 'reading') = 'reading'
              AND COALESCE(bc.context_type_id, 0) <> %d
              AND TRIM(p.content) LIKE '%% %%'
    """ % EXAMPLE_SENTENCE_CONTEXT
    out = set()
    for row in db.session.execute(text(q), {"user_id": user_id, "language_id": language_id}):
        if row.article_id is None:
            continue
        for word in split_words_from_text(row.content or ""):
            word = word.lower()
            if len(word) >= min_len:
                out.add((word, row.article_id))
    return out


def clicked_words(db, user_id, language_id, min_len=1):
    """Yield (word, time, article_id, session_key) for each explicitly clicked word.

    WHAT COUNTS AS A LOOK-UP (corrected 2026-09-09; see
    _meta/mwe-bookmark-provenance.md):

    * Only look-ups made WHILE READING. `translation_source` distinguishes
      'reading' from 'exercise' and 'article_preview', but is not sufficient on
      its own: the practice UI writes ExampleSentence rows (context type 9) that
      carry translation_source = 'reading' and no reading session. Those are
      excluded explicitly. Requiring a reading session instead was rejected --
      sessions were not recorded before 2019, which would silently delete the
      earliest two years of history.

    * A multi-word selection is NOT split into per-component look-ups. It used to
      be, on the reasoning that the learner clicked each word in turn. That is
      false in production: the client fuses the selection and deletes the partial
      bookmark, and `mwe_partner_token_i` is NULL in every bookmark in the
      database -- so the components were never separately clicked. Splitting
      manufactured 79.2% of all top-300 look-up evidence the model ever saw
      (120,304 spurious against 31,507 genuine), which made frequent words look
      like the ones learners look up most. Since a look-up is negative evidence,
      dropping the artefact RAISES P(known) on frequent words.

      A multi-word selection is genuine evidence about the EXPRESSION, which this
      word-level function cannot represent, so such rows are skipped here rather
      than reinterpreted. Modelling expressions as their own units is future work.

    `min_len` drops short forms; callers that count word forms of length >= 3
    elsewhere pass min_len=3 so the two sides stay comparable.
    """
    reading_only = "" if KEEP_NONREADING else f"""
              AND COALESCE(b.translation_source, 'reading') = 'reading'
              AND COALESCE(bc.context_type_id, 0) <> {EXAMPLE_SENTENCE_CONTEXT}
    """
    q = f"""
        SELECT LOWER(p.content) AS content, b.time AS click_time,
               s.article_id AS article_id,
               COALESCE(b.reading_session_id, -b.id) AS session
        FROM bookmark b
        JOIN user_word uw ON b.user_word_id = uw.id
        JOIN meaning m ON uw.meaning_id = m.id
        JOIN phrase p ON m.origin_id = p.id
        LEFT JOIN user_reading_session s ON b.reading_session_id = s.id
        LEFT JOIN bookmark_context bc ON bc.id = b.context_id
        WHERE uw.user_id = :user_id AND p.language_id = :language_id
              AND b.time IS NOT NULL
              AND COALESCE(b.is_mwe, 0) = 0
              {reading_only}
    """
    for row in db.session.execute(text(q), {"user_id": user_id, "language_id": language_id}):
        words = split_words_from_text(row.content or "")
        if len(words) > 1 and not SPLIT_MWE:
            continue          # evidence about the expression, not about its words
        for word in words:
            word = word.lower()
            if len(word) >= min_len:
                yield word, row.click_time, row.article_id, row.session
