#!/usr/bin/env python
"""
Compute monthly vocabulary-progression estimates per learner, using the P(know)
evidence-accumulation model over real reading + translation history.

This runs in the zeeguu-api venv (it needs the word
frequency lists via `wordstats` and the api tokenizer). It writes an anonymised
../data/monthly_progression.csv that the DB-free chart script then plots.

Method (see paper §4.2):

  For each learner we replay their history in time order. An "encounter" of a
  word = a plausibly-read article (duration > 30s) that contains that word form
  (counted once per article). A "translation" = an explicit click on the word;
  a multi-word selection is split into its component clicks, each a lookup of
  that word (see _common.clicked_words).

  P(know) per word is an online logistic evidence-accumulation model. We replay
  events in time order, keeping a running log-odds S per word:

    clean encounter:  S <- DECAY*S + ALPHA
    lookup:           S <- DECAY*S - (ALPHA + BETA)
    P(know) = sigmoid(S)

  (A looked-up word was also in the article it was looked up in, so the article
  event already added +ALPHA; the lookup cancels that and applies -BETA, netting
  -BETA per looked-up encounter. That is why the paper states it as -BETA.)

  A clean encounter (seen, not looked up) is evidence for knowing; a lookup is
  evidence against. DECAY (<1) fades old evidence so recent behaviour weighs more:
  a word looked up then later read cleanly recovers toward "known". Before any
  evidence S=0 and P(know)=sigmoid(0)=0.5, the prior. BETA>ALPHA: a lookup is an
  explicit "didn't know it", a clean pass may also be skimming or guessing.
  DECAY=1 recovers the static, order-invariant limit.

  Words are reduced to Snowball stems (approximate lemmas) first, so we count
  near-lemmas rather than raw forms. At each month boundary we snapshot and derive:
    - words_known_est : distinct known stems within the top TOP_VOCAB_CAP
    - topN coverage   : share of the N most-frequent stems that are "known"
                        (a never-encountered stem counts as NOT known -
                        we do not credit the 0.5 prior toward coverage)
    - cefr            : mapped from words_known_est via documented vocab-size bands

MODELLING DECISIONS worth revisiting (all flagged so the paper can defend them):
  * Snowball-stem level (approximate lemmas); Polish and other unsupported
    languages fall back to raw word forms.
  * "known" needs encounter evidence; the 0.5 never-seen prior is NOT counted as
    coverage, otherwise a beginner would show ~50% coverage of everything.
  * KNOWN_THRESHOLD and the CEFR vocab-size bands are heuristics (constants below).
"""
import csv
import math
import os
import sys
from collections import defaultdict

try:
    from zeeguu.api.app import create_app
except ImportError:
    _sibling_api = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "api"))
    if os.path.isdir(os.path.join(_sibling_api, "zeeguu")):
        sys.path.insert(0, _sibling_api)
    try:
        from zeeguu.api.app import create_app
    except ImportError as e:
        sys.exit(f"Could not import `zeeguu` (run in the api venv). {e}")

from zeeguu.core.model import db
from zeeguu.core.util.text import split_words_from_text
from wordstats import LanguageInfo
from nltk.stem import SnowballStemmer
from sqlalchemy import text
from _common import get_active_users, read_articles, clicked_words

app = create_app()
app.app_context().push()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, "..", "data", "monthly_progression.csv")

# Sample: ALL learners with >=20 read articles, no
# cap. This used to be the 30 most active at >=15 articles, which made §5 report
# a subset the paper never declared -- and the best-observed learners at that, so
# the coverage and CEFR figures came out optimistic. Override via env for a quick run.
USER_LIMIT = int(os.environ.get("USER_LIMIT", "100000"))
MIN_ARTICLES = int(os.environ.get("MIN_ARTICLES", "20"))
# A stem counts as "known" at P(know) >= 0.75, i.e. three clean encounters with no
# offsetting lookups. With ALPHA=0.5 and DECAY=0.9 the accumulated P(known) is
# 0.62 / 0.72 / 0.79 after 1 / 2 / 3 clean encounters, so 0.75 crosses at three.
# The §6.1 reliability curve is ~flat from two encounters (87 / 90 / 91% at 1/2/3),
# so two would validate too -- but band coverage is very SENSITIVE to the threshold
# near 0.70, where 2-clean-encounter words (P=0.72) are dense: dropping to 0.70
# jumps the example learners 1-2 CEFR levels and a light reader's estimate ~2.5x.
# We keep three so the reported coverage stays a stable, conservative floor.
KNOWN_THRESHOLD = float(os.environ.get("KNOWN_THRESHOLD", "0.75"))
BANDS = [100, 500, 1000]
# "words known" is reported as known words WITHIN the most-frequent TOP_VOCAB_CAP,
# a bounded, defensible proxy for functional vocabulary. Counting all distinct
# word-forms instead yields tens of thousands (inflected forms, names, typos) and
# is not a meaningful vocabulary size.
TOP_VOCAB_CAP = 5000
# Extrapolation. Our evidence only covers words the learner has actually met, so
# counting known/band-size makes a light reader look like a beginner however good
# they are -- and puts our numbers on a different scale from the published CEFR
# mappings, which come from tests that sample the whole frequency range. We do
# what a vocabulary size test does instead: within each narrow rank bucket, take
# the known rate over the stems the learner HAS met and apply it to the bucket.
# A bucket with fewer than MIN_SAMPLE encountered stems is credited nothing
# rather than guessed at, so the estimate stays a floor where we have no data.
BUCKET = 100
MIN_SAMPLE = 5
# vocab-size -> CEFR (receptive, rough; documented heuristic, not from the data)
# Published receptive vocabulary-size mapping (Milton & Alexiou 2009, via X_Lex,
# which samples the 5000 most frequent lemmas -- the same base as TOP_VOCAB_CAP):
#   A1 <1500 | A2 1500-2500 | B1 2750-3250 | B2 3250-3750 | C1 3750-4500 | C2 4500-5000
# Boundaries are the upper edge of each band; the published gap between A2 (2500)
# and B1 (2750) is assigned to B1. We apply the published bands rather than tuning
# our own: our count is bounded by the words a learner has actually encountered, so
# the level that comes out is a FLOOR, not an estimate, and we prefer to under-report
# against a citable mapping than to hand-set thresholds that flatter the estimate.
CEFR_BANDS = [(1500, "A1"), (2500, "A2"), (3250, "B1"), (3750, "B2"), (4500, "C1")]


def clicked_lookups(user_id, language_id, reduce):
    """(stem, time, article_id) lookups, one per (stem, reading session).

    Each explicitly clicked word is a lookup; a multi-word selection is split
    into its component clicks (see _common.clicked_words). Repeated clicks of
    the same stem within one reading session count once, so a looked-up
    encounter is penalised exactly once (otherwise -(ALPHA+BETA) fires per click
    but the article added +ALPHA only once).

    `article_id` is returned so process_user() can keep only the lookups made in
    a READ article. The -(ALPHA+BETA) update assumes the article's +ALPHA was
    already applied, which holds only when that article is in the event stream.
    """
    best = {}   # (stem, session) -> (click_time, article_id)
    for word, click_time, article_id, session in clicked_words(db, user_id, language_id):
        stem = reduce(word)
        key = (stem, session)
        if key not in best or click_time < best[key][0]:
            best[key] = (click_time, article_id)
    return [(stem, click_time, article_id) for (stem, _session), (click_time, article_id) in best.items()]


# Log-odds evidence weights. BETA > ALPHA: a lookup is an explicit "didn't know",
# a clean pass weaker evidence (could be knowing, skimming, or guessing).
ALPHA = 0.5   # per clean (untranslated) encounter
BETA = 1.0    # per lookup
# Recency: the running log-odds is scaled by DECAY at each event, so recent
# behaviour weighs more than old. DECAY<1 lets a word looked up then later read
# cleanly recover toward "known"; DECAY=1 is the static, order-invariant limit.
# Per-EVENT decay (a coarse proxy for time-based forgetting); the value is a
# deliberate guess, to be fit to data in future work (see §9).
DECAY = float(os.environ.get("DECAY", "0.9"))


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)            # avoid overflow for large negative x
    return ex / (1.0 + ex)


def cefr_for(words_known):
    for size, level in CEFR_BANDS:
        if words_known < size:
            return level
    return "C2"


def month_of(dt):
    return f"{dt.year:04d}-{dt.month:02d}"


# Snowball stems approximate lemmas, so we count near-lemmas rather than raw
# forms (the CEFR bands are lemma-based). Languages Snowball lacks (e.g. Polish)
# fall back to raw forms via an identity reducer.
SNOWBALL = {"da": "danish", "de": "german", "en": "english", "es": "spanish",
            "fr": "french", "it": "italian", "nl": "dutch", "no": "norwegian",
            "pt": "portuguese", "ro": "romanian", "ru": "russian", "sv": "swedish",
            "hu": "hungarian", "fi": "finnish"}

_model_cache = {}
def lang_model(code):
    """Return (reduce, bands, top_cap) for a language.

    `reduce` maps a word form to its stem (or itself if unsupported). `bands` is
    {N: set of top-N stems} and `top_cap` the set of the TOP_VOCAB_CAP most
    frequent stems, each stem ranked by the frequency of its most frequent form.
    """
    if code not in _model_cache:
        # Normalise apostrophes before stemming. The tokeniser splits elisions and
        # drops the apostrophe ("L'homme" -> "L", "homme"), while the frequency list
        # keeps it ("l'"), so elided clitics could never match and were credited to
        # nobody: for French that silently removed 9 of the top 100 (c' d' j' l' m'
        # n' qu' s' t') from every learner's reachable vocabulary.
        _strip = lambda w: w.replace("'", "").replace("\u2019", "")
        if code in SNOWBALL:
            _stem = SnowballStemmer(SNOWBALL[code]).stem
            reduce = lambda w: _stem(_strip(w))
        else:
            reduce = _strip
        ordered, seen = [], set()
        for f in LanguageInfo.load(code).all_words():   # already frequency-ranked
            s = reduce(f.lower())
            if s not in seen:
                seen.add(s)
                ordered.append(s)
        bands = {n: set(ordered[:n]) for n in BANDS}
        top_cap = set(ordered[:TOP_VOCAB_CAP])
        # Narrow rank buckets for extrapolation (see snapshot()). Within a bucket
        # of BUCKET consecutive ranks the words a learner happens to have met are
        # much closer to a random sample of the bucket than they would be across a
        # whole band, where the frequent end is met first and would bias the rate up.
        buckets = [set(ordered[i:i + BUCKET]) for i in range(0, TOP_VOCAB_CAP, BUCKET)]
        _model_cache[code] = (reduce, bands, top_cap, buckets)
    return _model_cache[code]


def process_user(learner_id, user, model):
    reduce, bands, top_cap, buckets = model
    articles = read_articles(db, user.id, user.language_id)
    lookups = clicked_lookups(user.id, user.language_id, reduce)

    # Only lookups made in a READ article are evidence: those articles are the
    # evidence base, and the -(ALPHA+BETA) update below cancels an +ALPHA that
    # only a read article contributes. A lookup in an article that failed the
    # read filter would otherwise be penalised at -(ALPHA+BETA) with no
    # offsetting +ALPHA, i.e. 50% harder than the model intends.
    read_ids = {article.id for article in articles}

    # merged, time-ordered event stream (words reduced to stems)
    events = []
    for article in articles:
        if not article.content or not article.read_time:
            continue
        stems = {reduce(w.lower()) for w in split_words_from_text(article.content)}
        events.append((article.read_time, "article", stems))
    for stem, click_time, article_id in lookups:
        if article_id not in read_ids:
            continue
        events.append((click_time, "translation", stem))
    events.sort(key=lambda x: (x[0], x[1]))   # tiebreak on kind, never on payload

    S = defaultdict(float)   # running (recency-decayed) log-odds per stem
    encountered = set()
    articles_seen = 0
    rows = []
    cur_month = None

    def snapshot(month):
        known = {stem for stem in encountered if _sigmoid(S[stem]) >= KNOWN_THRESHOLD}

        def rate(stems):
            """Known rate over the stems of `stems` the learner has encountered.

            Returns None when too few have been met to estimate one.
            """
            met = encountered & stems
            if len(met) < MIN_SAMPLE:
                return None
            return len(known & met) / len(met)

        # Band coverage and vocabulary size are both extrapolated from the
        # encountered sample, bucket by bucket, so that a learner is measured on
        # what they know rather than on how much of the band we happen to have
        # observed. Buckets with too little evidence contribute 0, keeping the
        # estimate a floor.
        est = 0.0
        for bucket in buckets:
            known_rate = rate(bucket)
            if known_rate is not None:
                est += known_rate * len(bucket)
        words_known = int(round(est))

        coverage = {}
        for n in BANDS:
            covered, seen_any = 0.0, False
            for bucket in buckets:
                inband = bucket & bands[n]
                if not inband:
                    continue
                known_rate = rate(inband)
                if known_rate is not None:
                    covered += known_rate * len(inband)
                    seen_any = True
            coverage[n] = round(100.0 * covered / n, 1) if seen_any else 0.0
        rows.append([
            learner_id, user.language, month, articles_seen, len(encountered), words_known,
            coverage[100], coverage[500], coverage[1000], cefr_for(words_known),
        ])

    for event_time, kind, payload in events:
        event_month = month_of(event_time)
        if cur_month is None:
            cur_month = event_month
        if event_month != cur_month:
            snapshot(cur_month)
            cur_month = event_month
        if kind == "article":
            articles_seen += 1
            for stem in payload:
                S[stem] = DECAY * S[stem] + ALPHA          # clean encounter
                encountered.add(stem)
        else:
            # lookup: net -BETA per looked-up encounter (the paper's model, §4.2).
            # The article event above already added +ALPHA to this stem, so we
            # cancel it (-ALPHA) and apply the penalty (-BETA) => -(ALPHA+BETA).
            S[payload] = DECAY * S[payload] - (ALPHA + BETA)
    if cur_month is not None:
        snapshot(cur_month)
    return rows


def main():
    users = get_active_users(db, MIN_ARTICLES, USER_LIMIT)
    print(f"Computing progression for {len(users)} learners")
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "learner_id", "language", "month", "articles_cumulative",
            "distinct_words_encountered", "words_known_est",
            "top100_cov", "top500_cov", "top1000_cov", "cefr",
        ])
        for i, user in enumerate(users, start=1):
            learner_id = f"L{i:02d}"
            rows = process_user(learner_id, user, lang_model(user.language))
            for r in rows:
                writer.writerow(r)
            last = rows[-1] if rows else None
            print(f"  {learner_id} ({user.language}): {len(rows)} months"
                  + (f", final ~{last[5]} words known, {last[9]}" if last else ""))
    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
