#!/usr/bin/env python
"""
Generate monthly vocabulary-progression figures for the paper from REAL data.

Standalone: needs pandas + matplotlib (see requirements.txt), no database. Reads
../data/monthly_progression.csv (produced by extract/extract_progression.py) and
writes the figures to ../figures/.

    python analysis/generate_progression_charts.py [learnerA] [learnerB]
    # defaults: L01 and L04 (any two learner ids present in the CSV)

Each learner gets a two-panel figure: frequency-band coverage (Top 100/500/1000)
and estimated words-known (of the 5000 most frequent) with CEFR markers. A
side-by-side comparison of the two learners is also written.

The series are the model's output over real reading history; see the extractor
for the P(know) model and its documented modelling choices.
"""
import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "monthly_progression.csv")
OUT_DIR = os.path.join(HERE, "..", "figures")

LANG_NAME = {"da": "Danish", "fr": "French", "de": "German", "es": "Spanish",
             "nl": "Dutch", "en": "English"}


def style_time_axis(ax):
    """Year labels on the major ticks, quarterly minor ticks, light grid."""
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))  # quarterly ticks
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def load(learner_id):
    df = pd.read_csv(CSV)
    d = df[df["learner_id"] == learner_id].copy()
    if d.empty:
        sys.exit(f"No rows for {learner_id} in {CSV}")
    d["date"] = d["month"].map(lambda m: datetime.strptime(m, "%Y-%m"))
    d = d[d["articles_cumulative"] > 0].sort_values("date")
    return d


def describe(d, learner_id):
    lang = LANG_NAME.get(d["language"].iloc[0], d["language"].iloc[0])
    arts = int(d["articles_cumulative"].max())
    months = len(d)
    return lang, f"{lang}\n{arts} articles over {months} active months"


def coverage_panel(ax, d, small=False):
    ms, size, lw = d["date"], (3 if small else 4), (1.5 if small else 2)
    ax.fill_between(ms, 0, d["top1000_cov"], alpha=0.3, color="#2ecc71", label="Top 1000")
    ax.fill_between(ms, 0, d["top500_cov"], alpha=0.5, color="#3498db", label="Top 500")
    ax.fill_between(ms, 0, d["top100_cov"], alpha=0.7, color="#9b59b6", label="Top 100")
    ax.plot(ms, d["top100_cov"], "o-", color="#9b59b6", markersize=size, linewidth=lw)
    ax.plot(ms, d["top500_cov"], "s-", color="#3498db", markersize=size, linewidth=lw)
    ax.plot(ms, d["top1000_cov"], "^-", color="#2ecc71", markersize=size, linewidth=lw)
    ax.axhline(y=80, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylabel("Coverage (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    style_time_axis(ax)


def vocab_panel(ax, d):
    ms = d["date"]
    ax.fill_between(ms, 0, d["words_known_est"], alpha=0.3, color="#e74c3c")
    ax.plot(ms, d["words_known_est"], "-", color="#e74c3c", linewidth=1.3, zorder=2)
    # each dot's area is proportional to the number of articles read that month, so
    # reading bursts (and gaps) are visible directly on the growth curve.
    read = d["articles_cumulative"].diff().fillna(d["articles_cumulative"]).clip(lower=0)
    ax.scatter(ms, d["words_known_est"], s=6 + 7 * read, color="#7f8c8d",
               edgecolor="white", linewidth=0.4, zorder=3)
    prev = None
    for _, row in d.iterrows():
        if row["cefr"] != prev:
            ax.annotate(
                row["cefr"], xy=(row["date"], row["words_known_est"]), xytext=(0, 12),
                textcoords="offset points", fontsize=9, fontweight="bold",
                ha="center", color="#2c3e50",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#2c3e50", alpha=0.8),
            )
            prev = row["cefr"]
    # size legend: reference circles for articles read that month (dot area ∝ this)
    for n in (10, 30, 50):
        ax.scatter([], [], s=6 + 7 * n, facecolor="#7f8c8d", edgecolor="white",
                   linewidth=0.4, label=str(n))
    ax.legend(title="articles / month", loc="upper left", fontsize=7.5,
              title_fontsize=7.5, labelspacing=2.1, borderpad=1.4,
              handletextpad=1.2, frameon=True, framealpha=0.85)
    ax.set_ylabel("Est. words known (of top 5000)", fontsize=10)
    ax.set_xlabel("Month", fontsize=10)
    ax.grid(True, alpha=0.3)
    style_time_axis(ax)


def single_chart(d, letter, out_path):
    lang, _ = describe(d, letter)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Vocabulary Progression — Learner {letter} ({lang})",
                 fontsize=14, fontweight="bold")
    coverage_panel(ax1, d)
    ax1.set_title("Frequency Band Coverage", fontsize=11)
    vocab_panel(ax2, d)
    ax2.set_title("Vocabulary Growth with CEFR Level", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def comparison_chart(da, db, la, lb, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Monthly Vocabulary Progression: Two Learner Profiles",
                 fontsize=16, fontweight="bold")
    coverage_panel(axes[0, 0], da, small=True)
    axes[0, 0].set_title(f"Learner {la[0]}: {la[1]}", fontsize=11)
    vocab_panel(axes[1, 0], da)
    coverage_panel(axes[0, 1], db, small=True)
    axes[0, 1].set_title(f"Learner {lb[0]}: {lb[1]}", fontsize=11)
    vocab_panel(axes[1, 1], db)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    a_id = sys.argv[1] if len(sys.argv) > 1 else "L01"
    b_id = sys.argv[2] if len(sys.argv) > 2 else "L04"
    # Panel letters. The paper's §5.3 table is sorted ascending by reading volume,
    # so the two plotted profiles are its last two rows: C (left, moderate) and
    # D (right, heavy). Override as argv[3] argv[4] if the table order changes.
    letter_a = sys.argv[3] if len(sys.argv) > 3 else "C"
    letter_b = sys.argv[4] if len(sys.argv) > 4 else "D"
    os.makedirs(OUT_DIR, exist_ok=True)
    da, db = load(a_id), load(b_id)
    la = (letter_a, describe(da, a_id)[1])
    lb = (letter_b, describe(db, b_id)[1])

    lang_a = LANG_NAME.get(da["language"].iloc[0], "").lower()
    lang_b = LANG_NAME.get(db["language"].iloc[0], "").lower()
    single_chart(da, letter_a, os.path.join(OUT_DIR, f"progression_learner_{letter_a.lower()}_{lang_a}.png"))
    single_chart(db, letter_b, os.path.join(OUT_DIR, f"progression_learner_{letter_b.lower()}_{lang_b}.png"))
    comparison_chart(da, db, la, lb, os.path.join(OUT_DIR, "progression_comparison.png"))
    print("\nDone (real data from monthly_progression.csv).")


if __name__ == "__main__":
    main()
