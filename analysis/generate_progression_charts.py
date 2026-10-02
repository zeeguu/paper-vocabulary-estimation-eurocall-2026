#!/usr/bin/env python
"""
Generate monthly vocabulary-progression figures for the paper from REAL data.

Standalone: needs pandas + matplotlib (see requirements.txt), no database. Reads
../data/monthly_progression.csv (produced by extract/extract_progression.py) and
writes the figures to ../paper/figures/.

    python analysis/generate_progression_charts.py [learnerA] [learnerB]
    # defaults: L03 (Danish, high activity) and L11 (French, moderate activity)

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
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "data", "monthly_progression.csv")
OUT_DIR = os.path.join(HERE, "..", "paper", "figures")

LANG_NAME = {"da": "Danish", "fr": "French", "de": "German", "es": "Spanish",
             "nl": "Dutch", "en": "English"}


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


def coverage_panel(ax, d, small=False, legend=True):
    ms, size, lw = d["date"], (3 if small else 4), (1.5 if small else 2)
    # Top 5000 is the lower panel's band as a share: words known / 5000.
    top5000 = 100.0 * d["words_known_est"] / 5000
    # Lines only: filled areas stacked into a muddy purple and drew the eye.
    # Sloped lines, not steps: this panel is for comparing the bands' shapes, and
    # four staircases are hard to trace (the lower panel shows the flat months).
    # Bands that run at nearly the same height (Top 100 and Top 500 near 100%)
    # would merge, so each is nudged one point vertically -- display only, less
    # than a line's width.
    for col, color, marker, label, k in (
            ("top100_cov", "#9b59b6", "o", "Top 100", 1.5),
            ("top500_cov", "#3498db", "s", "Top 500", 0.5),
            ("top1000_cov", "#2ecc71", "^", "Top 1000", -0.5),
            (None, "#f39c12", "d", "Top 5000", -1.5)):
        ys = top5000 if col is None else d[col]
        shift = mtransforms.offset_copy(ax.transData, fig=ax.figure, y=k, units="points")
        ax.plot(ms, ys, marker + "-", color=color, markersize=size, linewidth=lw,
                label=label, transform=shift)
    ax.set_ylabel("Band coverage: % of band known\n(extrapolated)", fontsize=10)
    ax.set_ylim(0, 102)  # headroom so a line at 100% is not hidden by the frame
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(loc="lower right", fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # one faint line per month (the snapshot unit)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def dot_area(n):
    """Marker area for n articles read in a month (area grows linearly with n)."""
    return 4 + 1.2 * n


def vocab_panel(ax, d, legend=True):
    ms = d["date"]
    # Discovery vs acquisition: of the known stems actually observed, the share
    # looked up at some point before becoming known (plausible acquisition) is
    # drawn as the darker band at the bottom; the rest, never looked up, is
    # prior knowledge being discovered (or learned without any lookup). The
    # observed share is applied to the extrapolated estimate.
    share = (d["known_after_lookup"] / d["known_observed"].where(d["known_observed"] > 0)).fillna(0)
    acquired = d["words_known_est"] * share
    ax.fill_between(ms, acquired, d["words_known_est"], alpha=0.22, color="#e74c3c",
                    step="post", label="never looked up")
    ax.fill_between(ms, 0, acquired, alpha=0.75, color="#c0392b",
                    step="post", label="known after a lookup")
    ax.plot(ms, d["words_known_est"], "-", color="#e74c3c", linewidth=1.3,
            drawstyle="steps-post", zorder=2)
    # each dot's area is proportional to the number of articles read that month, so
    # reading bursts (and gaps) are visible directly on the growth curve.
    read = d["articles_cumulative"].diff().fillna(d["articles_cumulative"]).clip(lower=0)
    # Kept small: reading volume is context, not the message (a heavy reader
    # logs 250+ articles in a month, which at the old scale swamped the curve).
    ax.scatter(ms, d["words_known_est"], s=dot_area(read), color="#7f8c8d",
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
    # component legend (the two bands), kept while the size legend is added
    if legend:
        # top right, kept compact so it fits above a curve near the 5,000 ceiling
        bands = ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.85,
                          ncol=2, borderpad=0.3, handlelength=1.2, columnspacing=1.0)
        ax.add_artist(bands)
        # size legend: reference circles for articles read that month (dot area ∝ this)
        sizes = [ax.scatter([], [], s=dot_area(n), facecolor="#7f8c8d", edgecolor="white",
                            linewidth=0.4, label=str(n)) for n in (10, 50, 250)]
        ax.legend(handles=sizes, title="articles / month", loc="upper left", fontsize=7.5,
                  title_fontsize=7.5, labelspacing=2.1, borderpad=1.4,
                  handletextpad=1.2, frameon=True, framealpha=0.85)
    ax.set_ylabel("Top 5000 band: words known\n(absolute, extrapolated; CEFR marked)", fontsize=10)
    ax.set_ylim(0, 5000)  # same scale in both panels: the top-5000 ceiling
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())  # one faint line per month (the snapshot unit)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(True, which="major", alpha=0.35)
    ax.grid(True, which="minor", axis="x", alpha=0.12)


def single_chart(d, letter, out_path):
    lang, _ = describe(d, letter)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Vocabulary Estimate: Learner {letter} ({lang})",
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
    # Equal month width in both columns, so growth rates compare visually:
    # each column is as wide as the time span it covers (plus a month either side).
    pad = pd.DateOffset(months=1)
    spans = [(d["date"].min() - pad, d["date"].max() + pad) for d in (da, db)]
    widths = [(hi - lo).days for lo, hi in spans]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             gridspec_kw={"width_ratios": widths})
    # Legends only in the wider right column; the left one is the same key.
    coverage_panel(axes[0, 0], da, small=True, legend=False)
    axes[0, 0].set_title(f"Learner {la[0]}: {la[1]}", fontsize=11)
    vocab_panel(axes[1, 0], da, legend=False)
    coverage_panel(axes[0, 1], db, small=True)
    axes[0, 1].set_title(f"Learner {lb[0]}: {lb[1]}", fontsize=11)
    vocab_panel(axes[1, 1], db)
    # Both columns share the same y scales (0-100%, 0-5000), so the right
    # column drops its axis label and tick numbers rather than repeat them.
    for ax in axes[:, 1]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)
    for col, (lo, hi) in enumerate(spans):
        for ax in axes[:, col]:
            ax.set_xlim(lo, hi)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    # The paper's Learner C (French, 88 articles) and D (Dutch, 2,854 articles).
    a_id = sys.argv[1] if len(sys.argv) > 1 else "L07"
    b_id = sys.argv[2] if len(sys.argv) > 2 else "L01"
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
