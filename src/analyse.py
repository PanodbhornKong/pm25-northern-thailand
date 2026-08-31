"""
analyse.py
----------
Part B (describe the problem with evidence) for HW4 (DS-270702).

Run this from the REPO ROOT:

    python3 src/analyse.py

What it does:
  1. Loads data/processed/daily_combined.csv (built by prepare_data.py).
  2. Produces four figures, each answering one of the questions listed
     in Section 3, Part B of the lab sheet:
       fig01 -- daily PM2.5 over time, both locations, with the PCD
                "Moderate" threshold marked (when does the season
                start/end, does it shift year to year?)
       fig02 -- exceedance days per year, by location (is the problem
                getting better or worse over time?)
       fig03 -- distribution of daily PM2.5 by location, side by side
                (does the problem look the same in different places?)
       fig04 -- distribution of "bad air episode" lengths (how many
                days in a row does a bad spell typically last? -- ties
                directly into the hospital sensitive-group angle)
  3. Runs a Welch's t-test comparing Chiang Mai vs Chiang Rai daily
     PM2.5 and writes the result to outputs/results/checkpoints_c4.txt
     -- this finishes Checkpoint C4, which prepare_data.py only set up.
  4. Prints a suggested one-line caption for each figure so you can
     paste it straight into the report (the lab sheet requires every
     figure to have one).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------

PROCESSED_PATH = os.path.join("data", "processed", "daily_combined.csv")
FIGURES_DIR = os.path.join("outputs", "figures")
RESULTS_DIR = os.path.join("outputs", "results")

PCD_MODERATE_THRESHOLD = 37.5  # ug/m3, matches prepare_data.py

LOCATION_LABELS = {"chiang_mai": "Chiang Mai", "chiang_rai": "Chiang Rai"}
LOCATION_COLORS = {"chiang_mai": "#d95f02", "chiang_rai": "#1b9e77"}

# --------------------------------------------------------------------------


def load_data() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["date"])
    df["year"] = df["date"].dt.year
    return df


def fig01_timeseries(df: pd.DataFrame, captions: list):
    fig, ax = plt.subplots(figsize=(11, 5))
    for loc, group in df.groupby("location"):
        ax.plot(
            group["date"], group["pm2_5_mean"],
            label=LOCATION_LABELS.get(loc, loc),
            color=LOCATION_COLORS.get(loc), linewidth=0.8,
        )
    ax.axhline(
        PCD_MODERATE_THRESHOLD, color="red", linestyle="--", linewidth=1,
        label=f"PCD Moderate threshold ({PCD_MODERATE_THRESHOLD} ug/m3)",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily mean PM2.5 (ug/m3)")
    ax.set_title("Daily PM2.5 over time, Chiang Mai vs Chiang Rai (2023-2026)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig01_pm25_timeseries.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig01] saved -> {path}")
    captions.append(
        "Fig01: PM2.5 spikes sharply above the PCD threshold every "
        "January-April burning season in both provinces, then falls back "
        "to low levels the rest of the year."
    )


def fig02_exceedance_per_year(df: pd.DataFrame, captions: list):
    yearly = (
        df.groupby(["year", "location"])["exceeds_pcd_moderate"]
        .sum()
        .reset_index()
    )
    years = sorted(yearly["year"].unique())
    locations = sorted(yearly["location"].unique())
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, loc in enumerate(locations):
        subset = yearly[yearly["location"] == loc].set_index("year").reindex(years)
        offset = (i - (len(locations) - 1) / 2) * width
        ax.bar(
            [y + offset for y in years], subset["exceeds_pcd_moderate"],
            width=width, label=LOCATION_LABELS.get(loc, loc),
            color=LOCATION_COLORS.get(loc),
        )
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Days exceeding {PCD_MODERATE_THRESHOLD} ug/m3")
    ax.set_title("Number of exceedance days per year, by location")
    ax.set_xticks(years)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig02_exceedance_days_per_year.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig02] saved -> {path}")
    captions.append(
        "Fig02: [FILL IN AFTER LOOKING AT THE PLOT] state whether exceedance "
        "days are trending up, down, or flat across the years shown -- note "
        "that the current year may be a partial year, not a full one."
    )


def fig03_distribution_by_location(df: pd.DataFrame, captions: list, log_lines: list):
    locations = sorted(df["location"].unique())
    data_by_loc = [df.loc[df["location"] == loc, "pm2_5_mean"].dropna() for loc in locations]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(data_by_loc, tick_labels=[LOCATION_LABELS.get(l, l) for l in locations])
    ax.set_ylabel("Daily mean PM2.5 (ug/m3)")
    ax.set_title("Distribution of daily PM2.5 by location")
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig03_pm25_distribution_by_location.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig03] saved -> {path}")

    # ---- Checkpoint C4: do the two locations actually differ? -------------
    group_a, group_b = data_by_loc[0], data_by_loc[1]
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)  # Welch's t-test

    log_lines.append("Checkpoint C4 -- comparing places")
    log_lines.append(f"  {LOCATION_LABELS[locations[0]]}: mean={group_a.mean():.2f}, "
                      f"sd={group_a.std():.2f}, n={len(group_a)}")
    log_lines.append(f"  {LOCATION_LABELS[locations[1]]}: mean={group_b.mean():.2f}, "
                      f"sd={group_b.std():.2f}, n={len(group_b)}")
    log_lines.append(f"  Welch's t-test: t={t_stat:.3f}, p={p_value:.6f}")
    if p_value < 0.05:
        log_lines.append(
            "  p < 0.05: the difference in mean daily PM2.5 between the two "
            "locations is statistically significant at the 5% level."
        )
    else:
        log_lines.append(
            "  p >= 0.05: we cannot reject the hypothesis that the two "
            "locations have the same mean daily PM2.5 at the 5% level."
        )

    captions.append(
        f"Fig03: {LOCATION_LABELS[locations[0]]} and {LOCATION_LABELS[locations[1]]} "
        f"show [SIMILAR / DIFFERENT -- see checkpoints_c4.txt for the t-test result] "
        f"PM2.5 distributions (Welch's t-test, p={p_value:.4f})."
    )


def episode_lengths(df_location: pd.DataFrame) -> list:
    """
    Collapse the day-by-day 'consecutive_bad_days' streak counter into one
    number per bad-air episode: the streak value on the LAST day of each
    run of exceedance days. Assumes df_location is sorted by date and
    covers one location only.
    """
    exceeds = df_location["exceeds_pcd_moderate"].to_numpy()
    streak = df_location["consecutive_bad_days"].to_numpy()
    n = len(exceeds)
    lengths = []
    for i in range(n):
        is_end_of_run = exceeds[i] and (i == n - 1 or not exceeds[i + 1])
        if is_end_of_run:
            lengths.append(streak[i])
    return lengths


def fig04_episode_lengths(df: pd.DataFrame, captions: list, log_lines: list):
    fig, ax = plt.subplots(figsize=(7, 5))
    all_lengths = {}
    for loc, group in df.sort_values("date").groupby("location"):
        lengths = episode_lengths(group)
        all_lengths[loc] = lengths
        ax.hist(
            lengths, bins=range(1, max(lengths, default=1) + 2),
            alpha=0.6, label=LOCATION_LABELS.get(loc, loc),
            color=LOCATION_COLORS.get(loc),
        )
    ax.set_xlabel("Length of a bad-air episode (consecutive days above threshold)")
    ax.set_ylabel("Number of episodes")
    ax.set_title("How long do bad-air episodes typically last?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(FIGURES_DIR, "fig04_episode_length_distribution.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[fig04] saved -> {path}")

    log_lines.append("\nBad-air episode lengths (supports the Part D angle)")
    for loc, lengths in all_lengths.items():
        if lengths:
            log_lines.append(
                f"  {LOCATION_LABELS.get(loc, loc)}: {len(lengths)} episodes, "
                f"longest={max(lengths)} days, mean length={sum(lengths)/len(lengths):.1f} days"
            )
        else:
            log_lines.append(f"  {LOCATION_LABELS.get(loc, loc)}: no exceedance episodes found")

    captions.append(
        "Fig04: [FILL IN] most bad-air episodes last around N days, but a "
        "smaller number stretch much longer -- this is the basis for "
        "recommending a multi-day threshold rather than a single-day one."
    )


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df = load_data()
    captions = []
    log_lines = ["Checkpoint C4 and figure notes -- generated by analyse.py"]

    fig01_timeseries(df, captions)
    fig02_exceedance_per_year(df, captions)
    fig03_distribution_by_location(df, captions, log_lines)
    fig04_episode_lengths(df, captions, log_lines)

    report_path = os.path.join(RESULTS_DIR, "checkpoints_c4.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nC4 checkpoint notes written to {report_path}")

    print("\nSuggested captions (edit the [FILL IN] parts after looking at the plots):")
    for cap in captions:
        print(f"  - {cap}")


if __name__ == "__main__":
    main()
