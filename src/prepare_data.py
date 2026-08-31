"""
prepare_data.py
----------------
Cleaning, joining, and feature construction for HW4 (DS-270702).

Run this from the REPO ROOT (the folder that contains both `data/`
and `src/`), not from inside `src/`:

    python3 src/prepare_data.py

What it does:
  1. Loads the four raw JSON files fetch_data.py produced
     (air_quality_<loc>.json, weather_<loc>.json for each location).
  2. Joins air quality + weather on their hourly timestamps, per
     location, and reports exactly how many rows matched (Checkpoint C2).
  3. Checks that both endpoints' time axes agree, and that hours are
     evenly spaced with no silent gaps (Checkpoint C1).
  4. Aggregates hourly data to daily values (mean for most variables,
     sum for precipitation, circular mean for wind direction).
  5. Reports the percentage of missing values in every column, both
     before and after aggregation (Checkpoint C3).
  6. Builds a few extra features useful for the "hospital sensitive-
     group alert" angle: whether the day exceeds the Thai PCD
     "Moderate" threshold (37.5 ug/m3), how many days in a row that
     has been true (consecutive_bad_days), and next-day targets for
     the modelling step (Part C).
  7. Saves the combined daily table to data/processed/daily_combined.csv
     and writes a plain-text checkpoint summary to
     outputs/results/checkpoints_c1_c3.txt that you can paste straight
     into the report's Section 7.
"""

import json
import os

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# CONFIG -- keep this in sync with fetch_data.py
# --------------------------------------------------------------------------

LOCATIONS = ["chiang_mai", "chiang_rai"]

RAW_DIR = os.path.join("data", "raw")
PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_DIR = os.path.join("outputs", "results")

PCD_MODERATE_THRESHOLD = 37.5  # ug/m3, 24-hr PM2.5 -- Thailand PCD standard

AIR_QUALITY_VARS = ["pm2_5", "pm10", "carbon_monoxide", "dust"]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure",
]

# How each hourly column should be aggregated into a daily value.
DAILY_AGG = {
    "pm2_5": "mean",
    "pm10": "mean",
    "carbon_monoxide": "mean",
    "dust": "mean",
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean",
    "precipitation": "sum",
    "surface_pressure": "mean",
    # wind_direction_10m is handled separately (circular mean)
}

# --------------------------------------------------------------------------


def load_hourly_dataframe(path: str, value_cols: list) -> pd.DataFrame:
    """
    Load one Open-Meteo JSON response and return an hourly DataFrame
    indexed by local timestamp (the API returns local time already,
    because we requested timezone=Asia/Bangkok in fetch_data.py).
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    hourly = payload["hourly"]
    df = pd.DataFrame({col: hourly[col] for col in value_cols})
    df.index = pd.to_datetime(hourly["time"])
    df.index.name = "time"
    return df


def circular_mean_degrees(degrees: pd.Series) -> float:
    """Mean of a set of compass directions (0-360 deg), handling wraparound."""
    degrees = degrees.dropna()
    if degrees.empty:
        return np.nan
    radians = np.deg2rad(degrees)
    sin_mean = np.mean(np.sin(radians))
    cos_mean = np.mean(np.cos(radians))
    mean_deg = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    return (mean_deg + 360) % 360


def consecutive_true_streak(flag: pd.Series) -> pd.Series:
    """
    For a boolean series indexed by consecutive calendar days, return
    the number of consecutive True values ending at each row (0 if
    that row itself is False). Assumes the series has no missing days
    (call after asfreq('D')).
    """
    reset_points = (~flag.fillna(False)).cumsum()
    streak = flag.fillna(False).groupby(reset_points).cumcount() + 1
    return streak.where(flag.fillna(False), other=0)


def report_missing(df: pd.DataFrame, label: str, log_lines: list):
    log_lines.append(f"\nMissing values -- {label}")
    pct_missing = (df.isna().mean() * 100).round(2)
    for col, pct in pct_missing.items():
        log_lines.append(f"  {col}: {pct}% missing")
        if pct == 0.0:
            log_lines.append(
                f"    -> 0.00% missing for '{col}': check whether this is a "
                f"modelled variable with no gaps by construction, or whether "
                f"missing values were silently filled upstream."
            )


def process_location(location: str, log_lines: list) -> pd.DataFrame:
    log_lines.append(f"\n{'=' * 60}\nLocation: {location}\n{'=' * 60}")

    aq_path = os.path.join(RAW_DIR, f"air_quality_{location}.json")
    wx_path = os.path.join(RAW_DIR, f"weather_{location}.json")

    aq_hourly = load_hourly_dataframe(aq_path, AIR_QUALITY_VARS)
    wx_hourly = load_hourly_dataframe(wx_path, WEATHER_VARS)

    # ---- Checkpoint C1: time axis agreement ------------------------------
    log_lines.append("\n[C1 -- Time]")
    log_lines.append(f"  Air quality range: {aq_hourly.index.min()} to {aq_hourly.index.max()}")
    log_lines.append(f"  Weather range:      {wx_hourly.index.min()} to {wx_hourly.index.max()}")

    aq_gaps = aq_hourly.index.to_series().diff().value_counts()
    wx_gaps = wx_hourly.index.to_series().diff().value_counts()
    log_lines.append(f"  Air quality hourly step sizes seen: {dict(aq_gaps)}")
    log_lines.append(f"  Weather hourly step sizes seen:      {dict(wx_gaps)}")
    log_lines.append(
        "  Both series were requested with timezone=Asia/Bangkok, so "
        "timestamps above are already Thailand local time, not UTC."
    )

    only_in_aq = aq_hourly.index.difference(wx_hourly.index)
    only_in_wx = wx_hourly.index.difference(aq_hourly.index)
    log_lines.append(
        f"  Timestamps present in air quality but not weather: {len(only_in_aq)}"
    )
    log_lines.append(
        f"  Timestamps present in weather but not air quality: {len(only_in_wx)}"
    )

    # ---- Checkpoint C2: the join -------------------------------------------
    common_index = aq_hourly.index.intersection(wx_hourly.index)
    log_lines.append("\n[C2 -- The join]")
    log_lines.append(f"  Air quality hourly rows before join: {len(aq_hourly)}")
    log_lines.append(f"  Weather hourly rows before join:      {len(wx_hourly)}")
    log_lines.append(f"  Matched hourly rows after inner join: {len(common_index)}")
    log_lines.append(
        f"  Rows dropped from air quality: {len(aq_hourly) - len(common_index)} "
        f"(rows dropped from weather: {len(wx_hourly) - len(common_index)})"
    )

    joined_hourly = aq_hourly.loc[common_index].join(wx_hourly.loc[common_index])
    joined_hourly = joined_hourly.sort_index()

    # ---- Missing values, hourly (part of C3) -------------------------------
    report_missing(joined_hourly, f"{location}, hourly, after join", log_lines)

    # ---- Aggregate to daily -------------------------------------------------
    daily = joined_hourly.resample("D").agg(DAILY_AGG)
    daily["wind_direction_10m"] = (
        joined_hourly["wind_direction_10m"].resample("D").apply(circular_mean_degrees)
    )

    # Make sure every calendar day in range exists as a row, even if the
    # API returned nothing for it -- this turns "silent gaps" into
    # visible NaN rows so missingness is honest, not hidden.
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    missing_days = full_range.difference(daily.index)
    daily = daily.reindex(full_range)
    daily.index.name = "date"

    log_lines.append("\n[C3 -- Missing values, continued]")
    log_lines.append(f"  Calendar days in range: {len(full_range)}")
    log_lines.append(f"  Days with no data at all: {len(missing_days)}")
    report_missing(daily, f"{location}, daily, after aggregation", log_lines)

    # ---- Feature construction -------------------------------------------
    daily["location"] = location
    daily["exceeds_pcd_moderate"] = daily["pm2_5"] > PCD_MODERATE_THRESHOLD
    daily["consecutive_bad_days"] = consecutive_true_streak(daily["exceeds_pcd_moderate"])

    # Targets for Part C modelling. Shifted within this single location's
    # continuous daily series, so "tomorrow" always means a real calendar
    # day ahead, not the next available row in some other location.
    daily["target_next_day_pm2_5"] = daily["pm2_5"].shift(-1)
    daily["target_next_day_exceeds"] = daily["exceeds_pcd_moderate"].shift(-1)

    daily = daily.rename(columns={
        "pm2_5": "pm2_5_mean",
        "pm10": "pm10_mean",
        "carbon_monoxide": "carbon_monoxide_mean",
        "dust": "dust_mean",
        "temperature_2m": "temperature_mean",
        "relative_humidity_2m": "humidity_mean",
        "wind_speed_10m": "wind_speed_mean",
        "wind_direction_10m": "wind_direction_mean_deg",
        "precipitation": "precipitation_sum",
        "surface_pressure": "pressure_mean",
    })

    return daily.reset_index()


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_lines = ["Checkpoints C1-C3 -- generated by prepare_data.py"]
    all_daily = []

    for location in LOCATIONS:
        daily = process_location(location, log_lines)
        all_daily.append(daily)

    combined = pd.concat(all_daily, ignore_index=True)

    # ---- Checkpoint C4 groundwork: do the two locations actually differ? --
    log_lines.append(f"\n{'=' * 60}\n[C4 groundwork -- comparing places]\n{'=' * 60}")
    means_by_location = combined.groupby("location")["pm2_5_mean"].mean().round(2)
    log_lines.append(f"  Mean daily PM2.5 by location:\n{means_by_location.to_string()}")
    log_lines.append(
        "  Fill in the rest of C4 in analyse.py: a t-test or simple "
        "distribution plot comparing the two locations' PM2.5 values."
    )

    processed_path = os.path.join(PROCESSED_DIR, "daily_combined.csv")
    combined.to_csv(processed_path, index=False)
    log_lines.append(f"\nSaved combined daily table -> {processed_path}")
    log_lines.append(f"Final table shape: {combined.shape[0]} rows x {combined.shape[1]} columns")

    report_path = os.path.join(RESULTS_DIR, "checkpoints_c1_c3.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))
    print(f"\nCheckpoint notes written to {report_path}")


if __name__ == "__main__":
    main()
