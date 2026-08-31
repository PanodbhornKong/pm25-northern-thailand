"""
fetch_data.py
-------------
Data acquisition script for HW4 (DS-270702, PM2.5 in Northern Thailand).

What it does:
  1. Fetches hourly PM2.5 (and related pollutants) from the Open-Meteo
     Air Quality API for one or more locations.
  2. Fetches hourly weather variables from the Open-Meteo Archive
     (ERA5 reanalysis) API for the same locations and date range.
  3. Saves every raw JSON response exactly as returned, under
     data/raw/ -- untouched, per the assignment's "save raw before
     you touch it" rule.
  4. Builds a small static reference table of Thailand PCD AQI
     breakpoints for PM2.5 (used later for the sensitive-group
     threshold analysis) and saves it alongside the raw data.
  5. Writes a fetch_log.csv recording exactly what was pulled: source,
     endpoint, parameters, date range, row count, and retrieval date --
     required for the "Record what you fetched" checklist item.

Re-run behaviour:
  Running this script again will overwrite the files in data/raw/ with
  a fresh pull. Open-Meteo's archive data for past dates should not
  change, but very recent days can be revised as more observations
  come in -- this is noted in fetch_log.csv via the "retrieved_at"
  column so the report can discuss it if two runs ever disagree.

Usage:
  python src/fetch_data.py
"""

import csv
import json
import os
import time
from datetime import date, datetime, timezone

import requests

# --------------------------------------------------------------------------
# CONFIG -- edit this section for your own choice of location(s) and dates
# --------------------------------------------------------------------------

# At least one location is required. A second location lets you answer
# Part B's "does the problem look the same in different places?" and
# Checkpoint C4. Coordinates are city-centre approximations.
LOCATIONS = {
    "chiang_mai": {"latitude": 18.7883, "longitude": 98.9853},
    "chiang_rai": {"latitude": 19.9105, "longitude": 99.8406},
}

# Open-Meteo Air Quality history starts 2023-01-01 (earlier requests
# return an empty series, not an error -- confirmed in the lab sheet).
START_DATE = "2023-01-01"
END_DATE = date.today().isoformat()  # up to today; re-running extends the range

TIMEZONE = "Asia/Bangkok"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

AIR_QUALITY_HOURLY_VARS = "pm2_5,pm10,carbon_monoxide,dust"
WEATHER_HOURLY_VARS = (
    "temperature_2m,relative_humidity_2m,wind_speed_10m,"
    "wind_direction_10m,precipitation,surface_pressure"
)

RAW_DIR = os.path.join("data", "raw")
LOG_PATH = os.path.join(RAW_DIR, "fetch_log.csv")

# --------------------------------------------------------------------------


def ensure_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)


def fetch_json(url: str, params: dict) -> dict:
    """GET a URL with params and return parsed JSON. Raises on HTTP error."""
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def save_raw_json(data: dict, filename: str) -> str:
    """Write JSON exactly as received to data/raw/<filename>."""
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def count_hourly_rows(payload: dict) -> int:
    """Open-Meteo responses store hourly series under payload['hourly']['time']."""
    try:
        return len(payload["hourly"]["time"])
    except (KeyError, TypeError):
        return 0


def fetch_air_quality(location_name: str, lat: float, lon: float, log_rows: list):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": AIR_QUALITY_HOURLY_VARS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    payload = fetch_json(AIR_QUALITY_URL, params)
    filename = f"air_quality_{location_name}.json"
    path = save_raw_json(payload, filename)
    rows = count_hourly_rows(payload)
    print(f"[air quality] {location_name}: {rows} hourly rows -> {path}")

    log_rows.append({
        "source": "Open-Meteo Air Quality API",
        "endpoint": AIR_QUALITY_URL,
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "parameters": AIR_QUALITY_HOURLY_VARS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "num_rows": rows,
        "output_file": filename,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def fetch_weather(location_name: str, lat: float, lon: float, log_rows: list):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": WEATHER_HOURLY_VARS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": TIMEZONE,
    }
    payload = fetch_json(ARCHIVE_WEATHER_URL, params)
    filename = f"weather_{location_name}.json"
    path = save_raw_json(payload, filename)
    rows = count_hourly_rows(payload)
    print(f"[weather]      {location_name}: {rows} hourly rows -> {path}")

    log_rows.append({
        "source": "Open-Meteo Archive API (ERA5 reanalysis)",
        "endpoint": ARCHIVE_WEATHER_URL,
        "location": location_name,
        "latitude": lat,
        "longitude": lon,
        "parameters": WEATHER_HOURLY_VARS,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "num_rows": rows,
        "output_file": filename,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def write_pcd_aqi_breakpoints(log_rows: list):
    """
    Static reference table: Thailand PCD Air Quality Index breakpoints
    for PM2.5 (24-hour average, ug/m3).

    Source: air4thai.pcd.go.th/webV3/#/AQIInfo
            Announcement of the Pollution Control Department on the Air
            Quality Index of Thailand B.E. 2566 (2023), Royal Thai
            Government Gazette, Vol. 140, Special Issue 157.
    This table does not change day to day, so it is written once as a
    fixed CSV rather than fetched from an API.
    """
    breakpoints = [
        {"level": "Very good", "pm25_low": 0, "pm25_high": 15.0},
        {"level": "Good", "pm25_low": 15.1, "pm25_high": 25.0},
        {"level": "Moderate", "pm25_low": 25.1, "pm25_high": 37.5},
        {"level": "Beginning to have health effects", "pm25_low": 37.6, "pm25_high": 75.0},
        {"level": "Impact on health", "pm25_low": 75.1, "pm25_high": None},
    ]
    filename = "pcd_aqi_breakpoints.csv"
    path = os.path.join(RAW_DIR, filename)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["level", "pm25_low", "pm25_high"])
        writer.writeheader()
        writer.writerows(breakpoints)
    print(f"[reference]    PCD AQI breakpoints -> {path}")

    log_rows.append({
        "source": "Thailand Pollution Control Department (PCD) AQI announcement B.E. 2566",
        "endpoint": "air4thai.pcd.go.th/webV3/#/AQIInfo (static table, not an API call)",
        "location": "n/a",
        "latitude": "",
        "longitude": "",
        "parameters": "PM2.5 24-hr breakpoints",
        "start_date": "",
        "end_date": "",
        "num_rows": len(breakpoints),
        "output_file": filename,
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


def write_log(log_rows: list):
    fieldnames = [
        "source", "endpoint", "location", "latitude", "longitude",
        "parameters", "start_date", "end_date", "num_rows",
        "output_file", "retrieved_at",
    ]
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\nFetch log written to {LOG_PATH}")


def main():
    ensure_dirs()
    log_rows = []

    for location_name, coords in LOCATIONS.items():
        fetch_air_quality(location_name, coords["latitude"], coords["longitude"], log_rows)
        time.sleep(1)  # be polite to the free API
        fetch_weather(location_name, coords["latitude"], coords["longitude"], log_rows)
        time.sleep(1)

    write_pcd_aqi_breakpoints(log_rows)
    write_log(log_rows)


if __name__ == "__main__":
    main()
