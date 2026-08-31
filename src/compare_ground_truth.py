"""
compare_ground_truth.py
------------------------
Checkpoint C6 (ground truth) for HW4 (DS-270702).

Run this from the REPO ROOT, AFTER fetch_data.py has already been run
at least once (this script reuses the Open-Meteo files already saved
in data/raw/ rather than re-fetching them):

    python3 src/compare_ground_truth.py

What it does:
  1. Fetches Air4Thai's current-reading endpoint (getNewAQI_JSON.php).
     This endpoint has NO history -- it only ever returns "right now" --
     so this comparison is necessarily a single snapshot in time, not a
     time series. Saves the raw response, unmodified, to data/raw/.
  2. Finds the Air4Thai station physically closest to each of our two
     study locations (Chiang Mai, Chiang Rai).
  3. Looks up the Open-Meteo hourly PM2.5 value for the SAME hour and
     location from the files fetch_data.py already produced.
  4. Reports the absolute and percentage difference between the
     measured (Air4Thai) and modelled (Open-Meteo) PM2.5 values, and
     writes Checkpoint C6's answer to outputs/results/checkpoints_c6.txt.

Known API quirks (from the lab sheet -- handled below):
  - Every value in the Air4Thai response is a STRING, including
    latitude/longitude and pollutant readings -- everything is cast
    to float explicitly.
  - Missing readings are encoded as -1, not null/None -- treated as
    missing rather than as a real (impossible) negative concentration.
  - The endpoint is plain http:// and is known to redirect to an
    https:// URL with an incomplete certificate chain. We first try
    the request normally; only if that specific SSL error occurs do we
    retry with verification disabled for this one, documented,
    government endpoint -- this is not a general practice for
    untrusted sites.
"""

import json
import math
import os
from datetime import datetime

import requests
import urllib3

AIR4THAI_URL = "http://air4thai.pcd.go.th/services/getNewAQI_JSON.php"
RAW_DIR = os.path.join("data", "raw")
RESULTS_DIR = os.path.join("outputs", "results")

LOCATIONS = {
    "chiang_mai": {"latitude": 18.7883, "longitude": 98.9853},
    "chiang_rai": {"latitude": 19.9105, "longitude": 99.8406},
}


def fetch_air4thai() -> dict:
    try:
        resp = requests.get(AIR4THAI_URL, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.SSLError:
        print("[warning] SSL error on the documented air4thai redirect -- "
              "retrying with certificate verification disabled for this "
              "one known government endpoint only.")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        resp = requests.get(AIR4THAI_URL, timeout=30, verify=False)
        resp.raise_for_status()
        return resp.json()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def safe_float(value, missing_value=-1.0):
    """Air4Thai sends every value as a string and encodes missing data as -1."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f == missing_value else f


def find_nearest_station(stations: list, target_lat: float, target_lon: float):
    best, best_dist = None, float("inf")
    for st in stations:
        lat = safe_float(st.get("lat"))
        lon = safe_float(st.get("long"))
        if lat is None or lon is None:
            continue
        dist = haversine_km(target_lat, target_lon, lat, lon)
        if dist < best_dist:
            best, best_dist = st, dist
    return best, best_dist


def extract_pm25(station: dict):
    """
    Air4Thai nests the latest reading under AQILast -> PM25 -> value.
    Wrapped defensively since this is an undocumented public API and
    the exact nesting has been known to vary.
    """
    try:
        last = station.get("AQILast", {})
        pm25_block = last.get("PM25", {})
        value = safe_float(pm25_block.get("value"))
        date = last.get("date")
        time = last.get("time")
        return value, date, time
    except AttributeError:
        return None, None, None


def load_openmeteo_hourly_pm25(location: str) -> dict:
    """Return {timestamp_str: pm2_5} from the file fetch_data.py already saved."""
    path = os.path.join(RAW_DIR, f"air_quality_{location}.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    times = payload["hourly"]["time"]
    values = payload["hourly"]["pm2_5"]
    return dict(zip(times, values))


def nearest_openmeteo_hour(hourly_dict: dict, target_dt: datetime):
    target_key = target_dt.strftime("%Y-%m-%dT%H:00")
    if target_key in hourly_dict:
        return target_key, hourly_dict[target_key]
    # fall back to the closest hour if the exact key isn't present
    closest_key = min(
        hourly_dict.keys(),
        key=lambda k: abs((datetime.fromisoformat(k) - target_dt).total_seconds()),
    )
    return closest_key, hourly_dict[closest_key]


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    payload = fetch_air4thai()
    raw_path = os.path.join(RAW_DIR, "air4thai_current.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved raw Air4Thai response -> {raw_path}")

    stations = payload.get("stations", [])
    print(f"Air4Thai returned {len(stations)} stations nationwide.")

    log_lines = ["Checkpoint C6 -- generated by compare_ground_truth.py"]
    log_lines.append(
        "NOTE: Air4Thai has no history endpoint, so this is a single "
        "snapshot-in-time comparison, not a time-series validation."
    )

    for location, coords in LOCATIONS.items():
        station, dist_km = find_nearest_station(stations, coords["latitude"], coords["longitude"])
        log_lines.append(f"\n{'=' * 60}\nLocation: {location}\n{'=' * 60}")

        if station is None:
            log_lines.append("  No usable Air4Thai station found (check the raw JSON structure).")
            continue

        station_name = station.get("nameEN") or station.get("nameTH") or station.get("stationID")
        log_lines.append(f"  Nearest Air4Thai station: {station_name} ({dist_km:.1f} km away)")

        pm25_measured, date_str, time_str = extract_pm25(station)
        if pm25_measured is None:
            log_lines.append("  Air4Thai's latest PM2.5 reading is missing (-1) or unparseable for this station.")
            continue
        log_lines.append(f"  Air4Thai measured PM2.5: {pm25_measured} ug/m3 at {date_str} {time_str} (local time)")

        try:
            target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            log_lines.append("  Could not parse Air4Thai's timestamp -- skipping the Open-Meteo comparison.")
            continue

        hourly = load_openmeteo_hourly_pm25(location)
        matched_key, pm25_modelled = nearest_openmeteo_hour(hourly, target_dt)
        log_lines.append(f"  Open-Meteo modelled PM2.5: {pm25_modelled} ug/m3 at {matched_key}")

        if pm25_modelled is not None:
            abs_diff = abs(pm25_measured - pm25_modelled)
            pct_diff = 100 * abs_diff / pm25_measured if pm25_measured else float("nan")
            log_lines.append(f"  Absolute difference: {abs_diff:.1f} ug/m3 ({pct_diff:.0f}% of the measured value)")
            if pct_diff > 30:
                log_lines.append(
                    "  This is a large discrepancy -- treat Open-Meteo's modelled values as directionally "
                    "useful (e.g. for detecting the shape of the burning season) rather than as a precise "
                    "substitute for a ground instrument reading at any single hour."
                )
            else:
                log_lines.append(
                    "  The two sources agree reasonably closely at this snapshot, which supports using "
                    "Open-Meteo's modelled values as a reasonable stand-in given Air4Thai's lack of history."
                )

    report_path = os.path.join(RESULTS_DIR, "checkpoints_c6.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print("\n".join(log_lines))
    print(f"\nC6 checkpoint notes written to {report_path}")


if __name__ == "__main__":
    main()
