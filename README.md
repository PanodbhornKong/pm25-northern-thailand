# PM2.5 in Northern Thailand — HW4 (DS-270702)

Analysis of PM2.5 air quality in Chiang Mai and Chiang Rai: data
acquisition, cleaning, exploratory analysis, a next-day exceedance /
regression model, and a comparison against ground-station readings.

## Requirements

- Python 3.10+
- Internet access (only needed when running `fetch_data.py` and
  `compare_ground_truth.py`, which call external APIs)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip3 install -r requirements.txt
```

## How to run everything, in order

All commands below are run from the **repository root** (this folder),
not from inside `src/`.

### 1. Fetch raw data

```bash
python3 src/fetch_data.py
```

Downloads hourly PM2.5/pollutant data (Open-Meteo Air Quality API) and
hourly weather data (Open-Meteo Archive/ERA5 API) for Chiang Mai and
Chiang Rai, 2023-01-01 to today. Also writes the static Thailand PCD
AQI breakpoint table. Everything is saved, unmodified, to `data/raw/`,
along with `data/raw/fetch_log.csv` recording exactly what was
requested and when.

Safe to re-run: it overwrites `data/raw/` with a fresh pull each time.

### 2. Clean, join, and build features

```bash
python3 src/prepare_data.py
```

Joins air quality + weather per location, aggregates to daily values,
and constructs the exceedance flag, the consecutive-bad-day streak
feature, and the next-day prediction targets. Writes:

- `data/processed/daily_combined.csv`
- `outputs/results/checkpoints_c1_c3.txt` (Checkpoints C1-C3)

### 3. Analyse and visualise

```bash
python3 src/analyse.py
```

Produces four figures answering the questions in Part B, and runs the
Checkpoint C4 statistical comparison between the two locations.
Writes:

- `outputs/figures/fig01_pm25_timeseries.png`
- `outputs/figures/fig02_exceedance_days_per_year.png`
- `outputs/figures/fig03_pm25_distribution_by_location.png`
- `outputs/figures/fig04_episode_length_distribution.png`
- `outputs/results/checkpoints_c4.txt`

### 4. Model

```bash
python3 src/model.py
```

Trains a classification model (will tomorrow exceed 37.5 ug/m3?) and a
regression model (tomorrow's PM2.5) against persistence and
majority-class/mean baselines, using a chronological train/test split
and date-based cross-validation. Writes:

- `outputs/results/metrics.json`
- `outputs/results/checkpoints_c5.txt`

### 5. Compare against ground-station data

```bash
python3 src/compare_ground_truth.py
```

Must be run **after** step 1, since it reuses the Open-Meteo files
already saved in `data/raw/` rather than re-fetching them. Fetches
Air4Thai's current reading (no history is available from this API),
matches it to the nearest station and the nearest hour of Open-Meteo
data, and reports the difference. Writes:

- `data/raw/air4thai_current.json`
- `outputs/results/checkpoints_c6.txt`

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── fetch_data.py            # Part A -- data acquisition
│   ├── prepare_data.py          # Part A -- cleaning, joining, features
│   ├── analyse.py                # Part B -- figures, C4
│   ├── model.py                  # Part C -- baseline, model, C5
│   └── compare_ground_truth.py  # C6 -- ground-truth comparison
├── data/
│   ├── raw/                      # exactly what each API returned
│   └── processed/                # daily_combined.csv
├── outputs/
│   ├── figures/                  # fig01-04.png
│   └── results/                  # metrics.json + checkpoint notes
└── report/
    └── report.pdf
```

## Notes on data sources

- **Open-Meteo Air Quality API** and **Open-Meteo Archive (ERA5) API** —
  free, no key required, modelled (not measured) values.
- **Thailand PCD AQI breakpoints** — a static reference table, not
  fetched live; sourced from the official PCD announcement (B.E. 2566).
- **Air4Thai** — ground-station readings, but the public endpoint only
  ever returns the *current* reading (no history), so
  `compare_ground_truth.py` produces a single snapshot comparison, not
  a time series. That endpoint is also known to redirect to an https
  URL with an incomplete certificate chain; the script handles this
  automatically (see the comments in `compare_ground_truth.py`).

## AI usage disclosure

See the final page of `report/report.pdf` for the completed AI
disclosure table required by Section 11 of the assignment.
