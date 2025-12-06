# Raspberry Pi Deployment Guide

This document summarizes the steps we took to deploy the `new_puck` analysis system to a Raspberry Pi server.

## 1. Environment Setup

### Prerequisites
- Raspberry Pi (running Raspberry Pi OS / Debian)
- Python 3.x
- Git

### Initial Setup
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/harrison-mcadams/new_puck.git
    cd new_puck
    ```

2.  **Create Virtual Environment:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 2. Generating Historical Data (Backfill)

Since the Pi starts with no data, we need to download past seasons and train the initial xG model.

**Command:**
```bash
python3 scripts/backfill_seasons.py > logs/backfill.log 2>&1 &
```

**What this does:**
- Downloads data for all seasons defined in `SEASONS` list (starting from oldest).
- Trains the xG model incrementally after each season.
- Saves the trained model to `analysis/xgs/xg_model_rf.pkl`.

## 3. Daily Updates

To keep the data current, run the daily update script. This script fetches the latest games, updates player/team stats, and refreshes the plots.

**Command:**
```bash
python3 scripts/daily.py > logs/daily.log 2>&1 &
```

**Running for a specific past season:**
If you want to re-run the daily analysis logic for a specific season (e.g., 2023-2024), you can use the `--season` argument:
```bash
python3 scripts/daily.py --season 20232024 > logs/season_20232024.log 2>&1 &
```

**Note:** The script automatically handles `sys.path` to find the `puck` package.

## 4. Web-Based Monitoring

We implemented a monitoring system to watch long-running jobs (like backfills or daily updates) via the web interface.

### How it works:
- **Log Directory:** All logs should be written to the `logs/` directory.
- **Redirecting Output:** Use `> logs/your_job.log 2>&1 &` when running scripts.
- **Web Interface:** Click "Monitor" in the navigation bar to see active jobs.

### Viewing Logs:
- The dashboard lists all files in `logs/` sorted by recency.
- Clicking a log uses the `/monitor/view/<filename>` route.
- **Auto-Refresh:** Toggle the "Auto-Refresh" button to tail the log file in real-time.

## 5. Running the Web Server

To serve the analysis dashboard:

```bash
python3 app.py
```
*Note: Make sure to run this in a separate terminal session or use `nohup`/`systemd` to keep it running.*

## Troubleshooting Scenarios

### Missing `logs/` Directory
- **Issue:** `bash: logs/daily.log: No such file or directory`
- **Cause:** Git does not track empty directories.
- **Fix:** We added a `.keep` file to `logs/` to force git tracking.
- **Manual Fix:** `mkdir logs`

### Import Errors
- **Issue:** `ModuleNotFoundError: No module named 'parse'`
- **Cause:** Scripts run from `scripts/` couldn't find the `puck` package in the parent directory.
- **Fix:** Updated `daily.py` (and others) to append the project root to `sys.path`.

### Unexpected Keyword Arguments
- **Issue:** `TypeError: _season() got an unexpected keyword argument...`
- **Cause:** `scripts/daily.py` was using outdated arguments not supported by `puck.parse._season`.
- **Fix:** Removed specific deprecated arguments (`save_elaborated`, etc.) from the function call.
