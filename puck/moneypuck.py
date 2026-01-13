"""moneypuck.py

Module for interacting with MoneyPuck data (http://moneypuck.com/data.htm).
Handles downloading, caching, and merging MoneyPuck shot data with local predictions
for benchmarking and debugging.
"""

import pandas as pd
import numpy as np
import requests
import io
import zipfile
import logging
from pathlib import Path

# Configure Logging
logger = logging.getLogger(__name__)

def download_shots(year: str, data_dir: str = 'data/moneypuck') -> pd.DataFrame:
    """
    Downloads or loads cached MoneyPuck shots data for a given starting year.
    Example: year="2025" for 2025-2026 season.
    """
    base_dir = Path(data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = base_dir / f"shots_{year}.csv"
    
    if csv_path.exists():
        logger.info(f"Loading cached MoneyPuck data from {csv_path}...")
        return pd.read_csv(csv_path)
        
    url = f"http://peter-tanner.com/moneypuck/downloads/shots_{year}.zip"
    logger.info(f"Downloading MoneyPuck shots from {url}...")
    
    try:
        r = requests.get(url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        
        # Assume first file is the CSV
        filename = z.namelist()[0]
        logger.info(f"Extracting {filename}...")
        
        df = pd.read_csv(z.open(filename))
        
        # Save cache
        df.to_csv(csv_path, index=False)
        return df
        
    except Exception as e:
        logger.error(f"Failed to download MoneyPuck data: {e}")
        return pd.DataFrame()

def enrich_with_moneypuck(df_local: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches a local DataFrame (with 'game_id', 'period', 'xgs') with MoneyPuck xG data.
    automatically detects seasons needed based on local data.
    """
    if df_local.empty:
        return df_local
        
    # Detect seasons
    # GameID format: 2025020123 -> Season 2025
    if 'game_id' not in df_local.columns:
        logger.warning("No game_id in local data, cannot merge with MoneyPuck.")
        return df_local
        
    # Extract unique seasons (first 4 digits of game_id)
    # Ensure game_id is string or handle int
    game_ids = df_local['game_id'].astype(str)
    seasons = game_ids.str[:4].unique()
    
    df_mp_all = []
    for s in seasons:
        df_mp = download_shots(s)
        if not df_mp.empty:
            df_mp_all.append(df_mp)
            
    if not df_mp_all:
        logger.warning("No MoneyPuck data found/downloaded.")
        return df_local
        
    df_mp_combined = pd.concat(df_mp_all, ignore_index=True)
    return merge_predictions(df_local, df_mp_combined)

def merge_predictions(df_local: pd.DataFrame, df_mp: pd.DataFrame) -> pd.DataFrame:
    """
    Merges local predictions with MoneyPuck data using loose time matching (merge_asof).
    """
    logger.info("Merging local predictions with MoneyPuck data...")
    
    # 1. Prepare Local Data
    local_copy = df_local.copy()
    
    # Ensure Game Seconds Calculated
    # Local usually has 'time_elapsed_in_period_s' or 'period_seconds'
    if 'game_seconds_calc' not in local_copy.columns:
        if 'total_time_elapsed_s' in local_copy.columns:
            local_copy['game_seconds_calc'] = local_copy['total_time_elapsed_s']
        elif 'time_elapsed_in_period_s' in local_copy.columns and 'period' in local_copy.columns:
            local_copy['game_seconds_calc'] = (local_copy['period'] - 1) * 1200 + local_copy['time_elapsed_in_period_s']
        else:
            logger.warning("Could not calculate match time for local data. Skipping merge.")
            return df_local

    # Normalize Keys
    # Local GameID often lacks season if loaded from daily? Or is full? 
    # Usually Local: 2025020249. MP might be 20249 or 2025020249.
    
    # Check MP ID format
    # Initialize mp_copy early
    mp_copy = df_mp.copy()
    mp_sample = mp_copy['game_id'].iloc[0]
    local_sample = local_copy['game_id'].iloc[0]
    
    # Heuristic: If MP IDs are short (e.g. < 1,000,000) and Local are long, truncate local.
    # Standard NHL ID: 2025020249 (10 digits)
    # Short ID: 20249 (Game Number?) or 20001. 
    # Usually MP uses 20001 for game 1. 
    # 2025020001 % 1000000 = 20001.
    
    local_copy['join_game_id'] = local_copy['game_id'].astype(float)
    local_copy['join_period'] = local_copy['period'].astype(float)
    local_copy['join_sec'] = local_copy['game_seconds_calc'].astype(float)
    
    # Drop rows with invalid keys (cannot join)
    before_len = len(local_copy)
    local_copy.dropna(subset=['join_game_id', 'join_period', 'join_sec'], inplace=True)
    if len(local_copy) < before_len:
        logger.warning(f"Dropped {before_len - len(local_copy)} rows with missing join keys (game_id/period/time).")

    local_copy['join_game_id'] = local_copy['join_game_id'].astype(int)
    local_copy['join_period'] = local_copy['join_period'].astype(int)
    
    # Apply truncation if needed
    if mp_copy['game_id'].max() < 1000000 and local_copy['join_game_id'].max() > 1000000:
        logger.info("Truncating Local GameIDs to match MoneyPuck format (last 6 digits)...")
        local_copy['join_game_id'] = local_copy['join_game_id'] % 1000000
    
    # Prepare MP Keys
    mp_copy['join_game_id'] = mp_copy['game_id'].astype(int)
    mp_copy['join_period'] = mp_copy['period'].astype(int)
    mp_copy['join_sec'] = mp_copy['time'].astype(float)
    
    # 3. Sort for merge_asof
    local_copy = local_copy.sort_values('join_sec')
    mp_copy = mp_copy.sort_values('join_sec')
    
    # Create Event Mapping for Strict Matching
    # Local: 'shot-on-goal', 'missed-shot', 'goal', 'blocked-shot'
    # MP: 'SHOT', 'MISS', 'GOAL'
    
    def _map_event(e):
        e = str(e).lower().strip()
        if 'goal' == e: return 'GOAL'
        if 'shot' in e and 'on-goal' in e: return 'SHOT'
        if 'miss' in e: return 'MISS'
        return 'OTHER' # Blocks, etc.

    local_copy['join_event'] = local_copy['event'].apply(_map_event)
    
    # MP Mapping
    # MP Events are usually SHOT, MISS, GOAL
    mp_copy['join_event'] = mp_copy['event'].str.upper().str.strip()
    
    # 4. Merge
    # We want to keep all local rows, attaching MP info where found
    # Now matching on Game, Period, AND Event Type
    merged = pd.merge_asof(
        local_copy,
        mp_copy[['join_game_id', 'join_period', 'join_sec', 'join_event', 'xGoal', 'shotID', 'shotType', 'shotDistance']],
        on='join_sec',
        by=['join_game_id', 'join_period', 'join_event'],
        direction='nearest',
        tolerance=1.0, # Reduced to 1.0 second for precision
        suffixes=('', '_mp')
    )
    
    # Rename for clarity
    merged.rename(columns={
        'xGoal': 'mp_xGoal',
        'shotID': 'mp_shotID',
        'shotType': 'mp_shotType',
        'shotDistance': 'mp_shotDistance'
    }, inplace=True)
    
    # Check match rate
    n_matched = merged['mp_shotID'].notna().sum()
    pct_matched = 100 * n_matched / len(local_copy)
    logger.info(f"Merged {n_matched} events ({pct_matched:.1f}%) with MoneyPuck data.")
    
    # Drop temp cols
    cols_to_drop = ['join_game_id', 'join_period', 'join_sec', 'game_seconds_calc']
    merged.drop(columns=[c for c in cols_to_drop if c in merged.columns], inplace=True)
    
    return merged
