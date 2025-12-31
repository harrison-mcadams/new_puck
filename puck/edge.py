"""NHL Edge Tracking Data Helpers.

This module provides tools to fetch and transform player/puck tracking data
from the internal 'sprites' API used by the Goal Visualizer.
"""

import logging
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from puck.nhl_api import SESSION

# Coordinate system constants derived from reverse-engineering
RINK_ORIGIN_X_RAW = 1200.0
RINK_ORIGIN_Y_RAW = 510.0
PIXELS_PER_FOOT = 12.0

def fetch_tracking_data(game_id: str, event_id: str, season: str) -> Optional[Dict[str, Any]]:
    """
    Fetches the tracking data for a specific event from the NHL Edge API.
    
    URL Pattern: https://wsr.nhle.com/sprites/{season}/{gameId}/ev{eventId}.json
    
    Returns:
        JSON dictionary if successful, None otherwise.
    """
    url = f"https://wsr.nhle.com/sprites/{season}/{game_id}/ev{event_id}.json"
    logging.info(f"Fetching tracking data from: {url}")
    
    # Add headers to mimic a browser to avoid 403
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nhl.com/",
        "Origin": "https://www.nhl.com"
    }

    try:
        resp = SESSION.get(url, headers=headers, timeout=10)
        if resp.status_code == 404:
            logging.warning(f"  -> Data not found (404) for {url}")
            return None
        elif resp.status_code == 403:
            logging.warning(f"  -> Forbidden (403) for {url}. Headers might be insufficient.")
            return None
        
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"  -> Error fetching data: {e}")
        return None

def transform_coordinates(x_raw: float, y_raw: float) -> Tuple[float, float]:
    """
    Transforms raw API coordinates to standard NHL rink coordinates (feet).
    
    The raw data appears to be in a pixel space where:
    - Center Ice is at approx (1200, 510)
    - Scale is approx 12 pixels per foot
    - Y-axis is inverted relative to standard cartesian plots
    """
    x = (x_raw - RINK_ORIGIN_X_RAW) / PIXELS_PER_FOOT
    y = -(y_raw - RINK_ORIGIN_Y_RAW) / PIXELS_PER_FOOT # Flip Y for correct orientation
    return x, y

def filter_data_to_goal_moment(df: pd.DataFrame, buffer_frames: int = 0) -> pd.DataFrame:
    """
    Trims the tracking data to end at the moment the puck enters the net.
    
    Logic: First frame where Puck X > 89 (or X < -89) AND |Y| <= 3.0.
    """
    if df.empty or 'x' not in df.columns or 'y' not in df.columns:
        return df

    # Normalize check (assuming feet). If pixels, this checks fails safe (as 89 pixels is mid-ice)
    # But usually this is called after coordinate transform.
    
    # 1. Isolate Puck
    # Try 'entity_type' == 'puck' or similar
    if 'entity_type' in df.columns:
        pucks = df[df['entity_type'] == 'puck']
    else:
        # Fallback or assume all is checked? No, usually mixed df.
        return df 
        
    if pucks.empty:
        return df

    # 2. Detect Goal Frame
    # Goal Line is 89ft. Net width is 6ft (+/- 3ft).
    # Use 3.0 as strict limit inside post.
    
    goals = pucks[
        (pucks['x'].abs() > 89.0) & 
        (pucks['y'].abs() <= 3.0)
    ]
    
    if goals.empty:
        return df
        
    start_goal_frame = goals['frame_idx'].min()
    end_frame = start_goal_frame + buffer_frames
    
    # 3. Filter
    return df[df['frame_idx'] <= end_frame].copy()
