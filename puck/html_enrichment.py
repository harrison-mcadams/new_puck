import re
import logging
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from puck import nhl_api

def mmss_to_seconds(time_str: str) -> int:
    """Extract the first MM:SS from a string and convert to seconds."""
    if not time_str:
        return 0
    # Match the first MM:SS pattern (handles merged strings like 15:264:34)
    m = re.search(r'(\d+):(\d+)', time_str)
    if m:
        mm = int(m.group(1))
        ss = int(m.group(2))
        return mm * 60 + ss
    return 0

def parse_html_pbp(html_text: str) -> List[Dict[str, Any]]:
    """Parse NHL HTML PBP report into a list of event dicts."""
    if not html_text:
        return []
    
    soup = BeautifulSoup(html_text, 'html.parser')
    all_rows = soup.find_all('tr')
    events = []
    current_period = 1
    
    for row in all_rows:
        tds = row.find_all('td')
        if len(tds) < 6:
            continue
            
        first_col = tds[0].get_text(strip=True)
        if not first_col.isdigit():
            continue
            
        event_num = int(first_col)
        
        # Period is in index 1
        try:
            current_period = int(tds[1].get_text(strip=True))
        except Exception:
            # Fallback to last known if missing
            pass
            
        # Time columns: tds[3] is Elapsed <br> Remaining
        raw_time = tds[3].get_text(" ", strip=True)
        elapsed_sec = mmss_to_seconds(raw_time)
        
        # Event code at index 4, Description at index 5
        event_code = tds[4].get_text(strip=True).upper()
        description = tds[5].get_text(strip=True)
        
        # Extract shot type from description
        shot_type = "Unknown"
        
        # 1. Blocked shot pattern
        # Handles: "...BLOCKED BY SMITH, Wrist, Def. Zone" or "...BLOCKED BY SMITH, Wrist"
        block_match = re.search(r'BLOCKED BY .*?,\s*([a-zA-Z\s\-]+)(?:,|$)', description, re.IGNORECASE)
        if block_match:
            shot_type = block_match.group(1).strip()
        else:
            # 2. General shot pattern: Event - ShotType
            # Handles: "Shot - Wrist, Off. Zone" or "Shot - Wrist"
            gen_match = re.search(r'-\s*([a-zA-Z\s\-]+)(?:,|$)', description)
            if gen_match:
                shot_type = gen_match.group(1).strip()
            else:
                 gen_match_simple = re.search(r'-\s*([a-zA-Z\s\-]+)', description)
                 if gen_match_simple:
                    candidate = gen_match_simple.group(1).strip()
                    if "Zone" not in candidate:
                        shot_type = candidate
        
        # Clean up shot type
        shot_type = shot_type.split(',')[0].strip() 

        events.append({
            'html_event_id': event_num,
            'period': current_period,
            'period_seconds': elapsed_sec,
            'event_code': event_code,
            'description': description,
            'shot_type': shot_type
        })
        
    return events

def enrich_blocks_with_html(api_df: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Enrich a DataFrame of events with shot types from HTML for blocked shots."""
    if api_df.empty:
        return api_df
        
    # Only try if we have blocked shots with MISSING or UNKNOWN shot types
    blocks_mask = (api_df['event'].str.lower() == 'blocked-shot')
    unknown_mask = api_df['shot_type'].isna() | (api_df['shot_type'].str.lower().isin(['unknown', 'none', '']))
    
    target_mask = blocks_mask & unknown_mask
    if not target_mask.any():
        return api_df
        
    # Fetch HTML
    try:
        html_text = nhl_api.get_pbp_from_nhl_html(game_id)
        if not html_text:
            return api_df
        html_events = parse_html_pbp(html_text)
    except Exception as e:
        logging.warning(f"Failed to fetch/parse HTML for {game_id}: {e}")
        return api_df
        
    if not html_events:
        return api_df
        
    html_df = pd.DataFrame(html_events)
    # Filter HTML for blocks
    html_blocks = html_df[html_df['event_code'] == 'BLOCK'].copy()
    
    # We need a time column in api_df for matching.
    # Typically 'periodTime_seconds_elapsed'
    t_col = 'time_elapsed_in_period_s'
    if t_col not in api_df.columns:
        if 'periodTime_seconds_elapsed' in api_df.columns:
            t_col = 'periodTime_seconds_elapsed'
        else:
            # Fallback to any time-like column
            time_cols = [c for c in api_df.columns if 'time' in c.lower()]
            if time_cols:
                t_col = time_cols[0]
            else:
                return api_df

    # Attempt to match
    updated_count = 0
    for idx, row in api_df[target_mask].iterrows():
        # Ensure row[t_col] is numeric
        api_time = row[t_col]
        if isinstance(api_time, str):
            api_time = mmss_to_seconds(api_time)
        try:
            api_time = float(api_time)
        except (ValueError, TypeError):
            continue

        # Match by period and time (exact or +/- 5s)
        matches = html_blocks[
            (html_blocks['period'] == row['period']) &
            (np.abs(html_blocks['period_seconds'] - api_time) <= 5)
        ]
        
        if not matches.empty:
            # If multiple, take closest time
            best = matches.assign(tdiff=np.abs(matches['period_seconds'] - api_time)).sort_values('tdiff').iloc[0]
            if best['shot_type'] != 'Unknown':
                # Update but keep original if it was somehow known (unlikely for blocks)
                api_df.at[idx, 'shot_type'] = best['shot_type'].lower()
                updated_count += 1
                
    if updated_count > 0:
        logging.info(f"Enriched {updated_count} blocked shots from HTML for game {game_id}")
        
    return api_df
