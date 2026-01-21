
import sys
import os
import pandas as pd
import numpy as np
import json

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from puck import timing

def run_debug():
    print("Loading CSV...")
    if os.path.exists('data/20252026/20252026.csv'):
        csv_path = 'data/20252026/20252026.csv'
    else:
        csv_path = 'data/20252026.csv'
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Check columns
    # print("Columns:", df.columns.tolist())

    # Filter for ANA 5v5 (API Method)
    # Using '5v5' state AND 'is_net_empty' == 0
    mask_api = (df['game_state'] == '5v5') & ((df['is_net_empty'] == 0) | (df['is_net_empty'].astype(str) == '0'))
    df_api = df[mask_api].copy()
    
    # Filter for ANA involved
    df_api = df_api[((df_api['home_abb'] == 'ANA') | (df_api['away_abb'] == 'ANA'))]
    
    # Get relevant counts
    attempt_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_api_atts = df_api[df_api['event'].isin(attempt_events)].copy()
    
    print(f"API-based ANA 5v5 Attempts: {len(df_api_atts)}")
    
    # --- Shift Method ---
    print("Computing Shift-based stats...")
    
    # Get unique ANA games
    ana_games = df_api['game_id'].unique()
    season = '20252026'
    condition = {'game_state': ['5v5']} 
    
    shift_attempts_count = 0
    discrepancy_log = []
    
    # Pre-filter main DF for raw events for these games (to apply intervals to)
    df_raw_games = df[df['game_id'].isin(ana_games)].copy()
    
    game_stats = []

    for gid in ana_games:
        # Get intervals
        intervals = timing.get_game_intervals_cached(gid, season, condition)
        
        # Get raw game events
        df_g = df_raw_games[df_raw_games['game_id'] == gid]
        attempt_mask_raw = df_g['event'].isin(attempt_events)
        df_g_atts = df_g[attempt_mask_raw].copy()
        
        # Apply Intervals
        times = df_g_atts['total_time_elapsed_seconds'].values
        
        if not intervals:
            shift_count = 0
        else:
            in_shift = np.zeros(len(df_g_atts), dtype=bool)
            for start, end in intervals:
                in_shift |= ((times >= start) & (times <= end))
            
            shift_subset = df_g_atts[in_shift]
            shift_count = len(shift_subset)
            
            # Identify Discrepancies
            api_g = df_api_atts[df_api_atts['game_id'] == gid]
            
            api_indices = set(api_g.index)
            shift_indices = set(shift_subset.index)
            
            api_only = api_indices - shift_indices
            shift_only = shift_indices - api_indices
            
            api_count = len(api_g)
            
            game_stats.append({
                'game_id': gid,
                'api_count': api_count,
                'shift_count': shift_count,
                'diff': api_count - shift_count
            })
            
            def log_row(idx, type_label):
                try:
                    row = df.loc[idx]
                    discrepancy_log.append({
                        'type': type_label,
                        'game_id': int(gid),
                        'time': row.get('period_time', 'N/A'),
                        'period': int(row.get('period', 0)) if pd.notnull(row.get('period')) else 0,
                        'event': row.get('event', 'N/A'),
                        'desc': row.get('description', row.get('event_description', 'N/A')),
                        'game_state_api': row.get('game_state', 'N/A'),
                        'is_net_empty': int(row.get('is_net_empty', -1)) if pd.notnull(row.get('is_net_empty')) else -1,
                        'total_time': float(row.get('total_time_elapsed_seconds', 0.0))
                    })
                except Exception as e:
                    print(f"Error log row {idx}: {e}")

            if api_only:
                for idx in list(api_only)[:3]:
                    log_row(idx, 'API_only (Excluded by Shifts)')
            if shift_only:
                 for idx in list(shift_only)[:3]:
                    log_row(idx, 'Shift_only (Excluded by API Labels)')

            shift_attempts_count += shift_count

    print(f"Shift-based ANA 5v5 Attempts: {shift_attempts_count}")
    print(f"Difference: {len(df_api_atts) - shift_attempts_count}")
    
    print("\n--- Top Discrepancy Games (API - Shift) ---")
    g_df = pd.DataFrame(game_stats)
    if not g_df.empty:
        g_df['abs_diff'] = g_df['diff'].abs()
        print(g_df.sort_values('abs_diff', ascending=False).head(10))
    
    print("\n--- Sample Discrepancies ---")
    print(json.dumps(discrepancy_log[:15], indent=2))

if __name__ == '__main__':
    run_debug()
