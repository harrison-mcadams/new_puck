
import sys
import os
import pandas as pd
import numpy as np
import json
import logging

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def audit_season():
    print("Starting Season Audit...")
    
    # 1. Load API Data (CSV)
    csv_path = 'data/20252026/20252026.csv'
    if not os.path.exists(csv_path):
        csv_path = 'data/20252026.csv'
        
    print(f"Loading API CSV: {csv_path}")
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return

    # Filter for 5v5 generic (State='5v5' & NetEmpty=0)
    api_mask = (df['game_state'] == '5v5') & ((df['is_net_empty'] == 0) | (df['is_net_empty'].astype(str) == '0'))
    df_api_5v5 = df[api_mask].copy()
    
    # Attempt codes
    attempts = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    
    # Get all unique game IDs
    if df['game_id'].isnull().any():
        print("Warning: Null game_ids found in CSV")
    
    all_games = df.dropna(subset=['game_id'])['game_id'].unique()
    all_games = [int(g) for g in all_games if str(g).isdigit()]
    all_games.sort()
    
    print(f"Found {len(all_games)} games.")
    
    results = []
    
    count = 0
    for gid in all_games:
        count += 1
        if count % 20 == 0:
            print(f"Processed {count}/{len(all_games)}...", end='\r')
            
        # API Count
        df_g_api = df_api_5v5[df_api_5v5['game_id'] == gid]
        api_atts = df_g_api[df_g_api['event'].isin(attempts)]
        api_count = len(api_atts)
        
        # Shift Count
        # Bypassing timing.get_game_intervals_cached to ensure FRESH calculation with new heuristic
        
        cond = {'game_state': ['5v5'], 'is_net_empty': [0]}
        
        try:
            res = timing.compute_intervals_for_game(gid, cond, net_empty_mode='either')
            intervals = res.get('intersection_intervals', [])
            
            # Apply intervals to Raw Events
            # Need raw attempts from FULL df for this game
            df_g_raw = df[df['game_id'] == gid]
            df_g_raw_atts = df_g_raw[df_g_raw['event'].isin(attempts)]
            
            times = pd.to_numeric(df_g_raw_atts['total_time_elapsed_seconds'], errors='coerce').values
            
            shift_count = 0
            if intervals:
                mask = np.zeros(len(times), dtype=bool)
                for s, e in intervals:
                   mask |= ((times >= s) & (times <= e))
                shift_count = np.sum(mask)
            
            diff = api_count - shift_count
            
            home = 'UNK'
            away = 'UNK'
            try:
                if not df_g_raw.empty:
                     home = df_g_raw.iloc[0]['home_abb']
                     away = df_g_raw.iloc[0]['away_abb']
            except: pass


            results.append({
                'game_id': gid,
                'home': home,
                'away': away,
                'api_n': api_count,
                'shift_n': shift_count,
                'diff': diff,
                'abs_diff': abs(diff),
                'error_pct': abs(diff) / api_count if api_count > 0 else 0.0,
                'status': 'OK'
            })
            
        except Exception as e:
            # print(f"Error game {gid}: {e}")
            results.append({'game_id': gid, 'status': 'ERROR', 'error': str(e)})

    # Summary
    df_res = pd.DataFrame(results)
    
    # Save
    out_csv = 'audit_shift_stats.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nAudit Complete. Saved to {out_csv}")
    
    # Analysis
    if 'abs_diff' in df_res.columns:
        print("\n--- Summary Stats ---")
        print(df_res['abs_diff'].describe())
        
        print("\n--- Top 20 Outliers ({API} > {Shift} indicates Shift data data missing) ---")
        print(df_res.sort_values('abs_diff', ascending=False).head(20)[['game_id', 'home', 'away', 'api_n', 'shift_n', 'diff']])
        

if __name__ == '__main__':
    audit_season()
