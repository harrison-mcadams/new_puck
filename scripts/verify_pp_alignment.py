import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing, fit_nested_xgs, fit_xgs

def main():
    # Use Game 2025020121 where COL (Home) was on PP (5v4) and Scored?
    # Or just check attempts
    gid = 2025020121
    print(f"--- Verifying Alignment for Game {gid} ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv('data/20252026.csv')
    except:
        return
        
    df = fit_nested_xgs.preprocess_features(df)
    
     # Ensure total_seconds
    if 'total_seconds' not in df.columns:
        def time_to_total_sec(row):
            try:
                m, s = row['period_time'].split(':')
                p_sec = int(m)*60 + int(s)
                return (row['period'] - 1) * 1200 + p_sec
            except: return 0
        df['total_seconds'] = df.apply(time_to_total_sec, axis=1)

    # Enrich Team Name
    def get_team_name(row):
        tid = row['team_id']
        hid = row['home_id']
        aid = row['away_id']
        try:
            if float(tid) == float(hid): return row['home_abb']
            if float(tid) == float(aid): return row['away_abb']
        except: pass
        return "UNKNOWN"

    if 'team_name' not in df.columns:
        df['team_name'] = df.apply(get_team_name, axis=1)
        
    g_df = df[df['game_id'] == gid].copy()
    home_abb = g_df['home_abb'].iloc[0] # COL
    away_abb = g_df['away_abb'].iloc[0] # CAR
    print(f"Home: {home_abb}, Away: {away_abb}")
    
    # We want to check:
    # 1. Global State 5v4 (Home Advantage) -> home_abb should get '5v4' stats
    # 2. Global State 4v5 (Away Advantage) -> away_abb should get '5v4' stats
    
    # Simulate generate_mixed_heatmaps logic
    grand_stats = {
        home_abb: {'5v4': {'attempts': 0}},
        away_abb: {'5v4': {'attempts': 0}}
    }
    
    # --- TEST 5v4 (Home/COL Advantage) ---
    print("\nTesting 5v4 (Home Advantage)...")
    state = '5v4'
    
    # Mapping Logic from script
    home_cond = None
    away_cond = None
    if state == '5v4':
        home_cond = '5v4'
        away_cond = '4v5'
    elif state == '4v5':
        home_cond = '4v5'
        away_cond = '5v4'
        
    print(f"Global State: {state}")
    print(f"  Home ({home_abb}) Condition: {home_cond}")
    print(f"  Away ({away_abb}) Condition: {away_cond}")
    
    # Fetch Intervals
    season = "20252026"
    cond = {'game_state': [state], 'is_net_empty': [0]}
    raw_intervals = timing.get_game_intervals_cached(gid, season, cond)
    
    # Filter
    valid_intervals = []
    for s, e in raw_intervals:
        dur = e - s
        if dur <= 0: continue
        events_in_window = g_df[(g_df['total_seconds'] >= s) & (g_df['total_seconds'] < e)]
        if not events_in_window.empty:
            n_5v5 = len(events_in_window[events_in_window['game_state'] == '5v5'])
            ratio = n_5v5 / len(events_in_window)
            if ratio > 0.5: continue
        valid_intervals.append((s, e))
        
    print(f"  Valid Intervals: {len(valid_intervals)}")
    
    # Count Events
    def in_valid_intervals(t):
        for s, e in valid_intervals:
            if s <= t < e: return True
        return False
    mask = g_df['total_seconds'].apply(in_valid_intervals)
    proc_df = g_df[mask].copy()
    
    home_events = proc_df[proc_df['team_name'] == home_abb]
    away_events = proc_df[proc_df['team_name'] == away_abb]
    
    print(f"  Home ({home_abb}) Events: {len(home_events)}")
    print(f"  Away ({away_abb}) Events: {len(away_events)}")
    
    # Accumulate
    if home_cond == '5v4':
        grand_stats[home_abb]['5v4']['attempts'] += len(home_events)
    if away_cond == '5v4':
        grand_stats[away_abb]['5v4']['attempts'] += len(away_events)
        
    print(f"  > Added {len(home_events)} attempts to {home_abb} [5v4] bucket? {'Yes' if home_cond=='5v4' else 'No'}")
    
    # --- RESULT ---
    print("\nFinal 5v4 Buckets:")
    print(f"{home_abb}: {grand_stats[home_abb]['5v4']['attempts']}")
    print(f"{away_abb}: {grand_stats[away_abb]['5v4']['attempts']}")
    
    if grand_stats[home_abb]['5v4']['attempts'] > 0 and grand_stats[away_abb]['5v4']['attempts'] == 0: # Assuming Away didn't have 4v5 events here
         print("\nSUCCESS: Home team correctly credited with 5v4 stats during 5v4 state.")
    else:
         print("\nWARNING: Check results.")

if __name__ == "__main__":
    main()
