import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from joblib import Parallel, delayed
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing, fit_nested_xgs, fit_xgs, mixed_effects

def main():
    season = "20252026"
    data_path = Path("data/20252026.csv")
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        # Fallback to concatenate
        data_dir = Path("data/20252026")
        files = list(data_dir.glob("*.csv"))
        if not files:
            print("No data found!")
            return
        df = pd.concat([pd.read_csv(f) for f in files])
    
    # Preprocess
    df = fit_nested_xgs.preprocess_features(df)
    
    # Enrich BIOS (Handedness etc)
    df = fit_xgs.enrich_data_with_bios(df)
    
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
        
    df = df[df['team_name'] != "UNKNOWN"].copy()

    # Compute Context Features (Deltas)
    if 'time_since_last_event' not in df.columns:
        print("  Computing Context Features (Deltas)...")
        if 'period_seconds' not in df.columns and 'period_time' in df.columns:
            def time_to_sec(x):
                try:
                    m, s = x.split(':')
                    return int(m)*60 + int(s)
                except: return 0
            df['period_seconds'] = df['period_time'].apply(time_to_sec)
        df.sort_values(['game_id', 'period', 'period_seconds'], inplace=True)
        df['last_event_time'] = df.groupby('game_id')['period_seconds'].shift(1)
        df['last_event_period'] = df.groupby('game_id')['period'].shift(1)
        df['time_since_last_event'] = df['period_seconds'] - df['last_event_time']
        df.loc[df['period'] != df['last_event_period'], 'time_since_last_event'] = 0
        df['time_since_last_event'] = df['time_since_last_event'].fillna(0).clip(lower=0)
        
        if 'x_adj' in df.columns:
            df['last_x'] = df.groupby('game_id')['x_adj'].shift(1).fillna(0)
            df['last_y'] = df.groupby('game_id')['y_adj'].shift(1).fillna(0)
            df['dist_from_last_event'] = np.sqrt((df['x_adj'] - df['last_x'])**2 + (df['y_adj'] - df['last_y'])**2)
            df.loc[df['period'] != df['last_event_period'], 'dist_from_last_event'] = 0
            df['speed_from_last_event'] = df['dist_from_last_event'] / df['time_since_last_event']
            df['speed_from_last_event'] = df['speed_from_last_event'].replace([np.inf, -np.inf], 0).fillna(0)
            df['angle_change_last_event'] = 0.0

    # Rebound features
    if 'rebound_time_diff' not in df.columns:
         df['rebound_time_diff'] = 0.0
         df['rebound_angle_change'] = 0.0
         df['rebound_speed'] = 0.0
         df['rebound_dist_change'] = 0.0
    
    # Filter for COL games
    team = "COL"
    col_games = df[(df['home_abb'] == team) | (df['away_abb'] == team)]['game_id'].unique()
    print(f"Investigating {len(col_games)} games for {team}...")

    # Load trained model for PP if available
    model_path = Path(f"analysis/mixed_effects_heatmaps_{season}/models/mixed_model_5v4.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        print("Model not found, skipping xG check.")
        return

    me_features = [
        'x_adj', 'y_adj', 'distance', 'angle_deg',
        'time_since_last_event', 'angle_change_last_event', 
        'speed_from_last_event', 'dist_from_last_event',
        'rebound_angle_change', 'rebound_time_diff', 'rebound_speed', 'rebound_dist_change'
    ]
    me_features = [f for f in me_features if f in df.columns]

    total_seconds = 0
    total_xg = 0
    total_attempts = 0
    
    per_game_log = []

    # Ensure total_seconds
    if 'total_seconds' not in df.columns:
        def time_to_total_sec(row):
            try:
                m, s = row['period_time'].split(':')
                p_sec = int(m)*60 + int(s)
                return (row['period'] - 1) * 1200 + p_sec
            except: return 0
        df['total_seconds'] = df.apply(time_to_total_sec, axis=1)

    for gid in col_games:
        # Get Intervals for this game
        # We need both 5v4 and 4v5 intervals to check for COL PP
        
        # Get Game Info from one row
        g_df_all = df[df['game_id'] == gid].copy()
        if g_df_all.empty: continue
        
        home_abb = g_df_all['home_abb'].iloc[0]
        away_abb = g_df_all['away_abb'].iloc[0]
        
        # Iterate over potential PP states
        # 5v4 (Home Advantage) vs 4v5 (Away Advantage)
        # We only care if COL is the Advantage team.
        
        relevant_states = []
        if home_abb == team:
            relevant_states.append('5v4')
        elif away_abb == team:
            relevant_states.append('4v5')
            
        for state in relevant_states:
            # Fetch Shift Intervals
            # We want standard 5v4/4v5 with Goalie (is_net_empty=0) to match cache keys
            cond = {'game_state': [state], 'is_net_empty': [0]}
            intervals = timing.get_game_intervals_cached(gid, season, cond)
            
            if not intervals: 
                continue
                
            seconds = sum(e-s for s,e in intervals)
            
            # Filter events in window
            # We want events where total_seconds is in ANY interval
            # AND is_net_empty == 0
            
            def in_intervals(t):
                for s, e in intervals:
                    if s <= t < e: return True
                return False
                
            mask_time = g_df_all['total_seconds'].apply(in_intervals)
            # Standard filters
            mask_valid = (g_df_all['is_net_empty'] == 0)
            
            col_events = g_df_all[mask_time & mask_valid & (g_df_all['team_name'] == team)].copy()
            
            if col_events.empty:
                xg = 0
                n = 0
            else:
                # DualMixedEffectsXG needs opp_team_name
                col_events['opp_team_name'] = away_abb if home_abb == team else home_abb
                probs = model.predict_proba(col_events)[:, 1]
                xg = probs.sum()
                n = len(col_events)
            
            total_seconds += seconds
            total_xg += xg
            total_attempts += n
            per_game_log.append({
                'gid': gid,
                'state': state,
                'seconds': seconds,
                'xg': xg,
                'n': n,
                'rate60': (xg/seconds)*3600 if seconds > 0 else 0
            })

    print(f"\nSummary for {team} PP (5v4):")
    print(f"Total Time: {total_seconds/3600:.2f} hours ({total_seconds:.1f} seconds)")
    print(f"Total xG: {total_xg:.3f}")
    print(f"Total Attempts: {total_attempts}")
    print(f"Calculated xG/60: {(total_xg/total_seconds)*3600:.3f}")
    
    print("\nPer Game Details (PP Only):")
    for log in per_game_log[:20]:
        print(f"Game {log['gid']} ({log['state']}): {log['seconds']:>5.1f}s | xG: {log['xg']:>5.3f} | N: {log['n']:>2} | Rate: {log['rate60']:>6.3f}")

if __name__ == "__main__":
    main()
