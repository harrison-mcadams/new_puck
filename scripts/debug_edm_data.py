
import pandas as pd
import numpy as np
import sys
import os
import joblib
from pathlib import Path

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing

def main():
    print("Loading Data...")
    df = pd.read_csv("data/20252026.csv")
    print("Columns:", list(df.columns))
    
    # Precompute time
    def time_to_sec(x):
        try:
            m, s = x.split(':')
            return int(m)*60 + int(s)
        except: return 0
    df['period_seconds'] = df['period_time'].apply(time_to_sec)
    df['total_seconds'] = (df['period'] - 1) * 1200 + df['period_seconds']
    
    # Filter EDM games
    edm_games = df[(df['home_abb'] == 'EDM') | (df['away_abb'] == 'EDM')]['game_id'].unique()
    print(f"Found {len(edm_games)} EDM games.")
    
    total_raw_duration = 0
    total_valid_duration = 0
    total_events_in_raw = 0
    total_events_in_valid = 0
    discarded_intervals = 0
    
    # Sample 5 games to debug in detail
    sample_games = edm_games[:5]
    
    print("Columns:", df.columns.tolist())
    
    # Load Model
    import joblib
    model_path = Path("analysis/mixed_effects_heatmaps_20252026/models/mixed_model_5v4.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        print("Model loaded.")
    else:
        model = None
        print("Model NOT found.")

    # Pre-compute opp_team_name for model
    if 'opp_team_name' not in df.columns and 'team_name' in df.columns:
         df['opp_team_name'] = np.where(df['team_name'] == df['home_abb'], df['away_abb'], df['home_abb'])

    for gid in edm_games:
        # Get raw 5v4 intervals
        g_df = df[df['game_id'] == gid].copy()
        if g_df.empty: continue
        
        home = g_df['home_abb'].iloc[0]
        away = g_df['away_abb'].iloc[0]
        
        # Determine 5v4 perspective
        # If EDM is Home, they are 5v4 if state='5v4'.
        # If EDM is Away, they are 5v4 if state='4v5'.
        target_state = '5v4' if home == 'EDM' else '4v5'
            
        cond = {'game_state': [target_state], 'is_net_empty': [0]}
        try:
            raw_intervals = timing.get_game_intervals_cached(gid, "20252026", cond)
        except:
            continue
        
        for s, e in raw_intervals:
            dur = e - s
            if dur <= 0: continue
            
            total_raw_duration += dur
            
            # Events in window
            # Use Time Mask
            mask = (g_df['total_seconds'] >= s) & (g_df['total_seconds'] < e)
            events = g_df[mask].copy()
            
            total_events_in_raw += len(events)
            
            # Check Consensus Logic
            n_5v5 = len(events[events['game_state'] == '5v5'])
            total = len(events)
            
            is_valid = True
            ratio = 0
            if total > 0:
                ratio = n_5v5 / total
                if ratio > 0.5:
                    is_valid = False
                    discarded_intervals += 1
            
            if is_valid:
                total_valid_duration += dur
                total_events_in_valid += len(events)
            else:
                # Analyze xG of discarded
                try:
                    if model and not events.empty:
                        # Check skater counts if available
                        cols_to_show = ['period_time', 'game_state', 'event', 'team_name']
                        if 'home_skaters' in g_df.columns: cols_to_show.append('home_skaters')
                        if 'away_skaters' in g_df.columns: cols_to_show.append('away_skaters')
                        
                        mislabeled = events[events['game_state'] == '5v5']
                        if not mislabeled.empty:
                            print(f"Game {gid} Interval {s}-{e}: Found {len(mislabeled)} '5v5' events in 5v4 time.")
                            print(mislabeled[cols_to_show].head(3))

                        # Filter for EDM shots
                        edm_events = events[events['team_name'] == 'EDM'].copy()
                        if not edm_events.empty:
                            try:
                                probs = model.predict_proba(edm_events)[:, 1]
                                xg_sum = probs.sum()
                                xg_rate = (xg_sum / dur) * 3600
                                # ...
                            except Exception as e:
                                pass 
                except Exception as e:
                    pass
            
    print("-" * 30)
    print(f"EDM PP Analysis ({len(edm_games)} games)")
    print(f"Total Raw Duration: {total_raw_duration:.1f} sec ({total_raw_duration/60:.1f} min)")
    # Check Penalties
    total_penalties = 0
    for gid in edm_games:
         g_df = df[df['game_id'] == gid]
         home = g_df['home_abb'].iloc[0]
         # EDM drawn penalties = Opponent penalties
         opp = g_df['away_abb'].iloc[0] if home == 'EDM' else g_df['home_abb'].iloc[0]
         
         # Assuming 'penalty' event. Check column 'team_name' or 'team_id_for'
         # We need to filter penalties AGAINST the opponent (drawn by EDM)
         # Or just count ALL penalties to see if they exist.
         
         pens = g_df[g_df['event'] == 'penalty']
         # Naive: Count all penalties
         total_penalties += len(pens)

    print(f"Total Penalties (Raw, Both Teams): {total_penalties}")
    print(f"Naive PP Duration (Est. 3 per game * 2 min): {len(edm_games)*3*2} mins")
    print(f"Total Raw Duration: {total_raw_duration:.1f} sec ({total_raw_duration/60:.1f} min)")
    print(f"Total Valid Duration: {total_valid_duration:.1f} sec ({total_valid_duration/60:.1f} min)")
    print(f"Discarded Intervals: {discarded_intervals}")
    print(f"Events in Raw: {total_events_in_raw}")
    print(f"Events in Valid: {total_events_in_valid}")
    
if __name__ == "__main__":
    main()
