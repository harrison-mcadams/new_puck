import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_nested_xgs, fit_xgs

def main():
    season = "20252026"
    print(f"--- Verifying CSV-Only xG Rate for {season} ---")
    
    print("Loading Data...")
    try:
        df = pd.read_csv('data/20252026.csv')
    except:
        print("Data not found")
        return
        
    df = fit_nested_xgs.preprocess_features(df)
    
    # Enrich BIOS
    try:
        df = fit_xgs.enrich_data_with_bios(df)
    except Exception as e:
        print(f"Warning: Bios enrichment failed: {e}")

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

    if 'rebound_time_diff' not in df.columns:
         df['rebound_time_diff'] = 0.0
         df['rebound_angle_change'] = 0.0
         df['rebound_speed'] = 0.0
         df['rebound_dist_change'] = 0.0
         
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
    
    # Filter for COL
    team = "COL"
    col_df = df[(df['home_abb'] == team) | (df['away_abb'] == team)].copy()
    print(f"Checking events for {team}...")
    
    # Load Model
    model_path = Path(f"analysis/mixed_effects_heatmaps_{season}/models/mixed_model_5v4.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        print("Model not found")
        return

    # Filter for PP Events (CSV Only)
    # 5v4 if Home=COL, 4v5 if Away=COL
    
    mask_pp = (
        ((col_df['home_abb'] == team) & (col_df['game_state'] == '5v4')) |
        ((col_df['away_abb'] == team) & (col_df['game_state'] == '4v5'))
    )
    mask_valid = (col_df['is_net_empty'] == 0)
    
    pp_df = col_df[mask_pp & mask_valid].copy()
    
    # Calculate Duration from delta_t
    # Sum of time_since_last_event for ALL events in the game that are PP?
    # No, that's not quite right. time_since_last_event covers the gap.
    # If the gap was PP, then yes.
    # But if state changed during gap?
    # Usually we assign state of the EVENT to the preceding gap.
    # This is an approximation.
    
    total_seconds = pp_df['time_since_last_event'].sum()
    
    # Calculate xG for COL attempts
    col_atts = pp_df[pp_df['team_name'] == team].copy()
    
    xg = 0
    if not col_atts.empty:
        col_atts['opp_team_name'] = col_atts.apply(lambda r: r['away_abb'] if r['home_abb'] == team else r['home_abb'], axis=1)
        probs = model.predict_proba(col_atts)[:, 1]
        xg = probs.sum()
        
    print("\n=== RESULTS (CSV ONLY) ===")
    print(f"Total PP Time (Sum Deltas): {total_seconds:.1f}s")
    print(f"Total xG: {xg:.2f}")
    if total_seconds > 0:
        rate = (xg / total_seconds) * 3600
        print(f"Rate: {rate:.2f} xG/60")
    else:
        print("Rate: 0.00")

if __name__ == "__main__":
    main()
