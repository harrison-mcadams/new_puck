import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing, fit_nested_xgs, fit_xgs

def main():
    season = "20252026"
    print(f"--- Verifying Phantom PP Intervals for {season} ---")
    
    # 1. Load Data
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

    # Rebound features
    if 'rebound_time_diff' not in df.columns:
         df['rebound_time_diff'] = 0.0
         df['rebound_angle_change'] = 0.0
         df['rebound_speed'] = 0.0
         df['rebound_dist_change'] = 0.0
    
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
    
    # Filter for COL
    team = "COL"
    games = df[(df['home_abb'] == team) | (df['away_abb'] == team)]['game_id'].unique()
    print(f"Checking {len(games)} games for {team}...")
    
    # Load Model
    model_path = Path(f"analysis/mixed_effects_heatmaps_{season}/models/mixed_model_5v4.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
    else:
        print("Model not found")
        return

    # Accumulators
    stats = {
        'all': {'seconds': 0, 'xg': 0, 'attempts': 0},
        'clean': {'seconds': 0, 'xg': 0, 'attempts': 0},
        'phantom': {'seconds': 0, 'xg': 0},
    }
    
    for gid in games:
        g_df_all = df[df['game_id'] == gid].copy()
        if g_df_all.empty: continue
        
        home_abb = g_df_all['home_abb'].iloc[0]
        away_abb = g_df_all['away_abb'].iloc[0]
        
        # COL PP State
        state = '5v4' if home_abb == team else '4v5'
        
        # Get Intervals
        cond = {'game_state': [state], 'is_net_empty': [0]}
        intervals = timing.get_game_intervals_cached(gid, season, cond)
        if not intervals: continue
        
        for s, e in intervals:
            dur = e - s
            if dur <= 0: continue
            
            # Check Events Consistency
            events_in_window = g_df_all[(g_df_all['total_seconds'] >= s) & (g_df_all['total_seconds'] < e)]
            
            # Consensus Check
            # If > 50% of events are 5v5, assume Phantom
            n_events = len(events_in_window)
            n_5v5 = len(events_in_window[events_in_window['game_state'] == '5v5'])
            
            is_phantom = False
            if n_events > 0:
                ratio_5v5 = n_5v5 / n_events
                if ratio_5v5 > 0.5:
                    is_phantom = True
            elif dur > 60:
                # Long interval with NO events? Suspicious but maybe no events.
                # But statistically unlikely during PP.
                pass
            
            # Calculate xG for interval (using interval-filtered events)
            mask_time = (g_df_all['total_seconds'] >= s) & (g_df_all['total_seconds'] < e)
            mask_valid = (g_df_all['is_net_empty'] == 0)
            col_events = g_df_all[mask_time & mask_valid & (g_df_all['team_name'] == team)].copy()
            
            xg = 0
            n = 0
            if not col_events.empty:
                col_events['opp_team_name'] = away_abb if home_abb == team else home_abb
                probs = model.predict_proba(col_events)[:, 1]
                xg = probs.sum()
                n = len(col_events)
                
            # Log to 'all'
            stats['all']['seconds'] += dur
            stats['all']['xg'] += xg
            stats['all']['attempts'] += n
            
            if is_phantom:
                stats['phantom']['seconds'] += dur
                stats['phantom']['xg'] += xg
                # print(f"  Phantom PP in {gid} ({dur:.1f}s): xG={xg:.2f}")
            else:
                stats['clean']['seconds'] += dur
                stats['clean']['xg'] += xg
                stats['clean']['attempts'] += n

    # Report
    def calc_rate(s):
        if s['seconds'] <= 0: return 0
        return (s['xg'] / s['seconds']) * 3600
        
    print("\n=== RESULTS ===")
    print(f"Total PP Time (Shift Chart): {stats['all']['seconds']:.1f}s")
    print(f"Total xG: {stats['all']['xg']:.2f}")
    print(f"Rate (All): {calc_rate(stats['all']):.2f} xG/60")
    
    print(f"\nPhantom Time Excluded: {stats['phantom']['seconds']:.1f}s ({stats['phantom']['seconds']/stats['all']['seconds']*100:.1f}%)")
    print(f"Phantom xG Included: {stats['phantom']['xg']:.2f}")
    
    print(f"\nClean PP Time: {stats['clean']['seconds']:.1f}s")
    print(f"Clean xG: {stats['clean']['xg']:.2f}")
    print(f"Rate (Clean): {calc_rate(stats['clean']):.2f} xG/60")

if __name__ == "__main__":
    main()
