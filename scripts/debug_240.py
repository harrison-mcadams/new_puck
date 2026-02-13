import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import timing, get_game_state

def main():
    gid = 2025020240
    print(f"--- Debugging Game {gid} ---")
    
    # 1. Get Shifts (Use private method that worked in CLI)
    print("Fetching shifts via timing._get_shifts_df...")
    df_shifts = timing._get_shifts_df(gid)
    
    if df_shifts is None or df_shifts.empty:
        print("!! Shifts Empty !!")
        return
    print(f"Shifts: {len(df_shifts)} rows")
    
    # 2. Compute Game State
    print("Computing Game State...")
    gs_df, _ = get_game_state.get_game_state(gid, return_df=True, df_shifts=df_shifts)
    if gs_df.empty:
        print("!! Game State Empty !!")
        return
        
    print("\nGame State Durations:")
    print(gs_df.groupby('label').apply(lambda x: (x['end'] - x['start']).sum()))
    
    intervals_4v5 = gs_df[gs_df['label'] == '4v5'][['start', 'end']].values
    print(f"\n4v5 Intervals (Count: {len(intervals_4v5)}):")
    # Print first few long ones
    long_ints = [i for i in intervals_4v5 if (i[1]-i[0]) > 20]
    for s, e in long_ints[:5]:
        print(f"  {s:.1f} - {e:.1f} ({e-s:.1f}s)")
        
    # 3. Check CSV Events
    print("\nLoading CSV Data...")
    try:
        df = pd.read_csv('data/20252026.csv')
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return
        
    g = df[df['game_id'] == gid].copy()
    print(f"Game Events: {len(g)} rows")
    
    # Ensure total_seconds
    if 'total_seconds' not in g.columns:
        def time_to_total_sec(row):
            try:
                m, s = row['period_time'].split(':')
                p_sec = int(m)*60 + int(s)
                return (row['period'] - 1) * 1200 + p_sec
            except: return 0
        g['total_seconds'] = g.apply(time_to_total_sec, axis=1)
        
    # 4. Cross-Reference
    print("\nChecking Mismatches (CSV vs Shift 4v5):")
    
    mismatch_count = 0
    total_4v5_csv_events = 0
    
    mislabelled_game_states = {}
    
    for s, e in intervals_4v5:
        # buffer? usually strictly within
        events_in_window = g[(g['total_seconds'] >= s) & (g['total_seconds'] < e)]
        
        for _, row in events_in_window.iterrows():
            total_4v5_csv_events += 1
            csv_state = row.get('game_state')
            if csv_state != '4v5':
                mismatch_count += 1
                mislabelled_game_states[csv_state] = mislabelled_game_states.get(csv_state, 0) + 1
                if mismatch_count <= 10:
                    print(f"  Time {row.get('period_time')} (Total: {row.get('total_seconds', 0):.1f}s): Shift=4v5, CSV={csv_state}, Event={row.get('event')}, Team={row.get('team_id')}")
                    
    print(f"\nSummary:")
    print(f"Total Events during Shift-4v5: {total_4v5_csv_events}")
    print(f"Mismatched States: {mismatch_count}")
    print(f"Breakdown of Mislabelled: {mislabelled_game_states}")

if __name__ == '__main__':
    main()
