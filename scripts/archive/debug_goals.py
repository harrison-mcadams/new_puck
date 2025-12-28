
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import config

def debug_goals(season='20252026', player_id=8471817): # Reaves
    print(f"--- Debugging Goals for Player {player_id} ---")
    
    # 1. Load Data
    print("Loading Season Data...")
    df_data = timing.load_season_df(season)
    
    # 2. Get Shifts
    print("Loading Shifts...")
    # We need to iterate all games involving this player? 
    # That's slow. Let's pick the specific game we looked at: 2025010073
    game_id = 2025010073
    print(f"Focusing on Game {game_id}")
    
    df_game = df_data[df_data['game_id'] == int(game_id)].copy()
    if df_game.empty:
        print("Game not found in season data.")
        return
    print("Columns:", list(df_game.columns))

    # 3. Intervals
    # Manual Interval Logic (replicating process_daily_cache)
    
    # Get Player Shifts
    df_shifts = timing._get_shifts_df(int(game_id), season=season)
    p_shifts = df_shifts[df_shifts['player_id'] == int(player_id)]
    print(f"Found {len(p_shifts)} shifts.")
    
    # Get 5v5 Intervals
    # We can use the logic from analyze.py if we construct the 'intervals_obj'
    # Or just fetch raw 5v5 intervals via timing (which might calculate from scratch if cache missing)
    # get_game_intervals_cached writes to disk. 
    # Let's bypass disk and compute in memory using compute_intervals_for_game
    print("Computing 5v5 Intervals (No Cache)...")
    
    # Needs: game_id, condition, season
    # Returns: dict matching structure (DIRECT, not wrapped in per_game)
    intervals_5v5 = timing.compute_intervals_for_game(game_id, {'game_state': ['5v5'], 'is_net_empty': [0]}, season=season)
    
    # Extract
    # compute_intervals_for_game returns: {intersection_intervals: [], ...}
    g5v5_list = intervals_5v5.get('intersection_intervals', [])
    if not g5v5_list and 'intervals_per_condition' in intervals_5v5:
        # Fallback if intersection logic inside timing didn't trigger because only 1 condition
        # But compute_intervals_for_game usually handles this.
        # Let's check keys
        print("Keys:", intervals_5v5.keys())
        # Try getting from game_state
        ipc = intervals_5v5.get('intervals_per_condition', {})
        g5v5_list = ipc.get('game_state', [])
        
    print(f"Game 5v5 Intervals: {len(g5v5_list)} segments.")

    # Intersect with Shifts
    p_intervals = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
    
    # Intersection func
    def intersect(a, b):
        res = []
        i=j=0
        a=sorted(a); b=sorted(b)
        while i < len(a) and j < len(b):
            s1,e1=a[i]; s2,e2=b[j]
            s=max(s1,s2); e=min(e1,e2)
            if e > s: res.append((s,e))
            if e1 < e2: i+=1
            else: j+=1
        return res
    
    intervals_to_use = intersect(g5v5_list, p_intervals)
    print(f"Player 5v5 Intervals: {len(intervals_to_use)} segments.")
    
    # 4. Filter Events
    matched = []
    times = df_game['total_time_elapsed_seconds'].values
    for s,e in intervals_to_use:
        mask = (times >= s) & (times <= e)
        matched.append(df_game[mask])
        
    if matched:
        df_filtered = pd.concat(matched).drop_duplicates()
        goals = df_filtered[df_filtered['event'].str.lower() == 'goal']
        print(f"\nFound {len(goals)} GOALS:")
        for idx, row in goals.iterrows():
            desc = row.get('event_description') or row.get('secondary_type') or row.get('event')
            gs = row.get('game_state', 'UNK')
            ne = row.get('is_net_empty', 'UNK')
            print(f" - {row['period_time']} P{row['period']} ({row['total_time_elapsed_seconds']}s): {desc} \n   [TeamID: {row['team_id']}] [State: {gs}] [NetEmpty: {ne}]")
    else:
        print("No matched events.")

if __name__ == "__main__":
    debug_goals()
