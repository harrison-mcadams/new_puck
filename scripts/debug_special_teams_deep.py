
import sys
import os
import pandas as pd
import numpy as np

# Add path
sys.path.append(os.path.abspath('.'))

from puck import timing

def debug_timing_states():
    game_id = 2025020007
    print(f"--- Debugging Timing for Game {game_id} ---")
    
    # 1. Get Shifts
    df_shifts = timing._get_shifts_df(game_id)
    print(f"Shifts: {len(df_shifts)} rows")
    
    # 2. Compute Game State Intervals explicitly
    # We want to use the internal helper that `compute_game_timing` uses
    # It seems to be `_compute_game_state_intervals` or inside `compute_intervals_for_game`
    
    # Let's call compute_game_timing with NO condition to see what it builds internally?
    # Actually `compute_game_timing` takes a df_all events usually.
    # But `get_game_intervals_cached` uses `timing.compute_intervals_for_game`.
    
    # Let's call compute_intervals_for_game with a condition that matches everything?
    # No, let's look at what states are available.
    
    # timing.py doesn't seem to expose a "get all states" function easily, 
    # but `compute_intervals_for_game` calculates state intervals.
    # Let's look at `timing.py` source again or just try to match specific states.
    
    states_to_test = ['5v5', '5v4', '4v5', '4v4', '5v3', '3v5']
    
    for s in states_to_test:
        cond = {'game_state': [s], 'is_net_empty': [0]}
        intervals = timing.compute_intervals_for_game(game_id, cond)
        
        # intervals is a dict with 'intervals_per_condition' etc.
        # But `compute_intervals_for_game` in `timing.py` returns a dict structure?
        # Let's check `timing.py` signature.
        # It calls `get_game_intervals_cached` which calls `compute_intervals_for_game`.
        
        # Actually `get_game_intervals_cached` returns a LIST of intervals (merged).
        # So let's use that.
        
        ivs = timing.get_game_intervals_cached(game_id, '20252026', cond)
        total = sum(e-s for s,e in ivs)
        print(f"State {s}: {total:.1f}s")
        
    print("\n--- Testing Flip Logic ---")
    cond_home = {'game_state': ['5v4'], 'is_net_empty': [0]}
    ivs_home = timing.get_game_intervals_cached(game_id, '20252026', cond_home)
    print(f"Home PP (5v4): {sum(e-s for s,e in ivs_home):.1f}s")
    
    cond_away = {'game_state': ['4v5'], 'is_net_empty': [0]}
    ivs_away = timing.get_game_intervals_cached(game_id, '20252026', cond_away)
    print(f"Away PP (4v5): {sum(e-s for s,e in ivs_away):.1f}s")

if __name__ == "__main__":
    debug_timing_states()
