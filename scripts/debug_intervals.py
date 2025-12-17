
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze

# Enable Logging
logging.basicConfig(level=logging.INFO)

def debug_intervals(game_id=2025010073, season='20252026'):
    print(f"--- Debugging Intervals for Game {game_id} ---")
    
    # 1. Load Data
    # timing uses load_season_df internally if needed, but compute_intervals_for_game loads it?
    # Actually, compute_intervals_for_game takes season arg and loads it.
    
    # 2. Call compute_intervals_for_game
    print("Computing 5v5 intervals...")
    cond = {'game_state': ['5v5']} # Simplified: Removed is_net_empty
    
    res = timing.compute_intervals_for_game(game_id, cond, season=season)
    
    print("\nResult Keys:", res.keys())
    
    if 'per_game' in res:
        gd = res['per_game'].get(int(game_id))
        if gd:
            print("\nGame Data Keys:", gd.keys())
            
            # Print intervals
            merged = gd.get('merged_intervals', [])
            print(f"Merged Intervals (5v5): {len(merged)}")
            if merged:
                print(f"Sample: {merged[:3]}...")
                total_dur = sum(e-s for s,e in merged)
                print(f"Total Duration: {total_dur:.2f}s ({total_dur/60:.1f} min)")
            else:
                print("No Merged Intervals found.")
                
            intersection = gd.get('intersection_intervals', [])
            print(f"Intersection Intervals: {len(intersection)}")
        else:
            print("Game ID not in per_game dict.")
    else:
        print("per_game key missing.")
        
    # 3. Check Shifts Raw
    print("\n--- Raw Shifts Check ---")
    shifts = timing.get_shifts_with_html_fallback(game_id)
    all_s = shifts.get('all_shifts', [])
    print(f"Total Shifts: {len(all_s)}")
    if not all_s:
        print("SHIFTS ARE EMPTY!")
        return
        
    # Check Skater Counts manually?
    # timing.compute_game_timing(..., debug=True)
    
    # Let's verify if '5v5' exists in output
    if res.get('intervals_per_condition'):
        print("\nIntervals Per Condition:")
        for k, v in res['intervals_per_condition'].items():
            dur = sum(e-s for s,e in v)
            print(f"  {k}: {len(v)} segments, {dur:.1f}s")
            
if __name__ == "__main__":
    debug_intervals()
