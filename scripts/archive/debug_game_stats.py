
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import nhl_api, parse, analyze, timing

def debug_game():
    game_id = '2025020510'
    print(f"Debugging Game {game_id}...")
    
    # 1. Fetch Data
    feed = nhl_api.get_game_feed(game_id)
    df = parse._game(feed)
    df = parse._elaborate(df)
    
    print(f"Loaded {len(df)} events.")
    
    # 2. Inspect Teams
    home_id = df['home_id'].iloc[0]
    away_id = df['away_id'].iloc[0]
    home_abb = df['home_abb'].iloc[0]
    away_abb = df['away_abb'].iloc[0]
    
    print(f"Home: {home_abb} ({home_id})")
    print(f"Away: {away_abb} ({away_id})")
    
    # 3. Check Goals
    goals = df[df['event'] == 'goal']
    print("\nGoals in DataFrame:")
    for _, row in goals.iterrows():
        print(f"  P{row['period']} {row['period_time']} - Team: {row['team_id']} ({row.get('team_abb', '?')})")
        
    # 4. Test Logic from analyze.py
    team_val = 'PHI'
    print(f"\nTesting Logic with team_val='{team_val}'")
    
    def _is_team_row(r):
        try:
            t = team_val
            tid = int(t) if str(t).strip().isdigit() else None
        except Exception:
            tid = None
        try:
            if tid is not None:
                return str(r.get('team_id')) == str(tid)
            tupper = str(t).upper()
            
            # Logic from analyze.py
            if r.get('home_abb') is not None and str(r.get('home_abb')).upper() == tupper:
                return str(r.get('team_id')) == str(r.get('home_id'))
            if r.get('away_abb') is not None and str(r.get('away_abb')).upper() == tupper:
                return str(r.get('team_id')) == str(r.get('away_id'))
        except Exception as e:
            print(f"Error in row check: {e}")
            return False
        return False

    mask = df.apply(_is_team_row, axis=1)
    phi_goals = goals[mask.reindex(goals.index)]
    other_goals = goals[~mask.reindex(goals.index)]
    
    print(f"PHI Goals Count: {len(phi_goals)}")
    print(f"Other Goals Count: {len(other_goals)}")
    
    # 5. Check Timing
    print("\nChecking Timing...")
    condition = {'team': 'PHI'}
    
    try:
        timing_res = timing.compute_game_timing(df, condition, force_refresh=True, season='20252026')
        agg = timing_res.get('aggregate', {})
        inter = agg.get('intersection_seconds_total', 0.0)
        print(f"Computed Intersection Seconds: {inter}")
        
    except Exception as e:
        print(f"Timing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_game()
