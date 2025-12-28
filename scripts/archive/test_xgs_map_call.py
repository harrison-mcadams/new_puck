
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from puck import analyze

# Mock DF
# We need enough columns for _predict_xgs and _apply_condition to not crash
df = pd.DataFrame({
    'x': [50, 60], 
    'y': [0, 5], 
    'event_type': ['shot-on-goal', 'missed-shot'],
    'team_id': [1, 1],
    'home_id': [1, 1],
    'away_id': [2, 2],
    'home_abb': ['PHI', 'PHI'],
    'away_abb': ['NYR', 'NYR'],
    'home_team_defending_side': ['left', 'left'],
    'game_id': [2025020001, 2025020001],
    'period': [1, 1],
    'period_seconds': [100, 200],
    'total_time_elapsed_seconds': [100, 200],
    'game_state': ['5v5', '5v5'],
    'is_net_empty': [0, 0],
    'shot_type': ['Wrist Shot', 'Snap Shot'], # Needed for model
    'distance': [30, 20],
    'angle': [0, 10],
    'scored': [0, 0] # target
})

# Mock Intervals
intervals = {
    'per_game': {
        2025020001: {
            'intersection_intervals': [(0, 300)]
        }
    }
}

print("--- Testing NEW Call (Fix) ---")
try:
    # New correct call: passing season and data_df as kwargs
    grid, stats = analyze.xgs_map(
        season='20252026',
        data_df=df,
        intervals_input=intervals,
        condition={},
        heatmap_only=True # Skip plotting to be faster
    )
    print("Success: xgs_map returned without error.")
    print("Grid shape:", grid['team'].shape if grid and grid.get('team') is not None else "None")
    print("Stats present:", bool(stats))
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n--- Testing OLD Call (Bug) ---")
try:
    # Original incorrect call: analyze.xgs_map(df_game, season=None, ...)
    # This passed DF as first positional arg (which maps to 'season') 
    # and season=None as kwarg.
    analyze.xgs_map(
        df, # Passed as positional 'season'
        season=None, # Passed as keyword 'season' -> Collision!
        intervals_input=intervals,
        heatmap_only=True
    )
    print("Unexpected: Old call did NOT raise TypeError.")
except TypeError as e:
    print(f"Confirmed expected TypeError: {e}")
except Exception as e:
    print(f"Unexpected exception type: {type(e).__name__}: {e}")
