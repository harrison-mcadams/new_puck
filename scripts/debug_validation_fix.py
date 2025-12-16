
import sys
import os
import pandas as pd

# Ensure local puck is importable
sys.path.insert(0, os.getcwd())

import puck.analyze as analyze
import puck.timing as timing

print(f"Using analyze from: {analyze.__file__}")

gid = 2025010073
season = "20252026"

print(f"--- Debugging Game {gid} Validation logic ---")

# Condition that should FAIL for Eklund goal (which is 5v4)
cond = {'game_state': ['5v5'], 'team': 'SJS'}

print(f"Calling xgs_map with condition: {cond}")
try:
    res = analyze.xgs_map(
        season=season,
        game_id=gid,
        condition=cond,
        return_filtered_df=True,
        force_refresh=True  # Force re-fetch to ensure clean state
    )
    
    if isinstance(res, tuple):
        df_filtered = res[0]
    else:
        df_filtered = res
        
    print(f"Returned {len(df_filtered)} rows.")
    
    # Check for Eklund goal
    eklund_goals = df_filtered[
        (df_filtered['player_name'].str.contains('Eklund', na=False)) & 
        (df_filtered['event'] == 'goal')
    ]
    
    if not eklund_goals.empty:
        print("\n[FAILURE] Eklund goal found in 5v5 set (Double Count persists):")
        print(eklund_goals[['period_time', 'game_state', 'total_time_elapsed_seconds']])
    else:
        print("\n[SUCCESS] Eklund goal filtered out of 5v5 set.")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
