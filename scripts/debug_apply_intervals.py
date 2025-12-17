
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze
from puck import timing

def test_apply_intervals():
    print("--- Testing _apply_intervals Logic ---")
    
    # Mock Dataframe: 3 Goals
    # 1. 5v5 Goal (Should keep)
    # 2. 5v4 Goal (Should drop)
    # 3. 5v5 Goal outside interval (Should drop)
    
    data = {
        'game_id': [2025010073]*3,
        'total_time_elapsed_seconds': [1000, 2000, 3000],
        'event': ['goal', 'goal', 'goal'],
        'game_state': ['5v5', '5v4', '5v5'],
        'is_net_empty': [0, 0, 0]
    }
    df = pd.DataFrame(data)
    
    print("Input DF:")
    print(df)
    
    # Condition
    cond = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    # Intervals: Cover 1000 and 2000. 3000 is outside.
    intervals_obj = {
        'per_game': {
            2025010073: {
                'merged_intervals': [(900, 2100)],
                'intersection_intervals': [(900, 2100)],
                'selected_team': 24 # irrelevant for absolute check?
            }
        }
    }
    
    print(f"\nCondition: {cond}")
    print(f"Intervals: 900-2100")
    
    # Run
    # team_val=None implies use selected_team from intervals checks
    res_df = analyze._apply_intervals(df, intervals_obj, condition=cond)
    
    print("\nResult DF:")
    print(res_df)
    
    # Check
    # Row 0 (1000, 5v5) should be IN.
    # Row 1 (2000, 5v4) should be OUT (fail validation).
    # Row 2 (3000, 5v5) should be OUT (fail interval).
    
    if len(res_df) == 1 and res_df.iloc[0]['total_time_elapsed_seconds'] == 1000:
        print("\nSUCCESS: Only verified 5v5 goal kept.")
    else:
        print("\nFAILURE: Wrong rows kept.")
        for i, r in res_df.iterrows():
            print(f" Kept: T={r['total_time_elapsed_seconds']}, State={r['game_state']}")

if __name__ == "__main__":
    test_apply_intervals()
