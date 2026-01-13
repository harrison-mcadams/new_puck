
import sys
import os
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from puck.moneypuck import merge_predictions

def test_merge_predictions_strict_matching():
    # Setup Local Data
    # 1 Blocked Shot, 1 Shot on Goal, 1 Miss
    df_local = pd.DataFrame({
        'game_id': [2024020001, 2024020001, 2024020001],
        'period': [1, 1, 1],
        'game_seconds_calc': [100.0, 200.0, 300.0],
        'event': ['blocked-shot', 'shot-on-goal', 'missed-shot'],
        'x': [0,0,0], 'y': [0,0,0]
    })
    
    # Setup MoneyPuck Data
    # 1 SHOT near the blocked shot (should NOT match)
    # 1 SHOT matching the shot-on-goal (should match)
    # 1 MISS matching the missed-shot (should match)
    df_mp = pd.DataFrame({
        'game_id': [20001, 20001, 20001],
        'period': [1, 1, 1],
        'time': [100.5, 200.5, 300.5], # Within 1s
        'event': ['SHOT', 'SHOT', 'MISS'],
        'xGoal': [0.5, 0.4, 0.3],
        'shotID': [1, 2, 3],
        'shotType': ['W', 'S', 'W'],
        'shotDistance': [10, 20, 30]
    })
    
    # Run Merge
    merged = merge_predictions(df_local, df_mp)
    
    print("\nMerged Cols:", merged.columns)
    print(merged[['event', 'mp_xGoal', 'mp_shotID']])
    
    # Verify Blocked Shot (Index 0) - Should NOT have MP data
    blocked_row = merged.iloc[0]
    assert blocked_row['event'] == 'blocked-shot'
    assert pd.isna(blocked_row['mp_xGoal']), "Blocked shot should not have mp_xGoal"
    
    # Verify Shot on Goal (Index 1) - Should have MP data
    shot_row = merged.iloc[1]
    assert shot_row['event'] == 'shot-on-goal'
    assert shot_row['mp_xGoal'] == 0.4, "Shot on goal should match"
    
    # Verify Missed Shot (Index 2) - Should have MP data
    miss_row = merged.iloc[2]
    assert miss_row['event'] == 'missed-shot'
    assert miss_row['mp_xGoal'] == 0.3, "Missed shot should match"

if __name__ == "__main__":
    test_merge_predictions_strict_matching()
    print("Test passed!")
