
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import puck
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from puck import data_pipeline

def verify_orientation():
    print("Verifying Orientation Logic...")
    
    # Create test dataframe
    # Scenario:
    # Period 1:
    # - Home Team (ID 1): Defending Left (-x), Attacking Right (+x).
    # - Away Team (ID 2): Defending Right (+x), Attacking Left (-x).
    
    data = {
        'game_id': [2023020001] * 4,
        'period': [1] * 4,
        'team_id': [1, 1, 2, 2],
        'home_id': [1] * 4,
        'away_id': [2] * 4,
        'home_team_defending_side': ['left'] * 4, # Standard start
        # Raw Coordinates (Rink Absolute)
        'x': [
            50.0,   # 1. Home Offense (Attacking Right, in Right Zone) -> Should stay +50
            -50.0,  # 2. Home Defense (Attacking Right, in Left Zone)  -> Should stay -50 (Long shot)
            -50.0,  # 3. Away Offense (Attacking Left, in Left Zone)   -> Should flip to +50
            50.0    # 4. Away Defense (Attacking Left, in Right Zone)  -> Should flip to -50 (Long shot)
        ],
        'y': [0.0] * 4,
        'event': ['shot-on-goal'] * 4
    }
    
    df = pd.DataFrame(data)
    
    print("\n[Input Data]")
    print(df[['team_id', 'x', 'y']])
    
    # Run Pipeline
    # Disable arena adjustments/imputation to isolate orientation logic
    processed = data_pipeline.preprocess_features(
        df, 
        apply_arena_adjustments=False, 
        apply_imputation=False,
        apply_dithering=False
    )
    
    print("\n[Processed Data]")
    print(processed[['team_id', 'x', 'y', 'distance', 'angle_deg']])
    
    # Assertions
    # Goal is at X = 89.0
    
    # 1. Home Offense (+50) -> +50
    # Dist = 89-50 = 39
    r1 = processed.iloc[0]
    pass_1 = (abs(r1['x'] - 50.0) < 0.1)
    
    # 2. Home Defense (-50) -> -50
    # Dist = 89 - (-50) = 139
    r2 = processed.iloc[1]
    pass_2 = (abs(r2['x'] - (-50.0)) < 0.1)
    
    # 3. Away Offense (-50) -> +50
    # Dist = 89 - 50 = 39
    r3 = processed.iloc[2]
    pass_3 = (abs(r3['x'] - 50.0) < 0.1)
    
    # 4. Away Defense (+50) -> -50
    # Dist = 89 - (-50) = 139
    r4 = processed.iloc[3]
    pass_4 = (abs(r4['x'] - (-50.0)) < 0.1)
    
    print("\n[Results]")
    print(f"1. Home Offense (+50 -> +50): {'PASS' if pass_1 else 'FAIL'} (Got {r1['x']})")
    print(f"2. Home Defense (-50 -> -50): {'PASS' if pass_2 else 'FAIL'} (Got {r2['x']}) - Expect Long Distance")
    print(f"3. Away Offense (-50 -> +50): {'PASS' if pass_3 else 'FAIL'} (Got {r3['x']})")
    print(f"4. Away Defense (+50 -> -50): {'PASS' if pass_4 else 'FAIL'} (Got {r4['x']}) - Expect Long Distance")
    
    if pass_1 and pass_2 and pass_3 and pass_4:
        print("\nSUCCESS: Orientation logic handles defensive zone shots correctly.")
    else:
        print("\nFAILURE: Orientation logic incorrect.")

if __name__ == "__main__":
    verify_orientation()
