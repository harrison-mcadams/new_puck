
import pandas as pd
import numpy as np
import sys
import os

# Adjust path to import puck
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from puck import data_pipeline

def verify_blocked_impute():
    print("Verifying Blocked Shot Imputation Logic...")
    
    # Create test dataframe with BLOCKED SHOTS
    # Scenario:
    # 1. Offensive Zone Block: X=50 (Attacking Right).
    # 2. Defensive Zone Block: X=-50 (Attacking Right).
    #    (Real Def Zone Shot: Block at -50, Goal at +89. Dist = 139)
    #    (If buggy, might flip to +50 -> Dist 39 -> Impute Origin +35 -> Flip back -35. Result: Origin closer to net).
    #    (Correct: Block -50 -> Dist 139 -> Impute Origin "Further" -> e.g. -65).
    
    data = {
        'game_id': [2023020001] * 2,
        'period': [1] * 2,
        'team_id': [1, 1],
        'home_id': [1] * 2,
        'away_id': [2] * 2,
        'home_team_defending_side': ['left', 'left'], # Home Attacking Right (+X)
        'x': [50.0, -50.0],
        'y': [0.0, 0.0],
        'event': ['blocked-shot', 'blocked-shot'],
        'shooter_role': ['F', 'D'] # Just to satisfy inputs
    }
    
    df = pd.DataFrame(data)
    
    print("\n[Input Data]")
    print(df[['x', 'y', 'event']])
    
    # Run Pipeline WITH IMPUTATION
    # We rely on data_pipeline to standardize first.
    processed = data_pipeline.preprocess_features(
        df, 
        apply_arena_adjustments=False, 
        apply_imputation=True, # Enable Imputation
        apply_dithering=False
    )
    
    print("\n[Processed Data]")
    print(processed[['x', 'imputed_x', 'distance']])
    
    cols = ['x', 'imputed_x', 'distance']
    
    # Check 1: Off Zone Block (+50)
    # Expect Imputed X < 50 (Further from net at 89).
    r1 = processed.iloc[0]
    print(f"\n1. Off Zone Block (+50): -> Imputed {r1['imputed_x']:.2f}")
    if r1['imputed_x'] < 50.0:
        print("PASS: Imputed origin is further from net (towards center/left)")
    else:
        print("FAIL: Imputed origin is closer to net?")
        
    # Check 2: Def Zone Block (-50)
    # Block is at -50. Net is at +89.
    # Origin should be "behind" the block relative to the net.
    # So X should be < -50 (more negative).
    r2 = processed.iloc[1]
    print(f"\n2. Def Zone Block (-50): -> Imputed {r2['imputed_x']:.2f}")
    
    if r2['imputed_x'] < -50.0:
         print(f"PASS: Imputed origin ({r2['imputed_x']:.2f}) is further away than block (-50).")
    else:
         print(f"FAIL: Imputed origin ({r2['imputed_x']:.2f}) is closer/wrong direction compared to block (-50).")
         # If it's -35 (mirrored logic), that means it flipped.

if __name__ == "__main__":
    verify_blocked_impute()
