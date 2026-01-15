
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.data_pipeline import preprocess_features

def verify_orientation():
    print("--- Verifying Orientation Standardization Flow ---")

    # Create Mock Data
    # Row 1: Attacking Right (Standard). Blocked Shot. X=70.
    # Row 2: Attacking Left (Needs Flip). Blocked Shot. X=-70.
    
    mock_data = [
        {
            'game_id': 2025010001,
            'event': 'blocked-shot',
            'team_id': 1, # Team 1 Attacking
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'left', # Home (1) Defends Left -> Attacking Right
            'x': 70.0,
            'y': 10.0,
            'period': 1,
            'shooter_role': 'F'
        },
        {
            'game_id': 2025010001,
            'event': 'blocked-shot',
            'team_id': 1, # Team 1 Attacking
            'home_id': 1,
            'away_id': 2,
            'home_team_defending_side': 'right', # Home (1) Defends Right -> Attacking Left
            'x': -70.0,
            'y': -10.0,
            'period': 2,
            'shooter_role': 'F'
        }
    ]
    
    df_raw = pd.DataFrame(mock_data)
    print("\nRaw Data:")
    print(df_raw[['x', 'y', 'home_team_defending_side']])
    
    # Run Pipeline (Training Mode to enable all steps if needed, but defaults are fine)
    # We want arena_adjustments=False to isolate orientation logic logic
    df_processed = preprocess_features(df_raw, 
                                       is_training=False, 
                                       apply_arena_adjustments=False, 
                                       apply_imputation=True, # Critical
                                       verbose=True)
    
    print("\nProcessed Data:")
    cols = ['x', 'y', 'distance', 'angle_deg', 'imputed_x', 'imputed_y']
    print(df_processed[cols])
    
    # Verification Checks
    
    # 1. Orientation
    # Both rows should end up with Positive X (Standardized)
    # Row 1: 70 -> 70
    # Row 2: -70 -> 70
    
    print("\nChecking Orientation...")
    row1 = df_processed.iloc[0]
    row2 = df_processed.iloc[1]
    
    if row1['x'] == 70.0 and row2['x'] == 70.0:
        print("PASS: Both shots standardized to Positive X.")
    else:
        print(f"FAIL: Expected X=70.0 for both. Got {row1['x']} and {row2['x']}.")
        
    # 2. Imputation
    # Blocked shots should have imputed origins ("imputed_x").
    # For a block at X=70, the shooter should be further back (e.g., < 70? Or > 70?).
    # Wait, standardized frame: Net is at 89.
    # Attacking Zone is X > 25.
    # Shooter is FURTHER from net than block.
    # Distance(Shooter, Net) > Distance(Block, Net)
    # Dist(70, 89) = 19.
    # So Shooter X should be < 70 (e.g. 50).
    
    print("\nChecking Imputation...")
    # Note: data_pipeline overwrites 'x' with 'imputed_x' for blocked shots (Shot Origin).
    # 'block_x' should contain the standardized block location.
    
    print(f"Row 1: Block X={row1.get('block_x')}, Shooter X (x)={row1.get('x')}")
    print(f"Row 2: Block X={row2.get('block_x')}, Shooter X (x)={row2.get('x')}")
    
    # We expect Imputed X to be roughly similar for both
    if abs(row1['x'] - row2['x']) < 20.0:
        print("PASS: Imputation is symmetric (roughly equal results for symmetric inputs).")
    else:
        print("WARNING: Imputation results diverge significantly.")
        
    # Check "Shooter Further than Block" logic
    # Imputed X (Shooter) should be < Block X (Further from Net@89)
    # Check using 'block_x' which preserves the standardized block location
    
    if row1.get('block_x') is not None:
         if row1['x'] < row1['block_x']:
             print("PASS: Row 1 Shooter is further from net than block.")
         else:
             print(f"FAIL: Row 1 Shooter X ({row1['x']}) >= Block X ({row1['block_x']}) - Closer to Net?")
    
    if row2.get('block_x') is not None:
         if row2['x'] < row2['block_x']:
             print("PASS: Row 2 Shooter is further from net than block.")
         else:
             print(f"FAIL: Row 2 Shooter X ({row2['x']}) >= Block X ({row2['block_x']}) - Closer to Net?")

if __name__ == "__main__":
    verify_orientation()
