
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import impute

def test_imputation():
    print("Testing Blocked Shot Imputation (Empirical Model)...")
    
    # Check Model Loading
    model = impute.load_blocked_model()
    if not model:
        print("ERROR: Could not load blocked_shot_model.json")
        return
        
    print(f"Model keys: {list(model.keys())}")
    print(f"Bin count: {len(model.get('bins', {}))}")
    
    # Create Dummy Data
    # 1. Standard Slot Block (Right Attack) -> x=60, y=0. Origin should be further back.
    # 2. Defensive Slot Block (Left Attack) -> x=-60, y=0. Origin should be further "left" (more negative).
    # 3. Flank Block -> x=70, y=20.
    
    df = pd.DataFrame({
        'event': ['blocked-shot', 'blocked-shot', 'blocked-shot', 'shot'],
        'x': [60.0, -60.0, 70.0, 50.0],
        'y': [0.0, 0.0, 20.0, 0.0],
    })
    
    print("\nOriginal DataFrame:")
    print(df[['event', 'x', 'y']])
    
    # Run Imputation
    df_imp = impute.impute_blocked_shot_origins(df, method='empirical_model', x_col='x', y_col='y')
    
    print("\nImputed DataFrame:")
    print(df_imp[['imputed_x', 'imputed_y', 'distance', 'angle_deg']])
    
    # Validations
    row0 = df_imp.iloc[0] # 60, 0
    # Mean Origin for block at 60 should be < 60 (closer to blue line / center).
    # Wait, X=0 is Center. X=89 is Net.
    # Block at 60 is Defensive Zone for Blocker (Attacking Zone for Shooter).
    # Origin should be "further from Net" -> Lower X (towards 0 or negative).
    print(f"Row 0 (60,0) -> Imp X: {row0['imputed_x']:.2f}")
    
    # Logic: Origin of shot is usually 'Point' (~X=30-60) or 'High Slot'.
    # If block is at 60, shot must be from < 60?
    # No, Net is at 89. Shot comes from e.g. 40. Block at 60.
    # Wait.
    # Frame: Net at 89.
    # Shooter at 40 (Blue Line approx 25). shot travels 40 -> 60 -> 89.
    # Correct. So Origin X should be < Block X (60).
    if row0['imputed_x'] < 60.0:
        print("PASS: Imputed Origin is further back (lower X) than Block.")
    else:
        print("FAIL: Imputed Origin is NOT further back.")

    row1 = df_imp.iloc[1] # -60, 0
    # Symmetric to Row 0. Net at -89.
    # Shooter at -40. Block at -60.
    # Origin Should be > -60 (closer to 0).
    print(f"Row 1 (-60,0) -> Imp X: {row1['imputed_x']:.2f}")
    if row1['imputed_x'] > -60.0:
        print("PASS: Imputed Origin is further back (higher X, closer to 0) than Block.")
    else:
        print("FAIL: Imputed Origin logic for negative coordinates seems wrong.")
        
    # Check Non-Blocked
    row3 = df_imp.iloc[3]
    if row3['imputed_x'] == 50.0 and row3['imputed_y'] == 0.0:
        print("PASS: Non-blocked shot preserved.")
    else:
        print("FAIL: Non-blocked shot modified.")

if __name__ == "__main__":
    test_imputation()
