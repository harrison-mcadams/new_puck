
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from puck.impute import impute_blocked_shot_origins
except ImportError:
    print("Could not import puck.impute")
    sys.exit(1)

def verify():
    print("--- Verifying Imputation Logic ---")
    
    # Create synthetic test cases
    # We want to test:
    # 1. Close Block (e.g., 15ft from net) -> Should map to Close Origin (e.g., 20-30ft)
    # 2. Far Block (e.g., 60ft from net) -> Should map to Far Origin (e.g., 60-70ft)
    # 3. Role Differences (F vs D)
    
    # Net is at X=89.
    # Close Block: 15ft away -> X = 89 - 15 = 74.
    # Far Block: 60ft away -> X = 89 - 60 = 29.
    
    test_data = {
        'x': [74.0, 74.0, 29.0, 29.0, 95.0], 
        'y': [0.0, 0.0, 0.0, 0.0, 10.0],
        'shooter_role': ['F', 'D', 'F', 'D', 'D'],
        'desc': ['Close Block (F)', 'Close Block (D)', 'Far Block (F)', 'Far Block (D)', 'Behind Net (D)'],
        'event': ['blocked-shot'] * 5
    }
    
    df = pd.DataFrame(test_data)
    
    print("\nInput Data:")
    print(df[['desc', 'x', 'y', 'shooter_role']])
    
    # Run Imputation
    df_imputed = impute_blocked_shot_origins(df, method='empirical_model', role_col='shooter_role')
    
    print("\nOutput Data:")
    cols = ['desc', 'x', 'imputed_x', 'distance', 'angle_deg'] 
    # Note: 'distance' in output is imputed distance? Or updated block distance?
    # Impute.py updates 'distance' to be the New Origin Distance.
    
    if 'distance' in df_imputed.columns:
        print(df_imputed[cols])
    else:
        print(df_imputed[['desc', 'x', 'imputed_x']])
        
    # Validation Checks
    print("\n\n--- Checks ---")
    vals = df_imputed.set_index('desc')
    
    # Check 1: Direct Mapping (Monotonicity)
    # Close Block (74) Origin X > Far Block (29) Origin X?
    # Normalized: High X = Close. Low X = Far.
    # So Imputed X(Close) should be > Imputed X(Far).
    
    x_close_F = vals.loc['Close Block (F)', 'imputed_x']
    x_far_F = vals.loc['Far Block (F)', 'imputed_x']
    
    print(f"Check Fwd Monotonicity: Close X ({x_close_F:.1f}) > Far X ({x_far_F:.1f})?")
    if x_close_F > x_far_F:
        print("PASS")
    else:
        print("FAIL - Inversion or Flat Mapping detected?")

    x_close_D = vals.loc['Close Block (D)', 'imputed_x']
    x_far_D = vals.loc['Far Block (D)', 'imputed_x']
    
    print(f"Check Def Monotonicity: Close X ({x_close_D:.1f}) > Far X ({x_far_D:.1f})?")
    if x_close_D > x_far_D:
        print("PASS")
    else:
        print("FAIL")

    # Check 2: Relative to Block
    # Origin should be further from net than block?
    # Net at 89. X values.
    # Origin X < Block X (Further away = Lower X)
    
    print(f"\nCheck Projection (Origin Further than Block):")
    for idx, row in vals.iterrows():
        bx = row['x']
        ox = row['imputed_x']
        desc = str(idx)
        
        if 'Behind Net' in desc:
            print(f"  {idx}: Block {bx:.1f} -> Origin {ox:.1f}. (Match?) {abs(bx - ox) < 0.01}")
        else:
            print(f"  {idx}: Block {bx:.1f} -> Origin {ox:.1f}. (Origin < Block?) {ox < bx}")
    
    # Check 3: Check Rink Boundaries
    if (df_imputed['imputed_x'].abs() > 99.0).any() or (df_imputed['imputed_y'].abs() > 42.0).any():
        print("\nFAIL: Imputed values outside Rink Boundaries!")
    else:
        print("\nPASS: Rink Boundary Check")

if __name__ == "__main__":
    verify()
