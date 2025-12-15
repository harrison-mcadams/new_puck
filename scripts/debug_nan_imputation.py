
import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import impute

def main():
    print("--- Verifying NaN Imputation Fix ---")
    
    # Create DataFrame with NaN Coords but Valid Distance (Simulating API return)
    df = pd.DataFrame({
        'event': ['blocked-shot', 'blocked-shot'],
        'x': [np.nan, 89.0], # One NaN, one Valid
        'y': [np.nan, 0.0],
        'distance': [178.0, 10.0],
        'angle_deg': [96.0, 0.0],
        'shot_type': ['Unknown', 'Unknown']
    })
    
    print("\n[Input]")
    print(df[['event', 'x', 'distance']])
    
    print("\n[Running Imputation]")
    df_out = impute.impute_blocked_shot_origins(df, method='mean_6')
    
    print("\n[Output]")
    print(df_out[['event', 'imputed_x', 'distance', 'angle_deg']])
    
    # Check Row 0
    dist_0 = df_out.loc[0, 'distance']
    if pd.isna(dist_0) or dist_0 == 0:
        print("\n[FAIL] Distance became NaN or 0 (Original Bug)")
    elif abs(dist_0 - 178.0) < 0.01:
        print(f"\n[PASS] Distance preserved: {dist_0}")
    else:
        print(f"\n[FAIL?] Distance changed to {dist_0}")

    # Check Row 1 (Should be imputed)
    dist_1 = df_out.loc[1, 'distance']
    if abs(dist_1 - 10.0) > 0.1:
         print(f"[PASS] Valid coords imputed. Old: 10.0, New: {dist_1}")
    else:
         print(f"[FAIL] Row 1 not imputed?")

if __name__ == "__main__":
    main()
