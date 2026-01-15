
import pandas as pd
import numpy as np
import sys
import os
import math

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.impute import impute_blocked_shot_origins

def stress_test_physics():
    print("--- Stress Testing Imputation Physics ---")
    
    # 1. Setup Mock Data
    # Block at x=60, y=0 (Standardized). Net at x=89.
    # Block Distance to Net = 29.
    # We expect Shooter Distance > 29.
    
    N = 1000
    df = pd.DataFrame({
        'x': [60.0] * N,
        'y': [0.0] * N,
        'event': ['blocked-shot'] * N,
        'shooter_role': ['F'] * N,
        'home_team_defending_side': ['left'] * N # Attack Right
    })
    
    # 2. Run Imputation
    # Use empirical model (default)
    print(f"Running {N} imputations for Block at (60, 0)...")
    df_imputed = impute_blocked_shot_origins(df, method='empirical_model', is_standardized=True)
    
    # 3. Check Distances
    # Net is at 89, 0
    df_imputed['dist_block'] = np.sqrt((df_imputed['block_x'] - 89)**2 + df_imputed['block_y']**2)
    df_imputed['dist_shooter'] = np.sqrt((df_imputed['imputed_x'] - 89)**2 + df_imputed['imputed_y']**2)
    
    # Violation: Shooter Closer (Dist < Block Dist)
    # Floating point tolerance
    df_imputed['violation'] = df_imputed['dist_shooter'] < df_imputed['dist_block'] - 0.1
    
    n_violations = df_imputed['violation'].sum()
    pct_violations = (n_violations / N) * 100
    
    print(f"\nResults:")
    print(f"Total Samples: {N}")
    print(f"Violations (Shooter closer than block): {n_violations} ({pct_violations:.1f}%)")
    
    if n_violations > 0:
        print("\nSample Violations:")
        print(df_imputed[df_imputed['violation']][['block_x', 'imputed_x', 'dist_block', 'dist_shooter']].head())
    else:
        print("\nNo violations found. Physics check appears robust (or model is very conservative).")

if __name__ == "__main__":
    stress_test_physics()
