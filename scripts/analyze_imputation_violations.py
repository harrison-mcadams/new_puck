
import pandas as pd
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.impute import impute_blocked_shot_origins

def analyze_violations():
    print("--- Analyzing Imputation Violations Across Zones ---")
    
    # Grid of Block Locations (Standardized: Attack Right, Net @ 89)
    # X: 30 to 80 (Offensive Zone)
    # Y: 0 (Center), 20 (Slot/Wing)
    
    test_points = [
        (40, 0), (60, 0), (75, 0),  # Center Lane
        (40, 20), (60, 20), (75, 20) # Side/Wing
    ]
    
    N_SAMPLES = 500
    
    print(f"{'Block (X, Y)':<15} | {'Violations %':<12} | {'Avg Violation Dist':<18} | {'Example Breach (Blk -> Imp)'}")
    print("-" * 85)
    
    for bx, by in test_points:
        # Create input dataframe
        df = pd.DataFrame({
            'x': [float(bx)] * N_SAMPLES,
            'y': [float(by)] * N_SAMPLES,
            'event': ['blocked-shot'] * N_SAMPLES,
            'shooter_role': ['F'] * N_SAMPLES,
            'home_team_defending_side': ['left'] * N_SAMPLES
        })
        
        # Run Imputation
        df_imp = impute_blocked_shot_origins(df, method='empirical_model', is_standardized=True)
        
        # Calculate Distances
        # Net at (89, 0) for standardized Attack Right
        df_imp['d_block'] = np.sqrt((df_imp['block_x'] - 89)**2 + df_imp['block_y']**2)
        df_imp['d_shooter'] = np.sqrt((df_imp['imputed_x'] - 89)**2 + df_imp['imputed_y']**2)
        
        # Violation: Shooter closer than block (with small tolerance)
        df_imp['violation'] = df_imp['d_shooter'] < (df_imp['d_block'] - 0.1)
        
        v_rows = df_imp[df_imp['violation']]
        pct = (len(v_rows) / N_SAMPLES) * 100
        
        avg_diff = 0.0
        example_str = "None"
        
        if len(v_rows) > 0:
            # How much closer? (Positive value = dist_block - dist_shooter)
            diffs = v_rows['d_block'] - v_rows['d_shooter']
            avg_diff = diffs.mean()
            
            # Pick worst case example
            worst_idx = diffs.idxmax()
            row = v_rows.loc[worst_idx]
            example_str = f"({row['block_x']:.1f}, {row['block_y']:.1f}) -> ({row['imputed_x']:.1f}, {row['imputed_y']:.1f})"
            
        print(f"({bx}, {by:<2}){'':<5} | {pct:>5.1f}%       | {avg_diff:>6.2f} units        | {example_str}")

if __name__ == "__main__":
    analyze_violations()
