"""debug_imputation.py

Script to visualize and verify blocked shot imputation logic.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import impute

def test_imputation():
    print("--- Testing Blocked Shot Imputation ---")
    
    # Create synthetic blocked shots demonstrating the issue
    # Row 1: Block at goal line (89, 5)
    # Row 2: Block behind net (-96, -2)
    # Row 3: Block at net (-89, 0) - Edge case
    # Row 4: Normal block (80, 0)
    
    data = {
        'game_id': [1, 1, 1, 1],
        'event': ['blocked-shot'] * 4,
        'x': [89.0, -96.0, -89.0, 80.0],
        'y': [5.0, -2.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    
    # Needs distance/angle for context? 
    # impute usually calculates them if missing, or uses them.
    # Let's let it calculate or we can provide.
    # impute logic checks for 'distance' to pick net end.
    # If not present, it guesses by x.
    
    print("\nOriginal Data:")
    print(df[['x', 'y']])
    
    # Run Imputation
    # Fix seed for reproducibility of stochastic distance
    np.random.seed(42)
    
    df_imp = impute.impute_blocked_shot_origins(df, method='mean_6')
    
    print("\nImputed Data:")
    cols = ['x', 'y', 'imputed_x', 'imputed_y', 'distance', 'angle_deg']
    print(df_imp[cols])
    
    # Analyze Vectors
    # Net is roughly 89 or -89.
    # Row 0: Net 89. Block 89, 5.
    # Vector: (0, 5).
    # d_proj: stochastic.
    
    r0 = df_imp.iloc[0]
    dx = r0['imputed_x'] - r0['x']
    dy = r0['imputed_y'] - r0['y']
    print(f"\nRow 0 (89, 5) Shift: dx={dx:.2f}, dy={dy:.2f}")
    
    if abs(dx) < 0.1 and abs(dy) > 1.0:
        print(">> CONFIRMED: Row 0 moved primarily in Y direction.")
    else:
        print(f">> Row 0 moved direction: ({dx:.2f}, {dy:.2f})")

if __name__ == "__main__":
    test_imputation()
