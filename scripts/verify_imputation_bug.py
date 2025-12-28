
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from puck import impute
from pathlib import Path

def main():
    data_file = Path('data/20252026.csv')
    if not data_file.exists():
        print(f"File not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    blocks = df[df['event'] == 'blocked-shot'].copy()
    
    if len(blocks) == 0:
        print("No blocked shots found.")
        return

    # Apply imputation
    df_imputed = impute.impute_blocked_shot_origins(blocks, method='mean_6')
    
    # Compare distances
    comparison = pd.DataFrame({
        'x': blocks['x'],
        'y': blocks['y'],
        'orig_dist': blocks['distance'],
        'new_dist': df_imputed['distance'],
        'diff': df_imputed['distance'] - blocks['distance']
    })
    
    print("\nDistance change after imputation (current implementation):")
    print(comparison['diff'].describe())
    
    pos_diff = (comparison['diff'] > 0.01).sum()
    neg_diff = (comparison['diff'] < -0.01).sum()
    zero_diff = len(comparison) - pos_diff - neg_diff
    
    print(f"\nIncreased distance (Expected): {pos_diff} ({pos_diff/len(comparison):.1%})")
    print(f"Decreased distance (BUG?): {neg_diff} ({neg_diff/len(comparison):.1%})")
    print(f"No change: {zero_diff} ({zero_diff/len(comparison):.1%})")
    
    # Check if decreased distance correlates with x < 0
    neg_blocks = comparison[comparison['diff'] < -0.01]
    if len(neg_blocks) > 0:
        print("\nStats for decreased distance shots:")
        print(neg_blocks['x'].describe())
    
    pos_blocks = comparison[comparison['diff'] > 0.01]
    if len(pos_blocks) > 0:
        print("\nStats for increased distance shots:")
        print(pos_blocks['x'].describe())

if __name__ == "__main__":
    main()
