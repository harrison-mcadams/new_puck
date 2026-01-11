
import pandas as pd
import numpy as np

def main():
    try:
        df = pd.read_csv('analysis/debug_imputation_pipeline.csv')
    except FileNotFoundError:
        print("CSV not found.")
        return

    # Filter for blocked shots
    blocks = df[df['event'] == 'blocked-shot'].copy()
    
    if len(blocks) == 0:
        print("No blocked shots in debug CSV.")
        return

    # Goal Location (Standard Rink)
    GOAL_X = 89
    GOAL_Y = 0

    # Calculate Distances
    # 1. Block Location (Input to imputation)
    # We used x_adj/y_adj if available, or x/y. The CSV has x_adj columns.
    # Let's check if they are populated.
    
    # Check for negative x_adj (Defensive Zone Artifacts)
    neg_count = (blocks['x_adj'] < 0).sum()
    print(f"Negative x_adj count: {neg_count}")
    if neg_count > 0:
        print("WARNING: Data still contains defensive zone coordinates!")
    
    # Use x_adj/y_adj for block location ref (or x/y if adj is tiny)
    blocks['dist_block'] = np.sqrt((blocks['x_adj'] - GOAL_X)**2 + blocks['y_adj']**2)
    
    # 2. Imputed Origin (Output)
    blocks['dist_origin'] = np.sqrt((blocks['imputed_x'] - GOAL_X)**2 + blocks['imputed_y']**2)
    
    # 3. Delta
    blocks['shift'] = blocks['dist_origin'] - blocks['dist_block']
    
    print(f"--- Imputation Shift Verification (N={len(blocks)}) ---")
    print(f"Mean Block Distance (Input):  {blocks['dist_block'].mean():.2f} ft")
    print(f"Mean Origin Distance (Output): {blocks['dist_origin'].mean():.2f} ft")
    print(f"Mean Shift (Origin - Block):   {blocks['shift'].mean():.2f} ft")
    
    print("\nQuantiles of Shift:")
    print(blocks['shift'].quantile([0.01, 0.1, 0.5, 0.9, 0.99]))
    
    print("\nSample Rows:")
    print(blocks[['x_adj', 'y_adj', 'dist_block', 'imputed_x', 'imputed_y', 'dist_origin', 'shift']].head(10))

if __name__ == "__main__":
    main()
