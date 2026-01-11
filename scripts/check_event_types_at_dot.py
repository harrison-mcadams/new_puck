
"""
Script to check ALL event types at (69, 22) to trace the 117k spike.
"""
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def main():
    print("Loading Data...")
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
        
    # Filter for Exact Match at (69, 22)
    mask = (df['x'] == 69.0) & (df['y'] == 22.0)
    subset = df[mask]
    
    print(f"\nTotal events exactly at (69, 22): {len(subset)}")
    print("\nEvent Counts:")
    print(subset['event'].value_counts())
    
    print("\nShooter Role counts for these events:")
    if 'shooter_role' in subset.columns:
        print(subset['shooter_role'].value_counts(dropna=False))
        
    print("\nCross Tab (Event vs Role):")
    if 'shooter_role' in subset.columns:
        print(pd.crosstab(subset['event'], subset['shooter_role']))

if __name__ == "__main__":
    main()
