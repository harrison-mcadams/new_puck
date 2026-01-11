
"""
Script to audit the most frequent exact coordinates in the dataset.
Helps identify "Default Coordinate" artifacts.
"""
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, data_pipeline

def print_top_coords(df, label):
    # Filter Forwards, Unblocked
    f_un = df[(df['shooter_role'] == 'F') & (df['event'] != 'blocked-shot')]
    total = len(f_un)
    
    print(f"\n--- {label} (N={total}) ---")
    
    top_coords = f_un.groupby(['x', 'y']).size().sort_values(ascending=False).head(10)
    
    print(f"{'Rank':<5} | {'(X, Y)':<15} | {'Count':<10} | {'%':<6} | {'Likely Location'}")
    print("-" * 65)
    
    for rank, ((x, y), count) in enumerate(top_coords.items(), 1):
        pct = (count / total) * 100
        
        # Guesses
        loc = "?"
        if abs(x - 69) < 1 and abs(y - 22) < 1: loc = "Faceoff Dot"
        elif abs(x - 69) < 1 and abs(y + 22) < 1: loc = "Faceoff Dot"
        elif abs(x - 20) < 1 and abs(y - 22) < 1: loc = "Neutral Dot?"
        elif abs(x - 0) < 1 and abs(y - 0) < 1: loc = "Center Ice"
        elif abs(x - 89) < 3 and abs(y - 0) < 3: loc = "Net / Crease"
        
        print(f"{rank:<5} | ({x:<.1f}, {y:<.1f})     | {count:<10} | {pct:<5.2f}% | {loc}")
    return top_coords

def main():
    print("Loading Data...")
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
        
    # Check RAW
    print_top_coords(df, "RAW DATA")
    
    # Check ADJUSTED
    print("\nApplying Arena Adjustments...")
    df_adj = data_pipeline.preprocess_features(
        df, 
        is_training=False,
        apply_imputation=False, # Don't impute, we want to check existing unblocked shots
        apply_dithering=False,   # Don't dither, we want to see exact spikes
        apply_arena_adjustments=True
    )
    
    print_top_coords(df_adj, "ADJUSTED DATA")

if __name__ == "__main__":
    main()
