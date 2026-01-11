
import sys
import os
import numpy as np
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, data_pipeline

def main():
    print("Generating Global Shot Prior...")
    
    # 1. Load All Seasons Data
    # This gives us the largest possible sample size for a smooth prior.
    try:
        df = fit_xgs.load_all_seasons_data()
    except:
        df = fit_xgs.load_data()
    
    print(f"  Loaded {len(df)} rows.")

    # 2. Filter for Unblocked Shots (Fenwick)
    # These represent the "natural" distribution of shots being taken.
    valid_unblocked = ['shot-on-goal', 'missed-shot', 'goal']
    df_un = df[df['event'].isin(valid_unblocked)].copy()
    
    # 3. Basic Preprocessing (Orientation only)
    # We want these in the same "Attacking Right" coordinate system as our imputation.
    df_un = data_pipeline.preprocess_features(
        df_un, 
        is_training=False, 
        apply_imputation=False, # Essential: don't loop!
        apply_arena_adjustments=True,
        apply_dithering=True # Add noise to smooth out the discrete API coordinates
    )
    
    print(f"  Processing {len(df_un)} unblocked shots...")

    # 4. Create 2D Density Map
    # We'll use a 2ft x 2ft binning for high resolution.
    x_bins = np.linspace(0, 100, 51)  # 2ft steps
    y_bins = np.linspace(-42.5, 42.5, 43) # ~2ft steps
    
    h, xedges, yedges = np.histogram2d(
        df_un['x'], df_un['y'], 
        bins=[x_bins, y_bins], 
        range=[[0, 100], [-42.5, 42.5]]
    )
    
    # Normalize to create a valid Probability Mass Function (PMF)
    # This allows us to sample from the rink proportional to shot frequency.
    total = h.sum()
    if total > 0:
        h_norm = h / total
    else:
        h_norm = h

    # 5. Save as JSON
    # We store the weights and the bin center coordinates.
    x_mids = (xedges[:-1] + xedges[1:]) / 2
    y_mids = (yedges[:-1] + yedges[1:]) / 2
    
    # Structure for easy consumption in puck/impute.py
    prior_data = {
        "meta": {
            "description": "Global distribution of unblocked shot origins (2ft bins)",
            "n_samples": int(len(df_un)),
            "x_edges": xedges.tolist(),
            "y_edges": yedges.tolist()
        },
        "x_mids": x_mids.tolist(),
        "y_mids": y_mids.tolist(),
        "weights": h_norm.tolist() # 2D array [nbinsx, nbinsy]
    }
    
    out_path = os.path.join('puck', 'data', 'global_shot_prior.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, 'w') as f:
        json.dump(prior_data, f)
        
    print(f"  Global Shot Prior saved to {out_path}")

if __name__ == "__main__":
    main()
