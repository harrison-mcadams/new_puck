
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, fit_xgs

def main():
    print("Loading data...")
    # Load default data (likely includes multiple seasons if available in default dir)
    df = fit_xgs.load_data()
    
    print(f"Loaded {len(df)} rows. Applying Pipeline (with Imputation)...")
    # We must apply the pipeline to get 'x' and 'y' for blocked shots (imputed)
    # matching what the model sees.
    df = data_pipeline.preprocess_features(
        df,
        is_training=True, # Enable dithering etc to match training
        apply_imputation=True,
        apply_arena_adjustments=True,
        apply_dithering=True,
        apply_filtering=True
    )
    
    # Define a "Mid-to-High Slot" Region
    # Standard Slot roughly: X=60 to 80 (Crease is ~89), Y=-15 to 15?
    # User said "Mid to High slot". 
    # Rink: X=0 is Center, X=89 is Goal Line (in offensive zone coords usually 0-100 where 89 is net)
    # Wait, our coords are typically 0=Center, 89=Net.
    # "High Slot" is usually around the circles/hashmarks, X ~ 40-60?
    # "Mid Slot" ~ 60-75?
    # Let's verify coordinate system. Standard NHL is X=0 center, X=89 goal.
    # "High Slot" (near blue line/circles top) is X ~ 40-55.
    # "Mid Slot" (hashmarks) is X ~ 55-70.
    # "Low Slot" (crease) is X ~ 70-85.
    
    # Let's inspect a few distinct zones.
    zones = {
        "High Slot (Blue Line/Circles)":  {'x_min': 35, 'x_max': 55, 'y_abs_max': 20},
        "Mid Slot (Hashmarks)":           {'x_min': 55, 'x_max': 70, 'y_abs_max': 15},
        "Low Slot (Net Mouth)":           {'x_min': 70, 'x_max': 85, 'y_abs_max': 10},
    }
    
    # Filter for valid events (shots + blocks)
    # Note: 'missed-shot' is unblocked. 'shot-on-goal' is unblocked. 'goal' is unblocked.
    mask_shot = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    mask_block = df['event'] == 'blocked-shot'
    
    df_events = df[mask_shot | mask_block].copy()
    df_events['is_blocked'] = (df_events['event'] == 'blocked-shot').astype(int)
    
    print("\n--- Empirical Block Rates by Zone (Processed Data) ---")
    
    for name, bounds in zones.items():
        # Filter by bounds
        mask_zone = (
            (df_events['x'] >= bounds['x_min']) & 
            (df_events['x'] <= bounds['x_max']) & 
            (df_events['y'].abs() <= bounds['y_abs_max'])
        )
        
        subset = df_events[mask_zone]
        total = len(subset)
        if total == 0:
            print(f"\n{name}: No events found.")
            continue
            
        n_blocked = subset['is_blocked'].sum()
        rate = n_blocked / total
        
        print(f"\n{name} [X:{bounds['x_min']}-{bounds['x_max']}, |Y|<{bounds['y_abs_max']}]:")
        print(f"  Total Events: {total}")
        print(f"  Blocked:      {n_blocked} ({rate:.2%})")
        print(f"  Unblocked:    {total - n_blocked}")
        
        # Check by Role if available
        if 'shooter_role' in subset.columns:
            for role in ['F', 'D']:
                sub_role = subset[subset['shooter_role'] == role]
                if len(sub_role) > 0:
                    rate_role = sub_role['is_blocked'].mean()
                    print(f"    Role {role}: {rate_role:.2%} ({sub_role['is_blocked'].sum()}/{len(sub_role)})")

if __name__ == "__main__":
    main()
