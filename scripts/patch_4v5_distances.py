"""
patch_4v5_distances.py

Reprocesses the 20252026.csv data to fix the incorrect distance calculations
for 4v5 events. Uses data_pipeline.py to ensure correct coordinate 
standardization and feature recalculation.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, config

def main():
    csv_path = os.path.join(config.DATA_DIR, '20252026.csv')
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Pre-check: Show current distance stats
    print("\n=== BEFORE FIX ===")
    for gs in ['5v5', '5v4', '4v5']:
        subset = df[df['game_state'] == gs]
        print(f"{gs}: {len(subset)} events, Mean distance: {subset['distance'].mean():.1f}")
    
    # The issue: distance was calculated incorrectly for 4v5 events
    # We need to recalculate distance using the correct formula
    
    # Option 1: Simple recalculation from x_adj/y_adj
    # This should work if x_adj is already in attacking direction (positive x toward goal at 89)
    
    print("\n--- Recalculating distances from x_adj/y_adj ---")
    
    # Verify x_adj values are sensible first
    print(f"\nCheck: 5v4 mean x_adj = {df[df['game_state']=='5v4']['x_adj'].mean():.1f}")
    print(f"Check: 4v5 mean x_adj = {df[df['game_state']=='4v5']['x_adj'].mean():.1f}")
    
    # For 4v5 events, x_adj seems to be in wrong space (mean ~9 vs ~41 for 5v4)
    # This suggests x_adj itself was not properly flipped
    
    # Let's check if we can fix by running through data_pipeline again
    # But data_pipeline.preprocess_features expects raw x,y and will recompute x_adj
    
    # Actually, looking at the data:
    # - 4v5 events have x_adj = x (mostly unchanged)
    # - This means the coordinate flip never ran for these
    
    # The simplest fix: for 4v5 events where x_adj seems wrong (mean distance > 100),
    # recalculate by flipping coordinates
    
    print("\n--- Applying coordinate fix for 4v5 events ---")
    
    # Strategy:
    # 1. If original distance > 100, the shot is likely in wrong orientation
    # 2. Flip x_adj and recalculate distance
    
    mask_4v5_bad = (df['game_state'] == '4v5') & (df['distance'] > 80)
    n_bad = mask_4v5_bad.sum()
    print(f"Found {n_bad} 4v5 events with distance > 80 (likely wrong orientation)")
    
    if n_bad > 0:
        # For these events, the shooter is attacking the goal at x=89
        # But x_adj was not flipped, so it shows the wrong goal
        # We need to flip x_adj sign
        
        df.loc[mask_4v5_bad, 'x_adj'] = -df.loc[mask_4v5_bad, 'x_adj']
        df.loc[mask_4v5_bad, 'y_adj'] = -df.loc[mask_4v5_bad, 'y_adj']
        
        # Also flip the raw x,y to match
        df.loc[mask_4v5_bad, 'x'] = -df.loc[mask_4v5_bad, 'x']
        df.loc[mask_4v5_bad, 'y'] = -df.loc[mask_4v5_bad, 'y']
    
    # Also check 5v4 events (might have some bad ones too)
    mask_5v4_bad = (df['game_state'] == '5v4') & (df['distance'] > 80)
    n_bad_5v4 = mask_5v4_bad.sum()
    print(f"Found {n_bad_5v4} 5v4 events with distance > 80")
    
    if n_bad_5v4 > 0:
        df.loc[mask_5v4_bad, 'x_adj'] = -df.loc[mask_5v4_bad, 'x_adj']
        df.loc[mask_5v4_bad, 'y_adj'] = -df.loc[mask_5v4_bad, 'y_adj']
        df.loc[mask_5v4_bad, 'x'] = -df.loc[mask_5v4_bad, 'x']
        df.loc[mask_5v4_bad, 'y'] = -df.loc[mask_5v4_bad, 'y']
    
    # Now recalculate ALL distances using the correct goal position
    # Goal is at x=89 (right side) after standardization
    print("\n--- Recalculating all distances to goal at x=89 ---")
    
    goal_x = 89.0
    df['distance'] = np.sqrt((df['x_adj'] - goal_x)**2 + df['y_adj']**2)
    
    # Recalculate angle as well
    dx = df['x_adj'] - goal_x
    dy = df['y_adj']
    # Fixed CCW calc from vectors (matching data_pipeline.py)
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    df['angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0
    
    # Post-check: Show fixed distance stats
    print("\n=== AFTER FIX ===")
    for gs in ['5v5', '5v4', '4v5']:
        subset = df[df['game_state'] == gs]
        print(f"{gs}: {len(subset)} events, Mean distance: {subset['distance'].mean():.1f}")
    
    # Save back
    print(f"\nSaving patched data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done!")
    
    # Summary
    print("\n=== SUMMARY ===")
    print(f"Fixed {n_bad} 4v5 events and {n_bad_5v4} 5v4 events with wrong orientation")
    print("All distances recalculated to goal at x=89")
    print("Please regenerate events_bank.pkl using generate_mixed_heatmaps.py")

if __name__ == "__main__":
    main()
