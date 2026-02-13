"""
fix_all_distances.py

Comprehensive fix for shot coordinate standardization.
Ensures ALL shots have:
1. x_adj > 0 (attacking towards goal at x=89)
2. Distance calculated correctly to goal at x=89

Uses the same logic as data_pipeline.py but applied as a one-time patch.
"""

import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def main():
    csv_path = os.path.join(config.DATA_DIR, '20252026.csv')
    
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Filter to shot events only for coordinate normalization
    shot_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    mask_shot = df['event'].isin(shot_events)
    
    print(f"\nTotal shots: {mask_shot.sum()}")
    
    # Pre-check
    print("\n=== BEFORE FIX ===")
    for gs in ['5v5', '5v4', '4v5']:
        subset = df[(df['game_state'] == gs) & mask_shot]
        print(f"{gs}: Mean x_adj={subset['x_adj'].mean():.1f}, Mean dist={subset['distance'].mean():.1f}")
    
    # CRITICAL FIX: 
    # All shots should have x_adj > 0 (attacking right towards goal at x=89)
    # If x_adj < 0, the shot is in the wrong orientation - flip it
    
    print("\n--- Standardizing all shots to attack right (x > 0) ---")
    
    # Find shots with negative x_adj (wrong orientation)
    mask_negative_x = mask_shot & (df['x_adj'] < 0)
    n_negative = mask_negative_x.sum()
    print(f"Found {n_negative} shots with x_adj < 0 (wrong orientation)")
    
    if n_negative > 0:
        # Flip to positive orientation
        df.loc[mask_negative_x, 'x_adj'] = -df.loc[mask_negative_x, 'x_adj']
        df.loc[mask_negative_x, 'y_adj'] = -df.loc[mask_negative_x, 'y_adj']
        df.loc[mask_negative_x, 'x'] = -df.loc[mask_negative_x, 'x']
        df.loc[mask_negative_x, 'y'] = -df.loc[mask_negative_x, 'y']
    
    # Now recalculate distance and angle for ALL shots
    # This ensures consistency even for shots that were already correct
    print("\n--- Recalculating distance and angle for all shots ---")
    
    goal_x = 89.0
    
    # Distance: Euclidean from (x_adj, y_adj) to (89, 0)
    df.loc[mask_shot, 'distance'] = np.sqrt(
        (df.loc[mask_shot, 'x_adj'] - goal_x)**2 + 
        df.loc[mask_shot, 'y_adj']**2
    )
    
    # Angle: Using the same formula as data_pipeline.py
    dx = df.loc[mask_shot, 'x_adj'] - goal_x
    dy = df.loc[mask_shot, 'y_adj']
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    df.loc[mask_shot, 'angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0
    
    # Post-check
    print("\n=== AFTER FIX ===")
    for gs in ['5v5', '5v4', '4v5']:
        subset = df[(df['game_state'] == gs) & mask_shot]
        print(f"{gs}: Mean x_adj={subset['x_adj'].mean():.1f}, Mean dist={subset['distance'].mean():.1f}")
    
    # Verify all shots now have positive x_adj
    remaining_negative = (mask_shot & (df['x_adj'] < 0)).sum()
    print(f"\nRemaining shots with x_adj < 0: {remaining_negative}")
    
    # Verify distance distribution
    print("\nDistance distribution for PP states:")
    for gs in ['5v4', '4v5']:
        subset = df[(df['game_state'] == gs) & mask_shot]
        print(f"  {gs}: 0-40ft: {(subset['distance'] < 40).sum()}, 40-80ft: {((subset['distance'] >= 40) & (subset['distance'] < 80)).sum()}, 80+ft: {(subset['distance'] >= 80).sum()}")
    
    # Save
    print(f"\nSaving fixed data to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print("Done!")
    
    print("\n=== NEXT STEPS ===")
    print("1. Regenerate events_bank.pkl: python scripts/generate_mixed_heatmaps.py")
    print("2. Test matchup simulation: python scripts/matchup.py NYR NJD")

if __name__ == "__main__":
    main()
