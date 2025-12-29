
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def test_flip_heuristic():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for significant location disagreement (> 20ft)
    mask_bad_loc = df['dist_diff'].abs() > 20.0
    bad_locs = df[mask_bad_loc].copy()
    
    print(f"Total shots with >20ft disagreement: {len(bad_locs)}")
    
    # Heuristic: If we flip our X, does it match MP better?
    # MP X is 'xCord'. Our X is 'x'.
    # Actually, MP 'shotDistance' is the ground truth we want to match.
    
    # Let's try flipping X and recalculating distance to whichever goal we were attacking.
    # Note: 'x' in deep dive is the raw x from our parser.
    # We need to know which goal we were attacking. 
    # Based on our parser logic:
    # Sergachev (82) -> 25ft to +89. So we were attacking +89.
    
    def get_corrected_dist(row):
        # We need to deduce which goal was used.
        # d_pos = sqrt((x-89)^2 + y^2)
        # d_neg = sqrt((x+89)^2 + y^2)
        # One of these is row['distance'].
        
        d_raw = row['distance']
        x = row['x']
        y = row['y']
        
        # d_alt = distance to the OTHER goal
        d_alt = np.sqrt(((-x) - 89)**2 + (-y)**2) # Simple mirroring
        # Or just calculate distance to the opposite intended goal?
        # If we were attacking 89, now we attack -89? 
        # No, if the coordinate was mirrored, it means x_actual = -x_raw and y_actual = -y_raw.
        # Goal stays the same? Or goal flips too?
        # In NHL API, usually the goal flips.
        
        # Let's try the simplest: If distance > 100, try the other goal.
        d1 = np.sqrt((x - 89)**2 + y**2)
        d2 = np.sqrt((x + 89)**2 + y**2)
        
        d_best_raw = min(d1, d2)
        return d_best_raw

    bad_locs['dist_best_fit'] = bad_locs.apply(get_corrected_dist, axis=1)
    
    # Check how many now agree with MP (within 5ft)
    bad_locs['new_diff'] = (bad_locs['dist_best_fit'] - bad_locs['shotDistance']).abs()
    resolved = bad_locs[bad_locs['new_diff'] < 5.0]
    
    print(f"Resolved with 'Best Goal' heuristic: {len(resolved)} ({len(resolved)/len(bad_locs)*100:.1f}%)")
    
    if not resolved.empty:
        print("\n--- Samples of Resolved Glitches ---")
        cols = ['game_id', 'shooterName', 'x', 'y', 'distance', 'shotDistance', 'dist_best_fit', 'new_diff']
        print(resolved[cols].head(10).to_string(index=False))

if __name__ == "__main__":
    test_flip_heuristic()
