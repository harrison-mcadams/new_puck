
import pandas as pd
import os
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config, rink

def patch_csv():
    csv_path = os.path.join(config.DATA_DIR, '20252026.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Original Mean Distance: {df['distance'].mean():.2f}")
    
    # Identify shots
    shot_types = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    mask_shot = df['event'].isin(shot_types)
    
    # For shots, check if they need flipping
    # We need to identify goal_x used.
    # We can deduce it: if (x-89)^2 + y^2 is closer to distance^2, it was 89.
    
    def apply_patch(row):
        if not mask_shot.loc[row.name]:
            return row['distance'], row['angle_deg']
            
        x, y, d = row['x'], row['y'], row['distance']
        if pd.isna(x) or pd.isna(y) or pd.isna(d):
            return d, row['angle_deg']
            
        if d > 100:
            # Check other goal
            # Goal positions
            g1, g2 = -89.0, 89.0
            
            # Which one was used?
            d1 = np.sqrt((x - g1)**2 + y**2)
            d2 = np.sqrt((x - g2)**2 + y**2)
            
            # If d is close to d1, then g1 was the target.
            if abs(d - d1) < 1.0:
                target_was = g1
                alt_target = g2
            else:
                target_was = g2
                alt_target = g1
                
            alt_d, alt_a = rink.calculate_distance_and_angle(x, y, alt_target, 0)
            if alt_d < 80:
                print(f"Patching shot: {row['player_name']} {d:.1f} -> {alt_d:.1f}")
                return alt_d, alt_a
                
        return d, row['angle_deg']

    # Apply
    # Using row-wise for safety but vectorized where possible would be faster.
    # For now, let's just do it.
    
    print("Applying patches...")
    res = df.apply(apply_patch, axis=1)
    df['distance'] = res.apply(lambda x: x[0])
    df['angle_deg'] = res.apply(lambda x: x[1])
    
    print(f"Patched Mean Distance: {df['distance'].mean():.2f}")
    
    # Save back
    df.to_csv(csv_path, index=False)
    print("Saved patched CSV.")

if __name__ == "__main__":
    patch_csv()
