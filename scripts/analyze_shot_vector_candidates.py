
import pandas as pd
import numpy as np
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import config

def analyze_vectors():
    target_game_id = 2024020202
    target_goal_id = 328
    
    SHOOTER_ID = 8483930
    BLOCKER_ID = 8476467
    
    print(f"--- Analyzing Shot Vectors for Game {target_game_id} Goal {target_goal_id} ---")
    
    # 1. Load Data
    base_dir = config.DATA_DIR
    file_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{target_game_id}_goal_{target_goal_id}_positions.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    if col_id in df.columns:
        df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
        
    # Puck Data
    if 'entity_type' in df.columns:
        df_puck = df[df['entity_type'] == 'puck'].copy().sort_values('frame_idx')
    else:
        df_puck = df[df[col_id].isnull()].copy().sort_values('frame_idx')

    # Kinematics
    df_puck['dx'] = df_puck['x'].diff()
    df_puck['dy'] = df_puck['y'].diff()
    df_puck['vx'] = df_puck['dx'] / 0.1
    df_puck['vy'] = df_puck['dy'] / 0.1
    df_puck['speed'] = np.sqrt(df_puck['vx']**2 + df_puck['vy']**2)
    df_puck['angle'] = np.degrees(np.arctan2(df_puck['vy'], df_puck['vx']))
    
    # Linearity (Rolling Angle Std)
    df_puck['angle_std'] = df_puck['angle'].rolling(window=3, center=True).std()

    # Determine Net Direction
    # Check mean Vx. If negative, attacking -89.
    mean_vx = df_puck['vx'].mean()
    net_x = -89.0 if mean_vx < 0 else 89.0
    print(f"Detected Attack Direction: Net at X={net_x} (Mean Vx: {mean_vx:.1f})")

    candidates = []

    # 2. Iterate Frames 0-60
    for idx, row in df_puck.iterrows():
        if row['frame_idx'] > 80: break # Late
        if row['frame_idx'] < 5: continue # Too early
        
        f_idx = row['frame_idx']
        px, py = row['x'], row['y']
        
        # 1. Angle to Net
        dx_net = net_x - px
        dy_net = 0 - py
        angle_to_net = np.degrees(np.arctan2(dy_net, dx_net))
        
        # Circular Diff
        angle_diff = abs(row['angle'] - angle_to_net)
        angle_diff = angle_diff % 360
        if angle_diff > 180: angle_diff = 360 - angle_diff
        
        # 2. Dist to Shooter
        # Find shooter at this frame
        df_shooter = df[(df[col_id] == float(SHOOTER_ID)) & (df['frame_idx'] == f_idx)]
        dist_shooter = 999.9
        if not df_shooter.empty:
            sx, sy = df_shooter.iloc[0]['x'], df_shooter.iloc[0]['y']
            dist_shooter = np.sqrt((px-sx)**2 + (py-sy)**2)

        # 3. Linearity Check (Is it ballistic?)
        is_linear = row['angle_std'] < 15.0 # Fairly forgiving
        
        # Score Candidates
        # Ideally: Pointing at Net (< 40 deg dev), High Speed (> 20), Linear, Close to Shooter
        
        print(f"F{f_idx}: Spd {row['speed']:.1f} Ang {row['angle']:.1f} (ToNet {angle_to_net:.1f}, Dev {angle_diff:.1f}) ShootDist {dist_shooter:.1f}")
        
        if angle_diff < 90.0: # Very relaxed filter to capture everything
            candidates.append({
                'frame_idx': f_idx,
                'x': px, 'y': py,
                'speed': row['speed'],
                'angle': row['angle'],
                'angle_to_net': angle_to_net,
                'dev_deg': angle_diff,
                'dist_shooter': dist_shooter,
                'vx': row['vx'],
                'vy': row['vy']
            })

    print(f"\n--- Saving {len(candidates)} Candidates to CSV ---")
    df_candidates = pd.DataFrame(candidates)
    out_path = os.path.join(base_dir, 'analysis', 'candidate_shot_vectors.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_candidates.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    
    # Print top 5 best aligned
    if not df_candidates.empty:
        print("\nTop 5 Best Aligned Candidates:")
        print(df_candidates.sort_values('dev_deg').head(5))

if __name__ == "__main__":
    analyze_vectors()
