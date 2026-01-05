
import pandas as pd
import numpy as np
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import config

def debug_immediate_block():
    # Parse Args
    if len(sys.argv) > 2:
        target_game_id = int(sys.argv[1])
        target_goal_id = int(sys.argv[2])
    else:
        target_game_id = 2024020202
        target_goal_id = 328
    
    # Confirmed Shot Origin
    SHOT_FRAME_IDX = 24
    
    # Target Blocker
    BLOCKER_ID = 8476467
    
    print(f"--- Debugging Immediate Block for Game {target_game_id} Goal {target_goal_id} ---")
    print(f"Focusing on Frames {SHOT_FRAME_IDX} to {SHOT_FRAME_IDX + 20}")
    
    # 1. Load Data
    base_dir = config.DATA_DIR
    file_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{target_game_id}_goal_{target_goal_id}_positions.csv')
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    
    # robust casting
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
    df_puck['d_angle'] = df_puck['angle'].diff() # Simple diff for visualization

    # 2. Iterate Frames of Interest (Release Instant)
    df_window = df_puck[(df_puck['frame_idx'] >= SHOT_FRAME_IDX - 1) & (df_puck['frame_idx'] <= SHOT_FRAME_IDX + 6)]
    
    for _, row in df_window.iterrows():
        f_idx = row['frame_idx']
        px, py = row['x'], row['y']
        
        # Find all players at this frame
        df_frame = df[df['frame_idx'] == f_idx]
        
        # Filter for players only
        if 'entity_type' in df.columns:
            df_players = df_frame[df_frame['entity_type'] == 'player']
        else:
             df_players = df_frame[df_frame[col_id].notnull()]

        # Distances
        closest_dist = 999.9
        closest_id = None
        target_dist = 999.9
        
        for _, p_row in df_players.iterrows():
            pid = p_row.get(col_id)
            dist = np.sqrt((px - p_row['x'])**2 + (py - p_row['y'])**2)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_id = pid
            
            if pid == float(BLOCKER_ID):
                target_dist = dist
                
        print(f"F{f_idx}: Loc({px:.1f}, {py:.1f}) Spd {row['speed']:.1f} Ang {row['angle']:.1f} (d {row['d_angle']:.1f})")
        print(f"    Closest: {closest_id} ({closest_dist:.1f}ft)")
        print(f"    Target ({BLOCKER_ID}): {target_dist:.1f}ft")
        
        # Heuristic check
        if target_dist < 5.0:
            print(f"    *** TARGET NEARBY ***")
        if closest_dist < 3.0:
            print(f"    *** CONTACT IMMINENT ***")

if __name__ == "__main__":
    debug_immediate_block()
