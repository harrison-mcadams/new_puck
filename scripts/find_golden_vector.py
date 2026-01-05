
import pandas as pd
import numpy as np
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import config

def find_golden_vector():
    target_game_id = 2024020202
    target_goal_id = 328
    
    # User Hint: "A couple of frames back" from 24
    SEARCH_WINDOW = range(20, 27) 
    
    BLOCKER_ID = 8476467
    SHOOTER_ID = 8483930
    
    print(f"--- Search for Golden Vector (F{min(SEARCH_WINDOW)}-F{max(SEARCH_WINDOW)}) ---")
    
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
    df_puck['d_angle'] = df_puck['angle'].diff()

    # 2. Iterate Search Window
    for start_frame in SEARCH_WINDOW:
        # Get Start Frame Puck Info
        row_start = df_puck[df_puck['frame_idx'] == start_frame]
        if row_start.empty: continue
        row_start = row_start.iloc[0]
        
        # Check Net Orientation
        # Net is at X=89, Y=0 (approx)
        # Vector angle to net
        dx_net = 89.0 - row_start['x']
        dy_net = 0.0 - row_start['y']
        angle_to_net = np.degrees(np.arctan2(dy_net, dx_net))
        
        # Deviation
        shot_angle = row_start['angle']
        deviation = abs(shot_angle - angle_to_net)
        # Normalize deviation to -180..180 if needed, but simple abs usually ok for close angles
        
        print(f"\nCandidate Frame {start_frame}:")
        print(f"  Loc: ({row_start['x']:.1f}, {row_start['y']:.1f})")
        print(f"  Speed: {row_start['speed']:.1f} fps")
        print(f"  Angle: {shot_angle:.1f} (To Net: {angle_to_net:.1f} -> Dev: {deviation:.1f} deg)")
        
        # Trace Forward
        # Look for the FIRST point where:
        # 1. Puck hits Blocker (Proximity < 5ft)
        # 2. Puck Deflects (Angle Change > 20 deg)
        # 3. Time is later than start
        
        df_trace = df_puck[df_puck['frame_idx'] > start_frame].copy()
        
        hit_found = False
        for _, row_trace in df_trace.iterrows():
            f_trace = row_trace['frame_idx']
            
            # Check for Deflection
            d_angle = row_trace['d_angle']
            # Also precise angle change from ORIGINAL vector?
            # Or just local deflection? User said "deflected precisely when..."
            
            # Check Blocker Proximity
            df_blocker = df[(df[col_id] == float(BLOCKER_ID)) & (df['frame_idx'] == f_trace)]
            dist_to_blocker = 999.9
            if not df_blocker.empty:
                bx, by = df_blocker.iloc[0]['x'], df_blocker.iloc[0]['y']
                px, py = row_trace['x'], row_trace['y']
                dist_to_blocker = np.sqrt((px-bx)**2 + (py-by)**2)
            
            if dist_to_blocker < 6.0:
                 print(f"    -> [HIT?] Frame {f_trace}: Dist {dist_to_blocker:.1f}ft (dAngle {d_angle:.1f})")
                 hit_found = True
                 
            # Stop if too far?
            if f_trace > start_frame + 20: break
            
        if not hit_found:
            print("    -> No proximity to blocker found in next 20 frames.")

if __name__ == "__main__":
    find_golden_vector()
