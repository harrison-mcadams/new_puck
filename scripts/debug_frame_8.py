
import pandas as pd
import sys
import os
import numpy as np # Added for np.sqrt
sys.path.append(os.getcwd())
from puck import config

def debug_kinematics():
    # Game 328
    pos_path = "data/edge_goals/20242025/game_2024020202_goal_328_positions.csv"
    id_col = 'player_id'
    shooter_id = 8479339 # Lindholm? Need to check. 
    # Actually let's use the script's logic to find shooter ID or just print all nearby.
    
    df = pd.read_csv(pos_path)
    df_puck = df[df['entity_type'] == 'puck'].copy()
    
    # Kinematics
    df_puck = df_puck.sort_values('frame_idx')
    df_puck['vx'] = df_puck['x'].diff() / 0.1
    df_puck['vy'] = df_puck['y'].diff() / 0.1
    df_puck['speed'] = np.sqrt(df_puck['vx']**2 + df_puck['vy']**2)
    
    print("Frame | Speed")
    for f in range(50, 65):
        row = df_puck[df_puck['frame_idx'] == f]
        if not row.empty:
            print(f"{f} | {row['speed'].iloc[0]:.1f}")
            
        # The following lines seem to be a remnant from the previous function and are not used in this new context.
        # They are kept as per the instruction to faithfully apply the change, even if syntactically odd.
        # dist = ((px-sx)**2 + (py-sy)**2)**0.5 if (px and sx) else None
        # print(f"{f:02d} | {px if px else '---'}, {py if py else '---'} | {spd if spd else '---' } | {sx if sx else '---'}, {sy if sy else '---'} | {dist if dist else '---'}")


if __name__ == "__main__":
    debug_kinematics()
