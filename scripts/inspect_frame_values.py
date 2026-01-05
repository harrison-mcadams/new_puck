
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))
from puck import config

def inspect():
    target_game_id = 2024020202
    target_goal_id = 328
    base_dir = config.DATA_DIR
    file_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{target_game_id}_goal_{target_goal_id}_positions.csv')
    df = pd.read_csv(file_path)
    
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    if col_id in df.columns: df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
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
    df_puck['angle_std'] = df_puck['angle'].rolling(window=3, center=True).std()
    
    print("--- Frame 20-30 Inspection ---")
    print(df_puck[(df_puck['frame_idx'] >= 20) & (df_puck['frame_idx'] <= 30)][['frame_idx', 'speed', 'angle', 'angle_std']])

if __name__ == "__main__":
    inspect()
