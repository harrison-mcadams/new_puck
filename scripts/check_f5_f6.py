
import pandas as pd
import sys
import os
import numpy as np

def check():
    pos_path = "data/edge_goals/20242025/game_2024020118_goal_319_positions.csv"
    df = pd.read_csv(pos_path)
    
    # Cast IDs
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
    
    shooter_id = 8475324.0 # Jensen
    
    for f in [5, 6]:
        print(f"--- FRAME {f} ---")
        puck = df[(df['entity_type'] == 'puck') & (df['frame_idx'] == f)]
        shooter = df[(df[col_id] == shooter_id) & (df['frame_idx'] == f)]
        
        px, py, sx, sy = None, None, None, None
        
        if not puck.empty:
            px = puck.iloc[0]['x']
            py = puck.iloc[0]['y']
            print(f"Puck: {px}, {py}")
        else:
            print("Puck: MISSING")
            
        if not shooter.empty:
            sx = shooter.iloc[0]['x']
            sy = shooter.iloc[0]['y']
            print(f"Shooter: {sx}, {sy}")
        else:
            print("Shooter: MISSING")
            
        if px is not None and sx is not None:
            dist = np.sqrt((px-sx)**2 + (py-sy)**2)
            print(f"Dist: {dist}")
        else:
            print("Dist: NaN")

if __name__ == "__main__":
    check()
