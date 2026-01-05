
import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from puck import config

def debug_frames():
    target_game_id = 2024020118
    target_goal_id = 319
    target_block_id = 317 # Jensen
    
    # Load Data
    pos_path = f"data/edge_goals/20242025/game_{target_game_id}_goal_{target_goal_id}_positions.csv"
    df = pd.read_csv(pos_path)
    
    # Normalize
    def norm_x(x): return (x - 1200.0) / 12.0
    def norm_y(y): return -(y - 510.0) / 12.0
    
    if df['x'].max() > 1000:
        df['x'] = norm_x(df['x'])
        df['y'] = norm_y(df['y'])
        
    df_puck = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
    
    # Calculate Kinematics manually
    df_puck['dx'] = df_puck['x'].diff()
    df_puck['dy'] = df_puck['y'].diff()
    df_puck['dt'] = 0.1
    df_puck['speed'] = (df_puck['dx']**2 + df_puck['dy']**2)**0.5 / 0.1
    
    # Determine ID column
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    df[col_id] = pd.to_numeric(df[col_id], errors='coerce')
    
    # Shooter
    # Nick Jensen ID: 8475324
    shooter_id = 8475324.0 
    df_shooter = df[(df[col_id] == shooter_id)]
    
    print("Frame | Puck X,Y | Speed | Shooter X,Y | Dist | AngleStd")
    
    for f in range(4, 11):
        pk = df_puck[df_puck['frame_idx'] == f]
        sh = df_shooter[df_shooter['frame_idx'] == f]
        
        if not pk.empty:
            px, py = pk['x'].iloc[0], pk['y'].iloc[0]
            spd = pk['speed'].iloc[0]
        else:
            px, py, spd = None, None, None
            
        if not sh.empty:
            sx, sy = sh['x'].iloc[0], sh['y'].iloc[0]
        else:
            sx, sy = None, None
            
        dist = ((px-sx)**2 + (py-sy)**2)**0.5 if (px and sx) else None
        
        print(f"{f:02d} | {px if px else '---'}, {py if py else '---'} | {spd if spd else '---' } | {sx if sx else '---'}, {sy if sy else '---'} | {dist if dist else '---'}")

if __name__ == "__main__":
    debug_frames()
