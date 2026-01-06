
import pandas as pd
import sys
import os
import numpy as np # Added for np.sqrt
sys.path.append(os.getcwd())
from puck import config

def debug_kinematics():
    if len(sys.argv) < 3:
        print("Usage: python debug_frame_8.py <game_id> <goal_id>")
        return

    game_id = sys.argv[1]
    goal_id = sys.argv[2]
    # Wildcard search for file
    search_path = os.path.join("data", "edge_goals", "20242025", f"game_{game_id}_goal_{goal_id}_positions.csv")
    
    # Handle wildcard? Actually let's assume standard naming or use glob if needed.
    # But batch script uses glob. Here we can just try the standard format or look it up.
    # The user provided valid IDs that worked for other scripts.
    
    if not os.path.exists(search_path):
        import glob
        pat = os.path.join("data", "edge_goals", "20242025", f"game_{game_id}_goal_{goal_id}_*.csv")
        files = glob.glob(pat)
        if files:
            search_path = files[0]
        else:
            print(f"File not found for {game_id} {goal_id}")
            return
            
    print(f"Analyzing {search_path}")
    df = pd.read_csv(search_path)
    print(f"Columns: {df.columns}")
    print(f"Types: {df.dtypes}")
    
    df_puck = df[df['entity_type'] == 'puck'].copy()
    print(f"Puck Rows: {len(df_puck)}")
    print(f"Unique Frames: {df_puck['frame_idx'].unique()}")
    
    # Kinematics
    df_puck = df_puck.sort_values('frame_idx').reset_index(drop=True) # Ensure diff works on consecutive rows
    df_puck['vx'] = df_puck['x'].diff() / 0.1
    df_puck['vy'] = df_puck['y'].diff() / 0.1
    df_puck['speed'] = np.sqrt(df_puck['vx']**2 + df_puck['vy']**2)
    
    # Calculate Angle and Std
    df_puck['angle'] = np.degrees(np.arctan2(df_puck['vy'], df_puck['vx']))
    df_puck['angle_unwrapped'] = pd.Series(np.degrees(np.unwrap(np.radians(df_puck['angle']))), index=df_puck.index).fillna(0)
    df_puck['angle_std'] = df_puck['angle_unwrapped'].rolling(window=5, center=True, min_periods=1).std()

    # Calculate Net Alignment
    # Net Center roughly (89, 0) - check coordinate system
    # Typically X ranges -100 to 100.
    # If shot is towards +89 or -89 depends on period/play.
    # Assume standard attacking zone? Script uses 'net_x' logic.
    # We'll just print trajectory angle and (x,y) to deduce.
    
    # Inspect Frames directly
    print("Direct content check of df_puck for frames 16-20:")
    subset = df_puck[(df_puck['frame_idx'] >= 16) & (df_puck['frame_idx'] <= 20)].copy()
    print(subset[['frame_idx', 'speed', 'angle', 'angle_std']].to_string())
            
        # The following lines seem to be a remnant from the previous function and are not used in this new context.
        # They are kept as per the instruction to faithfully apply the change, even if syntactically odd.
        # dist = ((px-sx)**2 + (py-sy)**2)**0.5 if (px and sx) else None
        # print(f"{f:02d} | {px if px else '---'}, {py if py else '---'} | {spd if spd else '---' } | {sx if sx else '---'}, {sy if sy else '---'} | {dist if dist else '---'}")


if __name__ == "__main__":
    debug_kinematics()
