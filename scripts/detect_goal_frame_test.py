import pandas as pd
import os
import glob
import sys

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"

def test_goal_detection():
    # Find a sample file
    files = glob.glob(os.path.join(DATA_DIR, "*positions.csv"))
    if not files: return
    
    # Check a few
    for fpath in files[:5]:
        print(f"\nChecking: {os.path.basename(fpath)}")
        df = pd.read_csv(fpath)
        
        # Puck data
        # Check units first (pixels vs feet)
        x_max = df['x'].abs().max()
        if x_max > 120: # Likely pixels
             df['x'] = (df['x'] - 1200.0) / 12.0
             df['y'] = -(df['y'] - 510.0) / 12.0
        
        df_puck = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
        
        if df_puck.empty:
            print("  No puck data.")
            continue
            
        # Detect Goal
        # x > 89 or x < -89 AND |y| < 3
        
        goal_frames = df_puck[
            (df_puck['x'].abs() > 89.0) & 
            (df_puck['y'].abs() <= 3.0)
        ]
        
        if not goal_frames.empty:
            first_goal_frame = goal_frames.iloc[0]['frame_idx']
            total_frames = df['frame_idx'].max()
            print(f"  Goal detected at Frame: {first_goal_frame}")
            print(f"  Total Frames: {total_frames}")
            print(f"  Post-Goal Frames: {total_frames - first_goal_frame}")
            
            # Show Puck Pos at goal frame
            row = goal_frames.iloc[0]
            print(f"  Puck Pos: ({row['x']:.2f}, {row['y']:.2f})")
        else:
            print("  No goal detected (Puck never crossed line in net area).")
            # Show max x to see how close
            max_x = df_puck['x'].abs().max()
            print(f"  Max Puck X (abs): {max_x:.2f}")

if __name__ == "__main__":
    test_goal_detection()
