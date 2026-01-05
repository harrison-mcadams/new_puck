
import pandas as pd
import sys
import os

def check():
    pos_path = "data/edge_goals/20242025/game_2024020118_goal_319_positions.csv"
    df_pos = pd.read_csv(pos_path)
    df_puck = df_pos[df_pos['entity_type'] == 'puck'].copy()
    
    # Check Frame 8 Raw
    p8 = df_puck[df_puck['frame_idx'] == 8].iloc[0]
    print(f"Raw Frame 8 Y: {p8['y']}")
    
    # Check Normalization Condition
    xmax = df_puck['x'].abs().max()
    print(f"Max Abs X: {xmax}")
    
    if xmax > 200:
        print("Normalization Triggered")
        df_puck['y'] = -(df_puck['y'] - 510.0) / 12.0
    else:
        print("Normalization Skipped")
        
    p8_new = df_puck[df_puck['frame_idx'] == 8].iloc[0]
    print(f"Processed Frame 8 Y: {p8_new['y']}")

if __name__ == "__main__":
    check()
