import pandas as pd
import numpy as np
import os

# We'll use the deep_dive_gravity.py logic but also capture speeds
cates_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\20242025\game_2024021146_goal_921_positions.csv"
michkov_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\20242025\game_2024020240_goal_609_positions.csv"

def get_avg_speed(file_path, target_pid):
    df = pd.read_csv(file_path)
    # Units
    if df['x'].abs().max() > 120:
         df['x'] = (df['x'] - 1200.0) / 12.0
         df['y'] = -(df['y'] - 510.0) / 12.0
         
    me = df[df['entity_id'].astype(str) == str(target_pid)].sort_values('frame_idx')
    if me.empty: return 0
    
    # Calculate velocity (ft/frame)
    # Assuming frames are consecutive. NHL Edge is usually 10Hz or 50Hz.
    # Let's just look at displacement per frame.
    dx = me['x'].diff()
    dy = me['y'].diff()
    dist = np.sqrt(dx**2 + dy**2)
    return dist.mean()

print(f"Cates Avg Displacement/Frame: {get_avg_speed(cates_file, 8480220):.4f}")
print(f"Michkov Avg Displacement/Frame: {get_avg_speed(michkov_file, 8484387):.4f}")
