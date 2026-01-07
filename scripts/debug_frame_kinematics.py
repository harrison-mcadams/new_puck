
import pandas as pd
import numpy as np
import os
import sys

def calculate_kinematics(df):
    df = df.sort_values('frame_idx')
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    
    # Logic from identify_shot_attempts.py
    if 'timestamp' in df.columns:
        t_diff = df['timestamp'].diff().median()
        # FIX: Tread delta=1 as 0.1s (10Hz)
        if 0.9 <= t_diff <= 1.1:
            df['dt'] = 0.1
        elif 90 <= t_diff <= 110: 
            df['dt'] = 0.1
        else:
            if t_diff < 0.2: df['dt'] = t_diff
            elif t_diff > 10.0: df['dt'] = t_diff / 1000.0
            else: df['dt'] = 0.1 
    else:
        df['dt'] = 0.1
        
    df['vx'] = df['dx'] / df['dt']
    df['vy'] = df['dy'] / df['dt']
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['angle'] = np.degrees(np.arctan2(df['vy'], df['vx']))
    
    # Linearity
    angles_rad = np.radians(df['angle'].fillna(0).values)
    unwrapped = np.unwrap(angles_rad)
    df['angle_unwrapped'] = np.degrees(unwrapped)
    df['angle_std_linear'] = df['angle_unwrapped'].rolling(window=5, center=True, min_periods=1).std()
    
    df['accel'] = df['speed'].diff()
    return df

def main():
    game_id = 2023020033
    goal_id = 195
    path = f"data/edge_goals/20232024/game_{game_id}_goal_{goal_id}_positions.csv"
    
    print(f"Loading {path}...")
    df = pd.read_csv(path)
    
    # Cast IDs
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    if col_id in df.columns:
        df[col_id] = pd.to_numeric(df[col_id], errors='coerce')

    df_puck = df[df['entity_type'] == 'puck'].copy()
    
    # Normalization check
    if df_puck['x'].abs().max() > 200:
         print("DEBUG: Normalizing coordinates...")
         def norm_x(x): return (x - 1200.0) / 12.0
         def norm_y(y): return -(y - 510.0) / 12.0
         df_puck['x'] = norm_x(df_puck['x'])
         df_puck['y'] = norm_y(df_puck['y'])

    df_puck = calculate_kinematics(df_puck)
    
    # Filter frames of interest
    subset = df_puck[(df_puck['frame_idx'] >= 45) & (df_puck['frame_idx'] <= 70)]
    
    print("\nFrame | X     | Y     | Speed | Accel | AngleStd | Trigger?")
    print("-" * 65)
    
    for _, row in subset.iterrows():
        f = int(row['frame_idx'])
        s = row['speed']
        a = row['accel']
        astd = row['angle_std_linear']
        
        # Original Conditions (for speed 100fps-like scale)
        # s > 15.0 and a > 10.0 was for the inflated values? 
        # Wait, previous values were ~800. So 15.0 was TINY.
        # Now speed will be ~84.8. 
        # Existing checks in identify_shot_attempts:
        # High Speed: speed > 10.0 & astd < 35.0
        # Burst: speed > 15.0 & accel > 10.0
        
        # Trigger conditions
        trigger_high_speed = (s > 10.0) and (astd < 35.0)
        trigger_burst = (s > 15.0) and (a > 10.0)
        
        mark = ""
        if trigger_high_speed: mark += " [HI-SPD]"
        if trigger_burst: mark += " [BURST]"
        
        print(f"{f:5d} | {row['x']:5.1f} | {row['y']:5.1f} | {s:5.1f} | {a:5.1f} | {astd:8.1f} | {mark}")

if __name__ == "__main__":
    main()
