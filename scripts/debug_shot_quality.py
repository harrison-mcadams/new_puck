
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add path for imports
sys.path.append(os.path.join(os.path.expanduser("~"), "Desktop", "new_puck"))

from puck import nhl_api
from puck import config

def analyze_shot_quality(game_id, goal_event_id):
    # 1. Load Tracking Data
    print(f"Loading data for Game {game_id}, Goal {goal_event_id}...")
    
    # Construct paths (assuming standard structure)
    # We'll use the discover_blocks helper logic simplistically or just raw verify path if known
    # Actually, let's look for the cached raw data if possible, or just re-fetch using nhl_api?
    # Better to rely on what discover_blocks does: load the json/csv from specific dir.
    
    # Load specific file
    base_dir = config.DATA_DIR
    file_path = os.path.join(base_dir, 'edge_goals', '20242025', f'game_{game_id}_goal_{goal_event_id}_positions.csv')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    
    print(f"Columns found: {df.columns.tolist()}")
    
    # Robust Puck Filter
    if 'entity_type' in df.columns:
        df_puck = df[df['entity_type'] == 'puck'].copy()
    elif 'player_id' in df.columns:
        df_puck = df[df['player_id'].isnull()].copy()
    else:
        # Fallback: ID is null? Or look for specific team_id?
        # Assuming entity_type exists based on previous script success
        if 'team_id' in df.columns:
             # Sometimes puck has nan team_id
             df_puck = df[df['team_id'].isnull()].copy()
        else:
             print("Cannot identify puck row.")
             return
            
    df_puck = df_puck.sort_values('frame_idx')
    
    if df_puck.empty:
        print("No puck data found!")
        return

    # 2. Calculate Kinematics
    # 10Hz data
    dt = 0.1
    df_puck['vx'] = df_puck['x'].diff() / dt
    df_puck['vy'] = df_puck['y'].diff() / dt
    df_puck['speed'] = np.sqrt(df_puck['vx']**2 + df_puck['vy']**2)
    df_puck['angle'] = np.degrees(np.arctan2(df_puck['vy'], df_puck['vx']))
    
    # 3. Calculate Linearity (Rolling R^2 or Angle Stability)
    # Angle Stability: Std Dev of Angle over 5 frames
    df_puck['angle_std'] = df_puck['angle'].rolling(window=5, center=True).std()
    
    # 4. Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Speed Plot
    ax1.plot(df_puck['frame_idx'], df_puck['speed'], label='Speed (ft/s)', color='blue')
    ax1.axhline(40, color='r', linestyle='--', label='Shot Threshold (40)')
    ax1.set_ylabel('Speed (ft/s)')
    ax1.legend()
    ax1.set_title(f'Puck Kinematics: Game {game_id} Goal {goal_event_id}')
    
    # Angle Stability Plot (Lower is straighter)
    ax2.plot(df_puck['frame_idx'], df_puck['angle_std'], label='Angle Std Dev (5-frame)', color='green')
    ax2.axhline(5, color='orange', linestyle='--', label='Stability Threshold (5deg)')
    ax2.set_ylabel('Angle Std Dev (deg)')
    ax2.set_xlabel('Frame Index')
    ax2.legend()
    
    # Highlight potential shots (High Speed + Low Angle Std)
    potential_shots = df_puck[ (df_puck['speed'] > 30) & (df_puck['angle_std'] < 5) ]
    if not potential_shots.empty:
        ax1.scatter(potential_shots['frame_idx'], potential_shots['speed'], color='red', s=10, label='Candidate')
        
    out_path = f'analysis/plots/diagnostic_shot_{game_id}_{goal_event_id}.png'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Diagnostic plot saved to {out_path}")
    
    # Load Shooter
    shooter_id = 8483930
    col_id = 'player_id' if 'player_id' in df.columns else 'entity_id'
    
    print("\nPotential Shot Segments (Speed > 30, Angle Std < 5):")
    if not potential_shots.empty:
        potential_shots['group'] = (potential_shots['frame_idx'].diff() > 1).cumsum()
        print(f"Found {potential_shots['group'].nunique()} segments.")
        for g, group in potential_shots.groupby('group'):
            head = group.head(1)
            f_idx = head['frame_idx'].iloc[0]
            spd = head['speed'].iloc[0]
            ang_std = head['angle_std'].iloc[0]
            
            # Check proximity
            df_shooter = df[(df[col_id] == float(shooter_id)) & (df['frame_idx'] == f_idx)]
            dist_str = "N/A"
            if not df_shooter.empty:
                sx, sy = df_shooter.iloc[0]['x'], df_shooter.iloc[0]['y']
                px, py = head['x'].iloc[0], head['y'].iloc[0]
                dist = np.sqrt((px-sx)**2 + (py-sy)**2)
                dist_str = f"{dist:.1f} ft"
                
            print(f"  Segment {g}: Frame {f_idx}, Speed {spd:.1f}, Angle Std {ang_std:.2f} -> Dist to Shooter: {dist_str}")

    else:
        print("No segments found matching criteria.")
        print(f"Max Speed found: {df_puck['speed'].max()}")
        print(f"Min Angle Std found: {df_puck['angle_std'].min()}")



if __name__ == "__main__":
    # We need to find where the csv is.
    # In `identify_shot_attempts.py`, it loads: 
    # f"data/edge_goals/{season_str}/{game_id}_{goal_id}.csv" or similar
    
    target_game = 2024020202
    target_goal = 328
    
    # Actually call the function
    analyze_shot_quality(target_game, target_goal)

    
