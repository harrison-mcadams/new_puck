import pandas as pd
import numpy as np
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck.possession import infer_possession_events

def analyze_play(file_path, target_pid, player_name):
    print(f"\n--- Analyzing {player_name} in {os.path.basename(file_path)} ---")
    df = pd.read_csv(file_path)
    
    # 1. Units & Norm
    if df['x'].abs().max() > 120:
         df['x'] = (df['x'] - 1200.0) / 12.0
         df['y'] = -(df['y'] - 510.0) / 12.0
    
    # Normalize attacking end (positive X is O-zone)
    last_frame = df['frame_idx'].max()
    end_data = df[(df['entity_type'] == 'player') & (df['frame_idx'] > last_frame - 50)]
    if not end_data.empty and end_data['x'].mean() < 0:
        df['x'] = -df['x']
        df['y'] = -df['y']
    
    # 2. Identify Teams
    puck_data = df[df['entity_type'] == 'puck']
    player_data = df[df['entity_type'] == 'player']
    
    # Find our player's team
    me = player_data[player_data['entity_id'] == target_pid]
    if me.empty:
        # Try string match
        me = player_data[player_data['entity_id'].astype(str) == str(target_pid)]
    
    if me.empty:
        print(f"Player {target_pid} not found in data.")
        return
    
    my_team_id = me.iloc[0]['team_id']
    
    # Defenders (Opposing team forwards/defense)
    opponents = player_data[player_data['team_id'] != my_team_id]
    
    # 3. Possession
    poss_events = infer_possession_events(df, threshold_ft=6.0)
    poss_map = {}
    if not poss_events.empty:
        # Convert to string for consistent lookup if needed
        # but entity_id is already used in poss_pid comparison in analyze_gravity.py
        for _, pev in poss_events.iterrows():
            if pev['is_possession']:
                for f in range(int(pev['start_frame']), int(pev['end_frame']) + 1):
                    poss_map[f] = str(pev['player_id'])

    # 4. Detailed Stats
    frames = sorted(me['frame_idx'].unique())
    play_stats = []
    
    for f in frames:
        my_frame = me[me['frame_idx'] == f].iloc[0]
        mx, my = my_frame['x'], my_frame['y']
        
        poss_pid = poss_map.get(f)
        on_puck = (poss_pid == str(target_pid))
        
        opps = opponents[opponents['frame_idx'] == f]
        if not opps.empty:
            dists = np.sqrt((opps['x'] - mx)**2 + (opps['y'] - my)**2)
            nearest = dists.min()
            mean_dist = dists.mean()
            
            play_stats.append({
                'frame': f,
                'x': mx, 'y': my,
                'on_puck': on_puck,
                'nearest_defender': nearest,
                'mean_defender_dist': mean_dist,
                'has_possession': (poss_pid is not None)
            })

    df_stats = pd.DataFrame(play_stats)
    
    # Summary
    print(f"Play Duration: {len(df_stats)} frames")
    print(f"X range: {df_stats['x'].min():.1f} to {df_stats['x'].max():.1f}")
    
    on_puck_df = df_stats[df_stats['on_puck']]
    off_puck_df = df_stats[~df_stats['on_puck']]
    
    if not on_puck_df.empty:
        print(f"On-Puck Frames: {len(on_puck_df)}")
        print(f"  Nearest Defender Avg: {on_puck_df['nearest_defender'].mean():.2f} ft")
        print(f"  Mean Defender Avg: {on_puck_df['mean_defender_dist'].mean():.2f} ft")
    else:
        print("Never had possession in this clip.")
        
    if not off_puck_df.empty:
        print(f"Off-Puck Frames: {len(off_puck_df)}")
        print(f"  Nearest Defender Avg: {off_puck_df['nearest_defender'].mean():.2f} ft")
        print(f"  Mean Defender Avg: {off_puck_df['mean_defender_dist'].mean():.2f} ft")

# Noah Cates (PHI) - 8480220
# Matvei Michkov (PHI) - 8484387

cates_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\20242025\game_2024021146_goal_921_positions.csv"
michkov_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\20242025\game_2024020240_goal_609_positions.csv"

analyze_play(cates_file, 8480220, "Noah Cates")
analyze_play(michkov_file, 8484387, "Matvei Michkov")
