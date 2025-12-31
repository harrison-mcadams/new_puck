import pandas as pd
import numpy as np
import os
import sys

POSSESSION_THRESHOLD_FT = 6.0 
DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"

def get_sample_file():
    for f in os.listdir(DATA_DIR):
        if f.endswith("game_2025020583_goal_1007_positions.csv"):
            return os.path.join(DATA_DIR, f)
    # Fallback
    for f in os.listdir(DATA_DIR):
        if f.endswith("_positions.csv"):
            return os.path.join(DATA_DIR, f)
    return None

def prototype_possession():
    fpath = get_sample_file()
    if not fpath:
        print("No position files found.")
        return

    print(f"Analyzing: {os.path.basename(fpath)}")
    df = pd.read_csv(fpath)
    
    df_puc = df[df['entity_type'] == 'puck'].set_index('frame_idx')[['x', 'y']]
    df_play = df[df['entity_type'] == 'player']
    
    frames = sorted(df['frame_idx'].unique())
    timeline = []
    
    for frame in frames:
        if frame not in df_puc.index: continue
        
        puck_pos = df_puc.loc[frame]
        px, py = puck_pos['x'], puck_pos['y']
        
        players_frame = df_play[df_play['frame_idx'] == frame]
        if players_frame.empty: continue
            
        dists = np.sqrt((players_frame['x'] - px)**2 + (players_frame['y'] - py)**2)
        min_dist = dists.min()
        
        closest_player = players_frame.loc[dists.idxmin()]
        
        possessor = "Loose"
        team = "-"
        
        if min_dist <= POSSESSION_THRESHOLD_FT:
            possessor = str(closest_player['entity_id'])
            team = str(closest_player['team_id'])
            
        timeline.append({
            'frame': frame,
            'possessor': possessor,
            'team': team,
            'dist': min_dist
        })
        
    df_time = pd.DataFrame(timeline)
    df_time['grp'] = (df_time['possessor'] != df_time['possessor'].shift()).cumsum()
    
    events = df_time.groupby('grp').agg(
        Start=('frame', 'first'),
        End=('frame', 'last'),
        Duration=('frame', 'count'),
        Possessor=('possessor', 'first'),
        Team=('team', 'first'),
        MinDist=('dist', 'mean')
    ).reset_index(drop=True)
    
    print(events.to_string())

if __name__ == "__main__":
    prototype_possession()
