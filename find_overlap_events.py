import pandas as pd
import os

file_path = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\gravity_analysis.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    # Noah Cates (8480220) and Matvei Michkov (8484387)
    cates = df[df['player_id'] == 8480220][['game_id', 'event_id', 'rel_off_puck_mean_dist_ft']]
    michkov = df[df['player_id'] == 8484387][['game_id', 'event_id', 'rel_off_puck_mean_dist_ft']]
    
    overlap = pd.merge(cates, michkov, on=['game_id', 'event_id'], suffixes=('_cates', '_michkov'))
    print(f"Found {len(overlap)} overlapping events.")
    print(overlap.to_string())
else:
    print("File not found.")
