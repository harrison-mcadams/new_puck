import pandas as pd
import os

analysis_file = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\gravity_analysis.csv"

if os.path.exists(analysis_file):
    df = pd.read_csv(analysis_file)
    # Noah Cates (8480220) and Matvei Michkov (8484387)
    target_pids = [8480220, 8484387]
    df_targets = df[df['player_id'].isin(target_pids)].copy()
    
    # Check average of the raw nearest distances
    summary = df_targets.groupby(['player_name', 'season']).agg({
        'on_puck_nearest_dist_ft': 'mean',
        'off_puck_nearest_dist_ft': 'mean',
        'rel_on_puck_mean_dist_ft': 'mean',
        'rel_off_puck_mean_dist_ft': 'mean',
        'game_id': 'count'
    })
    print(summary.to_string())
else:
    print(f"File not found: {analysis_file}")
