import pandas as pd
import os

file_path = r"c:\Users\harri\Desktop\new_puck\analysis\gravity\player_gravity_season.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    players = df[df['player_name'].str.contains('Cates|Michkov', case=False, na=False)]
    cols = ['player_name', 'rel_on_puck_mean_dist_ft', 'rel_off_puck_mean_dist_ft', 'goals_on_ice_count']
    print(players[cols].to_string())
else:
    print(f"File not found: {file_path}")
