import pandas as pd
import os

file_path = r"c:\Users\harri\Desktop\new_puck\analysis\gravity\player_gravity_season.csv"
out_path = r"c:\Users\harri\Desktop\new_puck\player_debug_stats.txt"

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    # Search for Noah Cates and Matvei Michkov. Also handle the case where "case" might be short for Cates.
    players = df[df['player_name'].str.contains('Cates|Michkov', case=False, na=False)]
    
    with open(out_path, 'w') as f:
        f.write("--- Detailed Gravity Stats ---\n")
        f.write(players.to_string())
    print(f"Results written to {out_path}")
else:
    print(f"File not found: {file_path}")
