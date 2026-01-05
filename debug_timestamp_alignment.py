import json
import pandas as pd
import glob
import os

# Game 2024020608, Goal 307 (Event 307? Filename uses event id?)
# Filename pattern: game_2024020608_goal_307_*.csv
season = '20242025'
game_id = 2024020608
goal_id = 307

# Find files
edge_dir = f'data/edge_goals/{season}'
import random

files = glob.glob(f'data/edge_goals/{season}/*_edge.json')
sample = random.sample(files, 20)

print(f"Checking {len(sample)} random files for timestamp alignment...")
for json_path in sample:
    with open(json_path, 'r') as f:
        meta = json.load(f)
    if isinstance(meta, list): meta = meta[0]
    json_ts = meta.get('timeStamp')
    
    csv_path = json_path.replace('_edge.json', '_positions.csv')
    if not os.path.exists(csv_path): continue
    
    df = pd.read_csv(csv_path)
    if df.empty: continue
    
    min_ts = df['timestamp'].min()
    diff = json_ts - min_ts
    
    print(f"File {os.path.basename(json_path)}: JSON={json_ts}, CSV_Min={min_ts}, Diff={diff}")
    if diff != 0:
        print("  -> MISMATCH!")

print("Done.")
