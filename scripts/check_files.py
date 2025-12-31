import pandas as pd
import os

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
meta_path = os.path.join(DATA_DIR, "metadata.csv")

df = pd.read_csv(meta_path)
df_focus = df[df['season'] == 20252026]

print(f"Checking 2025-2026 files (Total: {len(df_focus)})")

found = 0
missing = 0

for idx, row in df_focus.head(50).iterrows():
    game_id = str(row['game_id'])
    event_id = str(row['event_id'])
    fname = f"game_{game_id}_goal_{event_id}_positions.csv"
    path = os.path.join(DATA_DIR, fname)
    
    if os.path.exists(path):
        found += 1
    else:
        missing += 1
        # print(f"Missing: {fname}")

print(f"\nSample Result: Found {found}, Missing {missing}")
