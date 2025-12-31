import pandas as pd
import os

DATA_DIR = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
meta_path = os.path.join(DATA_DIR, "metadata.csv")
pos_path = os.path.join(DATA_DIR, "game_2025020583_goal_1007_positions.csv")

print("--- METADATA ---")
df_meta = pd.read_csv(meta_path)
row = df_meta[df_meta['event_id'] == 1007]
if row.empty:
    print("Event 1007 not found in metadata!")
else:
    print(row.iloc[0])
    print(f"Scorer ID type: {type(row.iloc[0]['scorer_id'])}")

print("\n--- POSITIONS ---")
if os.path.exists(pos_path):
    df_pos = pd.read_csv(pos_path)
    print(df_pos.head())
    print(f"Entity ID type: {df_pos['entity_id'].dtype}")
    print(f"Team ID type: {df_pos['team_id'].dtype}")
    
    # Check for Scorer
    scorer_id = row.iloc[0]['scorer_id']
    scorer_frames = df_pos[df_pos['entity_id'] == scorer_id]
    print(f"Frames for scorer {scorer_id}: {len(scorer_frames)}")
    
    if scorer_frames.empty:
        print("Scorer NOT found. Unique entities:")
        print(df_pos['entity_id'].unique())
else:
    print(f"File not found: {pos_path}")
