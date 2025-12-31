import pandas as pd
import json
import os

data_dir = r"c:\Users\harri\Desktop\new_puck\data\edge_goals"
csv_file = os.path.join(data_dir, "game_2025020583_goal_1007_positions.csv")
meta_file = os.path.join(data_dir, "metadata.csv")

try:
    print(f"--- {csv_file} ---")
    df = pd.read_csv(csv_file)
    print(df.head())
    print(df.columns)
except Exception as e:
    print(e)

try:
    print(f"\n--- {meta_file} ---")
    df_meta = pd.read_csv(meta_file)
    print(df_meta.head())
    print(df_meta.columns)
except Exception as e:
    print(e)
