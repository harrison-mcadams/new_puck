import pandas as pd
import glob
import os

def check_file(pattern, label):
    files = glob.glob(pattern)
    if not files:
        print(f"No files found for {label}")
        return
    
    fpath = files[0]
    print(f"\n--- {label} ({os.path.basename(fpath)}) ---")
    df = pd.read_csv(fpath)
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print(f"Rows: {len(df)}")

check_file("data/edge_goals/20232024/game_*_positions.csv", "2023-2024 Data")
check_file("data/edge_goals/20242025/game_*_positions.csv", "2024-2025 Data")
