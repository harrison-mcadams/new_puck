import pandas as pd
import glob
import os

def check_timestamp_unit():
    files = glob.glob('data/edge_goals/20242025/*_positions.csv')
    if not files:
        print("No files found.")
        return

    # Pick a file
    f = files[0]
    print(f"Inspecting {f}")
    
    df = pd.read_csv(f)
    if df.empty: return
    
    # Filter to one entity (puck)
    dp = df[df['entity_type'] == 'puck'].sort_values('frame_idx')
    
    print("\nFirst 10 Rows (Puck):")
    print(dp[['frame_idx', 'timestamp']].head(10))
    
    # Calculate diffs
    dp['diff'] = dp['timestamp'].diff()
    print("\nTimestamp Diffs:")
    print(dp['diff'].value_counts().head())
    
    # Total Range
    t_min = dp['timestamp'].min()
    t_max = dp['timestamp'].max()
    print(f"\nRange: {t_min} to {t_max} (Diff: {t_max-t_min})")
    print(f"Frame Count: {len(dp)}")
    
    # Check if Frame ID corresponds to timestamp
    # If 30fps, 1 sec = 30 frames.
    # If Diff is '33' (ms) -> 30fps.
    # If Diff is '1' -> ???

if __name__ == "__main__":
    check_timestamp_unit()
