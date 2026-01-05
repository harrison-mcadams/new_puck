import pandas as pd
from datetime import datetime

def verify_scale():
    # Load just the timestamp column
    df = pd.read_csv('data/edge_goals/20242025/game_2024020202_goal_328_positions.csv', usecols=['timestamp'])
    
    # Get first few values
    ts = df['timestamp'].unique()
    ts.sort()
    
    print(f"Total Unique Timestamps: {len(ts)}")
    print(f"First 5: {ts[:5]}")
    
    # Calculate intervals
    diffs = [ts[i+1] - ts[i] for i in range(len(ts)-1)]
    avg_diff = sum(diffs) / len(diffs)
    print(f"Average Interval: {avg_diff}")
    
    # Check conversion hypotheses
    sample = ts[0]
    print(f"\nSample Value: {sample}")
    
    # Case A: Seconds (10-digit) -> This is 11 digit, so maybe it's 10x seconds?
    seconds_val = sample / 10.0
    dt = datetime.fromtimestamp(seconds_val)
    print(f"Hypothesis A (Value/10 = Seconds): {dt} (UTC assuming local run)")
    
    # Case B: Milliseconds (13-digit) -> This is too small for ms if it's recent.
    ms_val = sample / 1000.0
    dt_ms = datetime.fromtimestamp(ms_val)
    print(f"Hypothesis B (Value/1000 = Seconds): {dt_ms}")

if __name__ == "__main__":
    verify_scale()
