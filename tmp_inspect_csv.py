import pandas as pd
import os

def inspect_csv(path):
    print(f"Inspecting {path}...")
    df = pd.read_csv(path, nrows=100)
    cols = ['game_id', 'event', 'team_id', 'home_id', 'x', 'y', 'home_team_defending_side']
    # Filter to existing columns
    available_cols = [c for c in cols if c in df.columns]
    
    # Filter to shots/goals
    mask = df['event'].astype(str).str.lower().str.contains('shot|goal')
    sdf = df[mask][available_cols].head(10)
    print(sdf)
    print("-" * 40)

if __name__ == "__main__":
    inspect_csv('data/20232024.csv')
    inspect_csv('data/20252026.csv')
