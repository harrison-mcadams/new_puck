
import pandas as pd
import sys
from pathlib import Path

def inspect_cache(game_id):
    cache_file = Path(f'data/20252026/shifts/shifts_{game_id}.pkl')
    if not cache_file.exists():
        print(f"Cache file {cache_file} does not exist.")
        return

    try:
        df = pd.read_pickle(cache_file)
        print(f"Loaded cache for {game_id}. Shape: {df.shape}")
        if 'end_total_seconds' in df.columns:
            max_end = df['end_total_seconds'].max()
            print(f"Max end_total_seconds: {max_end}")
            
            # Check for empty/nulls
            nulls = df['end_total_seconds'].isnull().sum()
            print(f"Null end_total_seconds: {nulls}")
        else:
            print("Column 'end_total_seconds' not found.")
            print(df.columns)
            
    except Exception as e:
        print(f"Failed to load cache: {e}")

if __name__ == "__main__":
    inspect_cache(2025020412)
