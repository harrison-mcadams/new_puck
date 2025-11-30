import pandas as pd
import os

files = [
    'static/20252026.csv',
    'data/20252026/20252026_df.csv'
]

for f in files:
    if os.path.exists(f):
        try:
            df = pd.read_csv(f)
            n_games = df['game_id'].nunique()
            print(f"{f}: {n_games} games, {len(df)} rows")
        except Exception as e:
            print(f"{f}: Error reading - {e}")
    else:
        print(f"{f}: Not found")
