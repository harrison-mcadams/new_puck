import timing
import pandas as pd

season = '20252026'
df = timing.load_season_df(season)
print("Columns:", df.columns.tolist())
if 'game_date' in df.columns:
    print("game_date sample:", df['game_date'].head())
else:
    print("game_date NOT found.")
    # Check for 'date' or similar
    possible_dates = [c for c in df.columns if 'date' in c.lower()]
    print("Possible date columns:", possible_dates)
