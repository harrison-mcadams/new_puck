import pandas as pd
from puck import fit_nested_xgs
try:
    df = fit_nested_xgs.load_data()
    team_cols = [c for c in df.columns if 'team' in c.lower()]
    print("TEAM columns:", team_cols)
    print("Game ID sample:", df['game_id'].head().tolist())
except Exception as e:
    print(e)
