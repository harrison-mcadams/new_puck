import pandas as pd
from puck import fit_nested_xgs
try:
    df = fit_nested_xgs.load_data()
    print("COLUMNS:", df.columns.tolist())
    if 'game_id' in df.columns:
        print("Sample Game IDs:", df['game_id'].head().tolist())
except Exception as e:
    print(e)
