import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from puck import nhl_api, parse, data_pipeline

def find_good_block(game_id):
    feed = nhl_api.get_game_feed(game_id)
    df_raw = parse._game(feed)
    df_proc = data_pipeline.preprocess_features(df_raw, apply_imputation=False)
    
    blocks = df_proc[df_proc['event'].str.lower() == 'blocked-shot']
    # Sort by distance
    blocks = blocks.sort_values('distance')
    
    if not blocks.empty:
        best = blocks.iloc[0]
        print(f"\nGame {game_id}: Nearest Block Dist={best['distance']:.1f}, X={best['x']:.1f}, Owner={best['team_id']}")
    else:
        print(f"No blocks in {game_id}")

find_good_block("2020020151")
find_good_block("2024020151")
