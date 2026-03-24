import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.getcwd())

from puck import nhl_api, parse, data_pipeline, correction

def trace_block(game_id):
    feed = nhl_api.get_game_feed(game_id)
    df = parse._game(feed)
    
    # Isolate first block
    blocks = df[df['event'].str.lower() == 'blocked-shot']
    if blocks.empty:
        print("No blocks.")
        return
    
    row = blocks.iloc[0:1].copy()
    print("\n--- INITIAL STATE ---")
    print(row[['event', 'team_id', 'x', 'y', 'home_team_defending_side']])
    
    # 1. Attribution Fix
    row = correction.fix_blocked_shot_attribution(row)
    print("\n--- AFTER ATTRIBUTION FIX ---")
    print(row[['event', 'team_id', 'x', 'y', 'home_team_defending_side']])
    
    # 2. Derive is_home
    tid_str = row['team_id'].fillna(-1).astype(str).str.split('.').str[0]
    hid_str = row['home_id'].fillna(-2).astype(str).str.split('.').str[0]
    row['is_home'] = (tid_str == hid_str).astype(int)
    print("\n--- DERIVED IS_HOME ---")
    print(row[['team_id', 'home_id', 'is_home']])
    
    # 3. Orientation Logic
    side_str = row['home_team_defending_side'].astype(str).str.lower().str.strip()
    def_side_map = side_str.map({'left': -1, 'right': 1})
    is_home_series = (row['is_home'] == 1)
    side_multiplier = np.where(is_home_series, -1, 1)
    attacking_side = def_side_map * side_multiplier
    mask_flip = (attacking_side == -1)
    
    print("\n--- ORIENTATION VARS ---")
    print(f"Def Side Map: {def_side_map.iloc[0]}")
    print(f"Is Home: {is_home_series.iloc[0]}")
    print(f"Side Multiplier: {side_multiplier[0]}")
    print(f"Attacking Side: {attacking_side.iloc[0]}")
    print(f"Mask Flip: {mask_flip.iloc[0]}")
    
    if mask_flip.any():
        row['x'] *= -1
        row['y'] *= -1
    
    print("\n--- FINAL STATE ---")
    print(row[['event', 'team_id', 'x', 'y', 'distance']])

trace_block("2024020151")
