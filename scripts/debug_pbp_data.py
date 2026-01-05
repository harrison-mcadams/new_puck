import pandas as pd
import sys
import os
sys.path.append(os.getcwd())
from puck import config

def debug_pbp():
    season = '20242025'
    pbp_path = os.path.join(config.DATA_DIR, season, f"{season}_df.csv")
    
    print(f"Loading PBP from {pbp_path}...")
    df = pd.read_csv(pbp_path)
    
    target_game = 2024020202
    period = 1
    
    # Filter
    rows = df[
        (df['game_id'] == target_game) & 
        (df['period'] == period) & 
        (df['event'].isin(['blocked-shot', 'goal']))
    ].sort_values('total_time_elapsed_s')
    
    print(f"\nPBP Events for Game {target_game} Period {period} (Blocks & Goals):")
    cols_to_show = ['game_id', 'period', 'period_time', 'total_time_elapsed_s', 'event', 'player_name', 'player_id']
    # Add optional cols if they exist
    full_cols = list(df.columns)
    valid_cols = [c for c in cols_to_show if c in full_cols]
    
    print(rows[valid_cols])

if __name__ == "__main__":
    debug_pbp()
