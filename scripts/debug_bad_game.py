
import numpy as np
import os
import sys
import pandas as pd

# Adjust path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck.analyze import xgs_map
from puck import timing

def debug_game():
    season = '20252026'
    game_id = '2025020460'
    
    print(f"Debugging Game {game_id}...")
    
    # Load season data first to extract game events
    print("Loading season data...")
    df = timing.load_season_df(season)
    
    if df is None:
        print("Failed to load season DF")
        return

    # Filter for game to print some raw stats
    game_df = df[df['game_id'] == int(game_id)].copy()
    print(f"Events: {len(game_df)}")
    
    # Check for NaNs in columns
    print("\n--- Raw Data QC ---")
    for col in ['x', 'y', 'xgs']:
        if col in game_df.columns:
            nans = game_df[col].isna().sum()
            print(f"{col}: {nans} NaNs")
            if nans > 0:
                print(game_df[game_df[col].isna()])
        else:
            print(f"{col} missing")

    print("\n--- Running xgs_map ---")
    try:
        _, heatmaps, _, stats = xgs_map(
            season=season,
            data_df=df, # Pass full DF, it filters internally
            game_id=game_id,
            return_heatmaps=True,
            show=False,
            condition={'game_state': ['5v5']}
        )
        
        home_grid = heatmaps.get('home')
        away_grid = heatmaps.get('away')
        
        print("\n--- Grid Results ---")
        if home_grid is not None:
            print(f"Home Grid: nan={np.isnan(home_grid).any()}, max={np.nanmax(home_grid)}")
        else:
            print("Home Grid is None")
            
        if away_grid is not None:
             print(f"Away Grid: nan={np.isnan(away_grid).any()}, max={np.nanmax(away_grid)}")
        else:
            print("Away Grid is None")
            
        print("\n--- Stats ---")
        print(stats)
             
    except Exception as e:
        print(f"xgs_map failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_game()
