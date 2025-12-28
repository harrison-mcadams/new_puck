
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_xgs

def check_nans_by_season():
    print("Loading all seasons data...")
    try:
        df = fit_xgs.load_all_seasons_data()
        print(f"Loaded {len(df)} rows.")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    if 'game_id' not in df.columns:
        print("Error: game_id missing")
        return

    # Extract season from game_id (first 4 digits of game_id typically indicate season start year)
    # Actually, usually game_id is YYYY02NNNN. So first 4 are year.
    df['season_start'] = df['game_id'].astype(str).str[:4]
    
    target_col = 'total_time_elapsed_s'
    if target_col not in df.columns:
        print(f"{target_col} missing in dataframe")
        return
        
    # Filter for NaNs in target column
    nans = df[df[target_col].isna()]
    
    if nans.empty:
        print(f"No NaNs found in {target_col} across any season.")
    else:
        print(f"Found {len(nans)} NaNs in {target_col}.")
        print("Breakdown by season:")
        season_counts = nans['season_start'].value_counts()
        print(season_counts)
        
        # Verify total rows per season to give context
        print("\nTotal rows per season (for context):")
        print(df['season_start'].value_counts().sort_index())

if __name__ == "__main__":
    check_nans_by_season()
