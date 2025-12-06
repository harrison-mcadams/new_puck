"""
Daily update routine for NHL Analysis.

Workflow:
1. Update Data: Fetch new games via parse._season.
2. Pre-Compute Intervals: Generate shared interval cache for 5v5, 5v4, 4v5.
3. Run Player Analysis: Incremental update of player stats and maps.
4. Run Team Analysis: Incremental update of team stats and maps.
"""

import os
import sys
import argparse
import pandas as pd

# Add project root to sys.path to allow importing puck package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import parse
from puck import timing
from puck import analyze

# Scripts in the same directory
import run_player_analysis
import run_league_stats

def main():
    parser = argparse.ArgumentParser(description="Daily NHL Analysis Update")
    parser.add_argument('--season', type=str, default='20252026', help='Season string (e.g., 20252026)')
    parser.add_argument('--force', action='store_true', help='Force full re-download/re-calc')
    args = parser.parse_args()
    
    season = args.season
    print(f"--- Starting Daily Update for Season {season} ---")
    
    # 1. Update Data
    print("\n[1/4] Updating Game Data...")
    # use_cache=True allows efficient fetching of only new games if parse supports it properly
    # parse._season with use_cache=True will check static/cache/game_ID.json
    # out_path='data' because parse._season will append /{season}/
    df_season = parse._season(
        season=season, 
        out_path='data', 
        use_cache=True
    )
    print(f"Season data updated. Total games: {len(df_season['game_id'].unique()) if not df_season.empty else 0}")
    
    if df_season.empty:
        print("No data found. Exiting.")
        return

    # 2. Pre-Compute Intervals (Shared Cache)
    print("\n[2/4] Pre-Computing Intervals...")
    # We want to ensure the cache is populated for standard conditions
    # This avoids race conditions or redundant calcs later
    game_ids = sorted(df_season['game_id'].unique())
    
    conditions_to_cache = [
        {'game_state': ['5v5'], 'is_net_empty': [0]},
        {'game_state': ['5v4'], 'is_net_empty': [0]},
        {'game_state': ['4v5'], 'is_net_empty': [0]}
    ]
    
    # We can just call get_game_intervals_cached for each game/condition
    # It handles the check/compute/save logic.
    count = 0
    for game_id in game_ids:
        for cond in conditions_to_cache:
            timing.get_game_intervals_cached(game_id, season, cond)
        count += 1
        if count % 50 == 0:
            print(f"Processed intervals for {count}/{len(game_ids)} games...")
            
    print("Interval cache updated.")

    # 3. Run Player Analysis
    print("\n[3/4] Running Player Analysis (Incremental)...")
    try:
        # We can import and run the main logic. 
        # run_player_analysis doesn't have a main() function exposed cleanly that accepts args,
        # but we can call the code block if we wrap it or just import and run if it was structured that way.
        # Looking at run_player_analysis.py, it runs on import if __name__ == "__main__".
        # We should probably refactor it slightly or just use subprocess to be safe and clean.
        # Subprocess is safer to avoid global state pollution between scripts.
        import subprocess
        subprocess.run([sys.executable, 'run_player_analysis.py'], check=True)
    except Exception as e:
        print(f"Player Analysis failed: {e}")

    # 4. Run Team Analysis
    print("\n[4/4] Running Team Analysis (Incremental)...")
    try:
        # Similarly for run_league_stats.py
        subprocess.run([sys.executable, 'run_league_stats.py'], check=True)
    except Exception as e:
        print(f"Team Analysis failed: {e}")

    print("\n--- Daily Update Complete ---")

if __name__ == "__main__":
    main()
