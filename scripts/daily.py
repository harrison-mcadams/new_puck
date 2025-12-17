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
import subprocess
import argparse
import pandas as pd
import gc

# Add project root to sys.path to allow importing puck package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import parse
from puck import timing
from puck import analyze
from puck import config

# Scripts in the same directory
import run_player_analysis
import run_league_stats

def main():
    parser = argparse.ArgumentParser(description="Daily NHL Analysis Update")
    parser.add_argument('--season', type=str, default='20252026', help='Season string (e.g., 20252026)')
    parser.add_argument('--force', action='store_true', help='Force full re-download/re-calc')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip data fetching (use existing CSV)')
    parser.add_argument('--only-5v5', action='store_true', help='Only process 5v5 data')
    args = parser.parse_args()
    
    season = args.season
    print(f"--- Starting Daily Update for Season {season} ---")
    
    # 1. Update Data
    df_season = pd.DataFrame()
    if args.skip_fetch:
        print("Skipping data fetch as requested.")
        csv_path = os.path.join('data', f"{season}.csv")
        if os.path.exists(csv_path):
            try:
                df_season = pd.read_csv(csv_path)
                print(f"Loaded existing data from {csv_path}. Shape: {df_season.shape}")
            except Exception as e:
                print(f"Failed to load existing CSV: {e}")
        else:
             print(f"Error: {csv_path} not found.")
    if df_season.empty:
        # Standard Update Path
        # If force is true, clear the nhl_api cache to ensure fresh schedule
        if args.force:
            print("Force flag set: Clearing caches...")
            import shutil
            cache_root = os.path.join(config.CACHE_DIR, 'nhl_api')
            if os.path.exists(cache_root):
                try:
                    shutil.rmtree(cache_root)
                    print(f"Cleared {cache_root}")
                except Exception as e:
                    print(f"Warning: Failed to clear cache: {e}")
            
            # Also remove potential shadowing CSVs that timing.load_season_df might prefer
            files_to_nuke = [
                os.path.join('data', season, f'{season}.csv'),
                os.path.join('data', season, f'{season}_df.csv'),
                os.path.join('data', season, f'{season}_game_feeds.csv'),
                os.path.join('data', season, f'{season}_game_feeds.json'),
                os.path.join('data', f'{season}.csv'),
                os.path.join('data', f'{season}_df.csv')
            ]
            
            for f in files_to_nuke:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                        print(f"Removed stale file: {f}")
                    except Exception as e:
                        print(f"Warning: Failed to remove {f}: {e}")
                    
        # parse._season with use_cache=True will check static/cache/game_ID.json
        # We disable cache if force is True
        df_season = parse._season(
            season=season, 
            out_path='data', 
            use_cache=not args.force
        )
    print(f"Season data updated. Total games: {len(df_season['game_id'].unique()) if not df_season.empty else 0}")
    
    # 1b. Update Teams List (Ensure analysis/teams.json is fresh)
    if not args.skip_fetch:
        print("Updating teams list...")
        try:
            subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), 'generate_teams.py')], check=False)
        except Exception as e:
            print(f"Warning: generate_teams.py failed: {e}")

    
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
    
    if args.only_5v5:
        print("Filtering to 5v5 only for interval cache.")
        conditions_to_cache = [c for c in conditions_to_cache if c['game_state'] == ['5v5']]
    
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

    # FREE MEMORY: We don't need the season dataframe anymore.
    # This is critical on low-memory devices (Raspberry Pi) as the subprocesses
    # will load their own copy of the data.
    del df_season
    del game_ids
    gc.collect()

    # 2b. Process Game Caches (The Heavy Lifting)
    # This runs the "Map" phase of Map-Reduce, creating .npz files for all games.
    print("\n[2b/4] Processing Game Caches (Map Phase)...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_script = os.path.join(script_dir, 'process_daily_cache.py')
    
    conditions_to_process = ['5v5', '5v4', '4v5']
    if args.only_5v5:
        print("Filtering to 5v5 only for cache processing.")
        conditions_to_process = ['5v5']

    for cond in conditions_to_process:
        print(f"  -> Processing {cond} cache...")
        try:
            cmd = [sys.executable, cache_script, '--season', season, '--condition', cond]
            if args.force:
                cmd.append('--force')
            
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"Cache processing failed for {cond}: {e}")


    # 3. Run Team Analysis (Generates Baseline)
    print("\n[3/4] Running Team Analysis (Incremental)...")
    try:
        # Similarly for run_league_stats.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        league_stats_script = os.path.join(script_dir, 'run_league_stats.py')
        
        cmd = [sys.executable, league_stats_script, '--season', season]
        if args.only_5v5:
            cmd.extend(['--condition', '5v5'])
        
        subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Team Analysis failed: {e}")

    # 4. Run Player Analysis (Uses Baseline)
    print("\n[4/4] Running Player Analysis (Incremental)...")
    try:
        # We can import and run the main logic. 
        # run_player_analysis doesn't have a main() function exposed cleanly that accepts args,
        # but we can call the code block if we wrap it or just import and run if it was structured that way.
        # Looking at run_player_analysis.py, it runs on import if __name__ == "__main__".
        # We should probably refactor it slightly or just use subprocess to be safe and clean.
        # Subprocess is safer to avoid global state pollution between scripts.
        # Determine paths to sibling scripts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        player_analysis_script = os.path.join(script_dir, 'run_player_analysis.py')
        
        subprocess.run([sys.executable, player_analysis_script, '--season', season], check=True)
    except Exception as e:
        print(f"Player Analysis failed: {e}")

    print("\n--- Daily Update Complete ---")

if __name__ == "__main__":
    main()
