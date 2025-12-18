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


    # --- Helper to parse max ---
    import re
    def parse_max(output):
        # Match floats including scientific notation (e.g., 1.23e-05 or 0.0001)
        match = re.search(r"Max 80th Percentile \(Saturated\):\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", output)
        if match:
            return float(match.group(1))
        return 0.0
        
    def run_cmd_capture(cmd):
        print(f"Running (Scan): {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            print(f"Error in scan: {res.stderr}")
        return res.stdout

    # 3. & 4. Run Analysis with Consistent Limits
    print("\n[3/4] Running Analysis (Scanning & Plotting)...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    league_script = os.path.join(script_dir, 'run_league_stats.py')
    player_script = os.path.join(script_dir, 'run_player_analysis.py')
    
    # Process 5v5 (Global Consistency: Team + Player)
    import math
    
    print("-> Processing 5v5...")
    
    # Scan League 5v5
    cmd_l_scan = [sys.executable, league_script, '--season', season, '--condition', '5v5', '--scan-limit']
    out_l = run_cmd_capture(cmd_l_scan)
    max_l = parse_max(out_l)
    print(f"   League 5v5 Max: {max_l}")
    
    # Scan Players 5v5
    # Note: run_player_analysis currently defaults to 5v5.
    cmd_p_scan = [sys.executable, player_script, '--season', season, '--scan-limit']
    out_p = run_cmd_capture(cmd_p_scan)
    max_p = parse_max(out_p)
    print(f"   Player 5v5 Max: {max_p}")
    
    # Determine Independent Max Limits
    # We decouple League (Team) and Player limits because Player variance is much higher (approx 20x).
    # Using a single global limit washes out the Team maps effectively to blank.
    
    import math
    def smart_ceil(x):
        if x == 0: return 0.001
        if x < 0.01:
            return math.ceil(x * 10000) / 10000.0
        else:
            return math.ceil(x * 100) / 100.0
            
    vmax_l = 0.02
    vmax_p = 0.02
    
    print(f"   League 5v5 VMAX: {vmax_l} (Raw: {max_l})")
    print(f"   Player 5v5 VMAX: {vmax_p} (Raw: {max_p})")
    
    # Plot League 5v5
    subprocess.run([sys.executable, league_script, '--season', season, 
                    '--condition', '5v5', '--vmax', str(vmax_l)], check=True)
                    
    # Plot Players 5v5
    subprocess.run([sys.executable, player_script, '--season', season, 
                    '--vmax', str(vmax_p)], check=True)

    # Process Other Conditions (League Only)
    if not args.only_5v5:
        for cond in ['5v4', '4v5']:
            print(f"-> Processing {cond}...")
            # Scan
            cmd_scan = [sys.executable, league_script, '--season', season, '--condition', cond, '--scan-limit']
            out_scan = run_cmd_capture(cmd_scan)
            raw_max_c = parse_max(out_scan)
            
            # Round
            vmax_c = smart_ceil(raw_max_c)
            if vmax_c < 0.0005: vmax_c = 0.0005
            print(f"   {cond} VMAX: {vmax_c} (Raw: {raw_max_c})")
            
            # Plot
            subprocess.run([sys.executable, league_script, '--season', season, 
                           '--condition', cond, '--vmax', str(vmax_c)], check=True)

    print("\n--- Daily Update Complete ---")

if __name__ == "__main__":
    main()
