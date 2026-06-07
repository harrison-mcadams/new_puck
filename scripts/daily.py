"""
Daily update routine for NHL Analysis.

Workflow:
1. Update Data: Fetch new games via parse._season.
2. Pre-Compute Intervals: Generate shared interval cache for 5v5, 5v4, 4v5.
3. Run Player Analysis: Incremental update of player stats and maps.
4. Run Team Analysis: Incremental update of team stats and maps.

ARCHITECTURAL WARNING:
Coordinate orientation and blocked-shot attribution are centralized in `puck/data_pipeline.py`.
The pipeline uses a "Right-Attack" standardized orientation (x towards +89).
DO NOT add coordinate flips in this script or sub-scripts; double-flipping causes severe xG inflation.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import gc
import logging

# Add project root to sys.path to allow importing puck package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import parse
from puck import timing
from puck import analyze
from puck import config
from puck import data_pipeline
from puck import playoffs

# Scripts in the same directory
import run_player_analysis
import run_league_stats

def main():
    # Setup Logging to show verify_df output
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    parser = argparse.ArgumentParser(description="Daily NHL Analysis Update")
    parser.add_argument('--season', type=str, default='20252026', help='Season string (e.g., 20252026)')
    parser.add_argument('--force', action='store_true', help='Force full re-download/re-calc')
    parser.add_argument('--skip-fetch', action='store_true', help='Skip data fetching (use existing CSV)')
    parser.add_argument('--only-5v5', action='store_true', help='Only process 5v5 data')
    parser.add_argument('--turbo', action='store_true', help='Enable parallel processing for intervals and analysis')
    parser.add_argument('--teams-only', action='store_true', help='Only process team intermediates and plots')
    parser.add_argument('--players-only', action='store_true', help='Only process player intermediates and plots')
    parser.add_argument('--playoffs', action='store_true', help='Process playoff games (Game Type 03)')
    parser.add_argument('--model-path', type=str, default=None, help='Path to joblib model')
    args = parser.parse_args()
    
    season = args.season
    model_path = args.model_path
    if model_path is None:
        model_path = os.path.join(config.ANALYSIS_DIR, 'xgs', 'xg_model_xgboost_tensor_final.joblib')
    target_season = f"{season}_playoffs" if args.playoffs else season
    
    print(f"--- Starting Daily Update for {target_season} (Turbo={'ON' if args.turbo else 'OFF'}) ---")
    
    # 1. Update Data
    df_season = pd.DataFrame()
    if args.skip_fetch:
        print("Skipping data fetch as requested.")
        csv_path = os.path.join(config.DATA_DIR, f"{target_season}.csv")
        # Also check Gold Standard path
        alt_path = os.path.join(config.DATA_DIR, target_season, f"{target_season}_df.csv")
        
        load_path = None
        if os.path.exists(csv_path):
            load_path = csv_path
        elif os.path.exists(alt_path):
            load_path = alt_path
            
        if load_path:
            try:
                df_season = pd.read_csv(load_path)
                print(f"Loaded existing data from {load_path}. Shape: {df_season.shape}")
            except Exception as e:
                print(f"Failed to load existing CSV: {e}")
        else:
             print(f"Error: {csv_path} or {alt_path} not found.")
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
                os.path.join(config.DATA_DIR, target_season, f'{target_season}.csv'),
                os.path.join(config.DATA_DIR, target_season, f'{target_season}_df.csv'),
                os.path.join(config.DATA_DIR, target_season, f'{target_season}_game_feeds.csv'),
                os.path.join(config.DATA_DIR, target_season, f'{target_season}_game_feeds.json'),
                os.path.join(config.DATA_DIR, f'{target_season}.csv'),
                os.path.join(config.DATA_DIR, f'{target_season}_df.csv')
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
        # In Turbo mode, we increase workers for fetching
        fetch_workers = 16 if args.turbo else 4
        
        # Fetch regular season ('02') and playoffs ('03') if in playoff mode
        game_types = ['02', '03'] if args.playoffs else ['02']
        
        df_season = parse._season(
            season=season, 
            out_path=None, # We manually save below to handle target_season suffix
            use_cache=not args.force,
            max_workers=fetch_workers,
            game_types=game_types
        )
        # Standardize orientation and features for ALL events
        print("Standardizing season orientation (Right-Attack Standard)...")
        df_season = data_pipeline.preprocess_features(
            df_season, 
            is_training=False, 
            apply_filtering=False,
            apply_imputation=True,
            apply_arena_adjustments=True,
            verbose=True
        )

        # Manually save to target_season/target_season_df.csv (Gold Standard Path)
        if not df_season.empty:
            season_dir = os.path.join(config.DATA_DIR, target_season)
            os.makedirs(season_dir, exist_ok=True)
            out_csv = os.path.join(season_dir, f"{target_season}_df.csv")
            df_season.to_csv(out_csv, index=False)
            print(f"Saved standardized Gold Standard data to {out_csv}")
    print(f"Season data updated. Total games: {len(df_season['game_id'].unique()) if not df_season.empty else 0}")
    
    # 1b. Update Teams List (Ensure analysis/teams.json is fresh)
    # Note: generate_teams.py is deprecated/missing, skipping.
            
    if df_season.empty:
        print("No data found. Exiting.")
        return

    # 1c. Centralized xG Prediction
    # We run this ONCE for the whole season to avoid redundant calculations in subprocesses.
    # We run this if we fetched data OR if --force is used (to ensure new model is applied to existing data).
    run_xg_calc = (not args.skip_fetch) or args.force
    if run_xg_calc and not df_season.empty:
        print(f"\n[1c/4] Running Centralized xG Prediction (20202021+ Nested Model)...")
        try:
            # Predict using the active model
            df_season, _, _ = analyze._predict_xgs(df_season, model_path=model_path, behavior='overwrite')
            
            # Save back to CSV to be used by subprocesses
            out_csv = os.path.join(config.DATA_DIR, f"{target_season}.csv")
            df_season.to_csv(out_csv, index=False)
            print(f"Saved updated xG data to {out_csv}")
            
        except Exception as e:
            print(f"Warning: Centralized prediction failed: {e}")

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
    
    # Parallel Interval Generation
    if args.turbo:
        from joblib import Parallel, delayed
        print(f"[TURBO] Generating intervals in parallel for {len(game_ids)} games...")
        
        # Flatten tasks
        tasks = []
        for game_id in game_ids:
            for cond in conditions_to_cache:
                tasks.append((game_id, season, cond))
        
        # Run
        Parallel(n_jobs=-1, verbose=1)(
            delayed(timing.get_game_intervals_cached)(gid, target_season, c, force_refresh=args.force) for gid, _, c in tasks
        )
        print("Interval cache updated (Parallel).")
        
    else:
        # Serial Interval Generation
        count = 0
        for game_id in game_ids:
            for cond in conditions_to_cache:
                timing.get_game_intervals_cached(game_id, target_season, cond, force_refresh=args.force)
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
            cmd = [sys.executable, cache_script, '--season', target_season, '--condition', cond, '--model-path', model_path]
            if args.force:
                cmd.append('--force')
            if args.turbo:
                cmd.append('--turbo') # Pass it down
            if args.teams_only:
                cmd.append('--teams-only')
            if args.players_only:
                cmd.append('--players-only')
            
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
    cmd_l_scan = [sys.executable, league_script, '--season', target_season, '--condition', '5v5', '--scan-limit']
    out_l = run_cmd_capture(cmd_l_scan)
    max_l = parse_max(out_l)
    print(f"   League 5v5 Max: {max_l}")
    
    # Scan Players 5v5
    # Note: run_player_analysis currently defaults to 5v5.
    max_p = 0.0
    if not args.teams_only:
        cmd_p_scan = [sys.executable, player_script, '--season', target_season, '--scan-limit']
        if args.turbo:
            cmd_p_scan.append('--turbo')
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
            
    vmax_l = smart_ceil(max_l)
    if vmax_l < 0.02: vmax_l = 0.02
    
    vmax_p = smart_ceil(max_p)
    if vmax_p < 0.02: vmax_p = 0.02
    
    print(f"   League 5v5 VMAX: {vmax_l} (Raw: {max_l})")
    print(f"   Player 5v5 VMAX: {vmax_p} (Raw: {max_p})")
    
    # Plot League 5v5
    if not args.players_only:
        subprocess.run([sys.executable, league_script, '--season', target_season, 
                        '--condition', '5v5', '--vmax', str(vmax_l)], check=True)
                    
    # Plot Players 5v5
    if not args.teams_only:
        cmd_p_plot = [sys.executable, player_script, '--season', target_season, '--vmax', str(vmax_p)]
        if args.turbo:
            cmd_p_plot.append('--turbo')
        subprocess.run(cmd_p_plot, check=True)

    # Process Other Conditions (League Only)
    if not args.only_5v5:
        for cond in ['5v4', '4v5']:
            print(f"-> Processing {cond}...")
            # Scan
            cmd_scan = [sys.executable, league_script, '--season', target_season, '--condition', cond, '--scan-limit']
            out_scan = run_cmd_capture(cmd_scan)
            raw_max_c = parse_max(out_scan)
            
            # Round
            vmax_c = smart_ceil(raw_max_c)
            if vmax_c < 0.0005: vmax_c = 0.0005
            print(f"   {cond} VMAX: {vmax_c} (Raw: {raw_max_c})")
            
            # Plot
            if not args.players_only:
                subprocess.run([sys.executable, league_script, '--season', target_season, 
                               '--condition', cond, '--vmax', str(vmax_c)], check=True)

    # 5. Mixed Effects Summaries
    print("\n[5/5] Generating Mixed Effects Summaries...")
    plot_mixed_script = os.path.join(script_dir, 'plot_mixed_effects_summaries.py')
    if os.path.exists(plot_mixed_script):
        try:
            subprocess.run([sys.executable, plot_mixed_script], check=True)
        except Exception as e:
            print(f"Warning: Failed to generate mixed effects summaries: {e}")

    # 6. Playoff Plots
    if args.playoffs:
        print("\n[6/6] Generating Playoff Plots...")
        try:
            playoffs.generate_playoff_plots(season=season, force=args.force)
        except Exception as e:
            print(f"Warning: Playoff plotting failed: {e}")

    print("\n--- Daily Update Complete ---")

if __name__ == "__main__":
    main()
