"""
run_league_stats.py (Optimized V2)

A memory-efficient "Map-Reduce" pipeline for calculating Team/League stats.

Architecture:
1. Map Phase: Iterate through games one by one (or in parallel).
   - Extract Home/Away team stats for the game (GF, GA, xGF, xGA, CF, CA, TOI).
   - Use strict 5v5 filtering (or other conditions) per game.
   - Accumulate lightweight results.
2. Reduce Phase:
   - Aggregate totals per Team.
3. Context Phase:
   - Calculate Per-60 Rates.
   - Calculate League-wide Percentiles.
4. Output Phase:
   - Save `league_stats.json`.
   - Generate Scatter Plots (xGF vs xGA) using the lightweight aggregated data.
"""

import os
import sys
import gc
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import parse
from puck import plot

def process_game_league(args):
    """
    Pure function to process a single game for league stats.
    Args:
        args (tuple): (game_id, df_game, condition, season)
    Returns:
        list: List of dicts (usually 2, one for home one for away).
    """
    game_id, df_game, condition, season = args
    results = []
    
    try:
        if df_game.empty: 
            return []
        
        # Get Intervals (Shared Cache)
        # We need 5v5 intervals to filter the TOI and Events correctly
        intervals = timing.get_game_intervals_cached(game_id, season, condition)
        if not intervals: 
            return []
        
        # Calculate TOI for this game's 5v5 state
        # Intervals is a list of [start, end]
        toi_seconds = sum(e - s for s, e in intervals)
        if toi_seconds <= 0: 
            return []
        
        # Filter DataFrame to 5v5 intervals
        # We use analyze helper or manual?
        # analyze._apply_intervals is robust.
        df_filtered = analyze._apply_intervals(df_game, {'per_game': {game_id: intervals}} if isinstance(intervals, list) else intervals, 
                                             time_col='total_time_elapsed_seconds')
        
        if df_filtered.empty: 
            return []

        # Extract Home/Away Teams
        home_team = df_game['home_abb'].iloc[0]
        away_team = df_game['away_abb'].iloc[0]
        
        # Basic Stats Aggregation
        # Goals
        goals = df_filtered[df_filtered['event'] == 'goal']
        home_goals = len(goals[goals['team_id'] == goals['home_id']]) if not goals.empty else 0
        away_goals = len(goals[goals['team_id'] == goals['away_id']]) if not goals.empty else 0
        
        # xG
        # xgs column is already populated (pre-calculated in main)
        # Sum xgs for home and away
        is_home = df_filtered['team_id'] == df_filtered['home_id']
        xgs_series = df_filtered['xgs'].fillna(0)
        
        home_xg = xgs_series[is_home].sum()
        away_xg = xgs_series[~is_home].sum()
        
        # Corsi (Shot Attempts)
        attempts_mask = df_filtered['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])
        home_cf = len(df_filtered[attempts_mask & is_home])
        away_cf = len(df_filtered[attempts_mask & (~is_home)])
        
        # Append Results
        # Record for Home
        results.append({
            'team': home_team,
            'game_id': game_id,
            'gp': 1,
            'toi': toi_seconds,
            'gf': home_goals,
            'ga': away_goals,
            'xg_for': home_xg,
            'xg_against': away_xg,
            'cf': home_cf,
            'ca': away_cf
        })
        
        # Record for Away
        results.append({
            'team': away_team,
            'game_id': game_id,
            'gp': 1,
            'toi': toi_seconds,
            'gf': away_goals,
            'ga': home_goals,
            'xg_for': away_xg,
            'xg_against': home_xg,
            'cf': away_cf,
            'ca': home_cf
        })
        
    except Exception as e:
        # print(f"Error processing game {game_id}: {e}")
        pass
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run League Stats (Optimized)")
    parser.add_argument('--season', type=str, default='20252026')
    parser.add_argument('--out_dir', type=str, default='analysis/league')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    parser.add_argument('--parallel', action='store_true', help="Use multiprocessing for faster execution on Mac")
    args = parser.parse_args()
    
    season = args.season
    out_dir = args.out_dir
    use_parallel = args.parallel
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Starting Optimized League Stats for Season {season} ---")
    print(f"Mode: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
    
    # 1. Load Full Season Data (Once)
    print("Loading season data...")
    df_data = timing.load_season_df(season)
    if df_data is None or df_data.empty:
        print("No data found. Exiting.")
        return

    # 2. Pre-Calculate xG (Batch)
    print("Pre-calculating xG predictions for the season...")
    df_data, _, _ = analyze._predict_xgs(df_data)
    
    # 3. Map Phase: Iterate Games
    game_ids = sorted(df_data['game_id'].unique())
    print(f"Processing {len(game_ids)} games...")
    
    team_stats_acc = [] 
    
    # Condition: 5v5
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    map_args = []
    if use_parallel:
        print("Preparing tasks for parallel execution...")
        for gid in game_ids:
            df_g = df_data[df_data['game_id'] == gid].copy()
            map_args.append((gid, df_g, condition, season))
            
        print(f"Dispatching {len(map_args)} tasks to worker pool...")
        with multiprocessing.Pool() as pool:
            results_nested = pool.map(process_game_league, map_args)
            
        for res_list in results_nested:
            team_stats_acc.extend(res_list)
            
    else:
        # Sequential
        for idx, game_id in enumerate(game_ids):
            if (idx + 1) % 50 == 0:
                print(f"  Game {idx+1}/{len(game_ids)}...", flush=True)
                gc.collect()
            
            df_g = df_data[df_data['game_id'] == game_id]
            res_list = process_game_league((game_id, df_g, condition, season))
            team_stats_acc.extend(res_list)

    # Free memory
    del df_data
    if 'map_args' in locals(): del map_args
    gc.collect()
    
    # 4. Reduce Phase: Aggregation
    print("Aggregating team stats...")
    if not team_stats_acc:
        print("No stats collected.")
        return
        
    df_raw = pd.DataFrame(team_stats_acc)
    # Sum by team
    numeric_cols = ['gp', 'toi', 'gf', 'ga', 'xg_for', 'xg_against', 'cf', 'ca']
    df_agg = df_raw.groupby('team')[numeric_cols].sum().reset_index()
    
    # 5. Context Phase: Rates & Percentiles
    print("Calculating rates and percentiles...")
    
    # Rate stats (/60)
    df_agg['xg_for_60'] = (df_agg['xg_for'] / df_agg['toi']) * 3600
    df_agg['xg_against_60'] = (df_agg['xg_against'] / df_agg['toi']) * 3600
    df_agg['goals_for_60'] = (df_agg['gf'] / df_agg['toi']) * 3600
    df_agg['goals_against_60'] = (df_agg['ga'] / df_agg['toi']) * 3600
    df_agg['cf_60'] = (df_agg['cf'] / df_agg['toi']) * 3600
    df_agg['ca_60'] = (df_agg['ca'] / df_agg['toi']) * 3600
    
    # Ratios
    df_agg['xg_pct'] = (df_agg['xg_for'] / (df_agg['xg_for'] + df_agg['xg_against'])) * 100
    df_agg['cf_pct'] = (df_agg['cf'] / (df_agg['cf'] + df_agg['ca'])) * 100
    
    # Percentiles
    rate_cols = ['xg_for_60', 'xg_against_60', 'goals_for_60', 'goals_against_60', 'cf_60', 'ca_60', 'xg_pct', 'cf_pct']
    
    for col in rate_cols:
        pct_col = f'{col}_pct'
        df_agg[pct_col] = df_agg[col].rank(pct=True) * 100
        df_agg[pct_col] = df_agg[pct_col].fillna(0)

    # 6. Output Phase
    records = df_agg.to_dict('records')
    out_file = os.path.join(out_dir, f'{season}_teams.json')
    with open(out_file, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"Saved league stats to {out_file}")
    
    # 7. Plotting (Optional / Lightweight)
    if True: # Always generate scatter for now, it's cheap on aggregated data (32 points)
        print("Generating Scatter Plot...")
        try:
            # Simple Matplotlib
            plt.figure(figsize=(10, 8))
            
            # Filter valid
            plot_df = df_agg[df_agg['gp'] >= 5]
            if plot_df.empty: plot_df = df_agg
            
            x = plot_df['xg_for_60']
            y = plot_df['xg_against_60']
            teams = plot_df['team']
            
            plt.scatter(x, y, alpha=0.5)
            
            # Add labels
            for i, team in enumerate(teams):
                plt.annotate(team, (x.iloc[i], y.iloc[i]), fontsize=8)
                
            plt.title(f"League xG Rates (5v5) - {season}")
            plt.xlabel("xG For / 60")
            plt.ylabel("xG Against / 60")
            plt.grid(True, alpha=0.3)
            
            # Use Inverted Y-Axis so Top-Right is High-For / Low-Against (Good/Good)
            # Low AGainst is 'Top' on Y graph if inverted
            plt.gca().invert_yaxis()
            
            plot_path = os.path.join(out_dir, f'{season}_scatter.png')
            plt.savefig(plot_path, dpi=100)
            plt.close()
            print(f"Saved scatter plot to {plot_path}")
            
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
