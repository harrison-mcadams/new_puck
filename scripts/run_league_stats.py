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
        Contains 'grid_data': {'team': grid, 'other': grid, 'extent': extent}
    """
    game_id, df_game, condition, season = args
    results = []
    
    try:
        if df_game.empty: 
            return []
        
        # Get Intervals (Shared Cache)
        intervals = timing.get_game_intervals_cached(game_id, season, condition)
        if not intervals: 
            return []
        
        # Calculate TOI
        toi_seconds = sum(e - s for s, e in intervals)
        if toi_seconds <= 0: 
            return []
        
        # Filter DataFrame
        df_filtered = analyze._apply_intervals(df_game, {'per_game': {game_id: intervals}} if isinstance(intervals, list) else intervals, 
                                             time_col='total_time_elapsed_seconds')
        
        if df_filtered.empty: 
            return []

        # Extract Home/Away Teams
        home_team = df_game['home_abb'].iloc[0]
        away_team = df_game['away_abb'].iloc[0]
        
        # --- HEATMAP GENERATION (Lightweight) ---
        # We need to compute the grids for both teams. 
        # xgs_map with split_mode='home_away' returns home and away grids.
        # This is efficient because it's just 2D histogramming.
        
        # We pass the full filtered dataframe (which has xgs) to xgs_map logic.
        # But xgs_map usually fetches data. We want to bypass fetching.
        # Calling compute_xg_heatmap_from_df directly is better if we have the df.
        # However, plot.py's helper usually handles adjustment.
        # Let's use analyze.xgs_map but passing the df? 
        # analyze.xgs_map doesn't accept a df argument directly in the current signature easily to bypass load.
        # It takes `game_id` and loads.
        # But wait, we already have `df_game` (full game data).
        # We can just call plot.plot_events with return_heatmaps=True?
        # That uses matplotlib which is not thread safe for parallel.
        # We must use `analyze.compute_xg_heatmap_from_df` directly.
        
        # 1. Adjust coordinates
        # We need to process the whole game df for xG heatmaps generally? 
        # Or just the filtered one? Usually filtered (5v5).
        # But xgs_map uses df_filtered.
        
        # Adjust coords (team-agnostic adjustment first?)
        # analyze.compute_xg_heatmap_from_df expects standardized coords? 
        # It takes x_col, y_col.
        # Let's simply run valid adjustment for home/away.
        df_adj = plot.adjust_xy_for_homeaway(df_filtered, split_mode='home_away')
        
        # 2. Compute Grids
        # Home (Team)
        # We need to know home_id
        home_id = df_game['home_id'].iloc[0]
        
        # Use simple standard grid params
        res = 1.0
        sigma = 6.0
        
        _, _, grid_home, _, _ = analyze.compute_xg_heatmap_from_df(
            df_adj, grid_res=res, sigma=sigma, 
            selected_team=home_id, selected_role='team', 
            total_seconds=toi_seconds
        )
        
        # Away (Other from perspective of Home, or Team from perspective of Away)
        # Actually, for the "Away Team's Heatmap", we want their shots.
        # `grid_home` contains shots BY the home team.
        # We also need shots BY the away team.
        _, _, grid_away, _, _ = analyze.compute_xg_heatmap_from_df(
            df_adj, grid_res=res, sigma=sigma,
            selected_team=home_id, selected_role='other',
            total_seconds=toi_seconds
        )
        
        # Basic Stats Aggregation (Same as before)
        goals = df_filtered[df_filtered['event'] == 'goal']
        home_goals = len(goals[goals['team_id'] == goals['home_id']]) if not goals.empty else 0
        away_goals = len(goals[goals['team_id'] == goals['away_id']]) if not goals.empty else 0
        
        is_home = df_filtered['team_id'] == df_filtered['home_id']
        xgs_series = df_filtered['xgs'].fillna(0)
        home_xg = xgs_series[is_home].sum()
        away_xg = xgs_series[~is_home].sum()
        
        attempts_mask = df_filtered['event'].isin(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'])
        home_cf = len(df_filtered[attempts_mask & is_home])
        away_cf = len(df_filtered[attempts_mask & (~is_home)])
        
        # Append Results with Grids
        # Grids are numpy arrays, pickleable.
        
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
            'ca': away_cf,
            # We save the grid of shots FOR this team (grid_home)
            # And shots AGAINST this team (grid_away)
            'grid_for': grid_home,
            'grid_against': grid_away
        })
        
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
            'ca': home_cf,
            # For away team, shots FOR are grid_away
            # Shots AGAINST are grid_home
            'grid_for': grid_away,
            'grid_against': grid_home
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
    print("Aggregating team stats and grids...")
    if not team_stats_acc:
        print("No stats collected.")
        return
        
    df_raw = pd.DataFrame(team_stats_acc)
    # Be careful with numpy arrays in DataFrame groupby sum, it works if they are object type?
    # Usually it doesn't sum arrays elementwise automatically in pandas groupby.
    # We should separate grids from scalar stats.
    
    # Separate scalar data for aggregation
    scalar_cols = ['team', 'game_id', 'gp', 'toi', 'gf', 'ga', 'xg_for', 'xg_against', 'cf', 'ca']
    df_scalar = df_raw[scalar_cols].copy()
    
    # Grids accumulation
    # Dictionary {team: {'for': sum_grid, 'against': sum_grid}}
    team_grids = {}
    
    print("  Summing heatmap grids...")
    for item in team_stats_acc:
        tm = item['team']
        if tm not in team_grids:
            team_grids[tm] = {'for': None, 'against': None}
        
        g_for = item.get('grid_for')
        g_ag = item.get('grid_against')
        
        if g_for is not None:
            if team_grids[tm]['for'] is None:
                team_grids[tm]['for'] = g_for.copy()
            else:
                team_grids[tm]['for'] += g_for
                
        if g_ag is not None:
            if team_grids[tm]['against'] is None:
                team_grids[tm]['against'] = g_ag.copy()
            else:
                team_grids[tm]['against'] += g_ag

    # Scalar aggregation
    numeric_cols = ['gp', 'toi', 'gf', 'ga', 'xg_for', 'xg_against', 'cf', 'ca']
    df_agg = df_scalar.groupby('team')[numeric_cols].sum().reset_index()
    
    # 5. Context Phase: Rates & Percentiles
    print("Calculating rates and percentiles...")
    
    df_agg['xg_for_60'] = (df_agg['xg_for'] / df_agg['toi']) * 3600
    df_agg['xg_against_60'] = (df_agg['xg_against'] / df_agg['toi']) * 3600
    df_agg['goals_for_60'] = (df_agg['gf'] / df_agg['toi']) * 3600
    df_agg['goals_against_60'] = (df_agg['ga'] / df_agg['toi']) * 3600
    df_agg['cf_60'] = (df_agg['cf'] / df_agg['toi']) * 3600
    df_agg['ca_60'] = (df_agg['ca'] / df_agg['toi']) * 3600
    
    df_agg['xg_pct'] = (df_agg['xg_for'] / (df_agg['xg_for'] + df_agg['xg_against'])) * 100
    df_agg['cf_pct'] = (df_agg['cf'] / (df_agg['cf'] + df_agg['ca'])) * 100
    
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
    
    # 7. Scatter Plot (Lightweight)
    print("Generating Scatter Plot...")
    try:
        plt.figure(figsize=(10, 8))
        plot_df = df_agg[df_agg['gp'] >= 5]
        if plot_df.empty: plot_df = df_agg
        x = plot_df['xg_for_60']
        y = plot_df['xg_against_60']
        teams = plot_df['team']
        plt.scatter(x, y, alpha=0.5)
        for i, team in enumerate(teams):
            plt.annotate(team, (x.iloc[i], y.iloc[i]), fontsize=8)
        plt.title(f"League xG Rates (5v5) - {season}")
        plt.xlabel("xG For / 60")
        plt.ylabel("xG Against / 60")
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        plot_path = os.path.join(out_dir, f'{season}_scatter.png')
        plt.savefig(plot_path, dpi=100)
        plt.close()
        print(f"Saved scatter plot to {plot_path}")
    except Exception as e:
        print(f"Plotting failed: {e}")
        
    # 8. Heatmap Plotting (Restored Feature)
    print("Generating Team Heatmaps...")
    # Define extent (standard NHL rink coords approx -100 to 100, -42.5 to 42.5)
    # The grid was computed with standard analyze defaults?
    # analyze.compute_xg_heatmap_from_df uses hardcoded bins or params?
    # It returns gx, gy which are the axes. We need those to compute extent.
    # We can just check one call to see extent.
    # Generally: extent = (min_x, max_x, min_y, max_y)
    # With grid_res=1.0 and standard rink:
    # x: -100 to 100 => 200 bins
    # y: -42.5 to 42.5 => 85 bins
    # Let's assume standard extent for now to match draw_rink
    # x_range = (-100, 100), y_range = (-42.5, 42.5)
    extent = (-100, 100, -42.5, 42.5)
    
    for team, grids in team_grids.items():
        # Plot "For"
        g_for = grids.get('for')
        if g_for is not None:
             out_p = os.path.join(out_dir, f"{season}_{team}_for.png")
             try:
                 plot.plot_heatmap_grid(g_for, extent, color='black', title=f"{team} xG For (5v5)", out_path=out_p)
             except Exception as e:
                 pass
                 
        # Plot "Against"
        g_ag = grids.get('against')
        if g_ag is not None:
             out_p = os.path.join(out_dir, f"{season}_{team}_against.png")
             try:
                 plot.plot_heatmap_grid(g_ag, extent, color='orange', title=f"{team} xG Against (5v5)", out_path=out_p)
             except Exception as e:
                 pass
    
    print("Team heatmaps generated.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
