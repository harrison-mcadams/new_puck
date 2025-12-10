"""
run_player_analysis.py (Optimized V2)

A memory-efficient "Map-Reduce" style pipeline for calculating player stats.

Architecture:
1. Map Phase: Iterate through games one by one (or in parallel).
   - For each game, distinct players are identified.
   - Per-player stats (TOI, xG, Goals) are calculated using `analyze.xgs_map(stats_only=True)`.
   - Results are purely lightweight dictionaries (no heavy DataFrames retained).
   - Results are appended to a list.
2. Reduce Phase:
   - Aggregating the list of stats by Player ID.
   - Summing totals (Season Totals).
   - Calculating Per-60 Rates.
3. Context Phase:
   - Calculating League-wide percentiles for all metrics.
4. Output Phase:
   - Saving the final JSON for the web app.

This approach prevents OOM errors on Raspberry Pi by avoiding repeated filtering of the massive season DataFrame,
and allows for parallel execution on more powerful hardware.
"""

import os
import sys
import gc
import json
import argparse
import pandas as pd
import numpy as np
import multiprocessing
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import parse
from puck import plot

def process_game(args):
    """
    Pure function to process a single game.
    Args:
        args (tuple): (game_id, df_game, condition, season)
    Returns:
        list: List of player stat dictionaries.
    """
    game_id, df_game, condition, season = args
    results = []
    
    try:
        if df_game.empty:
            return []
            
        # Get Shifts
        # Using cache=True to leverage disk/mem cache
        df_shifts = timing._get_shifts_df(int(game_id))
        if df_shifts.empty:
            return []
            
        # Get Intervals (Shared)
        common_intervals = timing.get_game_intervals_cached(game_id, season, condition)
        if not common_intervals:
            return []
            
        # Identify Players
        players_in_game = df_shifts['player_id'].unique()
        p_team_map = df_shifts.groupby('player_id')['team_id'].first().to_dict()
        
        # Loop Players in Game
        for pid in players_in_game:
            # Player Shifts
            try:
                p_shifts = df_shifts[df_shifts['player_id'] == pid]
                if p_shifts.empty: continue
                
                # Intersect
                p_intervals_raw = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
                
                # Let's do the intersection here manually to be safe and light.
                from puck.timing import _intersect_two
                final_intervals = _intersect_two(common_intervals, p_intervals_raw)
                
                if not final_intervals:
                    continue
                    
                toi_seconds = sum(e - s for s, e in final_intervals)
                if toi_seconds <= 0:
                    continue
                
                # Inputs for xgs_map
                intervals_input = {
                    'per_game': {
                        game_id: {'intersection_intervals': final_intervals}
                    }
                }
                
                p_team_id = p_team_map.get(pid)
                p_cond = condition.copy()
                p_cond['team'] = p_team_id # Perspective for this player
                
                # Call Kernel (Stats Only)
                # Ensure game_id is string/int consistent? xgs_map handles loose types usually.
                # Pass filtered df_game to prevent reloading.
                _, _, _, p_stats = analyze.xgs_map(
                    season=season,
                    data_df=df_game, # Passed explicitly
                    condition=p_cond,
                    return_heatmaps=False, # We compute separately
                    show=False,
                    stats_only=True, # Critical to avoid plotting overhead
                    total_seconds=toi_seconds,
                    use_intervals=True,
                    intervals_input=intervals_input,
                    interval_time_col='total_time_elapsed_seconds', # Match event data column
                )
                
                if p_stats:
                    # Enrich Basic Info
                    p_stats['player_id'] = int(pid)
                    p_stats['team'] = p_stats.get('team') # Abbrev usually returned by xgs_map
                    p_stats['game_id'] = int(game_id)
                    p_stats['toi'] = toi_seconds
                    
                    # --- HEATMAP GRID COMPUTATION ---
                    # We need the player's filtered dataframe to compute their specific heatmap.
                    # xgs_map did the filtering internally but didn't return the df because stats_only=True
                    # We should probably do the filtering ourselves to be efficient if we need the df for both.
                    # Or just call apply_intervals here.
                    
                    # Re-filter for grid (inefficient to do twice? xgs_map does it.)
                    # Let's use analyze._apply_intervals locally.
                    df_filtered = analyze._apply_intervals(
                        df_game, 
                        intervals_input['per_game'][game_id]['intersection_intervals'],
                        time_col='total_time_elapsed_seconds'
                    )
                    
                    if not df_filtered.empty:
                        # Adjust xy
                        df_adj = plot.adjust_xy_for_homeaway(df_filtered, split_mode='home_away')
                        
                        # Compute grid (Team Perspective - i.e. Player's perspective)
                        res = 1.0
                        sigma = 6.0
                        # We want shots BY the player's team (and specifically usually the player? No, usually "On Ice" calls are team stats)
                        # analyze.players() usually creates "Relative to League" maps which means we need On-Ice Team For.
                        
                        # Compute 'On-Ice For' Grid
                        # Note: we need to know the 'team_id' integer.
                        try:
                            tid_int = int(p_team_id) if p_team_id else None
                        except:
                            tid_int = None
                            
                        # If p_team_id is None, we can't filter 'For'.
                        # But we have map from shifts.
                        
                        _, _, grid_for, _, _ = analyze.compute_xg_heatmap_from_df(
                            df_adj, grid_res=res, sigma=sigma,
                            selected_team=tid_int, selected_role='team',
                            total_seconds=toi_seconds
                        )
                        
                        # We attach grid to result
                        p_stats['grid_for'] = grid_for
                        # Do we need 'Against'? Usually player maps are Offense.
                        # If we want Defense maps, we need Against.
                        # analyze.players() generates both?
                        # It generates "Relative xG" maps. Usually just offense shown? 
                        # Let's store Against too just in case.
                        _, _, grid_against, _, _ = analyze.compute_xg_heatmap_from_df(
                            df_adj, grid_res=res, sigma=sigma,
                            selected_team=tid_int, selected_role='other',
                            total_seconds=toi_seconds
                        )
                        p_stats['grid_against'] = grid_against
                    
                    # Accumulate
                    results.append(p_stats)
                    
            except Exception as e:
                # Log but don't crash loop
                # print(f"Error processing player {pid} in game {game_id}: {e}")
                pass
    except Exception as e:
        print(f"Error processing game {game_id}: {e}")
        
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Player Analysis (Optimized)")
    parser.add_argument('--season', type=str, default='20252026')
    parser.add_argument('--out_dir', type=str, default='analysis/players')
    parser.add_argument('--parallel', action='store_true', help="Use multiprocessing for faster execution on Mac")
    args = parser.parse_args()
    
    season = args.season
    out_dir = args.out_dir
    use_parallel = args.parallel
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Starting Optimized Player Analysis for Season {season} ---")
    print(f"Mode: {'PARALLEL' if use_parallel else 'SEQUENTIAL'}")
    
    # 1. Load Full Season Data (Once)
    print("Loading season data...")
    df_data = timing.load_season_df(season)
    if df_data is None or df_data.empty:
        print("No data found. Exiting.")
        return

    # 2. Pre-Calculate xG (Batch)
    # This adds 'xgs' column to df_data so we don't re-run the model 50,000 times
    print("Pre-calculating xG predictions for the season...")
    df_data, _, _ = analyze._predict_xgs(df_data)
    
    # 3. Optimizing Dataframe Memory
    # Build Player Name Map BEFORE dropping/deleting data
    print("Building player name map from season data...")
    player_name_map = {}
    try:
        # Extract from event participants (p1, p2, goalie, etc)
        # Union of p1 and p2 usually covers everyone except maybe backup goalies who never played?
        # Actually backup goalies won't have stats anyway.
        cols_to_scan = [('event_player_1_id', 'event_player_1_name'), 
                        ('event_player_2_id', 'event_player_2_name'),
                        ('event_goalie_id', 'event_goalie_name')]
        
        for id_col, name_col in cols_to_scan:
            if id_col in df_data.columns and name_col in df_data.columns:
                subset = df_data[[id_col, name_col]].dropna().drop_duplicates()
                for _, r in subset.iterrows():
                    try:
                        pid = int(r[id_col])
                        if pid not in player_name_map:
                            player_name_map[pid] = r[name_col]
                    except: pass
    except Exception as e:
        print(f"Warning: Failed to build name map: {e}")
    
    print(f"Season Data Shape: {df_data.shape}")
    
    # 4. Map Phase: Iterate Games
    game_ids = sorted(df_data['game_id'].unique())
    print(f"Processing {len(game_ids)} games...")
    
    # Pre-configure conditions we care about (usually just '5v5' for base stats)
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    all_player_stats = []
    
    # Prepare Map Arguments
    # Note: df_game slices are view/copies.
    # For parallel: we create a list of tuples.
    # For sequential: we can lazy load or just iterate.
    
    # To save memory during mapping preparation, we can iterate.
    map_args = []
    if use_parallel:
        print("Preparing tasks for parallel execution...")
        for gid in game_ids:
            # We must pass the dataframe slice.
            # Warning: Pickling huge dataframes can be slow.
            # But partitioning by game makes them small (~1MB).
            df_g = df_data[df_data['game_id'] == gid].copy()
            map_args.append((gid, df_g, condition, season))
            
        print(f"Dispatching {len(map_args)} tasks to worker pool...")
        with multiprocessing.Pool() as pool:
            # Chunksize: usually heuristics, maybe 1?
            results_nested = pool.map(process_game, map_args)
            
        # Flatten results
        for res_list in results_nested:
            all_player_stats.extend(res_list)
            
    else:
        # Sequential Execution (Pi Friendly)
        for idx, game_id in enumerate(game_ids):
            if (idx + 1) % 50 == 0:
                print(f"  Game {idx+1}/{len(game_ids)}...", flush=True)
                gc.collect()
            
            df_g = df_data[df_data['game_id'] == game_id]
            # Call process_game directly
            res_list = process_game((game_id, df_g, condition, season))
            all_player_stats.extend(res_list)
                
    # Free memory of full season data before aggregation if constrained
    print(f"Finished Map Phase. Accumulated stats for {len(all_player_stats)} player-games.")
    del df_data
    if 'map_args' in locals(): del map_args
    gc.collect()
    
    # 5. Reduce Phase: Aggregation
    print("Aggregating stats and grids...")
    if not all_player_stats:
        print("No stats collected.")
        return
        
    df_raw = pd.DataFrame(all_player_stats)
    
    # Separate Grids from Scalar
    # Accumulate grids in a dictionary first because DataFrame groupby sum is inconsistent with arrays
    player_grids = {}
    
    print("  Summing player heatmap grids...")
    # This loop might be slow if millions of rows?
    # Player-games: 1300 games * 36 players ~ 46,000 rows. Fast enough.
    for item in all_player_stats:
        pid = item['player_id']
        if pid not in player_grids:
            player_grids[pid] = {'for': None, 'against': None}
            
        g_for = item.get('grid_for')
        g_ag = item.get('grid_against')
        
        if g_for is not None:
            if player_grids[pid]['for'] is None:
                player_grids[pid]['for'] = g_for.copy()
            else:
                player_grids[pid]['for'] += g_for
                
        if g_ag is not None:
            if player_grids[pid]['against'] is None:
                player_grids[pid]['against'] = g_ag.copy()
            else:
                player_grids[pid]['against'] += g_ag
    
    # Define aggregation columns
    # We must exclude 'grid_for', 'grid_against' explicitly as they are in df_raw now
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # exclude game_id from sum
    numeric_cols = [c for c in numeric_cols if c != 'game_id' and c != 'player_id' and c != 'grid_for' and c != 'grid_against']
    
    df_agg = df_raw.groupby('player_id')[numeric_cols].sum().reset_index()
    
    # Recover Names and Teams
    records = df_agg.to_dict('records')
    print(f"Resolving names for {len(records)} players...")
    
    # We need to perform the team mode lookup efficiently.
    # Group by player_id -> team value counts -> idxmax
    print("Resolving player teams...")
    try:
        # Get most frequent team for each player
        player_teams = df_raw.groupby('player_id')['team'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'UNK').to_dict()
    except Exception as e:
        print(f"Warning deriving player teams: {e}")
        player_teams = {}

    for r in records:
        pid = r['player_id']
        try:
            # Trusted Source 1: Map from Season Data (built at start)
            name = player_name_map.get(pid)
            
            # Trusted Source 2: API Fallback (only if missing)
            if not name:
                 name = f"Player {pid}"
            
            r['player_name'] = name
            r['team'] = player_teams.get(pid, 'UNK')
                
        except Exception:
            r['player_name'] = f"Player {pid}"
            r['team'] = 'UNK'

    # 6. Output Phase
    out_file = os.path.join(out_dir, f'{season}_players.json')
    with open(out_file, 'w') as f:
        json.dump(records, f, indent=2)
        
    print(f"Saved player analysis to {out_file}")
    
    # 7. Generate Player Heatmaps
    print("Generating Player Heatmaps...")
    extent = (-100, 100, -42.5, 42.5)
    
    # Only generate for players with meaningful TOI?
    # e.g. > 60 mins
    
    count_plots = 0
    for pid, grids in player_grids.items():
        # Check TOI from agg
        p_rec = next((x for x in records if x['player_id'] == pid), None)
        if not p_rec or p_rec.get('toi', 0) < 3000: # 50 mins
            continue
            
        p_name = p_rec.get('player_name', f'Player {pid}')
        
        g_for = grids.get('for')
        if g_for is not None:
             out_p = os.path.join(out_dir, f"{season}_player_{pid}_for.png")
             try:
                 plot.plot_heatmap_grid(g_for, extent, color='black', title=f"{p_name} xG For (5v5)", out_path=out_p)
                 count_plots += 1
             except Exception as e:
                 pass
                 
    print(f"Generated heatmaps for {count_plots} players.")

if __name__ == "__main__":
    # On Mac/Windows, spawn is default in recent Pythons.
    # We need robust main check.
    multiprocessing.freeze_support()
    main()
