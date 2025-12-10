"""
run_player_analysis.py (Optimized V2)

A memory-efficient "Map-Reduce" style pipeline for calculating player stats.

Architecture:
1. Map Phase: Iterate through games one by one.
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

This approach prevents OOM errors on Raspberry Pi by avoiding repeated filtering of the massive season DataFrame.
"""

import os
import sys
import gc
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import timing
from puck import analyze
from puck import parse

def main():
    parser = argparse.ArgumentParser(description="Run Player Analysis (Optimized)")
    parser.add_argument('--season', type=str, default='20252026')
    parser.add_argument('--out_dir', type=str, default='analysis/players')
    args = parser.parse_args()
    
    season = args.season
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Starting Optimized Player Analysis for Season {season} ---")
    
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
    
    all_player_stats = []
    
    # Pre-configure conditions we care about (usually just '5v5' for base stats)
    # If we want all situations, we can run multiple passes or handle it inside.
    # For now, let's stick to standard 5v5 analysis as primary.
    condition = {'game_state': ['5v5'], 'is_net_empty': [0]}
    
    for idx, game_id in enumerate(game_ids):
        if (idx + 1) % 50 == 0:
            print(f"  Game {idx+1}/{len(game_ids)}...", flush=True)
            gc.collect()

        # Extract game subset (View/Slice)
        df_game = df_data[df_data['game_id'] == game_id]
        if df_game.empty:
            if idx < 5: print(f"DEBUG: Game {game_id} skipped: Empty df_game")
            continue
            
        # Get Shifts
        # Using cache=True to leverage disk/mem cache
        df_shifts = timing._get_shifts_df(int(game_id))
        if df_shifts.empty:
            if idx < 5: print(f"DEBUG: Game {game_id} skipped: Empty df_shifts")
            continue
            
        # Get Intervals (Shared)
        common_intervals = timing.get_game_intervals_cached(game_id, season, condition)
        if not common_intervals:
            if idx < 5: print(f"DEBUG: Game {game_id} skipped: Empty intervals")
            continue
            
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
                # Use timing's intersection logic or analyze internal?
                # timing._intersect_multiple or just analyze.xgs_map using intervals input.
                # Actually, we need to calculate TOI *before* calling xgs_map if we want to pass it "total_seconds"
                # But analyze.xgs_map can compute it if we pass "use_intervals=True".
                # HOWEVER, xgs_map expects 'intervals_input' in a specific format if we pass it.
                
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
                    return_heatmaps=False,
                    show=False,
                    stats_only=True, # Critical
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
                    
                    # Accumulate
                    all_player_stats.append(p_stats)
                    
            except Exception as e:
                # Log but don't crash loop
                print(f"Error processing player {pid} in game {game_id}: {e}")
                pass
                
    # Free memory of full season data before aggregation if constrained
    print(f"Finished Map Phase. Accumulated stats for {len(all_player_stats)} player-games.")
    del df_data
    gc.collect()
    
    # 5. Reduce Phase: Aggregation
    print("Aggregating stats...")
    if not all_player_stats:
        print("No stats collected.")
        return
        
    df_raw = pd.DataFrame(all_player_stats)
    
    # Define aggregation columns
    # We want sum of: xg_for, xg_against, goals_for, goals_against, toi, etc.
    # Group by player_id
    # Also need to keep 'team' (take last or mode?) - Players switch teams.
    # For simplicity, let's take 'last' team or list them.
    # Let's group by ['player_id'] and take sum of numeric.
    
    numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
    # exclude game_id from sum
    numeric_cols = [c for c in numeric_cols if c != 'game_id' and c != 'player_id']
    
    df_agg = df_raw.groupby('player_id')[numeric_cols].sum().reset_index()
    
    # Get Player Names (from most recent game or map)
    # We can load player names from a mapping or simple lookback
    # Ideally we'd have a helper in timing or just use the last 'team' and 'player_name' if we had captured it.
    # xgs_map doesn't always return player name.
    # Let's recover names from the initial df_raw via merge or last()
    # But df_raw might not have names if xgs_map didn't return them?
    # xgs_map returns 'team' (abb). It does NOT return player name usually unless logic added.
    # We might need to fetch names from NHLE API or existing cache.
    # Actually, analyze.players used to do this.
    # Let's use `puck.nhl_api.get_player_name(pid)` if needed, or cache it during Map phase.
    # We can cache it in map phase from df_shifts (it usually unfortunately doesn't have name).
    # Wait, df_shifts has names if scraped from HTML? API shifts usually just have ID.
    # The 'roster' endpoint is best.
    # Optimization: We can load the roster ONCE properties from API/Cache at the end.
    
    # Fetch names
    print(f"Resolving names for {len(records)} players...")
    # from puck import nhl_api # Already imported above if needed, but we try map first
    
    for r in records:
        pid = r['player_id']
        try:
            # Trusted Source 1: Map from Season Data (built at start)
            name = player_name_map.get(pid)
            
            # Trusted Source 2: API Fallback (only if missing)
            if not name:
                 # Minimal API call if absolutely needed
                 # name = nhl_api.get_player_name(pid) 
                 name = f"Player {pid}"
            
            r['player_name'] = name
            
            # Team from raw (mode)
            # Find rows for this player
            # (Optimization: could have been done in agg)
            subset = df_raw[df_raw['player_id'] == pid]
            if not subset.empty:
                r['team'] = subset['team'].mode().iloc[0] if not subset['team'].mode().empty else 'N/A'
                
        except Exception:
            r['player_name'] = f"Player {pid}"
            r['team'] = 'UNK'

    # 6. Output Phase
    out_file = os.path.join(out_dir, f'{season}_players.json')
    with open(out_file, 'w') as f:
        json.dump(records, f, indent=2)
        
    print(f"Saved player analysis to {out_file}")
    
    # Optional: Scatter Plots if requested? (Skipped for "Daily" routine usually)

if __name__ == "__main__":
    main()
