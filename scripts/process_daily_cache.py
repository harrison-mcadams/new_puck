import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
import gc
import shutil
import time

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config
from puck import timing
from puck import analyze


def ensure_dirs(season):
    base = config.get_cache_dir(season)
    partials = os.path.join(base, 'partials')
    os.makedirs(partials, exist_ok=True)
    return partials

def get_game_partials_path(partials_dir, game_id, condition_name):
    return os.path.join(partials_dir, f"{game_id}_{condition_name}.npz")

def process_game(game_id, df_game, season, condition, partials_dir, condition_name, force=False):
    """
    Process a single game:
    1. Calculate Team Stats & Maps (Home/Away).
    2. Calculate Player Stats & Maps (All players in game).
    3. Save to .npz.
    """
    try:
        out_path = get_game_partials_path(partials_dir, game_id, condition_name)
        if not force and os.path.exists(out_path):
            # We could check modification time vs data modification time?
            # For now, assume if it exists it's done unless force flag (handled by caller deleting).
            return True

        # Load Shifts (Cached internally by timing module, cleared regularly)
        df_shifts = timing._get_shifts_df(int(game_id))
        if df_shifts.empty:
            # Save empty placeholder
            np.savez_compressed(out_path, processed=True, empty=True)
            return True

        # Pre-compute Intervals
        common_intervals = timing.get_game_intervals_cached(game_id, season, condition)
        if not common_intervals:
            np.savez_compressed(out_path, processed=True, empty=True)
            return True

        # Prepare Data Container
        # keys: 'home_stats', 'away_stats', 'home_grid', 'away_grid', 
        #       'p_{pid}_stats', 'p_{pid}_grid'
        data_to_save = {}
        
        # --- Team Analysis ---
        # We need team IDs
        home_id = df_game.iloc[0]['home_id']
        away_id = df_game.iloc[0]['away_id']
        
        # Helper to run xgs_map for a specific entity condition
        def run_analysis(cond_local, entity_id, prefix):
            # We need to compute total_seconds for this entity to pass to xgs_map
            # For teams, it's just the common_intervals sum
            # For players, it's the intersection
            
            intervals_to_use = common_intervals
            if 'player_id' in cond_local:
                # Player Intersection
                pid = cond_local['player_id']
                p_shifts = df_shifts[df_shifts['player_id'] == pid]
                if p_shifts.empty:
                    # print(f"DEBUG: No shifts for player {pid}")
                    return
                p_intervals = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
                # Intersect
                # (Inline intersection for speed or use timing helper)
                # timing._intersect_intervals is not public/exposed easily, let's use the one in previous script logic
                # or verify timing.intersect_intervals exists.
                # It's timing._intersect_two usually.
                # Let's just reimplement simple one here to avoid internal dep issues if not exposed.
                intervals_to_use = timing._intersect_two(common_intervals, p_intervals)
            
            if not intervals_to_use:
                # print(f"DEBUG: intervals_to_use empty for {prefix} (common={len(common_intervals)})")
                return
            
            toi = sum(e-s for s,e in intervals_to_use)
            if toi <= 0:
                # print(f"DEBUG: TOI <= 0 for {prefix}")
                return

            print(f"DEBUG: Running xgs_map for {prefix} (TOI={toi:.1f})")

            # Construct intervals_input for xgs_map to force it to use these intervals
            intervals_input = {
                'per_game': {
                    game_id: {
                        'intersection_intervals': intervals_to_use
                    }
                }
            }
            
            # RUN XGS MAP
            # This calls analyze._predict_xgs internally if not present, but we already ensured it.
            # analyze.xgs_map returns: (xg_grid, stats_dict)
            try:
                # IMPORTANT: We need to pass df_game via data_df!
                # Prepare condition for xgs_map to filter/orient correctly
                # FIX: Start with global condition (e.g. {'game_state': ['5v5']}) to enforce strict filtering
                analysis_condition = condition.copy() if condition else {}
                
                if 'player_id' in cond_local:
                    analysis_condition['player_id'] = cond_local['player_id']
                    
                    # FIX: Players need 'team' in condition to split For/Against stats
                    # Infer team from shifts (use the team from the first shift found)
                    # We have p_shifts available in outer scope? No, need to pass it or re-derive.
                    # Wait, 'run_analysis' is a helper. We need access to p_shifts.
                    # Let's verify if we can get it.
                    # 'p_shifts' was local to the if block above (lines 75).
                    # We should restructure slightly to get team_id easily.
                    
                    # Re-fetch pid to be safe (cond_local has it)
                    pid_local = cond_local['player_id']
                    # We can't access p_shifts from here efficiently if variable scope is limited.
                    # But wait, python closures capture variables? 
                    # p_shifts is defined inside the if block earlier. 
                    
                    # Let's peek at df_shifts again to find the team for this player
                    # efficient lookup:
                    try:
                        # Find team_id for this player in this game's shifts
                        # Filter to just this player's rows
                        # We know df_shifts exists in outer scope
                        # Use iloc[0]
                        p_team = df_shifts.loc[df_shifts['player_id'] == pid_local, 'team_id']
                        if not p_team.empty:
                            analysis_condition['team'] = int(p_team.iloc[0])
                    except Exception:
                        pass
                else:
                    # For teams, we pass 'team' in the condition so xgs_map knows which team to orient for
                    analysis_condition['team'] = entity_id

                # xgs_map returns: (out_path, heatmaps, df_filtered, summary_stats)
                _, grid_raw, _, stats = analyze.xgs_map(
                    season=season,
                    data_df=df_game,
                    intervals_input=intervals_input,
                    condition=analysis_condition,
                    heatmap_only=True,  # Return data only, do not generate plots
                    total_seconds=toi   # PASS TOI explicitly to ensure team_seconds is correct
                )
                
                # Unwrap and Sanitize Grid
                if grid_raw is not None:
                    # Unwrap dict if xgs_map returned one (it returns {'team': ..., 'other': ...} when team is selected)
                    if isinstance(grid_raw, dict):
                        grid_final = grid_raw.get('team') # We only care about the entity's perspective here
                    else:
                        grid_final = grid_raw
                        
                    # Sanitize: Convert to float32 and fill NaNs with 0.0
                    try:
                        grid_final = np.asarray(grid_final, dtype=np.float32)
                        grid_final = np.nan_to_num(grid_final, nan=0.0)
                        data_to_save[f"{prefix}_grid_team"] = grid_final
                    except Exception as e:
                        print(f"Warning: Failed to sanitize grid for {prefix}: {e}")
                
                if stats:
                    data_to_save[f"{prefix}_stats"] = json.dumps(stats)
                    
            except Exception as e:
                print(f"Analysis failed for {prefix} in {game_id}: {e}")
                import traceback
                traceback.print_exc()
                pass

        # Helper for intersection
        def _intersect(int_a, int_b):
            # Assumes sorted, non-overlapping inputs
            result = []
            i, j = 0, 0
            while i < len(int_a) and j < len(int_b):
                a_start, a_end = int_a[i]
                b_start, b_end = int_b[j]
                
                start = max(a_start, b_start)
                end = min(a_end, b_end)
                
                if start < end:
                    result.append((start, end))
                
                if a_end < b_end:
                    i += 1
                else:
                    j += 1
            return result

        # 1. Teams
        run_analysis({}, home_id, f"team_{home_id}")
        run_analysis({}, away_id, f"team_{away_id}")
        
        # 2. Players
        # Get all players in game
        pids = set(df_shifts['player_id'].unique())
        for pid in pids:
             run_analysis({'player_id': pid}, pid, f"p_{pid}")
             
        # Save
        np.savez_compressed(out_path, processed=True, **data_to_save)
        return True

    except Exception as e:
        print(f"Failed to process game {game_id}: {e}")
        # Delete corrupt file if exists
        if os.path.exists(get_game_partials_path(partials_dir, game_id, condition_name)):
            try: os.remove(get_game_partials_path(partials_dir, game_id, condition_name)) 
            except: pass
        return False
        
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=str, default='20252026')
    parser.add_argument('--condition', type=str, default='5v5') # 5v5, 5v4, 4v5
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    
    season = args.season
    cond_name = args.condition
    
    # Resolve condition dict
    conditions_map = {
        '5v5': {'game_state': ['5v5'], 'is_net_empty': [0]},
        '5v4': {'game_state': ['5v4'], 'is_net_empty': [0]},
        '4v5': {'game_state': ['4v5'], 'is_net_empty': [0]}
    }
    condition = conditions_map.get(cond_name)
    if not condition:
        print(f"Unknown condition: {cond_name}")
        return

    print(f"--- Processing Cache for {season} [{cond_name}] ---")
    
    # Verify/Create Cache Dir
    partials_dir = ensure_dirs(season)
    print(f"Using Partial Cache Dir: {partials_dir}")
    
    # Load Season Data
    df_data = timing.load_season_df(season)
    if df_data is None or df_data.empty:
        print("No data found.")
        return
        
    # Ensure xGs
    df_data, _, _ = analyze._predict_xgs(df_data)
    
    partials_dir = ensure_dirs(season)
    
    # Filter games that need processing
    all_game_ids = sorted(df_data['game_id'].unique())
    games_to_process = []
    
    for gid in all_game_ids:
        path = get_game_partials_path(partials_dir, gid, cond_name)
        if args.force or not os.path.exists(path):
            games_to_process.append(gid)
            
    print(f"Found {len(games_to_process)} games to process out of {len(all_game_ids)} total.")
    
    if not games_to_process:
        return

    # Process Loop
    # Use config for batch size / GC
    count = 0
    start_time = time.time()
    
    for gid in games_to_process:
        # Extract single game DF
        df_game = df_data[df_data['game_id'] == gid]
        if df_game.empty: continue
        
        success = process_game(gid, df_game, season, condition, partials_dir, cond_name, force=args.force)
        
        if success:
            count += 1
            if count % 10 == 0:
                elapsed = time.time() - start_time
                rate = count / elapsed
                print(f"Processed {count}/{len(games_to_process)} ({rate:.2f} games/s)")
                
        # Memory Management
        if count % config.GC_FREQUENCY == 0:
            if hasattr(timing, '_SHIFTS_CACHE'):
                timing._SHIFTS_CACHE.clear()
            gc.collect()
            
    print("Cache processing complete.")

if __name__ == "__main__":
    main()
