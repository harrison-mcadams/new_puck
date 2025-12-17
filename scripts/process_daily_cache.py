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



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

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
            return True

        # Load Shifts
        df_shifts = timing._get_shifts_df(int(game_id), season=season)
        if df_shifts.empty:
            np.savez_compressed(out_path, processed=True, empty=True)
            return True

        # Helper to flip game state (5v4 <-> 4v5)
        def flip_condition(cond):
            if not cond: return {}
            new_cond = cond.copy()
            if 'game_state' in new_cond:
                flipped_states = []
                for s in new_cond['game_state']:
                    if s == '5v4': flipped_states.append('4v5')
                    elif s == '4v5': flipped_states.append('5v4')
                    elif s == '5v3': flipped_states.append('3v5')
                    elif s == '3v5': flipped_states.append('5v3')
                    else: flipped_states.append(s) # 5v5, 4v4, 3v3 stay same
                new_cond['game_state'] = flipped_states
            return new_cond

        # Compute Intervals for Home (Condition as-is) and Away (Flipped if needed)
        # Home Team "5v4" (PP) -> Game State 5v4
        # Away Team "5v4" (PP) -> Game State 4v5 (Home 4, Away 5)
        
        # Compute Intervals for Home and Away
        # Validation Condition (val_cond): The Relative State we want to measure (e.g. '4v5' for PK)
        val_cond_home = condition
        val_cond_away = condition

        # Global Condition (global_cond): The Global State that produces the Relative State
        # Home is aligned with Global (Global 4v5 -> Home 4v5)
        # Away is anti-aligned (Global 5v4 -> Away 4v5)
        global_cond_home = condition
        global_cond_away = flip_condition(condition)
        
        intervals_home = timing.get_game_intervals_cached(game_id, season, global_cond_home)
        intervals_away = timing.get_game_intervals_cached(game_id, season, global_cond_away)

        # Optimization: If both empty, skip game
        if not intervals_home and not intervals_away:
            np.savez_compressed(out_path, processed=True, empty=True)
            return True

        # Prepare Data Container
        data_to_save = {}
        
        # We need team IDs
        home_id = df_game.iloc[0]['home_id']
        away_id = df_game.iloc[0]['away_id']
        
        # Helper to run xgs_map
        def run_analysis(cond_local, entity_id, prefix, intervals_base, cond_base):

            if not intervals_base:
                return

            intervals_to_use = intervals_base
            if 'player_id' in cond_local:
                # Player Intersection
                pid = cond_local['player_id']
                p_shifts = df_shifts[df_shifts['player_id'] == pid]
                if p_shifts.empty:
                    return
                p_intervals = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
                intervals_to_use = timing._intersect_two(intervals_base, p_intervals)
            
            if not intervals_to_use:
                return
            
            toi = sum(e-s for s,e in intervals_to_use)
            if toi <= 0:
                return

            # Construct intervals_input
            intervals_input = {
                'per_game': {
                    game_id: {
                        'intersection_intervals': intervals_to_use
                    }
                }
            }
            
            # RUN XGS MAP
            try:
                # Use the passed base condition (flipped vs original) + local filters
                # REVERT: User wants strict label checking.
                # Since we correctly flipped the condition for Away team (5v4 -> 4v5),
                # the label check should pass for valid rows and correctly protect/filter.
                analysis_condition = cond_base.copy() if cond_base else {}
                
                # Copy orientation keys (already in cond_base usually, but ensure)
                if cond_base and 'team' in cond_base:
                     analysis_condition['team'] = cond_base['team']
                
                if 'player_id' in cond_local:
                    analysis_condition['player_id'] = cond_local['player_id']
                    # We pass the player's team explicitly if we know it (which we do, via selection of intervals_base)
                    # But xgs_map uses 'team' to orient.
                    # If this is an Away Player, we want stats oriented for Away Team.
                    # We should pass 'team' = entity's team.
                    
                    # Look up player team to be 100% sure we pass the right ID for orientation
                    try:
                        p_team = df_shifts.loc[df_shifts['player_id'] == cond_local['player_id'], 'team_id']
                        if not p_team.empty:
                            analysis_condition['team'] = int(p_team.iloc[0])
                    except: pass
                else:
                    analysis_condition['team'] = entity_id

                # Use existing xG for fake seasons (skip prediction)
                xg_behavior = 'skip' # We already computed xG for the whole season at start. Do NOT overwrite.


                _, grid_raw, _, stats = analyze.xgs_map(
                    season=season,
                    data_df=df_game,
                    intervals_input=intervals_input,
                    condition=analysis_condition,
                    heatmap_only=True,
                    total_seconds=toi,
                    behavior=xg_behavior
                )
                
                # if grid_raw is not None:
                if isinstance(grid_raw, dict):
                    grid_final = grid_raw.get('team')
                    grid_other = grid_raw.get('other')
                else:
                    grid_final = grid_raw
                    grid_other = None

                if grid_final is not None:
                    grid_final = np.nan_to_num(np.asarray(grid_final, dtype=np.float32), nan=0.0)
                    data_to_save[f"{prefix}_grid_team"] = grid_final
                        
                    if grid_other is not None:
                        # Rotate "Other" (Opponent Offense/Left) to "Defense" (Right)
                        grid_other_arr = np.nan_to_num(np.asarray(grid_other, dtype=np.float32), nan=0.0)
                        grid_other_rot = np.rot90(grid_other_arr, 2)
                        data_to_save[f"{prefix}_grid_other"] = grid_other_rot
                
                if stats:
                    data_to_save[f"{prefix}_stats"] = json.dumps(stats, cls=NumpyEncoder)
                    
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Analysis failed for {prefix}: {e}", flush=True)

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
        run_analysis({}, home_id, f"team_{home_id}", intervals_home, val_cond_home)
        run_analysis({}, away_id, f"team_{away_id}", intervals_away, val_cond_away)
        
        # 2. Players
        # Assign players to Home/Away interval sets based on their Team ID
        pids = set(df_shifts['player_id'].unique())
        for pid in pids:
             # Find team
             try:
                 p_team_rows = df_shifts.loc[df_shifts['player_id'] == pid, 'team_id']
                 if p_team_rows.empty: continue
                 p_tid = p_team_rows.iloc[0]
                 
                 # Compare Int to Int
                 if int(p_tid) == int(home_id):
                     run_analysis({'player_id': pid}, pid, f"p_{pid}", intervals_home, val_cond_home)
                 elif int(p_tid) == int(away_id):
                     run_analysis({'player_id': pid}, pid, f"p_{pid}", intervals_away, val_cond_away)
             except Exception:
                 continue
             
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
    xg_behavior = 'skip' if str(season).startswith('fake') else 'overwrite'
    df_data, _, _ = analyze._predict_xgs(df_data, behavior=xg_behavior)
    
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
            if count % 1 == 0:
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
