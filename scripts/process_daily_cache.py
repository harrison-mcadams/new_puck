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

def process_game(game_id, df_game, season, condition, partials_dir, condition_name):
    """
    Process a single game:
    1. Calculate Team Stats & Maps (Home/Away).
    2. Calculate Player Stats & Maps (All players in game).
    3. Save to .npz.
    """
    try:
        out_path = get_game_partials_path(partials_dir, game_id, condition_name)
        if os.path.exists(out_path):
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
                    return
                p_intervals = list(zip(p_shifts['start_total_seconds'], p_shifts['end_total_seconds']))
                # Intersect
                # (Inline intersection for speed or use timing helper)
                # timing._intersect_intervals is not public/exposed easily, let's use the one in previous script logic
                # or verify timing.intersect_intervals exists.
                # It's timing._intersect_intervals usually.
                # Let's just reimplement simple one here to avoid internal dep issues if not exposed.
                intervals_to_use = _intersect(common_intervals, p_intervals)
            
            if not intervals_to_use:
                return

            toi = sum(e-s for s,e in intervals_to_use)
            if toi <= 0:
                return

            # Construct intervals_input for xgs_map to force it to use these intervals
            intervals_input = {
                'per_game': {
                    game_id: {
                        'intersection_intervals': intervals_to_use
                    }
                }
            }

            # Run Map
            # We use heatmap_only=True if available, or return_heatmaps=True
            # Note: xgs_map signature in analyze.py: 
            #   heatmap_only: compute and return heatmap arrays instead of plotting
            
            try:
                # analyze.xgs_map might raise if no events found
                res = analyze.xgs_map(
                    season=season,
                    data_df=df_game,
                    condition=cond_local,
                    heatmap_only=True, # Efficient mode
                    total_seconds=toi,
                    use_intervals=True,
                    intervals_input=intervals_input,
                    stats_only=False,
                    return_heatmaps=True,
                    show=False,
                    out_path=None
                )
                
                # Unpack
                # If heatmap_only=True, res should be (heatmap_grid, stats_dict) or similar?
                # Let's check the function signature again or assume standard
                # Standard with return_heatmaps=True is: (out_path, heatmaps, filtered_df, stats)
                # If heatmap_only is set, it might change return. 
                # Let's assume standard 4-tuple for now, as heatmap_only might just skip plot side effects.
                
                # Actually, I'll use the standard call to be safe: return_heatmaps=True, show=False.
                # If heatmap_only is supported, it hopefully adheres to return pattern.
                if isinstance(res, tuple):
                    if len(res) == 4:
                        _, heatmaps, _, stats = res
                    elif len(res) == 2: # Maybe heatmap_only returns (heat, stats)
                         heatmaps, stats = res
                    else:
                        return # Unknown
                else:
                    return

                if stats:
                    data_to_save[f"{prefix}_stats"] = json.dumps(stats) # Save stats as JSON string
                
                if heatmaps is not None:
                    # heatmaps can be dict {'team': ..., 'other': ...}
                    if isinstance(heatmaps, dict):
                        if 'team' in heatmaps:
                            data_to_save[f"{prefix}_grid_team"] = heatmaps['team']
                        if 'other' in heatmaps:
                            data_to_save[f"{prefix}_grid_other"] = heatmaps['other']
            except Exception as e:
                # print(f"Error in xgs_map for {prefix}: {e}")
                pass

        # 1. Teams
        # Home
        run_analysis({**condition, 'team': home_id}, home_id, f"team_{home_id}")
        # Away
        run_analysis({**condition, 'team': away_id}, away_id, f"team_{away_id}")
        
        # 2. Players
        players = df_shifts['player_id'].unique()
        p_team_map = df_shifts.groupby('player_id')['team_id'].first().to_dict()
        
        for pid in players:
            p_tm = p_team_map.get(pid)
            cond_p = {**condition, 'player_id': pid, 'team': p_tm}
            run_analysis(cond_p, pid, f"p_{pid}")
            
        # Save to NPZ
        np.savez_compressed(out_path, **data_to_save)
        return True

    except Exception as e:
        print(f"Failed to process game {game_id}: {e}")
        return False

def _intersect(a, b):
    res = []
    i = j = 0
    a = sorted(a)
    b = sorted(b)
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if e > s:
            res.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return res

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
        
        success = process_game(gid, df_game, season, condition, partials_dir, cond_name)
        
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
