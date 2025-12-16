#!/usr/bin/env python3
"""
scripts/validate_time_filtering.py

Validates that xG and Event totals match between:
1. 'Naive' Filtering: df[state == '5v5'] (or whatever column is available)
2. 'Rigorous' Timing: using timing.compute_game_timing intervals to filter by time.

Usage:
    python3 scripts/validate_time_filtering.py --season 20252026 --games 50
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import puck
import puck.parse
import puck.timing as timing
import puck.analyze as analyze

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pandas as pd
import numpy as np

from puck import analyze
from puck import nhl_api

def get_game_ids(season, n_games=None):
    # Try to load season df
    df = timing.load_season_df(season)
    if df is not None and not df.empty:
        gids = df['game_id'].unique().tolist()
        if n_games:
            return gids[:n_games], df
        return gids, df
    return [], None

def validate_game(game_id, df_game, conditions=['5v5', '5v4', '4v5']):
    """
    Returns a list of result dicts (one per condition).
    """
    results = []
    
    # Pre-calculate timing for all needed conditions to save API calls?
    # Actually timing.compute_game_timing takes a specific condition.
    
    # We need predictions (xG) in the DF if not present.
    # We need predictions (xG) in the DF if not present.
    if 'xg_probs' not in df_game.columns and 'xg' not in df_game.columns:
        # Try to predict using analyze module
        try:
            print(f"Predicting xG for game {game_id}...")
            # analyze._predict_xgs returns (df, ...), so capture it
            # Note: _predict_xgs might expect the full season df or just game df. 
            # It usually works on whatever is passed.
            df_game, _, _ = analyze._predict_xgs(df_game)
        except Exception as e:
            print(f"Failed to predict xG for game {game_id}: {e}")
            return [{'error': f'Prediction failed: {e}'}]

    # Ensure we have numeric xg
    if 'xg' not in df_game.columns:
        if 'xgs' in df_game.columns:
            df_game['xg'] = df_game['xgs']
        elif 'xg_probs' in df_game.columns:
            df_game['xg'] = df_game['xg_probs']
    
    if 'xg' not in df_game.columns:
        # Debug: available columns
        # print(f"DEBUG: Columns available: {df_game.columns.tolist()}")
        return [{'error': 'No xG data (xg, xgs, xg_probs missing)'}]

    for cond_name in conditions:
        timing_ev_count = 0
        timing_xg_sum = 0.0
        game_state_filter = [cond_name]
        
        # 1. Naive Filter
        # Note: 'game_state' in DF might be '5v5', '5v4', etc.
        # But '5v4' is strictly HOME powerplay? 
        # In the DF, 'game_state' usually reflects the actual number of skaters, e.g. '5v4'.
        # However, for special teams, '5v4' from perspective of Home is '5v4', but for Away it is '4v5'.
        # The 'game_state' column in the DF is usually absolute (Home v Away).
        # So '5v4' means Home has 5, Away has 4.
        
        # Determine exact filter for naive approach
        # cond_name used here is typically '5v5', '5v4' (Home adv), '4v5' (Away adv)
        
        if 'game_state' not in df_game.columns:
            results.append({'condition': cond_name, 'error': 'No game_state column'})
            continue
            
        if 'period_type' in df_game.columns:
            df_naive = df_game[
                (df_game['game_state'] == cond_name) & 
                (df_game['period_type'] != 'SHOOTOUT')
            ]
            mask = (df_game['period_type'] != 'SHOOTOUT')
        else:
            # Fallback: assume all rows are valid or check period <= 4 if needed
            # For 20252026, usually SO is period 5
            mask = pd.Series([True] * len(df_game), index=df_game.index) # Initialize mask to all True
        
        if 'period' in df_game.columns:
             # Exclude period 5 (SO) just in case
             mask &= (df_game['period'] <= 4)

        # Enhance Naive: Use relative game state to handle 5v4 vs 4v5 correctly from team perspective
        # 1. Infer Team (default to Home if not specified)
        target_team = None
        if 'home_abb' in df_game.columns:
            target_team = df_game['home_abb'].iloc[0]
        elif 'home_id' in df_game.columns:
            target_team = df_game['home_id'].iloc[0]
            
        if target_team:
            # Add relative state column
            df_game = timing.add_game_state_relative_column(df_game, target_team)
            if 'game_state_relative_to_team' in df_game.columns:
                mask &= (df_game['game_state_relative_to_team'] == cond_name)
            else:
                mask &= (df_game['game_state'] == cond_name)
        else:
             mask &= (df_game['game_state'] == cond_name)
             
        df_naive = df_game[mask]
        
        naive_ev_count = len(df_naive)
        naive_xg_sum = df_naive['xg'].sum()
        
        # 2. Timing Interval Filter
        # We use timing.compute_game_timing to get intervals.
        # To match daily.py and standard analysis, we should technically filter for is_net_empty=0
        # if we want "true" 5v5 (no pulled goalie).
        # Naive 5v5 filter in DF usually implies 5 skaters vs 5 skaters. 
        # But wait, DF 'game_state' '5v5' usually means 5 skaters each. It DOES NOT imply goalies.
        # However, usually we analyze 5v5 with goalies.
        # Let's align with daily.py: {'game_state': ['5v5'], 'is_net_empty': [0]}
        
        # User requested apples-to-apples comparison with Naive (which ignores net empty).
        # So we remove is_net_empty constraint here.
        t_cond = {
            'game_state': [cond_name]
        }
        if target_team:
            t_cond['team'] = target_team
        # We need to compute intervals. compute_game_timing returns a complex object
        # but we can also use timing.compute_intervals_for_game directly or use the one that applies it to DF?
        # analyze.xgs_map uses timing to filter DF. Let's replicate that logic or use a helper if available.
        # Actually timing.compute_game_timing DOES return a 'filtered_df' or similar?
        # Checking timing.py source (from memory/previous view):
        # It has compute_game_timing(df, condition).
        
        timing_res = timing.compute_game_timing(df_game, t_cond)
        
        # The timing module might not return the filtered DF directly in the 'res' dict?
        # Let's check what it returns.
        # Based on previous file view: 
        # It calculates intervals but might not filter the DF for us?
        # Wait, analyze.xgs_map calls xgs_map -> which calls timing.compute_game_timing...
        # and then likely does the filtering itself using those intervals.
        
        if 'is_net_empty' in df_game.columns:
            print(f"[DEBUG] is_net_empty counts: {df_game['is_net_empty'].value_counts().to_dict()}")
        
        # Enable verbose
        timing_res = timing.compute_game_timing(df_game, t_cond, verbose=True)
        # print(f"[DEBUG] timing_res keys: {list(timing_res.keys())}")
        
        # Correctly extract intervals from per_game struct
        intervals = []
        if 'per_game' in timing_res and game_id in timing_res['per_game']:
             game_data = timing_res['per_game'][game_id]
             intervals = game_data.get('intersection_intervals', [])
             
        if not intervals:
            print(f"[DEBUG] Intervals are EMPTY for cond={cond_name}!")
            timing_ev_count = 0
            timing_xg_sum = 0.0
        else:
            print(f"[DEBUG] Found {len(intervals)} intervals. First: {intervals[0]}")
            # Use analyze._apply_intervals to reproduce logic EXACTLY
            # from puck.analyze import _apply_intervals
            
            # Call xgs_map to get valid filtered DF
            # analyze.xgs_map(..., return_filtered_df=True) returns: (fig, heatmaps, filtered_df, stats)
            # We need to pass data_df=df_game to avoid re-loading.
            
            try:
                # We need to construct intervals_input correctly.
                # timing_res is {per_game: {gid: ...}} which is fine.
                
                # Note: xgs_map expects game_id as argument if filtering for single game?
                # Actually if data_df is passed, xgs_map uses it.
                # However, _apply_intervals uses 'per_game' dictionary keyed by game_id.
                # so passing intervals_input=timing_res is correct.
                
                # force_refresh=False, show=False

                _, _, df_timing_filtered, _ = analyze.xgs_map(
                    season=None, # use data_df
                    game_id=game_id, # critical for logging/flow
                    data_df=df_game,
                    intervals_input=timing_res,
                    condition=t_cond,  # FIX: Pass condition so validation logic can run!
                    return_filtered_df=True,
                    return_heatmaps=False, # We don't need heatmaps
                    show=False,
                    stats_only=True, # Minimal processing
                    use_intervals=True # Ensure it uses the input
                )
                
                print(f"[DEBUG] xgs_map returned {len(df_timing_filtered)} rows", flush=True)
                
                # Use df_game['xg'] for rigorous sum to ensure values match naive sum source (Apples-to-Apples)
                if 'xg' in df_game.columns:
                     timing_xg_sum = df_game.loc[df_timing_filtered.index, 'xg'].sum()
                else:
                     # Fallback if df_game mysteriously lost xg (unlikely)
                     timing_xg_sum = 0.0
                
                # --- DEEP DIVE DIAGNOSTICS ---
                # Compare df_naive and df_timing_filtered
                # We need a stable identifier. Using index might work if both came from df_game with original indices.
                # df_game was copied, so indices should be preserved from df_season? 
                # Actually indices might be reset or preserved. Let's check.
                
                naive_idx = set(df_naive.index)
                timing_idx = set(df_timing_filtered.index)
                
                naive_only = naive_idx - timing_idx
                timing_only = timing_idx - naive_idx
                
                print(f"\n--- Discrepancy Analysis ({cond_name}) ---")
                print(f"Naive Count: {len(naive_idx)} | Rigorous Count: {len(timing_idx)}")
                print(f"Naive Only (False Positives?): {len(naive_only)}")
                print(f"Rigorous Only (False Negatives?): {len(timing_only)}")
                
                # Debug Value Mismatch if sets are identical but sums differ
                if not naive_only and not timing_only and abs(naive_xg_sum - timing_xg_sum) > 0.01:
                    print("\n[DEBUG VALUE MISMATCH] Sets identical but xG sums differ!")
                    print(f"Naive Sum: {naive_xg_sum:.4f} | Timing Sum: {timing_xg_sum:.4f}")
                    # Compare item by item
                    shared_idx = list(naive_idx)
                    df_n_vals = df_naive.loc[shared_idx, 'xg']
                    df_t_vals = df_timing_filtered.loc[shared_idx, 'xg']
                    
                    diffs = (df_n_vals - df_t_vals).abs()
                    mismatches = diffs[diffs > 0.0001]
                    if not mismatches.empty:
                        print(f"Found {len(mismatches)} rows with different xG values.")
                        print(f"Sample Mismatches (Naive vs Timing):\n")
                        for idx in mismatches.index[:5]:
                            print(f"Idx {idx}: Naive={df_n_vals[idx]:.4f} vs Timing={df_t_vals[idx]:.4f} | Event={df_naive.loc[idx, 'event']}")
                    else:
                        print("No per-row mismatch found? Summing difference must be weird floating point issue.")
                
                if naive_only:
                    print("\n[Sample Naive-Only Events] (Method says YES, Rigorous says NO)")
                    sample_idxs = list(naive_only)[:5]
                    
                    df_sample = df_game.loc[sample_idxs].copy()
                    # Ensure total_seconds... (same as before)
                    if 'total_seconds' not in df_sample.columns:
                        if 'period' in df_sample.columns and 'period_seconds' in df_sample.columns:
                            df_sample['total_seconds'] = (df_sample['period'] - 1) * 1200 + df_sample['period_seconds']
                        elif 'total_time_elapsed_seconds' in df_sample.columns:
                             df_sample['total_seconds'] = df_sample['total_time_elapsed_seconds']
                    
                    cols = ['period', 'period_time', 'game_state', 'total_seconds', 'event', 'xg', 'player_name']
                    print_cols = [c for c in cols if c in df_sample.columns]
                    print(df_sample[print_cols])
                    
                    print(f"Naive-Only Event Types:\n{df_game.loc[list(naive_only)]['event'].value_counts()}")
                    
                    # Show relevant intervals...
                    if not df_sample.empty and 'total_seconds' in df_sample.columns:
                        first_sec = df_sample.iloc[0]['total_seconds']
                        print(f"Event Time: {first_sec}")
                        sorted_ints = sorted(intervals)
                        found_nearby = False
                        for s, e in sorted_ints:
                            if s - 60 <= first_sec <= e + 60:
                                print(f"  Nearby Interval: {s} -> {e}")
                                found_nearby = True
                        if not found_nearby and sorted_ints:
                            print(f"  No nearby intervals. Closest starts at {min(sorted_ints, key=lambda i: abs(i[0]-first_sec))}")

                if timing_only:
                     print("\n[Sample Rigorous-Only Events] (Method says NO, Rigorous says YES)")
                     sample_idxs = list(timing_only)[:5]
                     
                     df_sample = df_game.loc[sample_idxs].copy()
                     if 'total_seconds' not in df_sample.columns:
                        if 'period' in df_sample.columns and 'period_seconds' in df_sample.columns:
                            df_sample['total_seconds'] = (df_sample['period'] - 1) * 1200 + df_sample['period_seconds']
                        elif 'total_time_elapsed_seconds' in df_sample.columns:
                             df_sample['total_seconds'] = df_sample['total_time_elapsed_seconds']
                     
                     cols = ['period', 'period_time', 'game_state', 'total_seconds', 'event', 'xg', 'player_name']
                     print_cols = [c for c in cols if c in df_sample.columns]
                     print(df_sample[print_cols])
                     
                     print(f"Rigorous-Only Event Types:\n{df_game.loc[list(timing_only)]['event'].value_counts()}")
                     
                     if not df_sample.empty and 'total_seconds' in df_sample.columns:
                        first_sec = df_sample.iloc[0]['total_seconds']
                        print(f"Rigorous Event Time: {first_sec}")
                        na_state = df_sample.iloc[0]['game_state']
                        print(f"Naive Game State: {na_state}")

                # --- EXPORT CLASSIFICATION ---
                # Label events in the main df_game for this condition
                col_name = f'status_{cond_name}'
                df_game.loc[list(naive_idx & timing_idx), col_name] = 'Both'
                df_game.loc[list(naive_idx - timing_idx), col_name] = 'Naive Only'
                df_game.loc[list(timing_idx - naive_idx), col_name] = 'Rigorous Only'
                # 'Neither' is default (NaN) or we can fill it
                
            except Exception as e:
                print(f"[ERROR] _apply_intervals/Analysis failed: {e}")
                timing_ev_count = 0
                timing_xg_sum = 0.0
        
        # Save Debug CSV
        out_csv = f'analysis/time_filtering_debug_{game_id}.csv'
        df_game.to_csv(out_csv, index=False)
        print(f"\n[SAVED] Game dataframe with classification columns to {out_csv}")

        results.append({
            'condition': cond_name,
            'naive_count': naive_ev_count,
            'naive_xg': naive_xg_sum,
            'timing_count': timing_ev_count,
            'timing_xg': timing_xg_sum,
            'diff_xg_abs': abs(naive_xg_sum - timing_xg_sum),
            'diff_xg_pct': 100 * abs(naive_xg_sum - timing_xg_sum) / naive_xg_sum if naive_xg_sum > 0 else 0
        })
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', default='20252026')
    parser.add_argument('--games', type=int, default=10)
    args = parser.parse_args()
    
    print(f"Validating Time Filtering for Season {args.season} (Analyzing {args.games} games)...")
    
    gids, df_season = get_game_ids(args.season, args.games)
    print(f"Loaded {len(gids)} games.")
    
    all_results = []
    
    for i, gid in enumerate(gids):
        print(f"Processing Game {gid} ({i+1}/{len(gids)})...")
        # Extract single game DF
        df_game = df_season[df_season['game_id'] == gid].copy().copy()
        
        # Ensure necessary columns
        # ...
        
        res = validate_game(gid, df_game)
        for r in res:
            r['game_id'] = gid
            all_results.append(r)
            
    # Aggregate results
    df_res = pd.DataFrame(all_results)
    if df_res.empty:
        print("No results generated.")
        return

    # Filter out errors
    if 'error' in df_res.columns:
        errors = df_res[df_res['error'].notna()]
        if not errors.empty:
            print(f"Encountered {len(errors)} errors:")
            print(errors[['game_id', 'condition', 'error'] if 'condition' in errors.columns else ['game_id', 'error']])
            df_res = df_res[df_res['error'].isna()]

    if df_res.empty:
        print("No valid results after filtering errors.")
        return

    print("\n--- Validation Summary ---")
    grp = df_res.groupby('condition').agg({
        'naive_xg': 'sum',
        'timing_xg': 'sum',
        'diff_xg_abs': 'mean'
    })
    grp['diff_pct_total'] = 100 * abs(grp['naive_xg'] - grp['timing_xg']) / grp['naive_xg']
    print(grp)
    
    # Check for big discrepancies
    print("\n--- Outliers (Diff > 5% in single game) ---")
    outliers = df_res[df_res['diff_xg_pct'] > 5.0]
    if not outliers.empty:
        print(outliers[['game_id', 'condition', 'naive_xg', 'timing_xg', 'diff_xg_pct']])
    else:
        print("No significant outliers found.")

if __name__ == '__main__':
    main()
