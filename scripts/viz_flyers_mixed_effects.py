
"""
scripts/viz_flyers_mixed_effects.py

Generates a Mixed Effects xG Heatmap for Flyers 5v5, matching daily.py style.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config
from puck import analyze
from puck import data_pipeline
from puck import timing
from puck import plot
from puck.rink import draw_rink

def main():
    season = "20252026"
    target_team = "PHI"
    condition = "5v5"
    
    print(f"--- Generating {target_team} {condition} Mixed Effects Heatmap ---")
    
    # 1. Load Data
    data_path = "data/20252026.csv"
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Enrichment Logic (Copied from update_mixed_effects.py)
    print("Enriching Team Abbreviations...")
    try:
        from puck import nhl_api
        games = nhl_api.get_season('PHI', season=season) # Fetch schedule to get team mapping
        
        id_map = {}
        for g in games:
            if 'awayTeam' in g:
                t = g['awayTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
                    
            if 'homeTeam' in g:
                t = g['homeTeam']
                if 'id' in t and 'abbrev' in t:
                    id_map[t['id']] = t['abbrev']
                    
        print(f"Built ID Map for {len(id_map)} teams.")
        
        df['team_id'] = pd.to_numeric(df['team_id'], errors='coerce')
        df['home_id'] = pd.to_numeric(df['home_id'], errors='coerce')
        df['away_id'] = pd.to_numeric(df['away_id'], errors='coerce')
        
        df['team_abbrev'] = df['team_id'].map(id_map)
        df['home_abb'] = df['home_id'].map(id_map)
        df['away_abb'] = df['away_id'].map(id_map)
        
        df['team_abbrev'] = df['team_abbrev'].fillna('Unknown')
        df['home_abb'] = df['home_abb'].fillna('Unknown')
        df['away_abb'] = df['away_abb'].fillna('Unknown')
        
    except Exception as e:
        print(f"Warning: Failed to enrich team abbrevs: {e}")
        
    # 2. Preprocess Features (Standard Pipeline)
    print("Preprocessing features...")
    df = data_pipeline.preprocess_features(
        df,
        is_training=False,
        apply_imputation=True,
        apply_arena_adjustments=True,
        apply_bio_enrichment=True,
        apply_filtering=True 
    )
    
    # Ensure is_goal exists
    if 'is_goal' not in df.columns:
        # Standardize event names first? Preprocessing does it?
        # Preprocessing standardizes 'event' column.
        # But let's be safe.
        df['is_goal'] = df['event'].apply(lambda x: 1 if str(x).lower() == 'goal' else 0)
    
    # 3. Load Mixed Effects Model
    model_path = "analysis/xgs/joint_mixed_effects.joblib"
    print(f"Loading Mixed Effects Model from {model_path}...")
    mixed = joblib.load(model_path)
    
    # 4. Filter for 5v5
    print("Filtering for 5v5...")
    # Using 'game_state' column (daily.py uses '5v5')
    # Preprocessing doesn't necessarily enforce strict '5v5' string unless mapped?
    # Let's inspect unique game states if possible or assume loose match?
    # daily.py uses '5v5' explicitly.
    df_5v5 = df[df['game_state'] == '5v5'].copy()
    
    # 5. Predict xG (Mixed Effects)
    print("Predicting xG using Mixed Effects...")
    # mix_effects.predict_proba adds 'xgs'
    # returns array of probas.
    probs = mixed.predict_proba(df_5v5)
    
    # DEBUG: Analyze Base Model vs Final
    print("--- DEBUG DIAGNOSITCS ---")
    try:
        base_probs = mixed.base_model_.predict_proba(df_5v5)[:, 1]
        print(f"Base Model Mean xG: {base_probs.mean():.4f}")
        print(f"Base Model min/max: {base_probs.min():.4f} / {base_probs.max():.4f}")
        
        # Check input features to base model
        if hasattr(mixed.base_model_, 'feature_names_in_'):
            booster_feats = mixed.base_model_.feature_names_in_
            missing = [f for f in booster_feats if f not in df_5v5.columns]
            print(f"Missing Base Features: {missing}")
            # print(f"Base Feats: {booster_feats}")
    except Exception as e:
        print(f"Failed to inspect base model: {e}")
        
    print(f"Final Model Mean xG: {probs[:, 1].mean() if probs.ndim==2 else probs.mean():.4f}")
    
    # DEBUG: Inspect 5v5 Model Internals
    try:
        if '5v5' in mixed.models_:
            m5v5 = mixed.models_['5v5']
            # Check Teams
            print(f"Model Teams (Sample): {m5v5.teams_[:5]}")
            if target_team in m5v5.teams_:
                print(f"Target Team '{target_team}' FOUND in model teams.")
            else:
                print(f"Target Team '{target_team}' NOT FOUND in model teams.")
                
            # Check Bias
            if m5v5.booster_:
                dump = m5v5.booster_.get_dump(dump_format='json')
                if dump:
                    import json
                    w = json.loads(dump[0])
                    print(f"Model Bias: {w.get('bias', 'N/A')}")
                    
                    # Check PHI Weights
                    # Find index
                    if target_team in m5v5.teams_:
                        idx = np.where(m5v5.teams_ == target_team)[0][0]
                        # 49 features per team?
                        n_feat = len(m5v5.feature_names)
                        start = idx * n_feat
                        w_arr = np.array(w['weight'])
                        phi_w = w_arr[start:start+n_feat]
                        print(f"PHI Weights Mean: {phi_w.mean():.4f}, Min: {phi_w.min():.4f}, Max: {phi_w.max():.4f}")
        else:
            print("No 5v5 model found!")
    except Exception as e:
        print(f"Failed to inspect 5v5 model: {e}")

    if probs.ndim == 2:
        df_5v5['xgs'] = probs[:, 1]
    else:
        df_5v5['xgs'] = probs
        
    print(f"Prediction complete. Mean xG: {df_5v5['xgs'].mean():.4f}")
    
    # 6. Calculate Team TOI for 5v5
    print(f"Calculating TOI for {target_team} 5v5...")
    # We need to find all games involving PHI
    # Then sum 5v5 TOI.
    # Use puck.timing logic? Or just sum observed time from data?
    # Accessing timing cache is best.
    # Cache: data/cache/{season}/partials/game_{game_id}_5v5.npz usually contains team_stats
    
    # Quick Scan of Cache
    cache_dir = f"data/cache/{season}/partials"
    files = [f for f in os.listdir(cache_dir) if f.endswith("_5v5.npz")]
    
    team_seconds = 0.0
    team_goals = 0
    other_goals = 0
    team_xgs = 0.0
    other_xgs = 0.0
    team_attempts = 0
    other_attempts = 0
    
    import json
    
    # Identify PHI ID
    # Usually in analysis/teams.json
    teams_json_path = "analysis/teams.json"
    phi_id = None
    if os.path.exists(teams_json_path):
        with open(teams_json_path, 'r') as f:
            tdata = json.load(f)
            for t in tdata:
                if t['abbr'] == target_team:
                    phi_id = int(t['id'])
                    break
    
    if phi_id is None:
        # Fallback: Inference from DF
        try:
            phi_id = int(df_5v5[df_5v5['team_abbrev'] == target_team]['team_id'].iloc[0])
        except:
            print(f"Could not ID {target_team}")
            return
            
    print(f"{target_team} ID: {phi_id}")
    
    # Accumulate Stats from Cache (Reliable TOI)
    # Note: We must also accumulate xG from OUR dataframe, NOT the cache (which has old xG).
    # TOI comes from cache. Goals/Attempts/xG comes from our DF.
    
    # Wait, we need to normalize the Grid by TOI.
    # So we strictly need correct TOI.
    
    for f in files:
        try:
            data = np.load(os.path.join(cache_dir, f), allow_pickle=True)
            k_stat = f"team_{phi_id}_stats"
            if k_stat in data:
                 if data[k_stat].dtype.kind in {'U', 'S'}:
                     s = json.loads(str(data[k_stat].item()))
                 else:
                     s = json.loads(str(data[k_stat]))
                 
                 team_seconds += s.get('team_seconds', 0.0)
        except:
            pass
            
    print(f"Total TOI (Seconds): {team_seconds}")
    
    # Accumulate Events from DF (using New xG)
    # FOR (Offense)
    mask_for = (df_5v5['team_id'] == phi_id) & (df_5v5['event'].isin(['Goal','Shot','Missed Shot','Blocked Shot','goal','shot','missed-shot','blocked-shot']))
    df_for = df_5v5[mask_for]
    
    # AGAINST (Defense) - Shots by Opponent when PHI is Home or Away
    # Opponent shots: team_id != phi_id AND (home_id == phi_id OR away_id == phi_id)
    mask_against = (df_5v5['team_id'] != phi_id) & ((df_5v5['home_id'] == phi_id) | (df_5v5['away_id'] == phi_id)) & (df_5v5['event'].isin(['Goal','Shot','Missed Shot','Blocked Shot','goal','shot','missed-shot','blocked-shot']))
    df_against = df_5v5[mask_against]
    
    team_xgs = df_for['xgs'].sum()
    other_xgs = df_against['xgs'].sum()
    team_goals = df_for['is_goal'].sum()
    other_goals = df_against['is_goal'].sum()
    team_attempts = len(df_for)
    other_attempts = len(df_against)
    
    # 7. Construct Full Rink Grids
    print("Constructing Grids...")
    # Grid Logic: analyze.compute_xg_heatmap_from_df
    # Grid Res: 0.5? daily.py uses analyze.bin_events which defaults?
    # run_league_stats uses analyze.compute_xg_heatmap_from_df implicitly? No, it uses cached grids.
    # Cached grids are generated by process_daily_cache using analyze.bin_events_to_grid?
    # Let's use analyze.compute_xg_heatmap_from_df directly.
    
    # We need to ensure grid dimensions match Baseline (200, 85 usually? or 100, 85?)
    # Baseline shape in run_league_stats is (85, 200)? daily.py saves 85, 200 via bin_events?
    # Range: x: -100 to 100, y: -42.5 to 42.5. Res=1.0?
    # Let's check Baseline shape.
    
    baseline_path = f"analysis/league/{season}/5v5/20252026_league_baseline.npy"
    if not os.path.exists(baseline_path):
        # try short name
        baseline_path = f"analysis/league/{season}/5v5/baseline.npy"
        
    baseline_grid = np.load(baseline_path)
    print(f"Baseline Shape: {baseline_grid.shape}")
    
    # Assume 85 rows, 200 cols?
    # x ranges -100 to 100. (200 units). y ranges -42.5 to 42.5 (85 units).
    # So Resolution is 1.0.
    
    # Compute FOR Grid
    # x_col='x_abs', y_col='y_abs' ? No, data has 'x', 'y' (oriented).
    # We want standard orientation?
    # data_pipeline orients everything to attacking right (x > 0)?
    # analyze.compute... handles it.
    
    # Note: run_league_stats uses:
    # grid_for + rot90(grid_against, 2)
    # This implies grid_for is "Attacking Right".
    # And Baseline is "Attacking Right" (on Right Side) + "Defending Left" (on Left Side).
    
    # compute_xg_heatmap_from_df returns (gx, gy, heatmap, xg_sum, seconds)
    # We just want the heatmap (sum of xgs).
    # Normalized? No, we want Sum.
    
    # For Grid
    _, _, grid_for, _, _ = analyze.compute_xg_heatmap_from_df(
        df_for, 
        grid_res=1.0, 
        sigma=2.0, # Smoothing? daily.py uses 2.0?
        x_col='x', 
        y_col='y', 
        amp_col='xgs',
        normalize_per60=False 
    )
    
    # Against Grid
    _, _, grid_against_raw, _, _ = analyze.compute_xg_heatmap_from_df(
        df_against, 
        grid_res=1.0, 
        sigma=2.0, 
        x_col='x', 
        y_col='y', 
        amp_col='xgs',
        normalize_per60=False 
    )
    
    # Rotate Against Grid (180 deg)
    grid_against = np.rot90(grid_against_raw, 2)
    
    # Combine
    team_grid = grid_for + grid_against
    
    # Verify shape
    if team_grid.shape != baseline_grid.shape:
        print(f"Shape Mismatch! Team: {team_grid.shape}, Baseline: {baseline_grid.shape}")
        # Identify padding/trimming needs
        # Usually bin_events handles ranges explicitly.
        # If shape mismatch, force reshape?
        # Re-run binning with explicit range if needed.
        pass
        
    # 8. Compute Relative Grid
    print("Computing Relative Map...")
    # team_norm = Sum / Seconds
    # rel = (team_norm - league_norm) * 3600 * 100
    
    # If baseline is already Rate per 60?
    # run_league_stats.py:
    # np.save(..., league_norm_grid * 3600.0 (Wait? No))
    # run_league_stats:
    #   league_norm_grid = league_sum / total_seconds (Rate per sec)
    #   np.save('baseline.npy', baseline_left * 3600.0)
    #   rel_grid = (team_norm - league_norm) * 3600 * 100.0
    
    # BUT if we load 'baseline.npy', it might be Scaled?
    # Let's inspect values of baseline.npy. If roughly 0-1, it's xG/60?
    # If it's xG/sec (1e-4), it's unscaled.
    # run_league_stats saves * 3600.
    
    # However, for REL calculation, run_league_stats uses `league_norm_grid` variable (unscaled rate per sec).
    # If we load from disk `baseline.npy`, we are loading the SCALED version (Per 60).
    # So we should compare:
    # Rel = (Team_Per_60 - Baseline_Per_60) * 100
    
    team_norm_grid = team_grid / team_seconds # xG per sec
    team_per_60 = team_norm_grid * 3600.0
    
    # Load Baseline (Assumed Per 60, but check filename/logic)
    # run_league_stats: np.save(..., baseline_left * 3600.0)
    # So stored baseline IS Per 60.
    
    # But Baseline Left zeroes out right. Baseline Right zeroes out left.
    # We want Full Baseline?
    # daily.py creates `20252026_league_baseline.npy` (Left only?)
    # Lines 416: `np.save(..., baseline_left * 3600.0)`
    
    # If standard baseline is only left side, we have a problem for the Right Side (Offense).
    # Check if `league_baseline.npy` is full or partial?
    # Code says: `baseline_left[:, mid:] = 0.0`
    
    # Does daily.py save a Full Baseline?
    # Line 409: `league_norm_grid` is full.
    # But it saves `baseline_left` to `baseline.npy`?
    # That implies `plot_relative` in daily might handle mirroring?
    # Or maybe I should reconstruct League Average from scratch or look for a full file?
    
    # Actually, `run_league_stats` iterates teams and uses `league_norm_grid` (in memory, full).
    # It does NOT load from disk for team plots.
    # So `baseline.npy` on disk might be insufficient (only Left?).
    
    # WORKAROUND:
    # Reconstruct Full Baseline from Cached Partials?
    # Or use `baseline.npy` (Left) + `baseline_right.npy` (Right) if it exists?
    # Line 424 saves `_right.npy`.
    
    baseline_right_path = f"analysis/league/{season}/5v5/20252026_league_baseline_right.npy"
    
    if os.path.exists(baseline_right_path):
        baseline_r = np.load(baseline_right_path)
        baseline_l = np.load(baseline_path)
        baseline_full = baseline_l + baseline_r
    else:
        # Assume symmetric?
        # Defense (Left) should be roughly equal to Offense (Right, rotated)?
        # For xG, yes.
        print("Warning: Only Left Baseline found. assuming symmetry/mirroring.")
        baseline_l = np.load(baseline_path)
        # Flip Left to Right
        baseline_r = np.rot90(np.rot90(baseline_l, 2)) # 180 (Wait, 180 of Left is Right)
        baseline_full = baseline_l + baseline_r
        
    rel_grid = (team_per_60 - baseline_full) * 100.0 # Per 100 sq ft scaling
    
    # Load League Average (if possible) for Relative Stats
    print("Loading League Summary for Baseline Stats...")
    league_sum_path = f"analysis/league/{season}/5v5/20252026_team_summary.csv"
    if not os.path.exists(league_sum_path):
        league_sum_path = f"analysis/league/{season}/5v5/team_summary.csv"
        
    league_xgf60_mean = 2.5 # Fallback
    league_xga60_mean = 2.5
    
    if os.path.exists(league_sum_path):
        try:
            df_sum = pd.read_csv(league_sum_path)
            # Filter valid (non-zero) or just mean?
            # data_pipeline usually filters < 1000 sec?
            # Just take mean.
            league_xgf60_mean = df_sum['team_xg_per60'].mean()
            league_xga60_mean = df_sum['other_xg_per60'].mean()
            print(f"League xGF/60 Baseline: {league_xgf60_mean:.3f}")
        except Exception as e:
            print(f"Failed to load league summary: {e}")
            
    # Calculate Stats
    xgf60 = team_xgs / team_seconds * 3600
    xga60 = other_xgs / team_seconds * 3600
    
    rel_off_pct = ((xgf60 - league_xgf60_mean) / league_xgf60_mean) * 100 if league_xgf60_mean > 0 else 0.0
    rel_def_pct = ((xga60 - league_xga60_mean) / league_xga60_mean) * 100 if league_xga60_mean > 0 else 0.0
    
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats = {
        'team_name': target_team,
        'home_goals': team_goals,
        'away_goals': other_goals,
        'home_xg': team_xgs,
        'away_xg': other_xgs,
        'home_attempts': team_attempts,
        'away_attempts': other_attempts,
        'team_xg_per60': xgf60,
        'other_xg_per60': xga60,
        'team_seconds': team_seconds,
        'game_ongoing': False,
        'have_xg': True,
        'rel_off_pct': rel_off_pct,
        'rel_def_pct': rel_def_pct,
        'off_percentile': 50, # Rough
        'def_percentile': 50,
        'home_shot_pct': team_attempts/(team_attempts+other_attempts)*100 if (team_attempts+other_attempts)>0 else 0,
        'away_shot_pct': other_attempts/(team_attempts+other_attempts)*100 if (team_attempts+other_attempts)>0 else 0
    }
    
    plot.plot_relative_map(
        ax=ax,
        rel_grid=rel_grid,
        title=f"{target_team} {condition} Mixed Effects (Relative)",
        stats=stats,
        team_name=target_team,
        full_team_name=target_team,
        cond=condition,
        mask_neutral_zone=True,
        vmax=None # Auto-scale
    )
    
    out_path = f"analysis/xgs/mixed_effects/{target_team}_{condition}_heatmap.png"
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    print(f"Saved heatmap to {out_path}")

if __name__ == "__main__":
    main()
