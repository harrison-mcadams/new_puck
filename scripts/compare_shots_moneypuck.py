
import pandas as pd
import numpy as np
import os
import sys
import requests
import io
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config, timing, analyze

def main():
    season = "20252026"
    mp_year = "2025" # Start year for MoneyPuck file
    
    # 1. Load Local Data & Generate Predictions
    print(f"Loading local data for {season}...")
    df_local = timing.load_season_df(season)
    
    if df_local is None or df_local.empty:
        print("Error: No local season data found.")
        return

    print("Generating local xG predictions...")
    # Explicitly use the new XGBoost model
    model_path = os.path.join(config.ANALYSIS_DIR, 'xgs', 'xg_model_nested.joblib')
    
    # Preprocess exactly like training/verification
    from puck import fit_xgboost_nested, impute
    df_local = fit_xgboost_nested.preprocess_data(df_local)
    df_local = impute.impute_blocked_shot_origins(df_local, method='point_pull')
    
    df_local, _, _ = analyze._predict_xgs(df_local, model_path=model_path, behavior='overwrite')
    
    # Filter for shots only
    df_local_shots = df_local[df_local['event'].isin(['shot-on-goal', 'goal', 'missed-shot'])].copy()
    print(f"Local Shots (All): {len(df_local_shots)}")
    
    # Normalize GameID
    # Local: 2025020001 (Int)
    
    # 2. Download MoneyPuck Shot Data
    mp_url = f"http://peter-tanner.com/moneypuck/downloads/shots_{mp_year}.zip"
    print(f"Downloading MoneyPuck shots from {mp_url}...")
    
    try:
        r = requests.get(mp_url)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        csv_name = z.namelist()[0] # Assume first file
        print(f"Extracting {csv_name}...")
        df_mp = pd.read_csv(z.open(csv_name))
    except Exception as e:
        print(f"Failed to download MoneyPuck data: {e}")
        return

    print(f"Loaded MoneyPuck Shots: {len(df_mp)}")
    
    # 3. Pre-process MoneyPuck for Merge
    # MP Columns: shotID, game_id, team, xGoal, ...
    # MP game_id is usually 2025020001
    
    # Filter MP for Regular Season only if needed (usually contained in file)
    # Check if 'game_id' exists
    if 'game_id' not in df_mp.columns:
        print("Error: 'game_id' not in MoneyPuck data.")
        print(df_mp.columns)
        return
        
    # MP uses 'team' as abbreviation (e.g. 'PHI'). Local uses 'team_id' usually, but might have 'team_abb' joined.
    # We loaded via timing.load_season_df, usually it sends raw.
    # We should add team abbr to local if missing.
    
    # Just match on GameID + Time + Event Type?
    # MP Time is 'time' (seconds).
    # Local Time is 'period_seconds' or 'game_seconds'?
    # Local: 'period' and 'period_seconds' (time elapsed in period).
    # MP: 'time'? Let's check MP columns if possible, but usually 'time' is seconds from start of GAME or PERIOD?
    # MoneyPuck 'time' is usually seconds elapsed in the GAME.
    # Local 'period_time' is string MM:SS.
    
    # Let's use 'game_seconds' if available in local, or calculate it.
    # period_seconds is usually elapsed in period.
    # game_seconds = (period - 1) * 1200 + period_seconds
    
    # Calculate game_seconds
    if 'total_time_elapsed_s' in df_local_shots.columns:
        df_local_shots['game_seconds_calc'] = df_local_shots['total_time_elapsed_s']
    elif 'time_elapsed_in_period_s' in df_local_shots.columns:
        df_local_shots['game_seconds_calc'] = (df_local_shots['period'] - 1) * 1200 + df_local_shots['time_elapsed_in_period_s']
    elif 'period_seconds' in df_local_shots.columns:
        df_local_shots['game_seconds_calc'] = (df_local_shots['period'] - 1) * 1200 + df_local_shots['period_seconds']
    else:
        # Fallback parsing period_time if needed, but total_time_elapsed_s should exist from parse.
        print("Warning: Could not find time column. Columns:", df_local_shots.columns)
        return

    # MoneyPuck 'time' column verification
    # Usually 'time' column in MP is seconds.
    
    # Match Keys
    # We will use: game_id, period, absolute_diff(time) < 3.0
    
    # Create simplified DFs for merging
    # Local Features
    cols_local = ['game_id', 'period', 'game_seconds_calc', 'team_id', 'event', 'xgs', 'player_id', 
                  'x', 'y', 'distance', 'angle_deg', 'is_net_empty', 'is_rebound', 'shot_type']
    
    # Ensure columns exist
    for c in cols_local:
        if c not in df_local_shots.columns:
            df_local_shots[c] = np.nan # or similar
            
    df_local_mini = df_local_shots[cols_local].copy()
    df_local_mini['id_local'] = df_local_mini.index
    
    # MP Features
    # MP Columns: shotDistance, xCord, yCord, shotAngle, shotOnEmptyNet
    # map to local names for clarity in merge but keep separate
    cols_mp = ['game_id', 'period', 'time', 'team', 'xGoal', 'shotID', 'shooterName', 'shotType',
               'shotDistance', 'shotAngle', 'xCord', 'yCord', 'shotOnEmptyNet', 'shotRebound', 'shotRush']
               
    df_mp_mini = df_mp[cols_mp].copy()
    # Handle simple rename or just use it
    if 'shotOnEmptyNet' in df_mp_mini.columns:
         df_mp_mini.rename(columns={'shotOnEmptyNet': 'emptyNet'}, inplace=True)
    elif 'emptyNet' not in df_mp_mini.columns:
        df_mp_mini['emptyNet'] = 0
    
    # Normalize GameID (Local % 1000000)
    df_local_mini['game_id'] = df_local_mini['game_id'].astype(int) % 1000000
    df_mp_mini['game_id'] = df_mp_mini['game_id'].astype(int)
    
    # Enforce Types for Keys
    df_local_mini['period'] = df_local_mini['period'].astype(int)
    df_mp_mini['period'] = df_mp_mini['period'].astype(int)
    
    df_local_mini['sec'] = df_local_mini['game_seconds_calc'].astype(float)
    df_mp_mini['sec'] = df_mp_mini['time'].astype(float)
    
    # Sort for merge_asof
    df_local_mini = df_local_mini.sort_values('sec')
    df_mp_mini = df_mp_mini.sort_values('sec')
    
    print("Merging data...")
    merged = pd.merge_asof(
        df_local_mini, 
        df_mp_mini, 
        on='sec', 
        by=['game_id', 'period'],
        direction='nearest',
        tolerance=3.0,
        suffixes=('_loc', '_mp')
    )
    
    matched = merged.dropna(subset=['shotID'])
    print(f"Matched {len(matched)} total shots.")
    
    # --- FILTERING ---
    # 1. Start with Empty Net Filter (User Request)
    # Filter if Local says empty OR MP says empty
    mask_empty = (matched['is_net_empty'] == 1) | (matched['emptyNet'] == 1)
    
    # Filter out empty nets
    matched_clean = matched[~mask_empty].copy()
    print(f"Removed {mask_empty.sum()} empty net events. Remaining: {len(matched_clean)}")
    
    # 2. Filter out explicit 0 xG from local if exists (User said model doesn't know, so maybe it predicts small non-zero?)
    
    matched = matched_clean
    
    matched['diff'] = matched['xgs'] - matched['xGoal']
    matched['abs_diff'] = matched['diff'].abs()
    
    # --- DEEP DIVE ANALYSIS ---
    corr = matched['xgs'].corr(matched['xGoal'])
    print(f"\nCorrelation (Non-Empty Net): {corr:.4f}")
    
    # Check Feature Agreements
    # Distance
    # MP likely uses feet. Local uses feet.
    matched['dist_diff'] = matched['distance'] - matched['shotDistance']
    
    print("\n--- Feature Agreements ---")
    print(f"Distance Correlation: {matched['distance'].corr(matched['shotDistance']):.4f}")
    print(f"Mean Distance Diff (Loc - MP): {matched['dist_diff'].mean():.2f} ft")
    print(f"MAE Distance Diff: {matched['dist_diff'].abs().mean():.2f} ft")
    
    # Discrepancy Buckets
    # 1. High xG Diff, Low feature Diff (Model Disagreement)
    # 2. High xG Diff, High feature Diff (Data Disagreement)
    
    threshold_xg = 0.20
    threshold_dist = 5.0 # ft
    
    mask_high_diff = matched['abs_diff'] > threshold_xg
    mask_bad_data = (matched['dist_diff'].abs() > threshold_dist)
    
    n_total = len(matched)
    n_high_diff = mask_high_diff.sum()
    n_bad_data = (mask_high_diff & mask_bad_data).sum()
    n_model_disagreement = (mask_high_diff & ~mask_bad_data).sum()
    
    print(f"\n--- Discrepancy Breakdown (Diff > {threshold_xg}) ---")
    print(f"Total Significant Discrepancies: {n_high_diff} ({n_high_diff/n_total*100:.1f}%)")
    print(f"  > Due to Location/Data Mismatch (> {threshold_dist}ft): {n_bad_data} ({n_bad_data/n_high_diff*100:.1f}%)")
    print(f"  > Due to Model Opinion (Loc Match):    {n_model_disagreement} ({n_model_disagreement/n_high_diff*100:.1f}%)")
    
    # Examples: Model Opinion
    print("Merged Columns:", matched.columns.tolist())
    
    # Resolve names
    c_shooter = 'shooterName' if 'shooterName' in matched.columns else 'shooterName_mp'
    c_st_mp = 'shotType_mp' if 'shotType_mp' in matched.columns else 'shotType'
    c_st_loc = 'shot_type_loc' if 'shot_type_loc' in matched.columns else 'shot_type'
    if 'shot_type' in matched.columns and c_st_loc == 'shot_type': pass 
    
    cols_show = ['game_id', 'period', 'sec', c_shooter, 'event', c_st_mp, c_st_loc, 
                 'xgs', 'xGoal', 'diff', 
                 'distance', 'shotDistance', 'dist_diff']
                 
    print("\n--- Top Model Disagreements (Data Agrees, Models Differ) ---")
    df_model_dis = matched[mask_high_diff & ~mask_bad_data].sort_values('abs_diff', ascending=False)
    print(df_model_dis.head(10)[cols_show].to_string(index=False))
    
    print("\n--- Top Data Disagreements (Location Mismatch) ---")
    df_data_dis = matched[mask_high_diff & mask_bad_data].sort_values('abs_diff', ascending=False)
    print(df_data_dis.head(10)[cols_show].to_string(index=False))
    
    # Save
    out_csv = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    matched.to_csv(out_csv, index=False)
    print(f"\nSaved deep dive data to {out_csv}")
    
    # Plotting
    # Scatter colored by Distance Diff?
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(matched['xGoal'], matched['xgs'], c=matched['dist_diff'].abs(), cmap='viridis', s=10, alpha=0.5, vmin=0, vmax=20)
    plt.colorbar(sc, label='Abs Distance Diff (ft)')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlabel("MoneyPuck xG")
    plt.ylabel("My Model xG")
    plt.title(f"xG Comparison (Non-Empty Net) | Color = Data Mismatch\nCorr={corr:.3f}")
    plt.savefig(os.path.join(config.ANALYSIS_DIR, 'comparison_deep_dive.png'))
    print("Saved plot.")
    plt.savefig(os.path.join(config.ANALYSIS_DIR, 'shot_comparison_scatter.png'))
    print("Saved scatter plot.")

if __name__ == "__main__":
    main()
