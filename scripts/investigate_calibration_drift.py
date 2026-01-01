
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve

sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_nested, fit_xgs, analyze, correction, impute, config

def evaluate_season(season_year, df, clf):
    print(f"\nEvaluating {season_year} (N={len(df)})...")
    
    # 1. Pipeline
    print("  Correcting Blocked Shot Attribution...")
    df = correction.fix_blocked_shot_attribution(df)
    
    print("  Imputing Blocked Shots...")
    # Need to filter to shot attempts first to impute safely
    mask_shots = df['event'].isin(['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot'])
    df_shots = df[mask_shots].copy()
    try:
        df_shots = impute.impute_blocked_shot_origins(df_shots, method='point_pull')
    except Exception as e:
        print(f"  Imputation warning: {e}")
        
    print("  Enriching Bios...")
    df_shots = fit_xgs.enrich_data_with_bios(df_shots)
    
    # 2. Predict
    # Prepare for XGBoost Nested
    print("  Predicting...")
    # Preprocess just like analyze.py / train loop
    # We need to manually prepare because analyze._predict_xgs handles a lot of logic
    # But we want to isolate the model behavior.
    
    # Ensure features exist
    input_features = getattr(clf, 'features', ['distance', 'angle_deg', 'game_state', 'shot_type', 'shoots_catches'])
    
    # Fill missing features
    for f in input_features:
        if f not in df_shots.columns:
            df_shots[f] = np.nan
            
    # Sub-select
    df_model = df_shots[input_features].copy()
    
    preds = clf.predict_proba(df_model)[:, 1]
    df_shots['xG'] = preds
    
    # 3. Metrics
    if 'is_goal' not in df_shots.columns:
        df_shots['is_goal'] = (df_shots['event'] == 'goal').astype(int)
        
    # Global
    total_xg = df_shots['xG'].sum()
    total_goals = df_shots['is_goal'].sum()
    global_ratio = total_xg / total_goals if total_goals > 0 else 0
    
    # Unblocked
    mask_unblocked = df_shots['event'] != 'blocked-shot'
    ub_xg = df_shots.loc[mask_unblocked, 'xG'].sum()
    ub_goals = df_shots.loc[mask_unblocked, 'is_goal'].sum()
    ub_ratio = ub_xg / ub_goals if ub_goals > 0 else 0
    
    # Blocked Share
    mask_blocked = df_shots['event'] == 'blocked-shot'
    blk_xg = df_shots.loc[mask_blocked, 'xG'].sum()
    blk_share_xg = blk_xg / total_xg if total_xg > 0 else 0
    blk_rate = mask_blocked.mean()
    
    # --- Sub-Model Diagnostics ---
    # 1. Block Model (All shots target: is_blocked)
    # Target: is_blocked
    is_blocked_target = (df_shots['event'] == 'blocked-shot').astype(int)
    prob_blocked = clf.predict_proba_layer(df_model, 'block')
    if is_blocked_target.mean() > 0:
        block_ratio = prob_blocked.mean() / is_blocked_target.mean()
    else:
        block_ratio = 0

    # 2. Accuracy Model (Unblocked only. Target: is_on_net)
    # Filter to unblocked
    mask_unblocked = df_shots['event'] != 'blocked-shot'
    if mask_unblocked.sum() > 0:
        df_ub_model = df_model[mask_unblocked]
        is_on_net = df_shots.loc[mask_unblocked, 'event'].isin(['shot-on-goal', 'goal']).astype(int)
        prob_accuracy = clf.predict_proba_layer(df_ub_model, 'accuracy')
        if is_on_net.mean() > 0:
            acc_ratio = prob_accuracy.mean() / is_on_net.mean()
        else:
            acc_ratio = 0
    else:
        acc_ratio = 0

    # 3. Finish Model (On Net only. Target: is_goal)
    mask_on_net = df_shots['event'].isin(['shot-on-goal', 'goal'])
    if mask_on_net.sum() > 0:
        df_on_net_model = df_model[mask_on_net]
        is_goal_target = (df_shots.loc[mask_on_net, 'event'] == 'goal').astype(int)
        prob_finish = clf.predict_proba_layer(df_on_net_model, 'finish')
        if is_goal_target.mean() > 0:
            finish_ratio = prob_finish.mean() / is_goal_target.mean()
        else:
             finish_ratio = 0
    else:
        finish_ratio = 0
    
    return {
        'df': df_shots,
        'metrics': {
            'Global Ratio': global_ratio,
            'Unblocked Ratio': ub_ratio,
            'Blocked xG Share': blk_share_xg,
            'Blocked Rate (Event)': blk_rate,
            'Total Events': len(df_shots),
            'Mean Blocked xG': df_shots.loc[mask_blocked, 'xG'].mean(),
            'Block Ratio': block_ratio,
            'Accuracy Ratio': acc_ratio,
            'Finish Ratio': finish_ratio
        }
    }

def run_investigation():
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_nested_all.joblib'
    print(f"Loading Model: {model_path}")
    clf = joblib.load(model_path)
    
    print(f"Loading Model: {model_path}")
    clf = joblib.load(model_path)
    
    # Scan All Seasons
    seasons = []
    base_dir = Path(config.DATA_DIR)
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and item.name.isdigit(): # Format: 20232024
             # Try standard file name
             csv_path = item / f"{item.name}_df.csv"
             if csv_path.exists():
                 # Format name as YYYY-YYYY
                 s_name = f"{item.name[:4]}-{item.name[4:]}"
                 seasons.append((s_name, str(csv_path)))
    
    # Also Check base dir for current season file if not in updated folder structure
    # (Handling the 20252026.csv case or similar)
    # Actually load_season_df logic handles that, but here we just list specific files we see
    # Let's rely on the folders + the known 2025 file
    
    # Add manual check for 2025 if missing
    if not any(s[0] == '2025-2026' for s in seasons):
         p = base_dir / '20252026' / '20252026_df.csv'
         if p.exists():
             seasons.append(('2025-2026', str(p)))
         else:
             # Fallback 2
             p = base_dir / '20252026.csv'
             if p.exists():
                  seasons.append(('2025-2026', str(p)))
    
    seasons = sorted(seasons, key=lambda x: x[0])
    print(f"Found {len(seasons)} seasons: {[s[0] for s in seasons]}")
    
    results = {}
    
    for name, path in seasons:
        if not os.path.exists(path):
            print(f"Skipping {name}, file not found: {path}")
            continue
            
        print(f"Loading {name}...")
        df = pd.read_csv(path)
        res = evaluate_season(name, df, clf)
        results[name] = res
        
    # --- Compare Metrics ---
    print("\n\n--- COMPARATIVE ANALYSIS ---")
    metrics_df = pd.DataFrame({name: res['metrics'] for name, res in results.items()})
    print(metrics_df.T)
    metrics_csv_path = 'analysis/calibration_drift_metrics.csv'
    metrics_df.T.to_csv(metrics_csv_path)
    print(f"Metrics saved to {metrics_csv_path}")
    
    # --- PLOTS ---
    import matplotlib.pyplot as plt
    
    # 1. Calibration Curves (Unblocked)
    plt.figure(figsize=(18, 6))
    
    # 1. Timeline of Ratios
    plt.subplot(1, 3, 1)
    
    # Process results for plotting
    rows = []
    for name, res in results.items():
        m = res['metrics']
        rows.append({
            'Season': name, 
            'Global Ratio': m['Global Ratio'],
            'Unblocked Ratio': m['Unblocked Ratio'],
            'Blocked xG Share': m['Blocked xG Share'],
            'Block Ratio': m['Block Ratio'],
            'Accuracy Ratio': m['Accuracy Ratio'],
            'Finish Ratio': m['Finish Ratio']
        })
    plot_df = pd.DataFrame(rows).sort_values('Season')
    
    plt.plot(plot_df['Season'], plot_df['Global Ratio'], marker='o', label='Global Ratio', color='tab:blue')
    plt.plot(plot_df['Season'], plot_df['Unblocked Ratio'], marker='s', label='Unblocked Ratio', color='tab:orange')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xticks(rotation=45)
    plt.title('Calibration Ratios over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Blocked Shot Distributions (Select recent vs oldest)
    plt.subplot(1, 3, 2)
    # Pick first, middle, last
    sorted_seasons = sorted(results.keys())
    if len(sorted_seasons) > 3:
        to_plot = [sorted_seasons[0], sorted_seasons[len(sorted_seasons)//2], sorted_seasons[-1]]
    else:
        to_plot = sorted_seasons
        
    for name in to_plot:
        df = results[name]['df']
        mask = df['event'] == 'blocked-shot'
        vals = df.loc[mask, 'xG']
        plt.hist(vals, bins=30, alpha=0.5, density=True, label=f"{name}")
        
    plt.xlabel('xG Value')
    plt.ylabel('Density')
    plt.title('Blocked Shot xG Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Sub-Model Ratios
    plt.subplot(1, 3, 3)
    plt.plot(plot_df['Season'], plot_df['Block Ratio'], marker='^', label='Block Ratio', linestyle=':')
    plt.plot(plot_df['Season'], plot_df['Accuracy Ratio'], marker='s', label='Accuracy Ratio', linestyle='--')
    plt.plot(plot_df['Season'], plot_df['Finish Ratio'], marker='o', label='Finish Ratio', linestyle='-')
    plt.axhline(1.0, color='gray', linestyle='--')
    plt.xticks(rotation=45)
    plt.title('Sub-Model Calibration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = 'analysis/calibration_investigation.png'
    plt.savefig(out_path)
    print(f"\nPlots saved to {out_path}")

if __name__ == "__main__":
    import os
    run_investigation()
