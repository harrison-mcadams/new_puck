
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs, fit_nested_xgs, impute
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

# Configuration
SEASONS_TO_LOAD = [f"{y}{y+1}" for y in range(2025, 2014, -1)] # 10 years

def load_combined_data():
    """Load all available data from disk."""
    dfs = []
    print("Loading data...")
    for season in SEASONS_TO_LOAD:
        p = Path(f"data/{season}/{season}_df.csv")
        if p.exists():
            try:
                # Only load essential cols to save RAM
                cols = ['event', 'x', 'y', 'distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
                # But we need to be careful if checking for missing cols
                df = pd.read_csv(p)
                dfs.append(df)
            except Exception as e:
                print(f"Skipping {season}: {e}")
    
    if not dfs:
        raise FileNotFoundError("No data found. Run backfill_seasons.py first?")
        
    full = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(full)} rows.")
    return full

def evaluate_pred(y_true, y_prob, name):
    ll = log_loss(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    print(f"[{name}] AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.4f}")
    return {'model': name, 'auc': auc, 'log_loss': ll, 'brier': brier}

def plot_merged_calibration(results_dict, out_path='analysis/xgs/comparison_calibration.png'):
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Ideal')
    
    for name, data in results_dict.items():
        y_true = data['y_true']
        y_prob = data['y_prob']
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='.', label=f"{name} (LL={data['log_loss']:.3f})")
        
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Fraction')
    plt.title('Calibration Comparison: Nested vs Single Layer')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")

def main():
    # 1. Load Data
    full_df = load_combined_data()
    
    # 2. Split Data (Train/Test)
    # We'll use a random split for general performance, 
    # but strictly we should respect time. Let's do random for now for robust "capability" check.
    train_df_raw, test_df_raw = train_test_split(full_df, test_size=0.2, random_state=42, shuffle=True)
    
    print(f"Train size: {len(train_df_raw)} | Test size: {len(test_df_raw)}")
    
    # --- MODEL A: SINGLE LAYER (Standard) ---
    # Rules: No blocked shots in training OR testing.
    # We must filter the train/test sets for this model specifically.
    
    print("\n--- Evaluating Single Layer Model ---")
    
    # Filter: Remove blocked shots
    train_single = train_df_raw[train_df_raw['event'] != 'blocked-shot'].copy()
    test_single = test_df_raw[test_df_raw['event'] != 'blocked-shot'].copy()
    
    features_single = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    
    # Clean/Preprocess
    train_single_mod, final_feats_s, _ = fit_xgs.clean_df_for_model(train_single, features_single)
    test_single_mod, _, _ = fit_xgs.clean_df_for_model(test_single, features_single) # Reuse logic
    
    # Fit
    clf_single, _, _ = fit_xgs.fit_model(
        train_single_mod, 
        feature_cols=final_feats_s, 
        n_estimators=100, # Lower estimators for speed in comparison script
        max_depth=10
    )
    
    # Predict
    # Ensure X_test has same cols
    X_test_s = test_single_mod[final_feats_s].values
    y_test_s = test_single_mod['is_goal'].values
    y_prob_s = clf_single.predict_proba(X_test_s)[:, 1]
    
    metrics_s = evaluate_pred(y_test_s, y_prob_s, "Single Layer")
    
    
    # --- MODEL B: NESTED MODEL ---
    # Rules: Uses "mean_6" imputation.
    # We train on full set (imputed).
    # We test on full set (imputed).
    # NOTE: The "Test Set" for Nested includes Blocked Shots (where goal=0).
    # The "Test Set" for Single Layer EXCLUDES Blocked Shots.
    # This makes direct LogLoss comparison tricky if the populations differ.
    # HOWEVER, the user asked to compare "metrics across these two paths".
    # Comparing AUC on their respective valid populations is fair for "how well does it model its domain".
    # Comparing on the INTERSECTION (Unblocked shots) is also interesting.
    
    print("\n--- Evaluating Nested Model ---")
    
    # Impute on the RAW splits (to avoid leakage)
    train_nested_imp = impute.impute_blocked_shot_origins(train_df_raw, method='mean_6')
    test_nested_imp = impute.impute_blocked_shot_origins(test_df_raw, method='mean_6')
    
    # Fit
    clf_nested = fit_nested_xgs.NestedXGClassifier(
        n_estimators=100,
        max_depth=10,
        prevent_overfitting=True
    )
    clf_nested.fit(train_nested_imp)
    
    # Predict (Full Population)
    # The predict_proba expects a DataFrame with raw features
    y_prob_n = clf_nested.predict_proba(test_nested_imp)[:, 1]
    
    # Construct Y truth
    # Nested model target is Goal.
    y_test_n = (test_nested_imp['event'] == 'goal').astype(int)
    
    metrics_n = evaluate_pred(y_test_n, y_prob_n, "Nested (Full)")
    
    
    # --- COMPARISON ON UNBLOCKED SUBSET ---
    # To compare apples-to-apples, let's see how Nested performs on just the Unblocked shots
    # (The same population Single Layer saw)
    
    mask_unblocked = (test_nested_imp['event'] != 'blocked-shot')
    y_test_n_unblocked = y_test_n[mask_unblocked]
    y_prob_n_unblocked = y_prob_n[mask_unblocked]
    
    metrics_n_un = evaluate_pred(y_test_n_unblocked, y_prob_n_unblocked, "Nested (Unblocked Only)")
    
    
    # --- PLOTTING ---
    results = {
        'Single Layer': {'y_true': y_test_s, 'y_prob': y_prob_s, 'log_loss': metrics_s['log_loss']},
        'Nested (Full)': {'y_true': y_test_n, 'y_prob': y_prob_n, 'log_loss': metrics_n['log_loss']},
        'Nested (Unblocked)': {'y_true': y_test_n_unblocked, 'y_prob': y_prob_n_unblocked, 'log_loss': metrics_n_un['log_loss']}
    }
    
    plot_merged_calibration(results)
    
    # Output JSON summary
    summary = [metrics_s, metrics_n, metrics_n_un]
    path = 'analysis/xgs/model_comparison_results.json'
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved results to {path}")

if __name__ == "__main__":
    main()
