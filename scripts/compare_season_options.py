"""scripts/compare_season_options.py

Rigorous comparison suite for analyzing multi-season NHL data representation:
1. Data Scope: Modern Era (20202021+) vs. Full History (20102011+)
2. Season Representation: No Season Feature vs. Categorical vs. Numerical

Uses 70/30 train/test splits collapsed across seasons for ultimate comparability.
Generates: `analysis/season_comparison_dashboard.png`
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.calibration import calibration_curve

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import fit_xgboost_tensor, config, analyze, data_pipeline, features as feature_util
from scripts.evaluate_predictive_power import DataUtils

# Set modern plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'DejaVu Sans', 'Arial']

def calculate_mace(y_true, y_prob, n_bins=10):
    """Calculate Weighted Mean Absolute Calibration Error (MACE) robustly."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    
    abs_diffs = []
    weights = []
    
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.any():
            p_mean = y_prob[mask].mean()
            y_mean = y_true[mask].mean()
            abs_diffs.append(abs(p_mean - y_mean))
            weights.append(mask.sum() / len(y_prob))
            
    if not weights:
        return 0.0
    return sum(d * w for d, w in zip(abs_diffs, weights))

def load_and_preprocess_scope(seasons):
    """Load and preprocess multiple seasons' data and aggregate."""
    dfs = []
    for s in seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            if csv_path:
                df_s = pd.read_csv(csv_path, low_memory=False)
                df_s['season'] = int(s)
                dfs.append(df_s)
                print(f"  [OK] Loaded {s} ({len(df_s)} events)")
        except Exception as e:
            print(f"  [Error] Failed to load season {s}: {e}")
            
    if not dfs:
        raise FileNotFoundError("No data loaded!")
        
    df_raw = pd.concat(dfs, ignore_index=True)
    
    print(f"  Preprocessing {len(df_raw)} raw events...")
    df = data_pipeline.preprocess_features(
        df_raw, 
        is_training=True, 
        verbose=False,
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=True,
        apply_filtering=True,
        apply_attribution_fix=True,
        apply_html_enrichment=False,
        impute_alpha=0.2
    )
    return df

def run_evaluation(df_train, df_test, season_mode, name):
    """Train the model and evaluate metrics on the test set."""
    feature_list = feature_util.get_features('all_inclusive')
    
    print(f"  Training model: {name} (Mode: {season_mode})...")
    start_t = time.time()
    
    clf = fit_xgboost_tensor.XGBTensorXGClassifier(
        features=feature_list,
        n_estimators=1000,  # Bounded for sweep speed but high enough for robustness
        max_depth=6,
        learning_rate=0.05,
        use_calibration=False,
        use_balancing=False,
        use_splines=True,
        season_mode=season_mode
    )
    
    clf.fit(df_train)
    train_t = time.time() - start_t
    print(f"    Training took {train_t:.1f}s.")
    
    # Predict Goal Probabilities
    y_test = (df_test['event'] == 'goal').astype(int)
    probs = clf.predict_proba(df_test)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    ll = log_loss(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    mace = calculate_mace(y_test, probs, n_bins=10)
    
    # Sub-model layer performance breakdown
    # Block Layer
    y_test_block = (df_test['event'] == 'blocked-shot').astype(int)
    probs_block = clf.predict_proba_layer(df_test, 'block')
    if len(probs_block.shape) > 1: probs_block = probs_block[:, 1]
    block_auc = roc_auc_score(y_test_block, probs_block)
    block_ll = log_loss(y_test_block, probs_block)
    
    # Accuracy Layer
    mask_unblocked = df_test['event'] != 'blocked-shot'
    df_test_unblocked = df_test[mask_unblocked]
    y_test_acc = df_test_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
    probs_acc = clf.predict_proba_layer(df_test_unblocked, 'accuracy')
    if len(probs_acc.shape) > 1: probs_acc = probs_acc[:, 1]
    acc_auc = roc_auc_score(y_test_acc, probs_acc)
    acc_ll = log_loss(y_test_acc, probs_acc)
    
    # Finish Layer
    mask_on_net = df_test['event'].isin(['shot-on-goal', 'goal'])
    df_test_on_net = df_test[mask_on_net]
    y_test_fin = (df_test_on_net['event'] == 'goal').astype(int)
    probs_fin = clf.predict_proba_layer(df_test_on_net, 'finish')
    if len(probs_fin.shape) > 1: probs_fin = probs_fin[:, 1]
    fin_auc = roc_auc_score(y_test_fin, probs_fin)
    fin_ll = log_loss(y_test_fin, probs_fin)
    
    metrics = {
        'name': name,
        'season_mode': season_mode,
        'auc': auc,
        'logloss': ll,
        'brier': brier,
        'mace': mace,
        'block_auc': block_auc,
        'block_ll': block_ll,
        'acc_auc': acc_auc,
        'acc_ll': acc_ll,
        'fin_auc': fin_auc,
        'fin_ll': fin_ll,
        'probs': probs,
        'y_true': y_test,
        'clf': clf
    }
    
    print(f"    [OK] AUC: {auc:.4f} | LogLoss: {ll:.4f} | Brier: {brier:.6f} | MACE: {mace:.5f}")
    return metrics

def main():
    print("============================================================")
    print("NHL SEASON FEATURE COMPARATIVE SWEEP SUITE")
    print("============================================================")
    
    # 1. Discover available seasons
    all_seasons = DataUtils.get_available_seasons()
    modern_seasons = sorted([s for s in all_seasons if int(s) >= 20202021])
    full_seasons = sorted([s for s in all_seasons if int(s) >= 20102011])
    
    print(f"Modern Era Seasons ({len(modern_seasons)}): {modern_seasons}")
    print(f"Full History Seasons ({len(full_seasons)}): {full_seasons}")
    
    # 2. Load and Preprocess datasets
    print("\n--- 1/3: Loading Modern Era Dataset ---")
    df_modern = load_and_preprocess_scope(modern_seasons)
    
    print("\n--- 2/3: Loading Full History Dataset ---")
    df_full = load_and_preprocess_scope(full_seasons)
    
    # 3. Create 70/30 Splits collapsed across seasons
    print("\nCreating 70/30 train/test splits...")
    modern_train, modern_test = train_test_split(df_modern, test_size=0.3, random_state=42)
    full_train, full_test = train_test_split(df_full, test_size=0.3, random_state=42)
    
    print(f"  Modern Train: {len(modern_train)} | Test: {len(modern_test)}")
    print(f"  Full Train: {len(full_train)} | Test: {len(full_test)}")
    
    # 4. Sweep Configurations
    configs = [
        # Modern Era variants
        ('Modern_No_Season', 'none', modern_train, modern_test),
        ('Modern_Categorical', 'categorical', modern_train, modern_test),
        ('Modern_Numerical', 'numerical', modern_train, modern_test),
        # Full History variants
        ('Full_No_Season', 'none', full_train, full_test),
        ('Full_Categorical', 'categorical', full_train, full_test),
        ('Full_Numerical', 'numerical', full_train, full_test),
    ]
    
    results = []
    
    print("\n--- 3/3: Executing Comparative Sweep ---")
    for name, mode, tr, te in configs:
        res = run_evaluation(tr, te, mode, name)
        results.append(res)
        
    # 5. Print Results Table
    print("\n" + "="*80)
    print("COMPARATIVE SWEEP METRICS SUMMARY")
    print("="*80)
    
    summary_data = []
    for r in results:
        summary_data.append({
            'Config': r['name'],
            'Season Mode': r['season_mode'],
            'Overall AUC': f"{r['auc']:.4f}",
            'LogLoss': f"{r['logloss']:.4f}",
            'Brier Score': f"{r['brier']:.6f}",
            'MACE': f"{r['mace']:.5f}",
            'Block AUC': f"{r['block_auc']:.4f}",
            'Acc AUC': f"{r['acc_auc']:.4f}",
            'Fin AUC': f"{r['fin_auc']:.4f}"
        })
        
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Save results as a JSON/CSV for walkthrough integration
    analysis_dir = Path(config.ANALYSIS_DIR)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(analysis_dir / 'season_comparison_results.csv', index=False)
    
    # 6. Generate Cohesive Dashboard (Calibration + Bar Metrics)
    print("\nGenerating comparative dashboard plot...")
    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor='#f8f9fa')
    
    # Left Panel: Calibration Curves
    ax_cal = axes[0]
    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
    
    colors = {
        'Modern_No_Season': '#3498db', 'Modern_Categorical': '#2980b9', 'Modern_Numerical': '#1abc9c',
        'Full_No_Season': '#e74c3c', 'Full_Categorical': '#c0392b', 'Full_Numerical': '#e67e22'
    }
    
    for r in results:
        y_t = r['y_true']
        probs = r['probs']
        name = r['name']
        prob_true, prob_pred = calibration_curve(y_t, probs, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='.', label=f"{name} (MACE={r['mace']:.4f})", color=colors.get(name, '#7f8c8d'), linewidth=2)
        
    ax_cal.set_title("Calibration Curves Comparison (Out-of-Sample)", fontsize=14, fontweight='bold')
    ax_cal.set_xlabel("Predicted Goal Probability", fontsize=12)
    ax_cal.set_ylabel("Observed Goal Rate", fontsize=12)
    ax_cal.set_xlim(-0.02, 0.6)  # Zoom in on realistic xG bounds
    ax_cal.set_ylim(-0.02, 0.6)
    ax_cal.grid(True, alpha=0.3)
    ax_cal.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    
    # Right Panel: Brier Score / LogLoss Comparison Bar Chart
    ax_bar = axes[1]
    names = [r['name'] for r in results]
    loglosses = [r['logloss'] for r in results]
    
    bars = ax_bar.bar(names, loglosses, color=[colors.get(n, '#7f8c8d') for n in names], alpha=0.8, width=0.5)
    ax_bar.set_title("Out-of-Sample LogLoss Comparison (Lower is Better)", fontsize=14, fontweight='bold')
    ax_bar.set_ylabel("LogLoss", fontsize=12)
    ax_bar.set_ylim(min(loglosses) - 0.005, max(loglosses) + 0.002)
    ax_bar.grid(True, alpha=0.3)
    plt.setp(ax_bar.get_xticklabels(), rotation=30, horizontalalignment='right')
    
    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f"{height:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
                    
    plt.suptitle("NHL Expected Goals (XGBoost Tensor) Season Feature Comparative Sweep\nOut-of-sample 70/30 split evaluation collapsed across seasons", 
                 fontsize=16, fontweight='bold', y=0.98, color='#2c3e50')
                 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    dashboard_path = analysis_dir / 'season_comparison_dashboard.png'
    fig.savefig(dashboard_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    print(f"  Dashboard saved to {dashboard_path}")
    
    # 7. Identify the Winner and Train Final Model
    best_res = min(results, key=lambda x: x['logloss'])
    print(f"\nWinner Variant identified: '{best_res['name']}' with LogLoss: {best_res['logloss']:.4f} and MACE: {best_res['mace']:.5f}")
    
    # Final Model Training
    final_save_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_final.joblib')
    print(f"\nTraining and saving the final model on 100% of all data using winning mode '{best_res['season_mode']}'...")
    
    df_winning_data = df_full if 'Full' in best_res['name'] else df_modern
    
    final_clf = fit_xgboost_tensor.XGBTensorXGClassifier(
        features=feature_util.get_features('all_inclusive'),
        n_estimators=3000,  # Max capacity for final production deployment
        max_depth=6,
        learning_rate=0.05,
        use_calibration=False,
        use_balancing=False,
        use_splines=True,
        season_mode=best_res['season_mode']
    )
    
    final_clf.fit(df_winning_data)
    
    # Save final model
    Path(final_save_path).parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(final_clf, final_save_path)
    print(f"  Final production-ready model saved to: {final_save_path}")
    
    # Write metadata
    meta = {
        'final_features': final_clf.features,
        'model_type': 'xgboost_tensor',
        'season_mode': final_clf.season_mode,
        'data_scope': 'full_history' if 'Full' in best_res['name'] else 'modern_era',
        'train_params': {
            'n_estimators': final_clf.n_estimators,
            'max_depth': final_clf.max_depth,
            'learning_rate': final_clf.learning_rate
        }
    }
    with open(final_save_path + '.meta.json', 'w') as f:
        json.dump(meta, f)

    # Generate and save modeled season DFs for the trained seasons
    try:
        print("\nGenerating modeled season dataframes...")
        fit_xgboost_tensor.save_modeled_seasons(final_clf, df_winning_data, final_save_path, verbose=True)
    except Exception as e:
        print(f"Warning: Failed to save modeled season DFs: {e}")
        
    print("\nSweep Complete! Proceeding to document findings.")

if __name__ == "__main__":
    main()
