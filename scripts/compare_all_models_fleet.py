"""compare_all_models_fleet.py

Comparative benchmarking for the 4 primary xG models:
1. Nested XGBoost
2. Non-Nested XGBoost
3. Nested GLM (Tensor Spline)
4. Non_Nested GLM (Tensor Spline)

Calculates ROC-AUC, Brier Score, and LogLoss.
Generates ROC and Calibration Curves.
Extracts Feature Analysis.
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve, auc, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, data_pipeline, config as puck_config

MODELS = {
    'Nested XGBoost': {
        'path': 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib',
        'color': '#8e44ad', # Purple
        'subfolder': 'xgboost_nested_xgs_20202021'
    },
    'Non-Nested XGBoost': {
        'path': 'analysis/xgs/xg_model_xgboost_non_nested_20202021.joblib',
        'color': '#3498db', # Blue
        'subfolder': 'xgboost_non_nested_xgs'
    },
    'Nested GLM': {
        'path': 'analysis/xgs/xg_model_nested_tensor_20202021.joblib',
        'color': '#27ae60', # Green
        'subfolder': 'nested_xgs_20202021'
    },
    'Non-Nested GLM': {
        'path': 'analysis/xgs/xg_model_non_nested_tensor_20202021.joblib',
        'color': '#e67e22', # Orange
        'subfolder': 'non_nested_xgs'
    }
}

def load_test_data():
    print("Loading test data (2024-2025)...")
    data_dir = Path(puck_config.DATA_DIR)
    # Using 2024-2025 as a dedicated test season if available, else standard split
    test_file = data_dir / '20242025.csv'
    if test_file.exists():
        df = pd.read_csv(test_file)
    else:
        df = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    
    # Ensure 'season' exists for filtering
    if 'season' not in df.columns:
        if 'game_id' in df.columns:
            df['season'] = (df['game_id'] // 1000000).astype(int)
        else:
            df['season'] = 20232024 # Fallback
            
    # Filter for a consistent test set (e.g. 2023-2024)
    print(f"Total rows before filter: {len(df)}")
    if 'season' in df.columns:
        print(f"Seasons available: {df['season'].unique()}")
        
    df = df[df['season'] == 20232024].copy()
    print(f"Total rows after filter (20232024): {len(df)}")
    
    if len(df) == 0:
        print("Empty test set! Defaulting to first 10k rows for evaluation.")
        df = fit_xgs.load_all_seasons_data(base_dir=str(data_dir)).head(10000).copy()
        df['is_goal'] = (df['event'] == 'goal').astype(int)
    
    # Ensure is_goal exists
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    
    # Run through full preprocessing pipeline to get derived features (spatial_xg, relative_game_state, etc.)
    print("Applying preprocessing to test set...")
    df = data_pipeline.preprocess_features(
        df, 
        is_training=False, 
        verbose=True,
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_attribution_fix=True,
        apply_filtering=True
    )
    
    # Filter for valid rows only (models can only predict on shots/misses/goals)
    df = df[df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])].copy()
    
    return df

def generate_feature_summary(model, model_name):
    """Extract top 10 features/coeffs for the summary."""
    summary = []
    
    # Nested handling
    if hasattr(model, 'model_block'):
        # Just grab Block for now as it's the anchor of current tasks
        for name, attr in [('Block', 'model_block'), ('Accuracy', 'model_acc'), ('Finish', 'model_finish')]:
            sub = getattr(model, attr, None)
            if sub:
                summary.append(f"### {name} Submodel")
                if hasattr(sub, 'feature_importances_'):
                    # XGBoost
                    importances = sub.feature_importances_
                    feature_names = getattr(model, 'features', [f"f{i}" for i in range(len(importances))])
                    indices = np.argsort(importances)[::-1][:10]
                    for i, idx in enumerate(indices):
                        summary.append(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
                elif hasattr(sub, 'named_steps') and 'clf' in sub.named_steps:
                    # GLM
                    clf = sub.named_steps['clf']
                    pre = sub.named_steps['preprocessor']
                    if hasattr(clf, 'coef_'):
                        coefs = clf.coef_.flatten()
                        try:
                            fnames = pre.get_feature_names_out()
                        except:
                            fnames = [f"f{i}" for i in range(len(coefs))]
                        indices = np.argsort(np.abs(coefs))[::-1][:10]
                        for i, idx in enumerate(indices):
                            summary.append(f"{i+1}. {fnames[idx]}: {coefs[idx]:+.4f}")
    else:
        # Non-Nested
        summary.append(f"### Single Pass Model")
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            # XGBoost Non-Nested
            importances = model.model.feature_importances_
            feature_names = model.features
            indices = np.argsort(importances)[::-1][:10]
            for i, idx in enumerate(indices):
                summary.append(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        elif hasattr(model, 'model') and'clf' in model.model.named_steps:
            # GLM Non-Nested
            clf = model.model.named_steps['clf']
            pre = model.model.named_steps['preprocessor']
            if hasattr(clf, 'coef_'):
                coefs = clf.coef_.flatten()
                try:
                    fnames = pre.get_feature_names_out()
                except:
                    fnames = [f"f{i}" for i in range(len(coefs))]
                indices = np.argsort(np.abs(coefs))[::-1][:10]
                for i, idx in enumerate(indices):
                    summary.append(f"{i+1}. {fnames[idx]}: {coefs[idx]:+.4f}")
                    
    return "\n".join(summary)

def main():
    print("--- Starting Fleet-Wide Benchmarking ---")
    
    test_df = load_test_data()
    y_true = (test_df['event'] == 'goal').astype(int)
    
    results = {}
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax_roc = axes[0]
    ax_cal = axes[1]
    
    ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Comparison (Modern Era)')
    
    ax_cal.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax_cal.set_xlabel('Predicted Probability')
    ax_cal.set_ylabel('Actual Goal Rate')
    ax_cal.set_title('Calibration Comparison')
    
    for name, info in MODELS.items():
        path = Path(puck_config.ANALYSIS_DIR).parent / info['path']
        if not path.exists():
            print(f"Skipping {name}: model not found at {path}")
            continue
            
        print(f"Evaluating {name}...")
        model = joblib.load(path)
        
        # Predict
        start_t = time.time()
        probs = model.predict_proba(test_df)[:, 1]
        elapsed = time.time() - start_t
        
        # Metrics
        fpr, tpr, _ = roc_curve(y_true, probs)
        score_auc = auc(fpr, tpr)
        score_ll = log_loss(y_true, probs)
        score_brier = brier_score_loss(y_true, probs)
        
        results[name] = {
            'auc': score_auc,
            'logloss': score_ll,
            'brier': score_brier,
            'latency': elapsed / len(test_df) * 1000 # ms per shot
        }
        
        # Plot
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC={score_auc:.4f})", color=info['color'], lw=2)
        
        prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='o', label=name, color=info['color'], lw=2)
        
        # Feature analysis
        feat_summary = generate_feature_summary(model, name)
        
        # Save model-specific stats
        sub_dir = Path(puck_config.ANALYSIS_DIR) / info['subfolder']
        sub_dir.mkdir(parents=True, exist_ok=True)
        with open(sub_dir / 'benchmarking_stats.txt', 'w') as f:
            f.write(f"Model: {name}\n")
            f.write(f"AUC: {score_auc:.4f}\n")
            f.write(f"LogLoss: {score_ll:.4f}\n")
            f.write(f"Brier: {score_brier:.6f}\n")
            f.write(f"Avg Latency: {results[name]['latency']:.4f} ms/shot\n\n")
            f.write("## Feature Analysis\n")
            f.write(feat_summary)
            
    ax_roc.legend()
    ax_cal.legend()
    plt.tight_layout()
    
    plot_path = Path(puck_config.ANALYSIS_DIR) / 'fleet_benchmarking_dashboard.png'
    plt.savefig(plot_path)
    print(f"Saved benchmarking dashboard to {plot_path}")
    
    # Save aggregate summary
    summary_path = Path(puck_config.ANALYSIS_DIR) / 'fleet_performance_summary.md'
    with open(summary_path, 'w') as f:
        f.write("# Fleet-Wide Performance Comparison\n\n")
        f.write("| Model | AUC | LogLoss | Brier Score | Latency (ms/shot) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for name, res in results.items():
            f.write(f"| {name} | {res['auc']:.4f} | {res['logloss']:.4f} | {res['brier']:.6f} | {res['latency']:.4f} |\n")
    
    print(f"Saved aggregate summary to {summary_path}")

if __name__ == "__main__":
    main()
