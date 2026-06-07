"""scripts/evaluate_season_performance.py

Evaluates the final optimal expected goals (xG) model season-by-season
across all available NHL history (2010-2026).
Generates an interactive, premium dashboard showing year-by-year performance,
with particular focus on the Modern Era (20202021+) and the current season (20252026).
"""

import sys
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import fit_xgboost_tensor, config, analyze, data_pipeline, features as feature_util
from scripts.compare_season_options import calculate_mace

# Set modern plotting aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Inter', 'Outfit', 'DejaVu Sans', 'Arial']

def evaluate_season(model, season):
    """Loads and evaluates the model on a specific season's data."""
    try:
        csv_path = analyze.locate_season_csv(season)
        if not csv_path:
            print(f"  [Warning] CSV not found for season {season}")
            return None
            
        print(f"  Loading & Preprocessing {season}...")
        df_raw = pd.read_csv(csv_path, low_memory=False)
        df_raw['season'] = int(season)
        
        df = data_pipeline.preprocess_features(
            df_raw, 
            is_training=False, 
            verbose=False,
            apply_arena_adjustments=True,
            apply_imputation=True,
            apply_dithering=True,
            apply_filtering=True,
            apply_attribution_fix=True,
            apply_html_enrichment=False,
            impute_alpha=0.2
        )
        
        if len(df) == 0:
            print(f"  [Warning] Empty dataset after preprocessing for {season}")
            return None
            
        y_true = (df['event'] == 'goal').astype(int)
        probs = model.predict_proba(df)[:, 1]
        
        # Core Metrics
        auc = roc_auc_score(y_true, probs)
        ll = log_loss(y_true, probs)
        brier = brier_score_loss(y_true, probs)
        mace = calculate_mace(y_true, probs, n_bins=10)
        
        # Goal Volumes
        actual_goals = y_true.sum()
        expected_goals = probs.sum()
        ratio = expected_goals / actual_goals if actual_goals > 0 else 0
        
        # Block Layer
        y_true_block = (df['event'] == 'blocked-shot').astype(int)
        probs_block = model.predict_proba_layer(df, 'block')
        if len(probs_block.shape) > 1: probs_block = probs_block[:, 1]
        block_auc = roc_auc_score(y_true_block, probs_block)
        block_ll = log_loss(y_true_block, probs_block)
        
        # Accuracy Layer
        mask_unblocked = df['event'] != 'blocked-shot'
        df_unblocked = df[mask_unblocked]
        y_true_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        probs_acc = model.predict_proba_layer(df_unblocked, 'accuracy')
        if len(probs_acc.shape) > 1: probs_acc = probs_acc[:, 1]
        acc_auc = roc_auc_score(y_true_acc, probs_acc)
        acc_ll = log_loss(y_true_acc, probs_acc)
        
        # Finish Layer
        mask_on_net = df['event'].isin(['shot-on-goal', 'goal'])
        df_on_net = df[mask_on_net]
        y_true_fin = (df_on_net['event'] == 'goal').astype(int)
        probs_fin = model.predict_proba_layer(df_on_net, 'finish')
        if len(probs_fin.shape) > 1: probs_fin = probs_fin[:, 1]
        fin_auc = roc_auc_score(y_true_fin, probs_fin)
        fin_ll = log_loss(y_true_fin, probs_fin)
        
        print(f"    [OK] AUC: {auc:.4f} | LogLoss: {ll:.4f} | Ratio (xG/G): {ratio:.3f}")
        
        return {
            'season': str(season),
            'shots': len(df),
            'actual_goals': actual_goals,
            'expected_goals': expected_goals,
            'xg_g_ratio': ratio,
            'auc': auc,
            'logloss': ll,
            'brier': brier,
            'mace': mace,
            'block_auc': block_auc,
            'block_ll': block_ll,
            'acc_auc': acc_auc,
            'acc_ll': acc_ll,
            'fin_auc': fin_auc,
            'fin_ll': fin_ll
        }
    except Exception as e:
        print(f"  [Error] Failed to evaluate season {season}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("============================================================")
    print("NHL expected GOALS MODEL SEASON-BY-SEASON PERFORMANCE TRACKER")
    print("============================================================")
    
    # 1. Load the Model
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_final.joblib'
    if not model_path.exists():
        # Fallback to full history model if final doesn't exist yet
        model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_full_history.joblib'
        
    if not model_path.exists():
        # Search for any recent xgboost tensor joblib
        model_dir = Path(config.ANALYSIS_DIR) / 'xgs'
        joblibs = sorted(list(model_dir.glob('xg_model_xgboost_tensor*.joblib')), key=os.path.getmtime)
        if joblibs:
            model_path = joblibs[-1]
            
    if not model_path.exists():
        print(f"[Error] No trained model found at {model_path} or in analysis/xgs/!")
        sys.exit(1)
        
    print(f"Loading trained expected goals model: {model_path}")
    model = joblib.load(model_path)
    print(f"  Model Type: {type(model).__name__}")
    print(f"  Season Mode: {getattr(model, 'season_mode', 'N/A')}")
    
    # 2. Discover available seasons
    from scripts.evaluate_predictive_power import DataUtils
    all_seasons = DataUtils.get_available_seasons()
    print(f"Discovered {len(all_seasons)} seasons: {all_seasons}")
    
    # 3. Loop and evaluate
    history = []
    for s in all_seasons:
        res = evaluate_season(model, s)
        if res:
            history.append(res)
            
    if not history:
        print("[Error] No seasons evaluated!")
        sys.exit(1)
        
    df_hist = pd.DataFrame(history)
    
    # Save CSV summary
    analysis_dir = Path(config.ANALYSIS_DIR)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df_hist.to_csv(analysis_dir / 'season_performance_history.csv', index=False)
    print(f"\nHistorical season performance metrics saved to {analysis_dir / 'season_performance_history.csv'}")
    
    # 4. Generate beautiful dashboard
    print("\nGenerating year-by-year performance dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), facecolor='#f8f9fa')
    
    seasons_labels = [f"{s[:4]}-{s[4:]}" for s in df_hist['season']]
    x_indices = np.arange(len(df_hist))
    
    # COLOR PALETTE
    c_primary = '#1abc9c'  # Teal
    c_secondary = '#3498db'  # Blue
    c_accent = '#e74c3c'  # Red
    c_dark = '#2c3e50'  # Slate
    
    # Panel 1: LogLoss & Brier Score (Quality of Probabilities)
    ax_loss = axes[0, 0]
    ax_brier = ax_loss.twinx()
    
    line_ll = ax_loss.plot(x_indices, df_hist['logloss'], marker='o', color=c_primary, linewidth=2.5, label='LogLoss (Left)')
    line_br = ax_brier.plot(x_indices, df_hist['brier'], marker='s', color=c_secondary, linewidth=2, linestyle='--', label='Brier Score (Right)')
    
    # Combine legends
    lines = line_ll + line_br
    labels = [l.get_label() for l in lines]
    ax_loss.legend(lines, labels, loc='upper right', frameon=True, facecolor='white')
    
    ax_loss.set_title("Probability Prediction Quality (LogLoss & Brier Score)", fontsize=13, fontweight='bold', color=c_dark)
    ax_loss.set_xlabel("Season", fontsize=11)
    ax_loss.set_ylabel("LogLoss", fontsize=11, color=c_primary)
    ax_brier.set_ylabel("Brier Score", fontsize=11, color=c_secondary)
    ax_loss.set_xticks(x_indices)
    ax_loss.set_xticklabels(seasons_labels, rotation=35, ha='right')
    ax_loss.grid(True, alpha=0.3)
    
    # Add vertical indicators for Modern Era and Current Season
    # Transition to Modern Era (20202021) is index of '20202021'
    if '20202021' in df_hist['season'].values:
        idx_modern = df_hist[df_hist['season'] == '20202021'].index[0]
        ax_loss.axvline(x=idx_modern, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=2)
        ax_loss.text(idx_modern - 0.2, ax_loss.get_ylim()[0] + (ax_loss.get_ylim()[1]-ax_loss.get_ylim()[0])*0.8, 
                     'Modern Era Starts', rotation=90, color='#7f8c8d', fontweight='bold', fontsize=10)
                     
    # Highlight current season (last season)
    idx_current = len(df_hist) - 1
    ax_loss.axvline(x=idx_current, color=c_accent, linestyle='--', alpha=0.7, linewidth=1.5)
    ax_loss.text(idx_current - 0.4, ax_loss.get_ylim()[0] + (ax_loss.get_ylim()[1]-ax_loss.get_ylim()[0])*0.5, 
                 'Current Season\n(2025-2026)', color=c_accent, fontweight='bold', fontsize=9, ha='right')
                 
    # Panel 2: Discriminative Performance (ROC-AUC)
    ax_auc = axes[0, 1]
    ax_auc.plot(x_indices, df_hist['auc'], marker='o', color=c_dark, linewidth=3, label='Overall xG AUC')
    ax_auc.plot(x_indices, df_hist['block_auc'], marker='^', color=c_accent, linewidth=1.5, linestyle=':', label='Block Layer AUC')
    ax_auc.plot(x_indices, df_hist['acc_auc'], marker='v', color=c_primary, linewidth=1.5, linestyle='--', label='Accuracy Layer AUC')
    ax_auc.plot(x_indices, df_hist['fin_auc'], marker='d', color=c_secondary, linewidth=1.5, linestyle='-.', label='Finish Layer AUC')
    
    ax_auc.set_title("Discriminative Power (ROC-AUC) By Layer", fontsize=13, fontweight='bold', color=c_dark)
    ax_auc.set_xlabel("Season", fontsize=11)
    ax_auc.set_ylabel("ROC-AUC Score", fontsize=11)
    ax_auc.set_xticks(x_indices)
    ax_auc.set_xticklabels(seasons_labels, rotation=35, ha='right')
    ax_auc.set_ylim(0.55, 0.85)
    ax_auc.grid(True, alpha=0.3)
    ax_auc.legend(loc='lower left', frameon=True, facecolor='white')
    
    # Highlight current season
    ax_auc.axvline(x=idx_current, color=c_accent, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Panel 3: Calibration Quality (MACE)
    ax_mace = axes[1, 0]
    ax_mace.plot(x_indices, df_hist['mace'] * 100, marker='o', color='#9b59b6', linewidth=2.5, label='MACE (%)')
    ax_mace.set_title("Calibration Error (Weighted MACE % - Lower is Better)", fontsize=13, fontweight='bold', color=c_dark)
    ax_mace.set_xlabel("Season", fontsize=11)
    ax_mace.set_ylabel("MACE (%)", fontsize=11)
    ax_mace.set_xticks(x_indices)
    ax_mace.set_xticklabels(seasons_labels, rotation=35, ha='right')
    ax_mace.grid(True, alpha=0.3)
    
    # Add values on top of points for recent seasons
    for i, m in enumerate(df_hist['mace']):
        if i >= len(df_hist) - 6:  # Only label recent years
            ax_mace.annotate(f"{m*100:.2f}%", 
                             xy=(i, m*100), 
                             xytext=(0, 8), 
                             textcoords="offset points", 
                             ha='center', va='bottom', fontsize=9, fontweight='bold', color='#8e44ad')
                             
    ax_mace.axvline(x=idx_current, color=c_accent, linestyle='--', alpha=0.7, linewidth=1.5)
    if '20202021' in df_hist['season'].values:
        ax_mace.axvline(x=idx_modern, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=2)
        
    # Panel 4: Actual vs Expected Goals (Calibration Volume)
    ax_vol = axes[1, 1]
    ax_vol.bar(x_indices - 0.2, df_hist['actual_goals'], width=0.4, color='#7f8c8d', alpha=0.6, label='Actual Goals')
    ax_vol.bar(x_indices + 0.2, df_hist['expected_goals'], width=0.4, color=c_primary, alpha=0.8, label='Expected Goals (xG)')
    
    ax_vol_twin = ax_vol.twinx()
    ax_vol_twin.plot(x_indices, df_hist['xg_g_ratio'], color=c_accent, marker='D', linewidth=2, label='xG / Actual Ratio')
    ax_vol_twin.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Legends
    handler1, label1 = ax_vol.get_legend_handles_labels()
    handler2, label2 = ax_vol_twin.get_legend_handles_labels()
    ax_vol.legend(handler1 + handler2, label1 + label2, loc='upper left', frameon=True, facecolor='white')
    
    ax_vol.set_title("Goal Volume Calibration (Expected vs Actual Goals)", fontsize=13, fontweight='bold', color=c_dark)
    ax_vol.set_xlabel("Season", fontsize=11)
    ax_vol.set_ylabel("Total Goals", fontsize=11)
    ax_vol_twin.set_ylabel("Ratio (Expected / Actual)", fontsize=11, color=c_accent)
    ax_vol.set_xticks(x_indices)
    ax_vol.set_xticklabels(seasons_labels, rotation=35, ha='right')
    ax_vol.grid(True, alpha=0.3)
    ax_vol_twin.set_ylim(0.8, 1.2)
    
    ax_vol.axvline(x=idx_current, color=c_accent, linestyle='--', alpha=0.7, linewidth=1.5)
    if '20202021' in df_hist['season'].values:
        ax_vol.axvline(x=idx_modern, color='#7f8c8d', linestyle=':', alpha=0.8, linewidth=2)
        
    plt.suptitle(f"NHL Expected Goals (XGBoost Tensor) Season-by-Season Performance Dashboard\nEvaluation over 16 seasons ({len(df_hist)} periods evaluated, final production model)",
                 fontsize=16, fontweight='bold', color='#2c3e50', y=0.98)
                 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plot_path = analysis_dir / 'season_performance_trends.png'
    fig.savefig(plot_path, dpi=300, facecolor='#f8f9fa')
    plt.close()
    print(f"Stunning performance trend dashboard saved to: {plot_path}")
    
    # 5. Copy to brain folder for user view
    brain_dir = Path(r"C:\Users\harri\.gemini\antigravity\brain\9d02cff6-8a0e-4709-ac4b-3ba370cc377b")
    if brain_dir.exists():
        import shutil
        shutil.copy(plot_path, brain_dir / 'season_performance_trends.png')
        print(f"Dashboard successfully copied to brain directory: {brain_dir / 'season_performance_trends.png'}")
        
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
