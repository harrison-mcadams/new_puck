
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve, auc
from sklearn.calibration import calibration_curve
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, fit_nested_xgs, impute, config as puck_config

def plot_roc_curve(y_true, y_prob, label, color, ax):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

def plot_calibration(y_true, y_prob, label, color, ax):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(prob_pred, prob_true, marker='o', color=color, label=label)

def plot_importance(model, feature_names, title, ax):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(title)
    ax.set_xlabel('Importance')

def main():
    print("Generating Nested Model Performance Report...")
    
    # 1. Load Model
    model_path = 'analysis/xgs/xg_model_Nested_Standard.joblib'
    if not os.path.exists(model_path):
        # Fallback
        model_path = 'analysis/xgs/xg_model_nested.joblib'
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    clf = joblib.load(model_path)
    
    # 2. Load Data
    data_path = 'data/20252026.csv'
    if not os.path.exists(data_path):
        data_path = 'data/20252026/20252026_df.csv'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Filter for regular season
    if 'game_id' in df.columns:
        df['game_id_str'] = df['game_id'].astype(str)
        df = df[df['game_id_str'].str.contains(r'^\d{4}02\d{4}$')]
    
    # Preprocess
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    
    # Filter for intended events
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df = df[df['event'].isin(valid_events)].copy()
    
    # Apply Imputation
    print("Applying imputation for blocked shots...")
    df_imp = impute.impute_blocked_shot_origins(df, method='mean_6')
    
    # 3. Evaluate
    print("Running predictions...")
    p_overall = clf.predict_proba(df_imp)[:, 1]
    
    # Layer specific predictions
    p_block = clf.predict_proba_layer(df_imp, 'block')
    
    mask_unblocked = (df['is_blocked'] == 0)
    df_unblocked = df_imp[mask_unblocked]
    p_accuracy = clf.predict_proba_layer(df_unblocked, 'accuracy')
    
    mask_on_net = (df['is_on_net'] == 1)
    df_on_net = df_imp[mask_on_net]
    p_finish = clf.predict_proba_layer(df_on_net, 'finish')
    
    # 4. Plots
    visuals_dir = Path('analysis/nested_xgs_report')
    visuals_dir.mkdir(parents=True, exist_ok=True)
    
    # A. Performance Metrics
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    plot_roc_curve(df['is_goal'], p_overall, 'Overall xG', 'black', axes[0])
    plot_roc_curve(df['is_blocked'], p_block, 'Block Layer', 'red', axes[0])
    plot_roc_curve(df_unblocked['is_on_net'], p_accuracy, 'Accuracy Layer', 'blue', axes[0])
    plot_roc_curve(df_on_net['is_goal'], p_finish, 'Finish Layer', 'green', axes[0])
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('ROC Curves')
    axes[0].legend()
    
    plot_calibration(df['is_goal'], p_overall, 'Overall xG', 'black', axes[1])
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_title('Calibration Curves')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'performance_metrics.png')
    print(f"Saved performance metrics to {visuals_dir / 'performance_metrics.png'}")
    
    # B. Feature Importance
    fig, axes = plt.subplots(3, 1, figsize=(10, 18))
    
    plot_importance(clf.model_block, clf.config_block.feature_cols, 'Block Layer Features', axes[0])
    plot_importance(clf.model_accuracy, clf.config_accuracy.feature_cols, 'Accuracy Layer Features', axes[1])
    plot_importance(clf.model_finish, clf.config_finish.feature_cols, 'Finish Layer Features', axes[2])
    
    plt.tight_layout()
    plt.savefig(visuals_dir / 'feature_importance.png')
    print(f"Saved feature importance to {visuals_dir / 'feature_importance.png'}")
    
    # Print Metrics
    print("\n" + "="*40)
    print("NESTED MODEL PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Overall AUC:    {roc_auc_score(df['is_goal'], p_overall):.4f}")
    print(f"Overall Brier:  {brier_score_loss(df['is_goal'], p_overall):.4f}")
    print("-" * 20)
    print(f"Block AUC:      {roc_auc_score(df['is_blocked'], p_block):.4f}")
    print(f"Accuracy AUC:   {roc_auc_score(df_unblocked['is_on_net'], p_accuracy):.4f}")
    print(f"Finish AUC:     {roc_auc_score(df_on_net['is_goal'], p_finish):.4f}")
    print("="*40)

if __name__ == "__main__":
    main()
