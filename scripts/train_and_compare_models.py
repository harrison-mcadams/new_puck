
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, fit_nested_xgs, compare_models
from puck import nhl_api # for any shared util if needed
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json

# Import Config for valid Data Directory
try:
    from puck import config as puck_config
except ImportError:
    try:
        import config as puck_config
    except ImportError:
        class DummyConfig:
            ANALYSIS_DIR = 'analysis'
        puck_config = DummyConfig()

def plot_importance(model, feature_names, title, ax):
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], align='center', color='skyblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(title)
    ax.set_xlabel('Importance')

def main():
    print("==================================================")
    print("   OPTIMAL xG MODEL TRAINING & COMPARISON SUITE   ")
    print("==================================================")
    
    # 1. Train Single Layer Model (Optimal)
    print("\n[1/3] Training Single Layer Model...")
    # Using fit_xgs defaults but explicit output
    # Note: fit_xgs.main() might be too simple, we might want to call internal functions unique to our optimal config.
    # But fit_xgs.py seems to default to standard config.
    # Let's import the specific training routine if possible, or just subprocess it.
    # Importing is better.
    
    # Define optimal single params
    # Single model usually uses: distance, angle, game_state, is_net_empty.
    # We DO NOT use shot_type in single model typically (or do we? User said "both tuned optimally").
    # If standard single model includes shot_type, it leaks blocked shot info (Unknown -> 0%).
    # "Optimally" for Single usually means: Don't use shot_type OR use it but handle Blocked carefully.
    # Let's stick to the "Baseline" (Dist/Angle/State/Net) for Single, or "With Shot Type" if properly imputed?
    # Nested handles Blocked properly. Single struggles.
    # I will stick to "Baseline Plus" for Single: Dist, Angle, GameState, NetEmpty. 
    # (No Shot Type to avoid leakage, or maybe OHE Shot Type with careful imputation? 
    # The Nested model exists specifically because Single sucks at Shot Type/Blocked interaction.
    # So "Optimal Single" is probably the robust base model.)
    
    # Actually, let's look at what fit_xgs does. It defaults to ['distance', 'angle_deg'].
    # We want valid features.
    
    # We will call fit_xgs.get_clf logic or similar.
    # Actually, better to just reimplement the high-level orchestration here to ensure control.
    
    # Load Data Once
    print("Loading all seasons data...")
    df = fit_xgs.load_all_seasons_data()
    print(f"Loaded {len(df)} rows.")
    
    # --- MODEL 1: SINGLE (Reference/Robust) ---
    print("Training Single Model (Dist, Angle, State)...")
    # Features
    feats_single = ['distance', 'angle_deg', 'game_state']
    
    # Clean/Prep
    # df_single will contain 'is_goal' automatically from clean_df_for_model
    # Capture final_feats_single which has encoded names (e.g. game_state_code instead of game_state)
    df_single, final_feats_single, single_map = fit_xgs.clean_df_for_model(df.copy(), feats_single)
    
    # Train
    clf_single = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42)
    # Target: is_goal (provided by clean_df_for_model)
    y_single = df_single['is_goal'].values
    # Train on FEATURES ONLY
    clf_single.fit(df_single[final_feats_single], y_single)
    
    # Save
    out_dir = Path(puck_config.ANALYSIS_DIR) / 'xgs'
    out_dir.mkdir(parents=True, exist_ok=True)
    path_single = out_dir / 'xg_model_single.joblib'
    joblib.dump(clf_single, path_single)
    
    # Save Meta
    meta_single = {
        'final_features': final_feats_single,
        'categorical_levels_map': single_map,
        'model_type': 'single'
    }
    with open(str(path_single) + '.meta.json', 'w') as f:
        json.dump(meta_single, f)
    print(f"Saved Single Model to {path_single}")


    # --- MODEL 2: NESTED (Advanced) ---
    print("\n[2/3] Training Nested Model (Optimal: OHE, Mean-6 Impute)...")
    # We can reuse fit_nested_xgs.main() logic but adapt it to use the already loaded df?
    # Or just instantiate and fit.
    
    # Filter valid events first (Nested expects only valid attempts)
    # Use the common preprocessing logic to handle exclusions (Empty Net, 1v0/0v1)
    df_nested_input = fit_nested_xgs.preprocess_features(df.copy())
    
    # Impute
    try:
        from puck import impute
    except ImportError:
        import impute
    
    print("Applying Mean-6 Imputation...")
    df_nested_imputed = impute.impute_blocked_shot_origins(df_nested_input, method='mean_6')
    
    # Initialize & Fit
    # Optimal: n_estimators=200, max_depth=10, prevent_overfitting=True
    clf_nested = fit_nested_xgs.NestedXGClassifier(n_estimators=200, max_depth=10, prevent_overfitting=True)
    
    # Fit (handles OHE internally)
    clf_nested.fit(df_nested_imputed)
    
    # Save
    path_nested = out_dir / 'xg_model_nested.joblib'
    joblib.dump(clf_nested, path_nested)
    
    # Save Meta
    meta_nested = {
        'model_type': 'nested',
        'imputation': 'mean_6',
        'final_features': ['distance', 'angle_deg', 'game_state', 'shot_type']
    }
    with open(str(path_nested) + '.meta.json', 'w') as f:
        json.dump(meta_nested, f)
    print(f"Saved Nested Model to {path_nested}")
    
    
    # --- COMPARISON ---
    print("\n[3/3] Generating Comparison Dashboard...")
    # We can invoke compare_models.main() but we might need to point it to the valid files if it hardcodes paths.
    # puck/compare_models.py defines 'models_to_load' in main().
    # It looks for:
    # ('Baseline', 'analysis/xgs/xg_model.joblib', 'blue'),
    # ('Nested xG', 'analysis/nested_xgs/nested_xg_model.joblib', 'purple')
    
    # We used:
    # Single -> analysis/xgs/xg_model_single.joblib
    # Nested -> analysis/xgs/xg_model_nested.joblib
    
    # We should update compare_models.py OR just implement the plotting here.
    # Implementation here is safer and avoids updating another file yet.
    # But reusing code is better.
    # I'll perform the comparison logic here using the loaded models directly (in-memory)!
    # Fast and robust.
    
    # 1. Split Eval Data (Stratified)
    df_eval_all = df.copy()
    df_eval_all['is_goal'] = (df_eval_all['event'] == 'goal').astype(int)
    _, df_test = train_test_split(df_eval_all, test_size=0.2, random_state=42, stratify=df_eval_all['is_goal'])
    
    # --- EVALUATION ---
    # --- EVALUATION ---
    print(f"Evaluating on {len(df_test)} raw test rows...")
    
    # 1. Filter out non-shots, empty nets, and extreme game states manually for alignment base
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    mask_eval = df_test['event'].isin(valid_events)
    
    # Empty Net Filter
    if 'is_net_empty' in df_test.columns:
        mask_eval &= (df_test['is_net_empty'] != 1) & (df_test['is_net_empty'] != True)
    
    # 1v0/0v1 Filter
    if 'game_state' in df_test.columns:
        mask_eval &= ~df_test['game_state'].isin(['1v0', '0v1'])
        
    # Regular Season Filter
    if 'game_id' in df_test.columns:
        df_test['game_id_str'] = df_test['game_id'].astype(str)
        mask_eval &= (df_test['game_id_str'].str.len() >= 6) & (df_test['game_id_str'].str[4:6] == '02')
    
    df_eval_base = df_test[mask_eval].copy()
    
    # 2. Prepare Single Model Eval
    # clean_df_for_model will handle encoding correctly (integer mode)
    df_single_eval, _, _ = fit_xgs.clean_df_for_model(df_eval_base.copy(), feats_single, fixed_categorical_levels=single_map)
    # y_true comes from the cleaned subset (in case NaNs were dropped)
    y_true = df_single_eval['is_goal'].values
    # Predict using the features determined at training time
    p_single = clf_single.predict_proba(df_single_eval[final_feats_single])[:, 1]
    
    print(f"Evaluation subset size: {len(df_single_eval)}")

    # 3. Prepare Nested Model Eval (on SAME indices)
    df_nested_eval = df_eval_base.loc[df_single_eval.index].copy()
    # Impute
    df_nested_imputed = impute.impute_blocked_shot_origins(df_nested_eval, method='mean_6')
    p_nested = clf_nested.predict_proba(df_nested_imputed)[:, 1]
    
    # Unblocked mask (must be same length as y_true)
    mask_unblocked = (df_nested_eval['event'] != 'blocked-shot').values
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_roc = axes[0]
    ax_cal = axes[1]
    
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title('ROC Curves')
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_cal.set_title('Calibration Curves')
    
    def plot_perf(y, p, name, col):
        fpr, tpr, _ = roc_curve(y, p)
        area = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=col, label=f'{name} (AUC={area:.3f})')
        
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='.', color=col, label=name)
        
    plot_perf(y_true, p_single, 'Single (Available)', 'blue')
    plot_perf(y_true, p_nested, 'Nested (Optimal)', 'purple')
    
    if mask_unblocked.any():
        y_ub = y_true[mask_unblocked]
        p_s_ub = p_single[mask_unblocked]
        p_n_ub = p_nested[mask_unblocked]
        
        # Helper for dotted
        def plot_perf_dot(y, p, col):
            fpr, tpr, _ = roc_curve(y, p)
            area = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, color=col, linestyle=':', label=f'Unblocked (AUC={area:.3f})')
            prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
            ax_cal.plot(prob_pred, prob_true, marker=None, linestyle=':', color=col)

        plot_perf_dot(y_ub, p_s_ub, 'blue')
        plot_perf_dot(y_ub, p_n_ub, 'purple')

    ax_roc.legend()
    ax_cal.legend()
    
    out_img = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'model_comparison_optimized.png')
    plt.savefig(out_img)
    print(f"Saved comparison to {out_img}")
    
    print("\nDone. Both models trained and saved.")
    print(f"Single: {path_single}")
    print(f"Nested: {path_nested}")

    # --- EXTRA PLOTS: FEATURE IMPORTANCE ---
    print("\n[Extra] Generating Feature Importance Comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_importance(clf_single, final_feats_single, 'Single Model Importance', axes[0])
    # For nested, we'll show Finish Layer (most critical for goal prediction)
    plot_importance(clf_nested.model_finish, clf_nested.config_finish.feature_cols, 'Nested Model (Finish Layer) Importance', axes[1])
    plt.tight_layout()
    importance_img = out_dir / 'feature_importance_comparison.png'
    plt.savefig(importance_img)
    plt.close()
    print(f"Saved feature importance comparison to {importance_img}")

    # --- EXTRA PLOTS: NESTED LAYER PERFORMANCE ---
    print("[Extra] Generating Nested Layer Performance Diagnostics...")
    # Get layer probabilities
    p_block = clf_nested.predict_proba_layer(df_nested_eval, 'block')
    
    mask_unblocked_eval = (df_nested_eval['event'] != 'blocked-shot')
    df_acc_eval = df_nested_eval[mask_unblocked_eval]
    p_acc = clf_nested.predict_proba_layer(df_acc_eval, 'accuracy')
    
    mask_on_net_eval = df_nested_eval['event'].isin(['shot-on-goal', 'goal'])
    df_fin_eval = df_nested_eval[mask_on_net_eval]
    p_fin = clf_nested.predict_proba_layer(df_fin_eval, 'finish')

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Layer ROC
    def plot_roc_layer(y, p, name, col, ax):
        fpr, tpr, _ = roc_curve(y, p)
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=col, label=f'{name} (AUC={area:.3f})')

    plot_roc_layer((df_nested_eval['event'] == 'blocked-shot').astype(int), p_block, 'Block Layer', 'red', axes[0])
    plot_roc_layer(df_acc_eval['event'].isin(['shot-on-goal', 'goal']).astype(int), p_acc, 'Accuracy Layer', 'blue', axes[0])
    plot_roc_layer((df_fin_eval['event'] == 'goal').astype(int), p_fin, 'Finish Layer', 'green', axes[0])
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[0].set_title('Nested Model Layer ROCs')
    axes[0].legend()

    # Layer Calibration
    def plot_cal_layer(y, p, name, col, ax):
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='.', color=col, label=name)

    plot_cal_layer((df_nested_eval['event'] == 'blocked-shot').astype(int), p_block, 'Block Layer', 'red', axes[1])
    plot_cal_layer(df_acc_eval['event'].isin(['shot-on-goal', 'goal']).astype(int), p_acc, 'Accuracy Layer', 'blue', axes[1])
    plot_cal_layer((df_fin_eval['event'] == 'goal').astype(int), p_fin, 'Finish Layer', 'green', axes[1])
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_title('Nested Model Layer Calibration')
    axes[1].legend()

    plt.tight_layout()
    nested_diag_img = out_dir / 'nested_layer_performance.png'
    plt.savefig(nested_diag_img)
    plt.close()
    print(f"Saved nested diagnostics to {nested_diag_img}")

if __name__ == "__main__":
    main()
