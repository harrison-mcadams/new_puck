
import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, fit_nested_xgs, compare_models
from puck import nhl_api # for any shared util if needed
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
    
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd
    import joblib
    import json
    
    # Load Data Once
    print("Loading all seasons data...")
    df = fit_xgs.load_all_seasons_data()
    print(f"Loaded {len(df)} rows.")
    
    # --- MODEL 1: SINGLE (Reference/Robust) ---
    print("Training Single Model (Dist, Angle, State, Net)...")
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
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_nested_input = df[df['event'].isin(valid_events)].copy()
    
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
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    # 1. Split Eval Data (Stratified)
    df_eval_all = df.copy()
    df_eval_all['is_goal'] = (df_eval_all['event'] == 'goal').astype(int)
    _, df_test = train_test_split(df_eval_all, test_size=0.2, random_state=42, stratify=df_eval_all['is_goal'])
    
    # --- EVALUATION ---
    # --- EVALUATION ---
    print(f"Evaluating on {len(df_test)} raw test rows...")
    
    # 1. Filter out non-shots and empty nets manually for alignment base
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    mask_eval = df_test['event'].isin(valid_events)
    if 'is_net_empty' in df_test.columns:
        # Filter is_net_empty is 0, False, or None
        mask_eval &= (df_test['is_net_empty'] != 1) & (df_test['is_net_empty'] != True)
    
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

if __name__ == "__main__":
    main()
