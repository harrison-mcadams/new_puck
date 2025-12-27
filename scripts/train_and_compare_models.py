
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
    
    from puck import features as puck_features

    # 1. Define Experiments
    experiments = [
        {'name': 'Single_Minimal', 'type': 'single', 'feature_set': 'minimal', 'color': 'gray'},
        {'name': 'Single_Standard', 'type': 'single', 'feature_set': 'standard', 'color': 'blue'},
        {'name': 'Single_All', 'type': 'single', 'feature_set': 'all_inclusive', 'color': 'cyan'},
        {'name': 'Nested_Standard', 'type': 'nested', 'feature_set': 'standard', 'color': 'purple'},
        {'name': 'Nested_All', 'type': 'nested', 'feature_set': 'all_inclusive', 'color': 'magenta'},
    ]

    # Load Data Once
    print("\nLoading all seasons data...")
    df_raw = fit_xgs.load_all_seasons_data()
    print(f"Loaded {len(df_raw)} rows.")
    
    out_dir = Path(puck_config.ANALYSIS_DIR) / 'xgs'
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for exp in experiments:
        name = exp['name']
        mtype = exp['type']
        fset = exp['feature_set']
        
        print(f"\n>>> Running Experiment: {name} (Type: {mtype}, Features: {fset})")
        
        features = puck_features.get_features(fset)
        
        model_filename = f"xg_model_{name.lower()}.joblib"
        model_path = out_dir / model_filename
        
        if mtype == 'single':
            # Train Single model
            # Use get_clf with 'train' behavior to standardize
            clf, final_feats, cat_map, meta = fit_xgs.get_clf(
                out_path=str(model_path),
                behavior='train',
                model_type='single',
                features=features,
                feature_set_name=fset,
                data_df=df_raw
            )
            exp['clf'] = clf
            exp['final_features'] = final_feats
            exp['cat_map'] = cat_map
            exp['meta'] = meta
            
            # DEBUG: Print Top Features
            print(f"   [DEBUG] Top 5 Features for {exp['name']}:")
            importances = clf.clf.feature_importances_
            top_indices = np.argsort(importances)[-5:][::-1]
            for idx in top_indices:
                print(f"     - {final_feats[idx]}: {importances[idx]:.4f}")
            print("")
        else:
            # Train Nested model
            # Preprocess (Standard exclusions)
            df_nested_input = fit_nested_xgs.preprocess_features(df_raw.copy())
            
            # Impute
            try:
                from puck import impute
            except ImportError:
                import impute
            
            df_nested_imputed = impute.impute_blocked_shot_origins(df_nested_input, method='mean_6')
            
            # Initialize & Fit
            clf_nested = fit_nested_xgs.NestedXGClassifier(
                n_estimators=200, 
                max_depth=10, 
                features=features,
                feature_set_name=fset
            )
            clf_nested.fit(df_nested_imputed)
            
            # Save
            joblib.dump(clf_nested, model_path)
            
            # Save Meta
            meta = {
                'model_type': 'nested',
                'feature_set_name': fset,
                'raw_features': features,
                'final_features': clf_nested.final_features,
                'imputation': 'mean_6'
            }
            with open(str(model_path) + '.meta.json', 'w') as f:
                json.dump(meta, f)
            
            exp['clf'] = clf_nested
            exp['final_features'] = clf_nested.final_features
            exp['meta'] = meta

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
    
    # 2. Prepare Evaluation Data
    print(f"Preparing evaluation data...")
    df_eval_raw = df_raw.copy()
    df_eval_raw['is_goal'] = (df_eval_raw['event'] == 'goal').astype(int)
    _, df_test_raw = train_test_split(df_eval_raw, test_size=0.2, random_state=42, stratify=df_eval_raw['is_goal'])
    
    # Standard exclusions for fair comparison
    valid_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    mask_eval = df_test_raw['event'].isin(valid_events)
    if 'is_net_empty' in df_test_raw.columns:
        mask_eval &= (df_test_raw['is_net_empty'] != 1)
    if 'game_state' in df_test_raw.columns:
        mask_eval &= ~df_test_raw['game_state'].isin(['1v0', '0v1'])
    
    df_test = df_test_raw[mask_eval].copy()
    print(f"Test set size: {len(df_test)}")

    # Plotting Setup
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_roc, ax_cal = axes[0], axes[1]
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title('ROC Curves (Comparative)')
    ax_cal.set_title('Calibration Curves')

    summary_stats = []

    # Evaluate Each Experiment
    for exp in experiments:
        name = exp['name']
        clf = exp['clf']
        mtype = exp['type']
        color = exp['color']
        
        print(f"Evaluating {name}...")
        
        if mtype == 'single':
            # Single model expects encoded DF
            df_e, final_feats, _ = fit_xgs.clean_df_for_model(
                df_test.copy(), 
                exp['meta']['raw_features'], 
                fixed_categorical_levels=exp['cat_map']
            )
            y_test = df_e['is_goal'].values
            p_test = clf.predict_proba(df_e[final_feats])[:, 1]
        else:
            # Nested model handles raw input (with imputation)
            from puck import impute
            df_imp = impute.impute_blocked_shot_origins(df_test, method='mean_6')
            y_test = (df_imp['event'] == 'goal').astype(int).values
            p_test = clf.predict_proba(df_imp)[:, 1]
        
        # Metrics
        fpr, tpr, _ = roc_curve(y_test, p_test)
        score_auc = auc(fpr, tpr)
        score_brier = brier_score_loss(y_test, p_test)
        
        results.append({
            'name': name,
            'auc': score_auc,
            'brier': score_brier,
            'type': mtype
        })
        
        # Plot
        ax_roc.plot(fpr, tpr, color=color, label=f'{name} (AUC={score_auc:.3f})')
        prob_true, prob_pred = calibration_curve(y_test, p_test, n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='.', color=color, label=name)

    ax_roc.legend()
    ax_cal.legend()
    
    comp_img = out_dir / 'feature_set_comparison.png'
    plt.savefig(comp_img)
    plt.close()
    print(f"\nSaved comparison plot to {comp_img}")

    # Display Summary Table
    print("\n" + "="*50)
    print(f"{'Model Name':<20} | {'Type':<8} | {'AUC':<8} | {'Brier':<8}")
    print("-" * 50)
    for res in results:
        print(f"{res['name']:<20} | {res['type']:<8} | {res['auc']:.4f} | {res['brier']:.4f}")
    print("="*50)

    # Importance Plots (Carousel style - one plot per important model)
    print("\nGenerating importance plots...")
    for exp in experiments:
        name = exp['name']
        clf = exp['clf']
        mtype = exp['type']
        
        if mtype == 'single':
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_importance(clf.clf, exp['final_features'], f'Importance: {name}', ax)
            plt.tight_layout()
            plt.savefig(out_dir / f'importance_{name.lower()}.png')
            plt.close()
        else:
            # Nested has multiple layers
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            plot_importance(clf.model_block, clf.config_block.feature_cols, f'{name}: Block', axes[0])
            plot_importance(clf.model_accuracy, clf.config_accuracy.feature_cols, f'{name}: Accuracy', axes[1])
            plot_importance(clf.model_finish, clf.config_finish.feature_cols, f'{name}: Finish', axes[2])
            plt.tight_layout()
            plt.savefig(out_dir / f'importance_{name.lower()}.png')
            plt.close()

    print("\nAll experiments complete.")

if __name__ == "__main__":
    main()
