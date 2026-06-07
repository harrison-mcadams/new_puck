"""scripts/optimize_tensor_hyperparameters.py

Rigorous hyperparameter optimization suite for XGBTensorXGClassifier:
1. Loads all 16 available seasons and preprocesses them sequentially.
2. Extracts stratified training sets for Block, Accuracy, and Finish layers.
3. Conducts independent RandomizedSearchCV sweeps for each layer to optimize max_depth, learning_rate, and L1/L2 regularizations.
4. Fits a final model on 100% of all data (2.2M events) with custom layer-specific optimized parameters.
5. Saves final model and refreshes all interactive HTML and performance dashboards.
"""

import sys
import os
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from puck import fit_xgboost_tensor, config, analyze, data_pipeline, features as feature_util, model_summary
from scripts.evaluate_predictive_power import DataUtils

def load_and_preprocess_all_history(seasons):
    """Memory-safe sequential loader and preprocessor for all seasons."""
    dfs = []
    for s in seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            if csv_path:
                print(f"  Loading & Preprocessing season {s}...")
                df_s = pd.read_csv(csv_path, low_memory=False)
                df_s['season'] = int(s)
                
                # Preprocess small chunks sequentially to be extremely memory safe
                df_proc = data_pipeline.preprocess_features(
                    df_s, 
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
                dfs.append(df_proc)
                print(f"    [OK] Preprocessed {len(df_proc)} events.")
        except Exception as e:
            print(f"    [Error] Failed to load/preprocess season {s}: {e}")
            
    if not dfs:
        raise FileNotFoundError("No data loaded successfully!")
        
    print("\n  Concatenating all seasons into a single historical corpus...")
    df_full = pd.concat(dfs, ignore_index=True)
    print(f"  Historical Corpus Concatenated. Total rows: {len(df_full)}")
    return df_full

def run_layer_tuning(df, target, name, feature_list):
    """Performs stratified hyperparameter search for a specific nested layer."""
    print(f"\n============================================================")
    print(f"TUNING HYPERPARAMETERS FOR LAYER: '{name.upper()}'")
    print(f"============================================================")
    
    # Cast categories correctly so XGBoost native categorical support splits optimally
    X_df = df[feature_list].copy()
    for col in feature_list:
        if col in fit_xgboost_tensor.CATEGORICAL_VOCABS or X_df[col].dtype == 'object':
            X_df[col] = X_df[col].astype('category')
            
    # Sample if too big (limit to 50k stratified samples for stable probability search)
    tuning_sample_size = 50000
    if len(X_df) > tuning_sample_size:
        print(f"  Stratifying dataset to {tuning_sample_size} samples for swift, robust search...")
        X_s, _, y_s, _ = train_test_split(
            X_df, target, 
            train_size=tuning_sample_size, 
            stratify=target, 
            random_state=42
        )
    else:
        X_s, y_s = X_df, target

    # Tuning Search Space (focused on depth, regularizations, and learning rate)
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [4, 5, 6, 7, 8],
        'learning_rate': [0.02, 0.03, 0.05, 0.07, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0.0, 1.0, 3.0, 5.0],
        'reg_lambda': [5.0, 15.0, 30.0],
        'gamma': [1.0, 3.0, 5.0, 7.0],
        'min_child_weight': [100, 300, 500, 750, 1000]
    }
    
    # Establish base score for convergence speed
    base_scores = {
        'block': 0.26,
        'accuracy': 0.70,
        'finish': 0.10
    }
    b_score = base_scores.get(name, 0.5)

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        device='cpu',
        enable_categorical=True,
        base_score=b_score,
        random_state=42,
        n_jobs=1
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=15,
        scoring='neg_log_loss',  # Optimize specifically for probability quality
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=4  # Run parallel across cores
    )
    
    start_t = time.time()
    search.fit(X_s, y_s)
    print(f"  Tuning took {time.time() - start_t:.1f}s.")
    print(f"  Best LogLoss for {name}: {-search.best_score_:.5f}")
    print(f"  Best Parameters: {search.best_params_}")
    
    return search.best_params_

def main():
    print("============================================================")
    print("NHL expected GOALS TENSOR MODEL ADVANCED OPTIMIZATION SUITE")
    print("============================================================")
    
    # 1. Discover and load all history
    all_seasons = DataUtils.get_available_seasons()
    print(f"Available History: {len(all_seasons)} seasons: {all_seasons}")
    
    print("\n--- 1/4: Loading and Preprocessing Historical Corpus ---")
    df = load_and_preprocess_all_history(all_seasons)
    
    # 2. Get Features list
    feature_list = feature_util.get_features('all_inclusive')
    if 'season' not in feature_list:
        feature_list.append('season')
        
    # 3. Layer Tuning
    print("\n--- 2/4: Tuning Individual Submodel Layer Hyperparameters ---")
    
    # A. Block Model (All shots)
    y_block = (df['event'] == 'blocked-shot').astype(int)
    best_block_params = run_layer_tuning(df, y_block, 'block', feature_list)
    
    # B. Accuracy Model (Unblocked shots)
    df_unblocked = df[df['event'] != 'blocked-shot'].copy()
    y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
    best_acc_params = run_layer_tuning(df_unblocked, y_acc, 'accuracy', feature_list)
    
    # C. Finish Model (Shots on net)
    df_on_net = df[df['event'].isin(['shot-on-goal', 'goal'])].copy()
    y_fin = (df_on_net['event'] == 'goal').astype(int)
    best_finish_params = run_layer_tuning(df_on_net, y_fin, 'finish', feature_list)
    
    # Save optimized parameters
    optimized_params = {
        'block': best_block_params,
        'accuracy': best_acc_params,
        'finish': best_finish_params
    }
    
    xgs_dir = Path(config.ANALYSIS_DIR) / 'xgs'
    xgs_dir.mkdir(parents=True, exist_ok=True)
    with open(xgs_dir / 'xgboost_tensor_optimized_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=4)
    print(f"\n  [OK] Optimized hyperparameters saved to {xgs_dir / 'xgboost_tensor_optimized_params.json'}")
    
    # 4. Final Model Training on 100% of Data
    print("\n--- 3/4: Training Final Production Model on 100% of Historical Corpus ---")
    
    # Construct final model with layer_params overrides
    final_clf = fit_xgboost_tensor.XGBTensorXGClassifier(
        features=feature_util.get_features('all_inclusive'),
        n_estimators=3000,  # Max capacity production tree depth
        max_depth=6,        # Backed up by layer_params overrides
        learning_rate=0.05,
        use_calibration=False,
        use_balancing=False,
        use_splines=True,
        season_mode='numerical',
        layer_params=optimized_params
    )
    
    start_t = time.time()
    final_clf.fit(df)
    print(f"  Final production fit completed in {time.time() - start_t:.1f}s.")
    
    # Save optimal final model
    final_save_path = xgs_dir / 'xg_model_xgboost_tensor_final.joblib'
    joblib.dump(final_clf, final_save_path)
    print(f"  [OK] Final production-ready model saved to: {final_save_path}")
    
    # Write metadata
    meta = {
        'final_features': final_clf.features,
        'model_type': 'xgboost_tensor',
        'season_mode': final_clf.season_mode,
        'optimized_hyperparameters': optimized_params,
        'train_params': {
            'n_estimators': final_clf.n_estimators,
            'max_depth': final_clf.max_depth,
            'learning_rate': final_clf.learning_rate
        }
    }
    with open(str(final_save_path) + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=4)
        
    # 5. Refresh Dashboards
    print("\n--- 4/4: Refreshing Performance and Interactive Dashboards ---")
    
    # A. Run season-by-season performance dashboard
    print("  Refreshing year-by-year performance dashboard...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, 'scripts/evaluate_season_performance.py'],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("    [OK] Year-by-year performance dashboard refreshed.")
        else:
            print(f"    [Warning] Failed to refresh performance dashboard: {result.stderr[:300]}")
    except Exception as e:
        print(f"    [Error] Failed to refresh performance dashboard: {e}")
        
    # B. Run model summary to refresh interactive HTML dashboard
    print("  Refreshing interactive HTML dashboard...")
    try:
        # Load a small 10% test slice of concatenated data for summary charts
        _, df_test = train_test_split(df, test_size=0.1, random_state=42)
        model_summary.generate_model_summary(
            model_path=str(final_save_path),
            test_df=df_test,
            output_dir=str(Path(config.ANALYSIS_DIR) / 'xgboost_tensor_xgs_modern'),
            verbose=True
        )
        print("    [OK] Interactive HTML dashboard refreshed.")
    except Exception as e:
        print(f"    [Error] Failed to refresh interactive HTML dashboard: {e}")
        
    print("\n============================================================")
    print("OPTIMIZATION & PRODUCTION TRAINING COMPLETE!")
    print("============================================================")

if __name__ == "__main__":
    main()
