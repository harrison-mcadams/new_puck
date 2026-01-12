import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from puck import features, data_pipeline, fit_xgs
import glob
import json
import joblib

def load_all_seasons_data(data_dir='data'):
    print("Loading all historical data...")
    # Use the robust loader from fit_xgs
    try:
        df = fit_xgs.load_all_seasons_data(base_dir=data_dir)
    except Exception as e:
        print(f"Error loading seasons via fit_xgs: {e}. Falling back to glob...")
        all_files = glob.glob(os.path.join(data_dir, "*", "*_df.csv"))
        df_list = [pd.read_csv(f) for f in all_files]
        df = pd.concat(df_list, ignore_index=True)
    
    print(f"Loaded {len(df)} rows.")
    return df

def optimize_layer(df, target_col, feature_cols, layer_name, param_dist, n_iter=15):
    print(f"\nOptimization for {layer_name} Layer ({len(df)} samples)...")
    
    X = df[feature_cols]
    y = df[target_col]
    
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        enable_categorical=True,
        eval_metric='logloss'
    )
    
    search = RandomizedSearchCV(
        clf, 
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='neg_log_loss',
        cv=3,
        verbose=1,
        n_jobs=1, # Single process for maximum stability on Windows
        random_state=42
    )
    
    search.fit(X, y)
    
    print(f"  Best {layer_name} Params: {search.best_params_}")
    print(f"  Best Score (LogLoss): {-search.best_score_:.4f}")
    
    return search.best_params_

def main():
    df = load_all_seasons_data()
    
    # Use Unified Preprocessing Pipeline (including Mixture Model Smoothing alpha=0.2)
    print("Applying Unified Preprocessing Pipeline (Alpha=0.2)...")
    df = data_pipeline.preprocess_features(
        df, 
        is_training=True, 
        verbose=True, 
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=True,
        apply_filtering=True,
        impute_alpha=0.2
    )

    # Ensure all object/string columns are category for XGBoost
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype('category')

    # Subsampling for optimization speed
    if len(df) > 300000:
        print("Subsampling to 300k rows for optimization speed and stability...")
        df_opt = df.sample(n=300000, random_state=42)
    else:
        df_opt = df
        
    results = {}
    
    # Define Parameter Spaces
    
    # 1. Block Layer: Relaxed now that Mixture Model handles the "Slot Blob"
    param_dist_block = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6],          
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'gamma': [0, 0.5, 1, 2],             # Relaxed from [1, 5, 10]
        'min_child_weight': [5, 10, 20, 50]  # Relaxed from [50, 100, 200]
    }
    
    # 2. General Layer: Standard optimization
    param_dist_default = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.5, 1, 2],
        'min_child_weight': [1, 5, 10]
    }
    
    # --- Optimization ---
    
    # 1. Block Layer
    feats_all = features.get_features('all_inclusive')
    feats_block = [f for f in feats_all if 'shot_type' not in f]
    
    df_opt['is_blocked'] = (df_opt['event'] == 'blocked-shot').astype(int)
    best_block = optimize_layer(df_opt, 'is_blocked', feats_block, 'Block', param_dist_block, n_iter=20)
    results['block'] = best_block
    
    # 2. Accuracy Layer
    df_unblocked = df_opt[df_opt['is_blocked'] == 0].copy()
    df_unblocked['is_on_net'] = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
    best_acc = optimize_layer(df_unblocked, 'is_on_net', feats_all, 'Accuracy', param_dist_default, n_iter=20)
    results['accuracy'] = best_acc
    
    # 3. Finish Layer
    df_on_net = df_unblocked[df_unblocked['is_on_net'] == 1].copy()
    df_on_net['is_goal'] = (df_on_net['event'] == 'goal').astype(int)
    best_finish = optimize_layer(df_on_net, 'is_goal', feats_all, 'Finish', param_dist_default, n_iter=20)
    results['finish'] = best_finish

    
    # SAVE
    out_path = 'analysis/nested_xgs/best_params_xgboost.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved optimized parameters to {out_path}")

if __name__ == "__main__":
    main()
