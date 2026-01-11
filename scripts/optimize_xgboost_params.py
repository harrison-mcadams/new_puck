import sys
import os
# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from puck import features, correction, impute, config as p_conf, fit_xgs
import glob
import json
import joblib

def load_all_seasons_data(data_dir='data'):
    print("Loading all historical data...")
    all_files = glob.glob(os.path.join(data_dir, "*", "*_df.csv"))
    df_list = []
    for f in all_files:
        try:
            temp = pd.read_csv(f)
            df_list.append(temp)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    
    if not df_list:
        raise ValueError("No data found!")
        
    df = pd.concat(df_list, ignore_index=True)
    print(f"Loaded {len(df)} rows.")
    return df

def preprocess_for_opt(df):
    # 1. Fix Attribution
    if 'event' in df.columns and 'blocked-shot' in df['event'].unique():
        print("  Fixing blocked shot attribution...")
        df = correction.fix_blocked_shot_attribution(df)
    
    # Enrich Bios (Shoots/Catches)
    print("  Enriching with player bios...")
    df = fit_xgs.enrich_data_with_bios(df)
        
    # 2. Impute (CDF Mapping)
    print("  Imputing blocked shots (CDF Mapping)...")
    suffix = getattr(p_conf, 'COORDINATE_SUFFIX', '_adj')
    cx, cy = f"x{suffix}", f"y{suffix}"
    use_x, use_y = ('x', 'y')
    if cx in df.columns and cy in df.columns:
        use_x, use_y = cx, cy
    
    try:
        df = impute.impute_blocked_shot_origins(df, method='cdf_mapping', x_col=use_x, y_col=use_y)
    except Exception as e:
        print(f"  Imputation warning: {e}")

    # 3. Basic Targets
    # Is Blocked
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    # Is On Net (Goal or Save)
    # Events: Goal, Shot
    # Missed, Blocked are NOT on net
    # We need strictly regex match usually but simpler here:
    valid_shots = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df = df[df['event'].isin(valid_shots)].copy()
    
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    df['is_goal'] = (df['event'] == 'goal').astype(int)
    
    # 4. Features Cleanup
    # Ensure all object/string columns are category for XGBoost
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = df[col].astype('category')
            except Exception:
                pass # Ignore if fails (e.g. weird types), likely not a feature or handled elsewhere
            
    return df

def optimize_layer(df, target_col, feature_cols, layer_name, n_iter=10):
    print(f"\nOptimization for {layer_name} Layer ({len(df)} samples)...")
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Define Parameter Space
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.5, 1],
        'min_child_weight': [1, 3, 5]
    }
    
    # Use fewer n_jobs to avoid crashing if memory tight, but -1 is usually fine
    clf = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist', # Faster
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
        n_jobs=-1,
        random_state=42
    )
    
    search.fit(X, y)
    
    print(f"  Best {layer_name} Params: {search.best_params_}")
    print(f"  Best Score: {search.best_score_}")
    
    return search.best_params_

def main():
    df = load_all_seasons_data()
    df = preprocess_for_opt(df)
    
    # Use a subset for optimization to speed it up?
    # 3M rows is a lot for RandomizedSearch. Let's sample 500k if larger.
    if len(df) > 500000:
        print("Subsampling to 500k rows for optimization speed...")
        df_opt = df.sample(n=500000, random_state=42)
    else:
        df_opt = df
        
    results = {}
    
    # 1. Block Layer
    # Feature extraction
    # Use 'all_inclusive' as the superset
    feats_all = features.get_features('all_inclusive')
    feats_block = [f for f in feats_all if 'shot_type' not in f]
    
    # Target: is_blocked. Input: All attempts.
    best_block = optimize_layer(df_opt, 'is_blocked', feats_block, 'Block', n_iter=30)
    results['block'] = best_block
    
    # 2. Accuracy Layer
    # Filter: Unblocked Only
    df_unblocked = df_opt[df_opt['is_blocked'] == 0].copy()
    
    # Target: is_on_net. Features: All standard (including shot_type now as it wasn't blocked)
    best_acc = optimize_layer(df_unblocked, 'is_on_net', feats_all, 'Accuracy', n_iter=30)
    results['accuracy'] = best_acc
    
    # 3. Finish Layer
    # Filter: On Net Only
    df_on_net = df_unblocked[df_unblocked['is_on_net'] == 1].copy()
    
    # Target: is_goal. Features: All standard
    best_finish = optimize_layer(df_on_net, 'is_goal', feats_all, 'Finish', n_iter=30)
    results['finish'] = best_finish
    
    # SAVE
    out_path = 'analysis/nested_xgs/best_params_xgboost.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Saved optimized parameters to {out_path}")

if __name__ == "__main__":
    main()
