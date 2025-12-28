
import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, fit_xgboost_nested, features as feature_util

def run_tuning(X, y, name, features_list):
    print(f"\n>>> Tuning {name} (N={len(X)}, Positive={y.sum()/len(y):.1%})")
    
    # 1. Prepare Data using the same logic as the class
    # We need to cast categoricals
    known_cats = ['game_state', 'shot_type', 'shoots_catches']
    X_node = X.copy()
    for col in features_list:
        if col in known_cats or (col in X_node.columns and X_node[col].dtype == object):
             X_node[col] = X_node[col].astype('category')
             
    # Sample if too big (limit to 50k for speed/stability)
    tuning_sample_size = 50000
    if len(X_node) > tuning_sample_size:
        # Stratified sample
        try:
            X_s, _, y_s, _ = train_test_split(X_node[features_list], y, train_size=tuning_sample_size, stratify=y, random_state=42)
        except ValueError: # e.g. too few samples of one class
             X_s, y_s = X_node[features_list], y
    else:
        X_s, y_s = X_node[features_list], y

    # 2. Define Search Space
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1, 5],
        'reg_lambda': [0, 0.01, 1, 5, 10],
        'gamma': [0, 0.1, 0.5, 1, 5],
        'min_child_weight': [1, 3, 5, 7]
    }

    xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        enable_categorical=True,
        use_label_encoder=False,
        n_jobs=1 # Use parallel via CV or external
    )

    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=20, # Reasonable budget
        scoring='neg_log_loss', # Optimize for probability calibration quality
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=4 
    )
    
    search.fit(X_s, y_s)
    
    print(f"Best Params for {name}: {search.best_params_}")
    print(f"Best LogLoss: {-search.best_score_:.4f}")
    return search.best_params_

def optimize_nested_layers():
    print("Loading all historical data...")
    df_raw = fit_xgs.load_all_seasons_data()
    
    print("Preprocessing for XGBoost...")
    # This filters to shot attempts, removes empty net, etc.
    df = fit_xgboost_nested.preprocess_data(df_raw)
    
    # Define Targets manually if not fully handled by preprocess
    if 'is_blocked' not in df.columns:
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    if 'is_on_net' not in df.columns:
        df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    if 'is_goal_layer' not in df.columns:
        df['is_goal_layer'] = (df['event'] == 'goal').astype(int)

    feature_list = feature_util.get_features('all_inclusive')
    
    # 1. Block Layer (No shot_type)
    block_feats = [f for f in feature_list if 'shot_type' not in f]
    print(f"Features for Block Model: {len(block_feats)}")
    block_params = run_tuning(df, df['is_blocked'], "block", block_feats)
    
    # 2. Accuracy Layer (Unblocked, Target: On Net)
    df_unblocked = df[df['is_blocked'] == 0].copy()
    print(f"Features for Accuracy Model: {len(feature_list)}")
    acc_params = run_tuning(df_unblocked, df_unblocked['is_on_net'], "accuracy", feature_list)
    
    # 3. Finish Layer (On Net, Target: Goal)
    df_on_net = df[df['is_on_net'] == 1].copy()
    finish_params = run_tuning(df_on_net, df_on_net['is_goal_layer'], "finish", feature_list)
    
    # Save Results
    results = {
        'block': block_params,
        'accuracy': acc_params,
        'finish': finish_params
    }
    
    out_path = Path('analysis/nested_xgs/best_params_xgboost.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\nOptimization Complete. Results saved to {out_path}")

if __name__ == "__main__":
    optimize_nested_layers()
