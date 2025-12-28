"""optimize_xgboost_params.py

Script to optimize hyperparameters for the Nested XGBoost xG Model.
Tunes Block, Accuracy, and Finish layers independently using RandomizedSearchCV.
Leverages GPU if available.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, make_scorer
from xgboost import XGBClassifier
import joblib

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgboost_nested, fit_xgs, features

# --- CONFIGURATION ---
OUTPUT_FILE = Path('analysis/nested_xgs/best_params_xgboost.json')
N_ITER = 20 # Number of parameter combinations to try
CV_FOLDS = 3
RANDOM_STATE = 42
NAN_MASK_RATE = 0.05

# GPU usage?
# User specified they have a graphics card.
# We try to use 'cuda' device.
TREE_METHOD = 'hist'
DEVICE = 'cuda' 

def get_param_dist():
    return {
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [3, 4, 5, 6, 8, 10],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.5],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.01, 0.1, 1],
        'reg_lambda': [1, 1.5, 2, 5]
    }

def optimize_layer(X, y, layer_name, existing_best=None):
    print(f"\n>>> Optimizing {layer_name} Layer (N={len(X)})...")
    
    if existing_best:
        print(f"Skipping {layer_name}: Already found in output file.")
        return existing_best

    # Base Estimator with GPU support
    # Note: enable_categorical=True is critical
    clf = XGBClassifier(
        objective='binary:logistic',
        tree_method=TREE_METHOD, 
        device=DEVICE,
        enable_categorical=True,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )

    # Scorer: LogLoss is primary optimization target for probability calibration
    # But usually RandomizedSearchCV maximizes score, so we use neg_log_loss
    
    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=get_param_dist(),
        n_iter=N_ITER,
        scoring='neg_log_loss', # Minimize LogLoss
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=1 # XGBoost handles parallelism internally (and on GPU)
    )
    
    search.fit(X, y)
    
    print(f"Best Params for {layer_name}:")
    print(search.best_params_)
    print(f"Best Neg LogLoss: {search.best_score_:.4f}")
    
    # Store simple dict
    best = search.best_params_
    best['score'] = float(search.best_score_)
    return best

def main():
    print("--- XGBoost Hyperparameter Optimization ---")
    
    # 1. Load All Data
    print("Loading data...")
    df = fit_xgs.load_data()
    
    # 2. Preprocess
    print("Preprocessing...")
    df = fit_xgboost_nested.preprocess_data(df)
    
    # Reload existing results to allow resuming
    results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, 'r') as f:
            try:
                results = json.load(f)
                print(f"Loaded existing results: {list(results.keys())}")
            except:
                pass

    # Features
    # Note: These match the XGBNestedXGClassifier defaults exactly
    feature_cols = ['distance', 'angle_deg', 'game_state', 'shot_type', 'shoots_catches']
    
    # --- BLOCK LAYER ---
    if 'block' not in results:
        # Exclude shot_type
        feat_block = [f for f in feature_cols if f != 'shot_type']
        
        # Target: is_blocked
        # Filter: All shot attempts
        X_block = df[feat_block]
        y_block = df['is_blocked']
        
        results['block'] = optimize_layer(X_block, y_block, "Block", results.get('block'))
        
        # Save intermediate
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    # --- ACCURACY LAYER ---
    if 'accuracy' not in results:
        # Target: is_on_net
        # Filter: Unblocked Only
        df_unblocked = df[df['is_blocked'] == 0].copy()
        
        # Apply NaN Injection for shot_type
        # This simulates the training conditions
        if NAN_MASK_RATE > 0:
            print(f"Applying NaN mask (rate={NAN_MASK_RATE}) for Accuracy optimization...")
            n_mask = int(len(df_unblocked) * NAN_MASK_RATE)
            rng = np.random.default_rng(RANDOM_STATE)
            mask_idx = rng.choice(df_unblocked.index, size=n_mask, replace=False)
            df_unblocked.loc[mask_idx, 'shot_type'] = np.nan

        X_acc = df_unblocked[feature_cols]
        y_acc = df_unblocked['is_on_net']
        
        results['accuracy'] = optimize_layer(X_acc, y_acc, "Accuracy", results.get('accuracy'))
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)
            
    # --- FINISH LAYER ---
    if 'finish' not in results:
        # Target: is_goal_layer (is_goal)
        # Filter: On Net Only (Unblocked & On Net)
        # We derive On Net from 'event' or use existing flag
        # Accuracy model predicts P(On Net | Unblocked). Finish model is P(Goal | On Net).
        df_on_net = df[df['is_on_net'] == 1].copy()
        
        # Apply NaN Injection
        if NAN_MASK_RATE > 0:
            print(f"Applying NaN mask (rate={NAN_MASK_RATE}) for Finish optimization...")
            n_mask = int(len(df_on_net) * NAN_MASK_RATE)
            rng = np.random.default_rng(RANDOM_STATE)
            mask_idx = rng.choice(df_on_net.index, size=n_mask, replace=False)
            df_on_net.loc[mask_idx, 'shot_type'] = np.nan
            
        X_fin = df_on_net[feature_cols]
        y_fin = df_on_net['is_goal_layer']
        
        results['finish'] = optimize_layer(X_fin, y_fin, "Finish", results.get('finish'))
        
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    print(f"\nOptimization Complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
