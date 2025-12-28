
import pandas as pd
import numpy as np
import sys
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_xgs, features

def run_tuning(X, y, name):
    print(f"\n>>> Tuning {name} (N={len(X)}, Positive={y.sum()/len(y):.1%})")
    
    if len(y.unique()) < 2:
        print(f"Skipping {name}: Only one class present in target.")
        return None

    param_dist = {
        'n_estimators': [200, 500],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [5, 10, 20, 50],
        'min_samples_leaf': [1, 5, 10, 20],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=1)
    
    # Sample if too big (limit to 30k for maximum stability)
    tuning_sample_size = 30000
    if len(X) > tuning_sample_size:
        X_s, _, y_s, _ = train_test_split(X, y, train_size=tuning_sample_size, stratify=y, random_state=42)
    else:
        X_s, y_s = X, y

    # RandomizedSearchCV (Lightweight Mode)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=10, 
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=1  # Total stability
    )
    
    search.fit(X_s, y_s)
    print(f"Best Params for {name}: {search.best_params_}")
    print(f"Best AUC: {search.best_score_:.4f}")
    return search.best_params_

def save_intermediate_results(results):
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)

def optimize_all():
    print("Loading data...")
    df = fit_xgs.load_all_seasons_data()
    
    # Define Targets BEFORE cleaning drops the 'event' column
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    df['is_goal_target'] = (df['event'] == 'goal').astype(int)

    feature_list = features.get_features('all_inclusive')
    print(f"Features: {feature_list}")
    
    # Clean the dataframe
    print("Cleaning base dataframe...")
    extra_cols = ['is_blocked', 'is_on_net', 'is_goal_target']
    df_clean, final_features, _ = fit_xgs.clean_df_for_model(df, feature_list + extra_cols, encode_method='integer')
    
    model_features = [f for f in final_features if f not in extra_cols]
    
    results = {}
    if os.path.exists('optimization_results.json'):
        with open('optimization_results.json', 'r') as f:
            try:
                results = json.load(f)
                print(f"Loaded existing results: {list(results.keys())}")
            except:
                pass

    # 1. Block Layer (Authentic features)
    if 'Nested_Block' not in results:
        block_features = [f for f in model_features if not f.startswith('shot_type')]
        results['Nested_Block'] = run_tuning(df_clean[block_features], df_clean['is_blocked'], "Nested_Block")
        save_intermediate_results(results)
    else:
        print("\n>>> Skipping Nested_Block (Already present in optimization_results.json)")
    
    # 2. Accuracy Layer (Unblocked only)
    if 'Nested_Accuracy' not in results:
        df_unblocked = df_clean[df_clean['is_blocked'] == 0]
        results['Nested_Accuracy'] = run_tuning(df_unblocked[model_features], df_unblocked['is_on_net'], "Nested_Accuracy")
        save_intermediate_results(results)
    else:
        print("\n>>> Skipping Nested_Accuracy (Already present in optimization_results.json)")
    
    # 3. Finish Layer (On-net only)
    if 'Nested_Finish' not in results:
        df_on_net = df_clean[df_clean['is_on_net'] == 1]
        results['Nested_Finish'] = run_tuning(df_on_net[model_features], df_on_net['is_goal_target'], "Nested_Finish")
        save_intermediate_results(results)
    else:
        print("\n>>> Skipping Nested_Finish (Already present in optimization_results.json)")
    
    # Save Results
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll tuning complete. Results saved to optimization_results.json")

if __name__ == "__main__":
    optimize_all()
