
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
        'n_estimators': [100, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 10],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Sample if too big (limit to 100k for speed)
    if len(X) > 100000:
        X_s, _, y_s, _ = train_test_split(X, y, train_size=100000, stratify=y, random_state=42)
    else:
        X_s, y_s = X, y

    # RandomizedSearchCV
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=15, 
        scoring='roc_auc',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    search.fit(X_s, y_s)
    print(f"Best Params for {name}: {search.best_params_}")
    print(f"Best AUC: {search.best_score_:.4f}")
    return search.best_params_

def optimize_all():
    print("Loading data...")
    df = fit_xgs.load_all_seasons_data()
    
    # Define Targets BEFORE cleaning drops the 'event' column
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    # is_goal is added by clean_df_for_model, but let's be explicit
    df['is_goal_target'] = (df['event'] == 'goal').astype(int)

    feature_list = features.get_features('all_inclusive')
    print(f"Features: {feature_list}")
    
    # Clean the dataframe
    print("Cleaning base dataframe...")
    # We pass the targets as part of the features so they are preserved
    extra_cols = ['is_blocked', 'is_on_net', 'is_goal_target']
    df_clean, final_features, _ = fit_xgs.clean_df_for_model(df, feature_list + extra_cols, encode_method='integer')
    
    # clean_df_for_model will have encoded is_blocked into is_blocked_code if it's an object, 
    # but it's already int. So they should remain as is in final_features.
    
    # Remove extra_cols from model features
    model_features = [f for f in final_features if f not in extra_cols]
    
    results = {}
    
    # 1. Single Model
    results['Single_All'] = run_tuning(df_clean[model_features], df_clean['is_goal_target'], "Single_All")
    
    # 2. Block Layer
    results['Nested_Block'] = run_tuning(df_clean[model_features], df_clean['is_blocked'], "Nested_Block")
    
    # 3. Accuracy Layer (Unblocked only)
    df_unblocked = df_clean[df_clean['is_blocked'] == 0]
    results['Nested_Accuracy'] = run_tuning(df_unblocked[model_features], df_unblocked['is_on_net'], "Nested_Accuracy")
    
    # 4. Finish Layer (On-net only)
    df_on_net = df_clean[df_clean['is_on_net'] == 1]
    results['Nested_Finish'] = run_tuning(df_on_net[model_features], df_on_net['is_goal_target'], "Nested_Finish")
    
    # Save Results
    with open('optimization_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("\nAll tuning complete. Results saved to optimization_results.json")

if __name__ == "__main__":
    optimize_all()
