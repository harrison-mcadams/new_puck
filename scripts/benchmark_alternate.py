"""benchmark_alternate.py

Script to compare the Nested GLM Spatial XGBoost vs. Pure XGBoost Spatial Alternate Model.
"""

import sys
import os
from pathlib import Path
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# Add project root to sys.path if running from child dir
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, config
from puck.fit_xgboost_nested import XGBNestedXGClassifier
from puck.fit_xgboost_alternate import XGBAlternateXGClassifier

def main():
    print("Loading recent data slice for benchmarking...")
    # Load 2023-2024 data as a benchmark slice
    data_path = Path(config.DATA_DIR) / '20232024.csv'
    if not data_path.exists():
        print("Data could not be loaded. Please ensure 20232024.csv is present.")
        return
    df_raw = pd.read_csv(data_path)
        
    print(f"Loaded {len(df_raw)} records. Preprocessing...")
    
    # Simple preprocessing
    df = data_pipeline.preprocess_features(
        df_raw, 
        is_training=True, 
        verbose=False,
        apply_filtering=True
    )
    
    # Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    y_test_goal = (df_test['event'] == 'goal').astype(int)
    
    print(f"\nEvaluating on {len(df_train)} train rows, {len(df_test)} test rows.")
    
    # 1. Baseline Nested Model
    print("\n--- Training Nested GLM XGBoost (Default) ---")
    start_nested = time.time()
    model_nested = XGBNestedXGClassifier(use_calibration=False)
    model_nested.fit(df_train)
    time_nested = time.time() - start_nested
    
    probs_nested = model_nested.predict_proba(df_test)[:, 1]
    auc_nested = roc_auc_score(y_test_goal, probs_nested)
    ll_nested = log_loss(y_test_goal, probs_nested)
    brier_nested = brier_score_loss(y_test_goal, probs_nested)
    
    # 2. Alternate Pure XGBoost Model
    print("\n--- Training Pure Spatial XGBoost (Alternate) ---")
    start_alt = time.time()
    model_alt = XGBAlternateXGClassifier(use_calibration=False)
    model_alt.fit(df_train)
    time_alt = time.time() - start_alt
    
    probs_alt = model_alt.predict_proba(df_test)[:, 1]
    auc_alt = roc_auc_score(y_test_goal, probs_alt)
    ll_alt = log_loss(y_test_goal, probs_alt)
    brier_alt = brier_score_loss(y_test_goal, probs_alt)
    
    # Results Summary
    print("\n==============================================")
    print("             BENCHMARK RESULTS")
    print("==============================================")
    print(f"{'Metric':<12} | {'Nested GLM (Old)':<20} | {'Pure XGB (New)':<20}")
    print("-" * 58)
    print(f"{'Training Time':<12} | {time_nested:<16.2f} sec | {time_alt:<16.2f} sec")
    print(f"{'AUC':<12} | {auc_nested:<20.4f} | {auc_alt:<20.4f}")
    print(f"{'LogLoss':<12} | {ll_nested:<20.4f} | {ll_alt:<20.4f}")
    print(f"{'Brier':<12} | {brier_nested:<20.6f} | {brier_alt:<20.6f}")
    print("==============================================")
    
if __name__ == "__main__":
    main()
