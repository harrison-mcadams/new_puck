
import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import data_pipeline, analyze, config

def test_single_layer_calibration():
    print("Loading 20252026 data...")
    csv_path = analyze.locate_season_csv("20252026")
    df = pd.read_csv(csv_path)
    
    print("Preprocessing...")
    df_std = data_pipeline.preprocess_features(df, is_training=True)
    
    # Target: Goal
    y = (df_std['event'] == 'goal').astype(int)
    
    # Features (simplified list)
    features = [
        'distance', 'angle_deg', 'is_home', 'is_rebound', 'is_rush',
        'shooter_role', 'relative_game_state',
        'last_event_time_diff', 'dist_from_last_event'
    ]
    
    # Ensure they are numeric
    X = df_std[features].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training single XGBoost model on {len(X_train)} rows...")
    model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=6, 
        learning_rate=0.1, 
        tree_method='hist',
        enable_categorical=True
    )
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    probs = model.predict_proba(X_test)[:, 1]
    
    total_xg = probs.sum()
    total_goals = y_test.sum()
    ratio = total_xg / total_goals
    
    print(f"Results:")
    print(f"  Total Shots: {len(X_test)}")
    print(f"  Total Goals: {total_goals}")
    print(f"  Total xG:    {total_xg:.2f}")
    print(f"  Ratio:       {ratio:.4f}")
    print(f"  AUC:         {roc_auc_score(y_test, probs):.4f}")

if __name__ == "__main__":
    test_single_layer_calibration()
