import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from puck import analyze, data_pipeline

def test_raw_xgb():
    print("Loading data...")
    df = data_pipeline.preprocess_features(pd.read_csv(analyze.locate_season_csv('20232024')), apply_filtering=True)
    
    y = (df['event'] == 'blocked-shot').astype(int)
    X = df[['distance', 'angle_deg', 'shot_type', 'shooter_role']].copy()
    
    for col in ['shot_type', 'shooter_role']:
        X[col] = X[col].astype('category')
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} rows. Target mean: {y_train.mean():.4f}")
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=4, enable_categorical=True, tree_method='hist')
    model.fit(X_train, y_train)
    
    probs = model.predict_proba(X_test)[:, 1]
    print(f"Test Mean Prob: {probs.mean():.4f}")
    print(f"Test Max Prob:  {probs.max():.4f}")
    
    # Check OWEN TIPPETT point
    test_pt = pd.DataFrame([{
        'distance': 29.0, 'angle_deg': 0.0,
        'shot_type': 'wrist', 'shooter_role': 'F'
    }])
    for col in ['shot_type', 'shooter_role']:
        test_pt[col] = pd.Categorical(test_pt[col], categories=X[col].cat.categories)
        
    p = model.predict_proba(test_pt)[0, 1]
    print(f"Owen Tippett P(Block): {p:.4f}")

if __name__ == "__main__":
    test_raw_xgb()
