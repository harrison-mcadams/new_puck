
import numpy as np
import pandas as pd
import xgboost as xgb

def main():
    print(f"XGBoost version: {xgb.__version__}")
    X = pd.DataFrame({
        'dist': [10, 20, 30, 40],
        'type': ['A', 'B', 'A', 'B']
    })
    X['type'] = X['type'].astype('category')
    y = [0, 1, 0, 1]
    
    params = {
        'n_estimators': 10,
        'max_depth': 3,
        'learning_rate': 0.1,
        'enable_categorical': True,
        'tree_method': 'hist',
        'eval_metric': 'logloss'
    }
    
    print("Testing XGBClassifier...")
    clf = xgb.XGBClassifier(**params)
    clf.fit(X, y)
    print("Fit successful!")
    
    print("Predictions:", clf.predict_proba(X)[:, 1])

if __name__ == "__main__":
    main()
