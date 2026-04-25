import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_alternate

def test_inference():
    # Create dummy model
    model = fit_xgboost_alternate.XGBAlternateXGClassifier(n_estimators=10, max_depth=2, use_splines=True)
    
    # Create dummy data
    data = {
        'x': np.random.uniform(0, 100, 100),
        'y': np.random.uniform(-42.5, 42.5, 100),
        'event': np.random.choice(['shot-on-goal', 'blocked-shot', 'goal'], 100),
        'distance': np.random.uniform(0, 100, 100),
        'angle_deg': np.random.uniform(0, 180, 100),
        'shot_type': np.random.choice(['wrist', 'slap'], 100)
    }
    df = pd.DataFrame(data)
    
    print("Fitting model...")
    model.fit(df)
    
    print("Feature list (acc):", model.features_acc[:5], "...", len(model.features_acc))
    
    print("Running predict_proba...")
    probs = model.predict_proba(df)
    print("Success! First 5 probs:", probs[:5, 1])

if __name__ == "__main__":
    test_inference()
