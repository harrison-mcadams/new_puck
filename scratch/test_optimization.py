import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck.fit_xgboost_tensor import XGBTensorXGClassifier

def test_optimization():
    print("Generating dummy data...")
    # Create 1000 rows of dummy hockey data
    n = 1000
    data = {
        'x': np.random.uniform(-100, 100, n),
        'y': np.random.uniform(-42, 42, n),
        'distance': np.random.uniform(0, 100, n),
        'angle': np.random.uniform(-90, 90, n),
        'event': np.random.choice(['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'], n),
        'shot_type': np.random.choice(['wrist', 'snap', 'slap'], n),
        'game_state': ['5v5'] * n
    }
    df = pd.DataFrame(data)
    
    print("Initializing XGBTensorXGClassifier...")
    clf = XGBTensorXGClassifier(
        n_estimators=100, # Small for testing
        max_depth=3,
        learning_rate=0.1,
        use_splines=False # Simplify for test
    )
    
    print("Fitting model (this should trigger early stopping logic)...")
    clf.fit(df)
    
    print("Fit successful!")
    
    print("Testing prediction...")
    probs = clf.predict_proba(df[:5])
    print(f"Predictions:\n{probs}")

if __name__ == "__main__":
    test_optimization()
