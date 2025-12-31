
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import config, fit_nested_xgs

def main():
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested.joblib'
    model = joblib.load(model_path)
    
    print(f"Model: {type(model).__name__}")
    print(f"Use Calibration: {model.use_calibration}")
    print(f"Calibrator: {type(model.calibrator).__name__ if model.calibrator else 'None'}")
    
    if model.calibrator:
        # Check mapping
        test_inputs = np.linspace(0, 1, 20)
        calibrated_outputs = model.calibrator.predict(test_inputs)
        
        print("\nCalibration Mapping (Raw -> Calibrated):")
        for inp, out in zip(test_inputs, calibrated_outputs):
             print(f"  {inp:.4f} -> {out:.4f}")
             
    # Test on a small slice of data
    df = fit_nested_xgs.load_data()
    df = df.sample(10000, random_state=42)
    df = fit_nested_xgs.preprocess_features(df)
    
    # Predict with and without calibration manually
    temp_cal = model.calibrator
    model.calibrator = None
    raw_preds = model.predict_proba(df)[:, 1]
    
    model.calibrator = temp_cal
    cal_preds = model.predict_proba(df)[:, 1]
    
    print("\nSample Statistics (10k shots):")
    print(f"  Mean Raw xG: {np.mean(raw_preds):.4f}")
    print(f"  Mean Cal xG: {np.mean(cal_preds):.4f}")
    print(f"  Actual Rate: {(df['event'] == 'goal').mean():.4f}")
    
    ratio_raw = np.sum(raw_preds) / (df['event'] == 'goal').sum()
    ratio_cal = np.sum(cal_preds) / (df['event'] == 'goal').sum()
    
    print(f"\n  Raw Ratio: {ratio_raw:.4f}")
    print(f"  Cal Ratio: {ratio_cal:.4f}")

if __name__ == "__main__":
    main()
