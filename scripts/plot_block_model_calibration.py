
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import features as feature_util, data_pipeline, fit_xgs

def plot_calibration():
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    if not model_path.exists():
        print("Model not found.")
        return
    
    model = joblib.load(model_path)
    
    print("Loading data for calibration check...")
    df = fit_xgs.load_data()
    if df.empty:
        print("No data.")
        return
        
    df_p = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True)
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_eval = df_p[df_p['event'].isin(valid_events)].copy()
    y_true = (df_eval['event'] == 'blocked-shot').astype(int)
    
    print("Predicting Block Prob...")
    # Use the internal method to get calibrated probabilities
    p_block = model.predict_proba_layer(df_eval, layer='block')
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(y_true, p_block, n_bins=10, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Block Model (Nested)')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Ideal')
    plt.title('Calibration Curve: Blocked Shot Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives (Actual Blocks)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/diagnostic_block_calibration.png')
    print("Saved plot to analysis/diagnostic_block_calibration.png")
    
    # Also print the bins
    print("\nCalibration Bins:")
    for pt, pp in zip(prob_true, prob_pred):
        print(f"Pred: {pp:.3f} | Actual: {pt:.3f}")

if __name__ == "__main__":
    plot_calibration()
