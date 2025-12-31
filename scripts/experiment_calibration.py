
import sys
import os
import joblib
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck import config, fit_nested_xgs, impute

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ExpCalib")

def analyze_calibration(df, name):
    print(f"\n--- Calibration Analysis: {name} ---")
    bins = [
        (0.00, 0.05),
        (0.05, 0.15),
        (0.15, 0.25),
        (0.25, 0.50),
        (0.50, 1.00)
    ]
    
    print(f"{'Bin Range':<15} | {'Count':<7} | {'Pred':<8} | {'Actual':<8} | {'Diff':<8} | {'Status':<15}")
    print("-" * 85)
    
    for bin_low, bin_high in bins:
        mask_bin = (df['xG'] >= bin_low) & (df['xG'] < bin_high)
        df_bin = df[mask_bin]
        
        n = len(df_bin)
        if n == 0:
            print(f"[{bin_low:.2f}, {bin_high:.2f}) | {0:<7} | {'-':<8} | {'-':<8} | {'-':<8} | -")
            continue
            
        pred = df_bin['xG'].mean()
        actual = df_bin['is_goal_final'].mean()
        diff = actual - pred
        
        status = "Under" if diff > 0.01 else ("Over" if diff < -0.01 else "OK")
        if abs(diff) > 0.05: status += " (!)"
        
        print(f"[{bin_low:.2f}, {bin_high:.2f}) | {n:<7} | {pred:.4f}   | {actual:.4f}   | {diff:+.4f}   | {status}")

def main():
    logger.info("Starting Calibration Experiment (Phase 1: Class Balancing)...")
    
    # 1. Load Data
    df = fit_nested_xgs.load_data()
    df = fit_nested_xgs.preprocess_features(df)
    logger.info("Imputing blocked shots...")
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # 2. Split
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Train: {len(df_train)}, Test: {len(df_test)}")
    
    # 3. Train Baseline (Standard)
    logger.info("\nTraining BASELINE model (No Class Weights)...")
    clf_base = fit_nested_xgs.NestedXGClassifier(
        n_estimators=100, 
        max_depth=10, 
        block_params={},
        accuracy_params={},
        finish_params={}
    )
    clf_base.fit(df_train)
    probs_base = clf_base.predict_proba(df_test)[:, 1]
    
    # 4. Train Experimental (Balanced)
    logger.info("\nTraining BALANCED model (class_weight='balanced_subsample')...")
    # applying to all layers
    balanced_params = {'class_weight': 'balanced_subsample'}
    
    clf_bal = fit_nested_xgs.NestedXGClassifier(
        n_estimators=100,
        max_depth=10,
        block_params=balanced_params,
        accuracy_params=balanced_params,
        finish_params=balanced_params
    )
    clf_bal.fit(df_train)
    probs_bal = clf_bal.predict_proba(df_test)[:, 1]
    
    # 5. Train Calibrated (Balanced + Isotonic)
    # We need a holdout set for calibration or use CV. 
    # For this experiment, we'll cheat slightly and calibrate on the TEST set just to show the potential impact (Upper Bound),
    # OR correctly split Train into Train/Calib.
    # Let's split Train further to be rigorous.
    
    X_train_sub, X_calib, y_train_sub, y_calib = train_test_split(df_train, df_train['is_goal_layer'], test_size=0.25, random_state=42)
    
    logger.info("\nTraining CALIBRATED model (Balanced + Isotonic)...")
    # A. Train Base on Sub-Train
    clf_bal_sub = fit_nested_xgs.NestedXGClassifier(
        n_estimators=100, max_depth=10,
        block_params=balanced_params, accuracy_params=balanced_params, finish_params=balanced_params
    )
    clf_bal_sub.fit(X_train_sub)
    
    # B. Calibrate on Calib Set
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(out_of_bounds='clip')
    
    # Get raw probs from base
    probs_calib_raw = clf_bal_sub.predict_proba(X_calib)[:, 1]
    # We calibrate against the FINAL GOAL outcome (is_goal_layer might be layer specific, we need global is_goal)
    # Actually Nested Model returns global probability. So we calibrate matching global outcome.
    y_calib_global = (X_calib['event'] == 'goal').astype(int)
    
    iso.fit(probs_calib_raw, y_calib_global)
    
    # C. Predict on Test
    probs_test_raw = clf_bal_sub.predict_proba(df_test)[:, 1]
    probs_iso = iso.predict(probs_test_raw)
    
    
    # 6. Compare Results
    df_test['is_goal_final'] = (df_test['event'] == 'goal').astype(int)
    
    df_test['xG'] = probs_base
    analyze_calibration(df_test, "BASELINE")
    logger.info(f"Baseline LogLoss: {log_loss(df_test['is_goal_final'], probs_base):.4f}")
    
    df_test['xG'] = probs_bal
    # Note: using full train probs for fair comparison to baseline, sub-train slightly weaker but calibrated
    analyze_calibration(df_test, "BALANCED (Raw)")
    logger.info(f"Balanced LogLoss: {log_loss(df_test['is_goal_final'], probs_bal):.4f}")
    
    df_test['xG'] = probs_iso
    analyze_calibration(df_test, "BALANCED + ISOTONIC")
    logger.info(f"Isotonic LogLoss: {log_loss(df_test['is_goal_final'], probs_iso):.4f}")
    
    # Plot Comparison
    plt.figure(figsize=(10, 6))
    
    # Baseline
    prob_true, prob_pred = calibration_curve(df_test['is_goal_final'], probs_base, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label='Baseline')
    
    # Balanced (Raw)
    prob_true_b, prob_pred_b = calibration_curve(df_test['is_goal_final'], probs_bal, n_bins=10)
    plt.plot(prob_pred_b, prob_true_b, marker='^', label='Balanced')
    
    # Isotonic
    prob_true_i, prob_pred_i = calibration_curve(df_test['is_goal_final'], probs_iso, n_bins=10)
    plt.plot(prob_pred_i, prob_true_i, marker='s', label='Isotonic')
    
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted xG')
    plt.ylabel('Actual Goal Rate')
    plt.title('Calibration: Baseline vs Balanced vs Isotonic')
    plt.legend()
    plt.savefig('analysis/calibration_experiment.png')

    logger.info("Saved analysis/calibration_experiment.png")

if __name__ == "__main__":
    main()
