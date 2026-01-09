
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_nested, fit_xgs, analyze, config as puck_config, features as feature_util, correction, impute

# Path to the model to evaluate
MODEL_PATH = 'analysis/xgs/xg_model_nested_all.joblib'

def plot_calibration_curve(y_true, y_prob, title, ax):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", label=title)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.set_title(f"Calibration: {title}")
    ax.legend(loc="lower right")

def main():
    print(f"--- Establishing Robust Baseline for Block Model ---")
    
    # Check Model
    if not Path(MODEL_PATH).exists():
        # Fallback
        model_candidate = 'analysis/xgs/xg_model_nested.joblib'
        if Path(model_candidate).exists():
            print(f"Using alternative model: {model_candidate}")
            model_path_final = model_candidate
        else:
            print(f"CRITICAL: No model found at {MODEL_PATH}")
            return
    else:
        model_path_final = MODEL_PATH

    print(f"Loading Model: {model_path_final}...")
    try:
        clf = joblib.load(model_path_final)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load Data
    print("Loading FULL available dataset...")
    # fit_xgs.load_data() usually loads 'data/20252026/20252026_df.csv' or similar
    df = fit_xgs.load_data()
    print(f"Loaded {len(df)} rows.")
    
    # Pre-process Pipeline
    print("Running Pre-processing Pipeline:")
    
    # 1. Attribution
    print("  1. Fixing Attribution...")
    df = correction.fix_blocked_shot_attribution(df)
    
    # 2. Imputation (The New Standard)
    print("  2. Imputing Blocked Shot Origins (Empirical Model + Adj)...")
    
    # Check for arena adjustments
    suffix = getattr(puck_config, 'COORDINATE_SUFFIX', '_adj')
    cx, cy = f"x{suffix}", f"y{suffix}"
    use_x, use_y = 'x', 'y'
    if cx in df.columns and cy in df.columns:
        print(f"     Using Adjusted Coordinates: {cx}, {cy}")
        use_x, use_y = cx, cy
    
    df = impute.impute_blocked_shot_origins(df, method='empirical_model', x_col=use_x, y_col=use_y)
    
    # 3. XGB Preprocess
    print("  3. Preprocessing features...")
    df_prep = fit_xgboost_nested.preprocess_data(df)
    
    # Count Events
    n_blocked = (df_prep['is_blocked'] == 1).sum()
    n_total = len(df_prep)
    print(f"Data Ready: {n_total} events. {n_blocked} Blocked Shots ({n_blocked/n_total:.1%})")
    
    # Evaluate
    print("Evaluating Block Model Layer...")
    try:
        if hasattr(clf, 'predict_proba_layer'):
            p_block = clf.predict_proba_layer(df_prep, 'block')
        else:
            # Fallback if it's not the wrapper class or different version
            # But based on previous checks, it should be.
            print("Model does not support 'predict_proba_layer'. Trying direct access to model_block...")
            if hasattr(clf, 'model_block'):
                # Need to manually select features
                feat_block = clf.config_block.feature_cols
                # Check for missing cols
                missing = [f for f in feat_block if f not in df_prep.columns]
                if missing:
                    print(f"Error: Missing columns for block model: {missing}")
                    # Try to fix by filling?
                    for m in missing: df_prep[m] = np.nan # or sensible default
                
                # We need to ensure categorical types match what model expects
                # This is tricky without the wrapper's _prepare_df logic.
                # Hopefully preprocess_data did enough.
                p_block = clf.model_block.predict_proba(df_prep[feat_block])[:, 1]
            else:
                print("Cannot find block model component.")
                return

        y_true = df_prep['is_blocked']
        
        # Metrics
        auc = roc_auc_score(y_true, p_block)
        brier = brier_score_loss(y_true, p_block)
        ll = log_loss(y_true, p_block)
        
        print(f"\n[BASELINE PERFORMANCE on {n_total} events]")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Log Loss:  {ll:.4f}")
        print(f"  Brier:     {brier:.4f}")
        print(f"  Pred Mean: {p_block.mean():.4f}")
        print(f"  True Mean: {y_true.mean():.4f}")
        print(f"  Bias:      {p_block.mean() - y_true.mean():.4f}")
        
        # Plot Calibration
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_calibration_curve(y_true, p_block, "Blocked Shot Model", ax)
        plot_path = "analysis/nested_xgs/baseline_block_calibration.png"
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        print(f"Calibration plot saved to {plot_path}")
        
        # Dist Plot
        plt.figure(figsize=(10,6))
        plt.hist(p_block[y_true==0], bins=50, alpha=0.5, label='Unblocked', density=True)
        plt.hist(p_block[y_true==1], bins=50, alpha=0.5, label='Blocked', density=True)
        plt.title("Predicted Probability Distribution by Class")
        plt.xlabel("P(Blocked)")
        plt.ylabel("Density")
        plt.legend()
        dist_path = "analysis/nested_xgs/baseline_block_distribution.png"
        plt.savefig(dist_path)
        print(f"Distribution plot saved to {dist_path}")

    except Exception as e:
        print(f"Scoring Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
