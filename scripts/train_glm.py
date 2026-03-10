"""train_glm.py

Script to train the Non-Nested Polynomial Logistic Regression (GLM) xG model.
Used for structural comparison against the Nested GLM.
"""

# %%
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
import joblib
import json
import logging
import time

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_glm, fit_xgs, analyze, config as puck_config, features as feature_util, data_pipeline, moneypuck, model_summary

def plot_calib(y_true, y_prob, name, ax):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    ax.set_title(name)
    ax.legend()
    ax.grid(True, alpha=0.3)

print("Imports complete.")

# %%
print("--- Training Non-Nested GLM (Poly) Model ---")

# 1. Load Data
print("Loading data from ALL seasons...")
try:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    if data_dir.exists():
        df = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    else:
        df = fit_xgs.load_data()
except Exception as e:
    print(f"Error loading all seasons: {e}")
    df = fit_xgs.load_data()

print(f"Loaded {len(df)} rows.")

# 2. Unified Preprocessing Pipeline
print("Applying Unified Preprocessing Pipeline (excluding blocked shots)...")
try:
    df = data_pipeline.preprocess_features(
        df, 
        is_training=True, 
        verbose=True, 
        apply_arena_adjustments=True,
        apply_imputation=False, # We don't need imputation if blocked are excluded
        apply_dithering=True,
        apply_filtering=True,
        impute_alpha=0.2, # Irrelevant here
        exclude_blocked=True
    )
except Exception as e:
    print(f"Pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# 4. Train
print(f"Training on {len(df_train)} rows...")
feature_list = feature_util.get_features('all_inclusive')
print(f"Using {len(feature_list)} features.")

# Initialize GLM
clf = fit_glm.NonNestedGLM(
    features=feature_list,
    poly_degree=2,      # Ignored if splines are active, but safe default
    use_splines=True,   # ENABLE SPLINES
    enable_marginalization=True
)

start_time = time.time()
clf.fit(df_train)
print(f"Training took {time.time() - start_time:.1f} seconds.")

# 5. Evaluate
print("\n--- Evaluation (Test Set) ---")
y_test_goal = (df_test['event'] == 'goal').astype(int)
probs = clf.predict_proba(df_test)[:, 1]

# Predictions to DF for Analysis
df_test = df_test.copy()
df_test['xG'] = probs

# Metrics
auc = roc_auc_score(y_test_goal, probs)
ll = log_loss(y_test_goal, probs)
print(f"Overall xG AUC:     {auc:.4f}")
print(f"Overall xG LogLoss: {ll:.4f}")
print(f"Avg Pred xG:        {probs.mean():.4f}")
print(f"Actual Goal Rate:   {y_test_goal.mean():.4f}")

# 6. Diagnostics
fig, ax = plt.subplots(figsize=(6, 5))
plot_calib(y_test_goal, probs, "Non-Nested Model", ax)

out_path = Path('analysis/non_nested_xgs/glm_calibration.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print(f"Saved calibration plot to {out_path}")

# 7. Text Report
report_path = Path('analysis/non_nested_xgs/training_report.txt')

metrics = {
    'AUC': auc,
    'LogLoss': ll,
    'Brier': brier_score_loss(y_test_goal, probs),
    'Count': len(df_test),
    'EventRate': y_test_goal.mean()
}

with open(report_path, 'w') as f:
    f.write("Non-Nested GLM Model Training Report\n")
    f.write("======================================\n\n")
    model_desc = "Non-Nested Spline Logistic Regression" if clf.use_splines else f"Non-Nested Polynomial Logistic Regression (Degree={clf.poly_degree})"
    f.write(f"Model: {model_desc}\n")
    f.write(f"Marginalization: {getattr(clf, 'enable_marginalization', 'Unknown')}\n\n")
    
    f.write("Overall Performance (xG):\n")
    f.write(f"  AUC:     {auc:.4f}\n")
    f.write(f"  LogLoss: {ll:.4f}\n")
    f.write(f"  Avg xG:  {probs.mean():.4f} (Actual: {y_test_goal.mean():.4f})\n\n")
    
    f.write("Model Performance:\n")
    f.write(f"    Brier:   {metrics['Brier']:.4f}\n\n")

    f.write("High Danger Stats (Test Set):\n")
    f.write(f"  > 0.3 xG: {len(df_test[df_test['xG'] > 0.3])}\n")
    f.write(f"  > 0.5 xG: {len(df_test[df_test['xG'] > 0.5])}\n")
    f.write(f"  Max xG:   {df_test['xG'].max():.4f}\n")

# 7b. Benchmarking (MoneyPuck)
print("Enriching test predictions with MoneyPuck data...")
try:
    df_test_mp = moneypuck.enrich_with_moneypuck(df_test)
    
    if 'mp_xGoal' in df_test_mp.columns:
        valid = df_test_mp.dropna(subset=['xG', 'mp_xGoal'])
        if len(valid) > 0:
            corr = valid['xG'].corr(valid['mp_xGoal'])
            
            with open(report_path, 'a') as f:
                f.write(f"\nBenchmark (MoneyPuck):\n")
                f.write(f"  Correlation (All): {corr:.4f}\n")
                f.write(f"  Matched Events: {len(valid)} / {len(df_test)}\n")
            
            print(f"MoneyPuck Correlation (All): {corr:.4f}")
        else:
            print("No matching MoneyPuck events found.")
    
    pred_path = Path('analysis/non_nested_xgs/test_predictions.csv')
    df_test_mp.to_csv(pred_path, index=False)
    print(f"Saved enriched predictions to {pred_path}")
    
except Exception as e:
    print(f"MoneyPuck enrichment failed: {e}")
    pred_path_bare = Path('analysis/non_nested_xgs/test_predictions_bare.csv')
    df_test.to_csv(pred_path_bare, index=False)

print(f"Saved report to {report_path}")

# 8. SAVE MODEL
model_path = Path('analysis/xgs/xg_model_non_nested_tensor.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Saving model to {model_path}...")
joblib.dump(clf, model_path)
print(f"Model saved successfully (NEW NON-NESTED TENSOR MODEL)")

# 8b. SAVE METADATA (Compatibility)
meta_path = str(model_path) + '.meta.json'
try:
    meta = {
        'final_features': clf.features,
        'categorical_levels_map': {}, 
        'feature_set_name': 'non_nested_glm_tensor',
        'model_type': 'non_nested_tensor',
        'raw_features': clf.features
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f)
    print(f"Saved metadata to {meta_path}")
except Exception as e:
    print(f"Warning: Failed to save metadata: {e}")

# 9. Verify Pipeline
print("\n--- Verifying analyze._predict_xgs pipeline ---")
try:
    df_raw = fit_xgs.load_data().sample(50, random_state=42)
    
    print("Calling analyze._predict_xgs...")
    df_pred, loaded_clf, _ = analyze._predict_xgs(df_raw, model_path=str(model_path))
    
    if 'xgs' in df_pred.columns:
        print(f"Prediction successful. Mean xG: {df_pred['xgs'].mean():.4f}")
    else:
        print("Error: 'xgs' column missing.")
        
    print(f"Loaded CLF type: {type(loaded_clf).__name__}")
except Exception as e:
    print(f"Pipeline verification failed: {e}")
    import traceback
    traceback.print_exc()

# 10. Generate Comprehensive Model Summary
print("\n--- Generating Model Summary ---")
try:
    out_dir = 'analysis/non_nested_xgs'
    model_summary.generate_model_summary(model_path=str(model_path), output_dir=out_dir, verbose=True)
except Exception as e:
    print(f"Model summary generation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== TRAINING COMPLETE ===")
