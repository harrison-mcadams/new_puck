"""train_xgboost_model.py

Script to train the XGBoost Nested xG model and produce diagnostic plots.
Comparisons against the RandomForest baseline can be done by inspecting the metrics.
"""

# %% [markdown]
# # XGBoost Nested Model Training
# This notebook/script trains the new XGBoost model.

# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, classification_report
from sklearn.calibration import calibration_curve
import joblib
import json
import logging

# Configure Logging to show INFO logs from modules
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_nested, fit_xgs, analyze, config as puck_config, features as feature_util, correction, data_pipeline

def plot_calib(y_true, y_prob, name, ax):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    ax.set_title(name)
    ax.legend()

print("Imports complete.")

# %%
print("--- Training XGBoost Nested Model ---")


# 1. Load Data
print("Loading data from ALL seasons...")
try:
    # Try explicit path to data dir relative to project root
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    if data_dir.exists():
        df = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    else:
        # Fallback to default
        df = fit_xgs.load_data()
except Exception as e:
    print(f"Error loading all seasons: {e}")
    print("Falling back to default load_data()...")
    df = fit_xgs.load_data()

print(f"Loaded {len(df)} rows.")
if len(df) < 10000:
    print("WARNING: Dataset size is small. Verify 'data/' directory contains all seasons.")

# 1.4 - 3. Unified Preprocessing Pipeline
# (Standardization, Dithering, Attribution, Arena Adj, Imputation, Recalc)
print("Applying Unified Preprocessing Pipeline...")
# Note: debug_imputation_pipeline.csv saving logic is specific to this script's debugging needs.
# The pipeline returns the FINAL processed df. 
# But the script wanted to save intermediate states (x_adj, imputed_x) too.
# The pipeline DOES return a DF with 'x_adj', 'imputed_x' preserved (it doesn't drop them).

try:
    df = data_pipeline.preprocess_features(
        df, 
        is_training=True, 
        verbose=True, 
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=True,
        apply_filtering=True,
        impute_alpha=0.2
    )
    
    # Save Debug CSV
    print("Saving pre-processed debug CSV (imputed + adjusted)...")
    debug_cols = ['game_id', 'event', 'x', 'y', 'x_adj', 'y_adj', 'imputed_x', 'imputed_y', 'shooter_role', 'distance', 'angle_deg']
    save_cols = [c for c in debug_cols if c in df.columns]
    
    mask_blocks = df['event'] == 'blocked-shot'
    mask_other = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
    
    df_debug = pd.concat([
        df[mask_blocks],
        df[mask_other].sample(min(10000, mask_other.sum()), random_state=42)
    ])
    
    debug_path = Path('analysis/debug_imputation_pipeline.csv')
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    df_debug[save_cols].to_csv(debug_path, index=False)
    print(f"  Saved {len(df_debug)} rows to {debug_path}")

except Exception as e:
    print(f"Pipeline failed: {e}")
    import traceback
    traceback.print_exc()


# %%
# 4. Split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

# %%
# 5. Train
print(f"Training on {len(df_train)} rows...")

# Define Features Explicitly
# This ensures we match the "all_inclusive" set defined in puck.features
feature_list = feature_util.get_features('all_inclusive')
print(f"Using {len(feature_list)} features: {feature_list}")

# Load optimized params if available
params_path = Path('analysis/nested_xgs/best_params_xgboost.json')
layer_params = {}
if params_path.exists():
    try:
        with open(params_path, 'r') as f:
            layer_params = json.load(f)
        print(f"Loaded optimized parameters from {params_path}")
    except Exception as e:
        print(f"Warning: Could not load optimized params: {e}")

# The training script will now use the optimized parameters loaded from 
# analysis/nested_xgs/best_params_xgboost.json if available.
# We've removed Strategy 1/Fixed overrides to allow the auto-optimizer to take lead.
print("Using parameters from best_params_xgboost.json (if available)...")







clf = fit_xgboost_nested.XGBNestedXGClassifier(
    features=feature_list,
    n_estimators=200, # Default (will be overridden if layer_params has them)
    max_depth=6,      # Default
    learning_rate=0.05, # Default
    layer_params=layer_params
)
clf.fit(df_train)

# %%
# 6. Evaluate
print("\n--- Evaluation (Test Set) ---")

# Re-derive targets for test set
y_test_goal = (df_test['event'] == 'goal').astype(int)

probs = clf.predict_proba(df_test)[:, 1]

# --- SAVE PREDICTIONS TO DF ---
df_test = df_test.copy()
df_test['xG'] = probs
df_test['prob_block'] = clf.predict_proba_layer(df_test, 'block')
df_test['prob_accuracy'] = clf.predict_proba_layer(df_test, 'accuracy')
import time
df_test['prob_finish'] = clf.predict_proba_layer(df_test, 'finish')

# Save to CSV for manual inspection
debug_csv_path = f'analysis/nested_xgs/test_predictions_{int(time.time())}.csv'
Path(debug_csv_path).parent.mkdir(parents=True, exist_ok=True)
df_test.to_csv(debug_csv_path, index=False)
print(f"Predictions saved to {debug_csv_path}")
# -----------------------------

auc = roc_auc_score(y_test_goal, probs)
ll = log_loss(y_test_goal, probs)

print(f"Overall xG AUC:     {auc:.4f}")
print(f"Overall xG LogLoss: {ll:.4f}")
print(f"Avg Pred xG:        {probs.mean():.4f}")
print(f"Actual Goal Rate:   {y_test_goal.mean():.4f}")
print(f"Ratio (Pred/Act):   {probs.mean() / y_test_goal.mean():.4f}")

# %%
# 7. Layer Diagnostics
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Block
df_test['is_blocked'] = (df_test['event'] == 'blocked-shot').astype(int)
p_block = clf.predict_proba_layer(df_test, 'block')
plot_calib(df_test['is_blocked'], p_block, "Block Layer", axes[0])
print(f"Block AUC: {roc_auc_score(df_test['is_blocked'], p_block):.4f}")

# Accuracy (Unblocked)
mask_unblocked = df_test['is_blocked'] == 0
if mask_unblocked.any():
    df_acc = df_test[mask_unblocked].copy()
    df_acc['is_on_net'] = df_acc['event'].isin(['shot-on-goal', 'goal']).astype(int)
    p_acc = clf.predict_proba_layer(df_acc, 'accuracy')
    plot_calib(df_acc['is_on_net'], p_acc, "Accuracy Layer", axes[1])
    print(f"Accuracy AUC: {roc_auc_score(df_acc['is_on_net'], p_acc):.4f}")
    
# Finish (On Net)
mask_on_net = (df_test['is_blocked'] == 0) & (df_test['event'].isin(['shot-on-goal', 'goal']))
if mask_on_net.any():
    df_fin = df_test[mask_on_net].copy()
    df_fin['is_goal'] = (df_fin['event'] == 'goal').astype(int)
    p_fin = clf.predict_proba_layer(df_fin, 'finish')
    plot_calib(df_fin['is_goal'], p_fin, "Finish Layer", axes[2])
    print(f"Finish AUC: {roc_auc_score(df_fin['is_goal'], p_fin):.4f}")

out_path = Path('analysis/nested_xgs/xgboost_calibration.png')
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path)
print(f"Saved calibration plots to {out_path}")

# %%
# 8. Full Summary Diagnostics
print("\n--- Generating Full Diagnostics ---")

# A. Feature Importance Plots
def plot_importance(model, feature_names, title, ax):
    if model is None:
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Relative Importance')
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plot_importance(clf.model_block, clf.config_block.feature_cols, "Block Model Importance", axes[0])
plot_importance(clf.model_accuracy, clf.config_accuracy.feature_cols, "Accuracy Model Importance", axes[1])
plot_importance(clf.model_finish, clf.config_finish.feature_cols, "Finish Model Importance", axes[2])
plt.tight_layout()
imp_path = Path('analysis/nested_xgs/feature_importance.png')
plt.savefig(imp_path)
print(f"Saved feature importance to {imp_path}")

# B. Textual Report
report_path = Path('analysis/nested_xgs/training_report.txt')
with open(report_path, 'w') as f:
    f.write("XGBoost Nested Model Training Report\n")
    f.write("====================================\n\n")
    
    # Model Config
    f.write("Model Configuration:\n")
    f.write(f"  n_estimators: {clf.n_estimators}\n")
    f.write(f"  max_depth: {clf.max_depth}\n")
    f.write(f"  learning_rate: {clf.learning_rate} (Base/Default)\n")
    f.write(f"  nan_mask_rate: {getattr(clf, 'nan_mask_rate', 'N/A')}\n")
    f.write(f"  enable_categorical: {clf.enable_categorical}\n")
    
    if clf.layer_params:
        f.write("\n  Optimized Layer Parameters:\n")
        for layer, params in clf.layer_params.items():
            f.write(f"    {layer.upper()}: {json.dumps(params, indent=None)}\n")
    f.write("\n")
    
    # Overall Metrics
    f.write("Overall Performance (Test Set):\n")
    f.write(f"  AUC: {auc:.4f}\n")
    f.write(f"  LogLoss: {ll:.4f}\n")
    f.write(f"  Avg Predicted xG: {probs.mean():.4f}\n")
    f.write(f"  Actual Goal Rate: {y_test_goal.mean():.4f}\n")
    f.write(f"  Ratio (Pred/Act): {probs.mean() / y_test_goal.mean():.4f}\n\n")
    
    # Layer Metrics
    f.write("Layer Performance:\n")
    # Block
    block_auc = roc_auc_score(df_test['is_blocked'], p_block)
    block_brier = brier_score_loss(df_test['is_blocked'], p_block)
    f.write(f"  Block Model:\n    AUC: {block_auc:.4f}\n    Brier: {block_brier:.4f}\n")
    
    # Accuracy
    if mask_unblocked.any():
        acc_auc = roc_auc_score(df_acc['is_on_net'], p_acc)
        acc_brier = brier_score_loss(df_acc['is_on_net'], p_acc)
        f.write(f"  Accuracy Model:\n    AUC: {acc_auc:.4f}\n    Brier: {acc_brier:.4f}\n")
        
    # Finish
    if mask_on_net.any():
        fin_auc = roc_auc_score(df_fin['is_goal'], p_fin)
        fin_brier = brier_score_loss(df_fin['is_goal'], p_fin)
        f.write(f"  Finish Model:\n    AUC: {fin_auc:.4f}\n    Brier: {fin_brier:.4f}\n")

    # Blocked Shot Analysis (Post-fix verification)
    f.write("\nBlocked Shot Analysis (Test Sample):\n")
    mask_test_blocked = df_test['is_blocked'] == 1
    if mask_test_blocked.any():
        blocked_xg_mean = df_test.loc[mask_test_blocked, 'xG'].mean()
        blocked_xg_max  = df_test.loc[mask_test_blocked, 'xG'].max()
        blocked_fin_mean = df_test.loc[mask_test_blocked, 'prob_finish'].mean()
        blocked_fin_max = df_test.loc[mask_test_blocked, 'prob_finish'].max()
        
        f.write(f"  Count: {mask_test_blocked.sum()}\n")
        f.write(f"  Mean xG: {blocked_xg_mean:.4f}\n")
        f.write(f"  Max xG: {blocked_xg_max:.4f}\n")
        f.write(f"  Mean Finish Prob: {blocked_fin_mean:.4f} (Goal: Low due to fix)\n")
        f.write(f"  Max Finish Prob:  {blocked_fin_max:.4f}\n")

print(f"Saved training report to {report_path}")

# %%
# 9. SAVE MODEL
# We save to the location analyze.py expects
# Using "all" as suffix since we might retrain on full data, but for dev we save this one
model_path = Path('analysis/xgs/xg_model_nested.joblib')
model_path.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# %%
# 9. VERIFY PREDICTION PIPELINE (analyze.py)
print("\n--- Verifying analyze._predict_xgs pipeline ---")
try:
    # Use a small sample of test data
    # Note: df_test ALREADY has imputed coordinates and processed columns.
    # _predict_xgs expects somewhat raw data (though it handles imputation internally).
    # Ideally we pass it data BEFORE imputation/preprocessing to test full flow.
    # We can load a few raw rows.
    
    # Load raw again for a small batch
    df_raw = fit_xgs.load_data().sample(50, random_state=42)
    # Ensure raw df doesn't have prediction columns which would skip processing
    for col in ['xgs', 'xG', 'prob_block', 'prob_accuracy', 'prob_finish']:
        if col in df_raw.columns:
            df_raw = df_raw.drop(columns=[col])
    
    # Run _predict_xgs
    # This will load the model we just saved!
    print("Calling analyze._predict_xgs...")
    df_pred, loaded_clf, _ = analyze._predict_xgs(df_raw, model_path=str(model_path))
    
    if 'xgs' in df_pred.columns:
        print(f"Prediction successful. Mean xG: {df_pred['xgs'].mean():.4f}")
        print("Sample predictions:\n", df_pred[['event', 'xgs']].head())
    else:
        print("Error: 'xgs' column missing after prediction.")
        
    # Check if correct class loaded
    print(f"Loaded CLF type: {type(loaded_clf).__name__}")
    if type(loaded_clf).__name__ == 'XGBNestedXGClassifier':
        print("SUCCESS: Loaded correct XGBoost class.")
    else:
        print("FAILURE: Loaded wrong class!")

except Exception as e:
    print(f"Pipeline verification failed: {e}")
    import traceback
    traceback.print_exc()
