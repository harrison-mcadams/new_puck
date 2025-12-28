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

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_nested, fit_xgs, analyze, config as puck_config, features as feature_util

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
print("Loading data...")
df = fit_xgs.load_data()

# %% 
# 2. Preprocess (Filter & Clean) using XGBoost specific routine
print("Preprocessing data for XGBoost...")
df = fit_xgboost_nested.preprocess_data(df)

# --- DEBUG: Verify Data Integrity ---
blocked_mask = df['is_blocked'] == 1
n_blocked = blocked_mask.sum()
# Check nulls (handle categorical NaN which might not show as isna() if not nullable, though typically does)
n_nan = df.loc[blocked_mask, 'shot_type'].isnull().sum()

print(f"\n[DEBUG] Blocked Shot Analysis:")
print(f"  Total Blocked Shots: {n_blocked}")
print(f"  Blocked with NaN shot_type: {n_nan}")
if n_blocked > 0:
    print(f"  Sample shot_types (Blocked): {df.loc[blocked_mask, 'shot_type'].unique().tolist()[:5]}")
# ------------------------------------

# %%
# 3. Impute Coordinates (for blocked shots)
# Note: shot_type remains NaN for blocked shots (handled by XGBoost)
try:
    from puck import impute
    print("Applying blocked shot coordinate imputation...")
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
except Exception as e:
    print(f"Warning: Could not impute coordinates: {e}")

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
df_test['prob_finish'] = clf.predict_proba_layer(df_test, 'finish')

# Save to CSV for manual inspection
debug_csv_path = 'analysis/nested_xgs/test_predictions_new.csv'
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
model_path = Path('analysis/xgs/xg_model_nested_all.joblib')
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
