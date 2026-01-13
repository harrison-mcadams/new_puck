import pandas as pd
import numpy as np
import sys
import os
import joblib
from sklearn.metrics import brier_score_loss, log_loss

sys.path.append(os.getcwd())
from puck import data_pipeline, fit_xgs

print("--- Calibration Verification ---")

# 1. Load Data (Last Season Test)
print("Loading 2023-2024 data (proxy for test set)...")
csv_path = 'data/20232024/20232024_df.csv'

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Try global loader as fallback
    try:
        df = fit_xgs.load_data(years=[20232024])
    except:
        print(f"Could not load data from {csv_path} or via fit_xgs.")
        sys.exit(1)

if df.empty:
    print("No data.")
    sys.exit(1)

# 2. Filter to Shot Events ONLY
shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
df = df[df['event'].isin(shot_events)].copy()
print(f"Filtered to {len(df)} shot events.")

# 3. Preprocess
print("Preprocessing...")
df = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=True)

# 3. Load Model
model_path = "analysis/xgs/xg_model_nested.joblib"
if not os.path.exists(model_path):
    print("Model not found.")
    sys.exit(1)
    
model = joblib.load(model_path)

# 4. Predict
print("Predicting xG...")
# Predict on ALL data (including blocked shots -> Unknown marginalization)
# Note: 'shot_type' column might need normalization for marginalization to work?
# The training script lowercased it. The raw data is normalized in data_pipeline?
# Let's check if we need to force lowercase here or if model handles it.
# fit_glm_nested checks: X['shot_type'].astype(str).str.lower() == 'unknown'
# And it lowercases training data to learn priors.
# But does it lowercase INPUT data for prediction?
# IN PREDICT:
# mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
# ... 
# df_nan_imputed['shot_type'] = st (where st is from self.shot_type_priors_ keys)
# The OneHotEncoder was fitted on... what?
# In fit(): "cat_trans = Pipeline([ ... OneHotEncoder(...)])"
# The pipeline is fitted on whatever X[cat_features] was passed.
# In fit(), we did NOT modify X inplace to be lowercase before passing to sub-pipelines.
# We only lowercased to calculate PRIORS.
# The sub-pipelines operate on the raw X passed to fit().
# If raw X had 'Wrist', 'Slap', then OHE learned 'Wrist', 'Slap'.
# The PRIORS keys are 'wrist', 'slap' (from VOCAB_SHOT_TYPE).
# If we impute 'wrist' (lowercase) but OHE knows 'Wrist' (Title), we have a mismatch!
# This might be another bug. Let's check.

# We will run predictions and see.
probs = model.predict_proba(df)[:, 1]
df['xg'] = probs

# 5. Analysis
goals = (df['event'] == 'goal').astype(int)
total_goals = goals.sum()
total_xg = df['xg'].sum()

print(f"\nTotal Actual Goals: {total_goals}")
print(f"Total Predicted xG: {total_xg:.2f}")
print(f"Ratio (xG/Goals):   {total_xg/total_goals:.3f}")
print(f"Difference:         {total_xg - total_goals:.2f}")

# Breakdown by Event Type
print("\n--- Breakdown by Event Type ---")
df['is_blocked'] = (df['event'] == 'blocked-shot')
df['is_goal'] = (df['event'] == 'goal').astype(int)

stats = df.groupby('is_blocked').agg({
    'event': 'count',
    'xg': ['sum', 'mean'],
    'is_goal': 'sum' 
})
stats.rename(columns={'is_goal': 'actual_goals'}, inplace=True)

print(stats)

print("\n--- Averages Comparison ---")
mean_xg_blocked = df[df['is_blocked']]['xg'].mean()
mean_xg_unblocked = df[~df['is_blocked']]['xg'].mean()
print(f"Average xG (Blocked):   {mean_xg_blocked:.4f}")
print(f"Average xG (Unblocked): {mean_xg_unblocked:.4f}")
print(f"Ratio (Blocked/Unblocked): {mean_xg_blocked/mean_xg_unblocked:.2f}")

# Also check missed shots for context
mean_xg_missed = df[df['event'] == 'missed-shot']['xg'].mean()
print(f"Average xG (Missed):    {mean_xg_missed:.4f}")

# Check specifically for "Unknown" shot types
if 'shot_type' in df.columns:
    df['st_norm'] = df['shot_type'].astype(str).str.lower()
    unknowns = df[df['st_norm'] == 'unknown']
    print(f"\nUnknown Shot Types: {len(unknowns)}")
    print(f"Unknown Total xG: {unknowns['xg'].sum():.2f}")
    if len(unknowns) > 0:
        print(f"Unknown Mean xG:  {unknowns['xg'].mean():.4f}")

