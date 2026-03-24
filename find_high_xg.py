import pandas as pd
import numpy as np
import joblib
from puck import data_pipeline

# Use 20202021 as it's small enough
df_raw = pd.read_csv('data/20202021/20202021_df.csv')
print(f"Loaded {len(df_raw)} rows.")

df = data_pipeline.preprocess_features(df_raw, apply_filtering=True, exclude_blocked=True)
print(f"Preprocessed {len(df)} rows.")

model = joblib.load('analysis/xgs/xg_model_xgboost_nested_20202021.joblib')

# Just take first 10000 rows to find some examples
test_subset = df.head(10000).copy()
probs = model.predict_proba(test_subset)[:, 1]
test_subset['xg'] = probs

high_xg = test_subset[test_subset['xg'] > 0.15].sort_values('xg', ascending=False)
print(f"Found {len(high_xg)} shots with xG > 15%")

if not high_xg.empty:
    cols = ['xg', 'distance', 'angle_deg', 'shot_type', 'is_rush', 'is_rebound', 'speed_from_last_event', 'dist_from_last_event', 'last_event_time_diff']
    print(high_xg[cols].head(20).to_string())
else:
    print("No high xG shots found in head.")
