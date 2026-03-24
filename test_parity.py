import sys
import os
import joblib
import pandas as pd
import numpy as np

model_path = 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib'
print(f"Loading {model_path}...")
model = joblib.load(model_path)

# Test 1: Wrist Shot vs Wrap-Around vs Backhand
tests = []

base = {
    'x': 80, 'y': 0, 'distance': 9.0, 'angle_deg': 0.0,
    'game_state': '5v5', 'relative_game_state': '5v5',
    'shooter_role': 'F', 'shoots_catches': 'L',
    'last_event_type': 'giveaway', 'speed_from_last_event': 0.0,
    'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
    'period_number': 2, 'time_since_last_event': 2.0
}

for is_rush in [0, 1]:
    for shot_type in ['wrist', 'slap', 'backhand', 'wrap-around']:
        row = base.copy()
        row['is_rush'] = is_rush
        row['shot_type'] = shot_type
        
        # We need the spatial GLM values
        df_single = pd.DataFrame([{'x': row['x'], 'y': row['y']}])
        row['spatial_block'] = model.spatial_glm_block_.predict_proba(df_single)[0, 1]
        row['spatial_acc'] = model.spatial_glm_acc_.predict_proba(df_single)[0, 1]
        row['spatial_fin'] = model.spatial_glm_fin_.predict_proba(df_single)[0, 1]
        
        tests.append(row)

df_test = pd.DataFrame(tests)

# Predict in python
df_c = model._prepare_inference_df(df_test)
p_block = model.predict_proba_layer(df_test, 'block')
p_acc = model.predict_proba_layer(df_test, 'accuracy')
p_fin = model.predict_proba_layer(df_test, 'finish')
p_xg_raw = (1 - p_block) * p_acc * p_fin
p_xg = model.predict_proba(df_test)[:, 1]

for i in range(len(df_test)):
    print(f"Rush={df_test['is_rush'].iloc[i]}, Shot={df_test['shot_type'].iloc[i]:<12} | Block={p_block[i]:.3f}, Acc={p_acc[i]:.3f}, Fin={p_fin[i]:.3f} => RawXG={p_xg_raw[i]:.4f}, CalXG={p_xg[i]:.4f}")
