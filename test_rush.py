import sys
import os
import joblib
import pandas as pd
import numpy as np

model_path = 'analysis/xgs/xg_model_xgboost_nested_20202021.joblib'
print(f"Loading {model_path}...")
model = joblib.load(model_path)

base = {
    'x': 80, 'y': 0, 'distance': 9.0, 'angle_deg': 0.0,
    'game_state': '5v5', 'relative_game_state': '5v5',
    'shooter_role': 'F', 'shoots_catches': 'L',
    'last_event_type': 'giveaway', 'speed_from_last_event': 10.0,
    'last_event_time_diff': 2.0, 'dist_from_last_event': 15.0,
    'is_rush': 0, 'is_rebound': 0, 'is_home': 1, 'score_diff': 0,
    'period_number': 2, 'shot_type': 'wrist'
}

tests = []
for is_rush in [0, 1]:
    for speed in [0.0, 10.0, 20.0, 30.0]:
        row = base.copy()
        row['is_rush'] = is_rush
        row['speed_from_last_event'] = speed
        df_single = pd.DataFrame([{'x': row['x'], 'y': row['y']}])
        row['spatial_block'] = model.spatial_glm_block_.predict_proba(df_single)[0, 1]
        row['spatial_acc'] = model.spatial_glm_acc_.predict_proba(df_single)[0, 1]
        row['spatial_fin'] = model.spatial_glm_fin_.predict_proba(df_single)[0, 1]
        tests.append(row)

df_test = pd.DataFrame(tests)
df_c = model._prepare_inference_df(df_test)
p_block = model.predict_proba_layer(df_c, 'block')
p_acc = model.predict_proba_layer(df_c, 'accuracy')
p_fin = model.predict_proba_layer(df_c, 'finish')
p_xg = (1 - p_block) * p_acc * p_fin

for i in range(len(df_test)):
    print(f"Rush={df_test['is_rush'].iloc[i]}, Speed={df_test['speed_from_last_event'].iloc[i]:<4} | "
          f"Spatial=({df_test['spatial_block'].iloc[i]:.2f}, {df_test['spatial_acc'].iloc[i]:.2f}, {df_test['spatial_fin'].iloc[i]:.2f}) | "
          f"Final=({p_block[i]:.2f}, {p_acc[i]:.2f}, {p_fin[i]:.2f}) => RawXG={p_xg[i]:.4f}")
