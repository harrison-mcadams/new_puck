
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from puck import data_pipeline

# Load Model
model = joblib.load('analysis/xgs/xg_model_nested.joblib')

# Create a Single Test Case (Center Slot, 20ft away)
row = {
    'x': 89 - 20, 
    'y': 0, 
    'event': 'shot-on-goal', # Dummy
    'shooter_role': 'F',
    'game_state': '5v5',
    'score_diff': 0,
    'shot_type': 'Unknown', # TESTING THIS
    'shoots_catches': 'L',
    'home_team_defending_side': 'left',
    'is_rebound': 0,
    'is_rush': 0,
    'period_time_type': 'elapsed',
    'total_time_elapsed_s': 1000
}

df = pd.DataFrame([row])

# Preprocess
df_proc = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=False)

# Predict
# Inject priors to test marginalization (since model on disk doesn't have them yet)
model.shot_type_priors_ = {
    'wrist': 0.5,
    'snap': 0.2,
    'slap': 0.2,
    'backhand': 0.05,
    'tip-in': 0.05
} # Approximate

p_block = model.predict_proba_layer(df_proc, layer='block')[0]
p_finish = model.predict_proba_layer(df_proc, layer='finish')[0]
p_accuracy = model.predict_proba_layer(df_proc, layer='accuracy')[0]

p_xg = model.predict_proba(df_proc)[:, 1][0]

print(f"--- Unknown Shot Type Probe (Marginalized) ---")
print(f"Location: (69, 0) [20ft from net]")
print(f"P(Block): {p_block:.4f}")
print(f"P(Accuracy - On Net): {p_accuracy:.4f}")
print(f"P(Finish - Goal): {p_finish:.4f}")
print(f"Final xG: {p_xg:.4f}")
