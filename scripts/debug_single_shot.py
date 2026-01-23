import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Load Models
print("Loading models...")
path_5v5 = "analysis/mixed_effects_heatmaps_20252026/models/mixed_model_5v5.pkl"
model_def = joblib.load(path_5v5)
model_off = model_def.base_model_
base_glm = model_off.base_model_

print(f"Model Structure:")
print(f"  Defense Layer Group: {model_def.group_col}")
print(f"  Offense Layer Group: {model_def.base_model_.group_col}")
print(f"  Base GLM: {type(base_glm)}")

# Create a synthetic shot
# High danger spot: Slot
row = pd.DataFrame([{
    'distance': 15.0,
    'angle_deg': 0.0,
    'game_state': '5v5',
    'shot_type': 'wrist-shot',
    'is_rebound': 0,
    'is_rush': 0,
    'time_since_last_event': 10.0,
    'speed_from_last_event': 5.0,
    'dist_from_last_event': 10.0,
    'angle_change_last_event': 0.0,
    'rebound_angle_change': 0.0, 
    'rebound_time_diff': 0.0, 
    'rebound_speed': 0.0, 
    'rebound_dist_change': 0.0,
    'last_event_type': 'pass',
    'last_event_team': 'same',
    'shoots_catches': 'L',
    'shooter_role': 'Forward',
    'team_name': 'NYR',      # Good Offense
    'opp_team_name': 'EDM',  # Bad Defense
    # Dummy cols for GLM if needed
    'period': 2,
    'period_seconds': 600,
    'score_diff': 0,
    'is_home': 1,
    'x_adj': 85,
    'y_adj': 0,
    # Required Base GLM Features (Found from inspection)
    'total_time_elapsed_s': 1200, 
    'last_event_time_diff': 10.0,
    'time_diff': 10.0,
    'dist_diff': 10.0,
    'angle_diff': 0.0,
    'dist': 15.0,
    'angle': 0.0,
    'shot_type': 'wrist-shot', # Needs OHE? NestedGLM likely handles categorical if passed as df?
    # Or does it expect pre-encoded? It has OHE internals.
    
    # Just in case:
    'shot_type_wrist-shot': 1,
    'is_rebound_True': 0,
    'is_rush_True': 0,
    'shooter_role': 'Forward', # Categorical
    'shoots_catches': 'L'
}])

# Force fill 0 for anything else in "features" list of base model
base_feats = base_glm.features
for f in base_feats:
    if f not in row.columns:
        row[f] = 0

dummy_feats = base_glm.feature_names_ if hasattr(base_glm, 'feature_names_') else []
for f in dummy_feats:
    if f not in row.columns:
        row[f] = 0

print("\n--- TRACING PREDICTION ---")

# 1. Base GLM
p0 = base_glm.predict_proba(row)[:, 1][0]
m0 = np.log(p0 / (1 - p0))
print(f"1. Base GLM Prob: {p0:.4f} (Margin: {m0:.4f})")

# 2. Offense Layer (NYR)
# Manual extract
off_booster = model_off.group_models_['NYR'].get_booster()
off_dmat = xgb.DMatrix(row[model_off.group_models_['NYR'].feature_names_in_])
# We must use base_margin=0 to see pure effect?
off_dmat.set_base_margin(np.zeros(1)) 
off_delta = off_booster.predict(off_dmat, output_margin=True)[0]
print(f"2. Offense (NYR) Delta: {off_delta:.4f}")

# Verify via predict_proba
p1 = model_off.predict_proba(row)[:, 1][0]
m1 = np.log(p1 / (1 - p1))
print(f"   Model Offense Prob: {p1:.4f} (Margin: {m1:.4f})")
print(f"   Implied Delta: {m1 - m0:.4f}")

# 3. Defense Layer (EDM)
# Manual extract
def_booster = model_def.group_models_['EDM'].get_booster()
def_dmat = xgb.DMatrix(row[model_def.group_models_['EDM'].feature_names_in_])
def_dmat.set_base_margin(np.zeros(1))
def_delta = def_booster.predict(def_dmat, output_margin=True)[0]
print(f"3. Defense (EDM) Delta: {def_delta:.4f}")

# Verify via predict_proba
p2 = model_def.predict_proba(row)[:, 1][0]
m2 = np.log(p2 / (1 - p2))
print(f"   Model Defense (Final) Prob: {p2:.4f} (Margin: {m2:.4f})")
print(f"   Implied Delta: {m2 - m1:.4f}")

# 4. Total Check
print(f"\nTotal Margin: {m0:.4f} + {off_delta:.4f} + {def_delta:.4f} = {m0 + off_delta + def_delta:.4f}")
final_prob_manual = 1 / (1 + np.exp(-(m0 + off_delta + def_delta)))
print(f"Manual Total Prob: {final_prob_manual:.4f}")
print(f"Model Output Prob: {p2:.4f}")

if abs(p2 - final_prob_manual) > 0.001:
    print("MISMATCH! The model predict code is inconsistent with components!")
else:
    print("MATCH. Logic is consistent.")

# 5. Check Values
# We expect NYR Delta > 0 (Good O)
# We expect EDM Delta > 0 (Bad D -> allows more goals)
if off_delta > 0: print("NYR Offense Effect: POSITIVE (Correct)")
else: print("NYR Offense Effect: NEGATIVE (Suspicious)")

if def_delta > 0: print("EDM Defense Effect: POSITIVE (Correct - Bad D)")
else: print("EDM Defense Effect: NEGATIVE (Suspicious - implies Good D)")
