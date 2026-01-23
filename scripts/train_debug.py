import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
import logging
from puck import mixed_effects, fit_nested_xgs, fit_xgs

logging.basicConfig(level=logging.INFO)

print("=== LOADING DATA ===")
df = pd.read_csv('data/20252026.csv')
df = fit_nested_xgs.preprocess_features(df)
df = fit_xgs.enrich_data_with_bios(df)

# Enrich Team Name
def get_team_name(row):
    tid = row['team_id']
    hid = row['home_id']
    aid = row['away_id']
    try:
        if float(tid) == float(hid): return row['home_abb']
        if float(tid) == float(aid): return row['away_abb']
    except: pass
    return "UNKNOWN"

if 'team_name' not in df.columns:
    if 'home_abb' in df.columns:
        df['team_name'] = df.apply(get_team_name, axis=1)
    else:
        df['team_name'] = df['team_id'].astype(str)

df = df[df['team_name'] != "UNKNOWN"].copy()

# Add context features if missing (simplified for debug)
if 'time_since_last_event' not in df.columns:
    df['time_since_last_event'] = 0.0
    df['dist_from_last_event'] = 0.0
    df['speed_from_last_event'] = 0.0
    df['angle_change_last_event'] = 0.0


# Filter 5v5
df_5v5 = df[df['game_state'] == '5v5'].copy()
df_5v5['opp_team_name'] = np.where(df_5v5['team_name'] == df_5v5['home_abb'], df_5v5['away_abb'], df_5v5['home_abb'])

print(f"5v5 Data: {len(df_5v5)} events")

print("\n=== TRAINING OFFENSE LAYER (TEAM) ===")
me_model_off = mixed_effects.MixedEffectsXG(
    n_estimators=500, # Use more conservative params
    learning_rate=0.1,
    group_col='team_name'
)
me_model_off.fit(df_5v5)

print("\n=== TRAINING DEFENSE LAYER (OPPONENT) ===")
me_model_def = mixed_effects.MixedEffectsXG(
    base_model=me_model_off, # Wrap Offense
    n_estimators=500,
    learning_rate=0.1,
    group_col='opp_team_name'
)
me_model_def.fit(df_5v5)

print("\n=== VERIFICATION ===")
# Predict on entire set
probs = me_model_def.predict_proba(df_5v5)[:, 1]
df_5v5['xgs'] = probs

# Check EDM
print("\n--- EDM ANALYSIS ---")
# 1. Base Model Over-prediction check
base_probs = me_model_off.base_model_.predict_proba(df_5v5)[:, 1]
print(f"Global Base Mean xG: {base_probs.mean():.4f}")
print(f"Global Actual Goal Rate: {(df_5v5['event']=='goal').mean():.4f}")

# 2. EDM Defense Specifics
edm_def = df_5v5[df_5v5['opp_team_name'] == 'EDM']
actual_edm = (edm_def['event']=='goal').mean()
base_edm = base_probs[df_5v5['opp_team_name'] == 'EDM'].mean()
print(f"\nEDM Defense (Shots Against EDM):")
print(f"  Actual Goals: {actual_edm:.4f}")
print(f"  Base Model xG: {base_edm:.4f}")
print(f"  Base Residual (y - p): {actual_edm - base_edm:.4f}")

# 3. Layer 1 Predictions vs EDM
l1_probs = me_model_off.predict_proba(df_5v5)[:, 1]
l1_edm = l1_probs[df_5v5['opp_team_name'] == 'EDM'].mean()
print(f"\nLayer 1 (Offense Adj) Predictions vs EDM:")
print(f"  L1 Model xG: {l1_edm:.4f}")
print(f"  L1 Residual (y - p): {actual_edm - l1_edm:.4f}")
if (actual_edm - l1_edm) > 0:
    print("  => EXPECT POSITIVE DEFENSE ADJUSTMENT")
else:
    print("  => EXPECT NEGATIVE DEFENSE ADJUSTMENT")

# 4. Final Predictions
final_edm = probs[df_5v5['opp_team_name'] == 'EDM'].mean()
print(f"\nFinal Layer (L2) Predictions vs EDM:")
print(f"  Final xG: {final_edm:.4f}")
print(f"  Did it go UP from L1? {final_edm > l1_edm}")
