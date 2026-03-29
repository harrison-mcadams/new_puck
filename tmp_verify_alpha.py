import pandas as pd
import numpy as np
from puck import data_pipeline, impute

# Create a dummy blocked shot
# Defensing side: Home (id=1) is defending right (x > 0). Attacking side: Away (id=2).
# Goal is at x=89.
# Blocked shot at x=60, y=10.
df = pd.DataFrame([{
    'game_id': 1,
    'event': 'blocked-shot',
    'team_id': 1, # RAW: Defender
    'event_owner_team_id': 1,
    'home_id': 1,
    'away_id': 2,
    'home_abb': 'HOME',
    'away_abb': 'AWAY',
    'x': 60.0,
    'y': 10.0,
    'home_team_defending_side': 'right', 
    'period': 1,
    'game_state': '5v5'
}])

print("--- Original ---")
print(df[['event', 'team_id', 'x', 'y']])

# Run pipeline with 0.2
df_proc = data_pipeline.preprocess_features(df.copy(), is_training=False, impute_alpha=0.2, apply_arena_adjustments=False)

print("\n--- Processed (alpha=0.2) ---")
# After pipeline: 
# 1. Team ID should be 2 (attacker)
# 2. X/Y should be standardized. 
#    Original x=60 for defender on right mean Attacker is attacking right. 
#    Actually if home is defending right, they face left (x < 0). 
#    Wait. If home is defending right (goal at +89), Attacker (Away) is attacking right (+89).
#    So x=60 is in the attacking zone for Away.
# 3. Imputation: relocates shot FROM block (60) TO shooter (somewhere further out).
#    If alpha=0.2, it moves it 20% back? Or 80% back?
#    Logic usually: shooter = block + alpha * (block - goal)? No.
#    Let's check the code or just see output.
print(df_proc[['event', 'team_id', 'x', 'y']])

# Run pipeline with 0.5
df_proc_5 = data_pipeline.preprocess_features(df.copy(), is_training=False, impute_alpha=0.5, apply_arena_adjustments=False)
print("\n--- Processed (alpha=0.5) ---")
print(df_proc_5[['event', 'team_id', 'x', 'y']])
