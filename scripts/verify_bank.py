import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import joblib
from scripts.matchup import precalculate_matchup_xg

print("Loading assets...")
models = {
    '5v5': joblib.load('analysis/mixed_effects_heatmaps_20252026/models/mixed_model_5v5.pkl')
}
bank = pd.read_pickle('analysis/mixed_effects_heatmaps_20252026/events_bank.pkl')

print(f"Bank Rows: {len(bank)}")
print("Pre-calculating...")
bank = precalculate_matchup_xg(bank, models, 'NYR', 'EDM')

# Check Results
mask_5v5 = bank['game_state'] == '5v5'
xg_vals = bank.loc[mask_5v5, 'xg_home_context']

print(f"\n--- Validation Results (NYR vs EDM 5v5) ---")
print(f"Mean xG: {xg_vals.mean():.4f}")
print(f"Max xG: {xg_vals.max():.4f}")
print(f"Min xG: {xg_vals.min():.4f}")
print(f"Negative xG Count: {(xg_vals < 0).sum()}")

if xg_vals.mean() > 0.03 and xg_vals.min() >= 0:
    print("SUCCESS: Values are reasonable and positive.")
else:
    print("FAILURE: Values are suspicious.")
