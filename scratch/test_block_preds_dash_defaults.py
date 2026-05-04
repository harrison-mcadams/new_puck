import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from puck import config

model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_modern_era.joblib'
print(f"Loading model: {model_path}")
model = joblib.load(str(model_path))

# Generate a synthetic grid of shots in the offensive zone
xs = np.linspace(0, 89, 20)
ys = np.linspace(-42, 42, 20)
xv, yv = np.meshgrid(xs, ys)

df = pd.DataFrame({
    'x': xv.flatten(),
    'y': yv.flatten(),
    'event': 'shot-on-goal',
    'shot_type': 'wrist',
    'period_number': 2,
    'total_time_elapsed_s': 600,
    'time_elapsed_in_period_s': 600,
    'score_diff': 0,
    'is_home': 1,
    'relative_game_state': '5v5',
    'shoots_catches': 'L',
    'is_rebound': 0,
    'rebound_angle_change': 0,
    'rebound_time_diff': 0,
    'rebound_source': 'none',
    'is_rush': 0,
    'last_event_type': 'giveaway',
    'last_event_time_diff': 2.0,
    'dist_from_last_event': 15.0,
    'speed_from_last_event': 7.5,
    'angle_change_last_event': 0,
    'shooter_role': 'F',
    'season': '20252026'
})

from puck import rink
dists, angles = [], []
for i in range(len(df)):
    d, a = rink.calculate_distance_and_angle(df['x'].iloc[i], df['y'].iloc[i], 89, 0)
    dists.append(d)
    angles.append(a)

df['distance'] = dists
df['angle_deg'] = angles

p_block = model.predict_proba_layer(df, 'block')
df['p_block'] = p_block

print(f"Average predicted block probability: {df['p_block'].mean():.3f}")

z1 = df[(df['distance'] < 30) & (df['distance'] >= 10)]
print(f"Predicted Block rate (10-30ft): {z1['p_block'].mean():.3f}")

z2 = df[(df['distance'] < 60) & (df['distance'] >= 30)]
print(f"Predicted Block rate (30-60ft): {z2['p_block'].mean():.3f}")

z3 = df[(df['distance'] >= 60)]
print(f"Predicted Block rate (>60ft): {z3['p_block'].mean():.3f}")
