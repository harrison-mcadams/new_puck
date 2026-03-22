import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from puck import data_pipeline
from puck import features as feature_util

print("Loading data...")
df_raw = pd.read_csv('data/20202021/20202021_df.csv')
df = data_pipeline.preprocess_features(df_raw, apply_filtering=True, apply_arena_adjustments=True, apply_dithering=True)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

print("Loading nested model...")
model = joblib.load('analysis/xgs/xg_model_xgboost_nested_20202021.joblib')

df_c = model._prepare_inference_df(df_test)
feat_block = [f for f in model.features if f != 'shot_type']
y = (df_c['event'] == 'blocked-shot').astype(int)

# Find a real row at ~39ft, 5v5, Forward
mask = (df_c['distance'] >= 34) & (df_c['distance'] <= 44)
mask = mask & (df_c['game_state'] == '5v5') & (df_c['shooter_role'] == 'F')
real_rows = df_c[mask].copy()

# Print XGBoost prediction for 5 real rows
print("\n--- Real Rows from Dataset (dist ~39ft, 5v5, F) ---")
for idx in range(20):
    single_row_c = real_rows.iloc[[idx]].copy()
    
    p_xgb = model.model_block.predict_proba(single_row_c[feat_block])[:, 1][0]
    
    is_blocked = y[real_rows.index[idx]]
    dist = single_row_c['distance'].iloc[0]
    speed = single_row_c['speed_from_last_event'].iloc[0] if 'speed_from_last_event' in single_row_c else 0
    last_event = single_row_c['last_event_type'].iloc[0] if 'last_event_type' in single_row_c else 'unknown'
    time_since = single_row_c['time_since_last_event'].iloc[0] if 'time_since_last_event' in single_row_c else 0
    
    print(f"Dist={dist:.1f}ft, Speed={speed:.1f}, TimeSince={time_since:.1f}, Last={last_event} => Actual={is_blocked}, XGB={p_xgb:.4f}")

# Re-test Exact Dashboard vs 0.0 vs avg
print("\n--- Testing specific Dashboard rows vs empirical means ---")
# Dashboard defaults
test_rows = []
for x in [89, 70, 50, 20]:
    dist = ((x - 89)**2 + 0**2)**0.5
    # angle is 0
    row = {f: 0 for f in feat_block}
    row['x'] = x
    row['y'] = 0
    row['distance'] = dist
    row['angle_deg'] = 0.0
    row['is_home'] = 1
    row['period_number'] = 2
    
    row['game_state'] = '5v5'
    row['relative_game_state'] = '5v5'
    row['shooter_role'] = 'F'
    row['shoots_catches'] = 'L'
    # Try with 'faceoff' vs 'giveaway'
    row['last_event_type'] = 'faceoff'
    test_rows.append(row.copy())
    
    row['last_event_type'] = 'giveaway'
    test_rows.append(row.copy())
    
    row['last_event_type'] = 'hit'
    test_rows.append(row.copy())

df_dash = pd.DataFrame(test_rows)
# prepare inference df
df_dash_c = model._prepare_inference_df(df_dash)
p_raw_dash = model.model_block.predict_proba(df_dash_c[feat_block])[:, 1]

for i in range(0, 12, 3):
    x = df_dash['x'].iloc[i]
    print(f"X={x} -> Dist={89-x:.1f}ft: Faceoff XGB={p_raw_dash[i]:.4f} | Giveaway XGB={p_raw_dash[i+1]:.4f} | Hit XGB={p_raw_dash[i+2]:.4f}")
