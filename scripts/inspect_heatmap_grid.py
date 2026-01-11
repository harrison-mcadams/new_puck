
import sys, os
sys.path.append(os.getcwd())
import joblib, pandas as pd, numpy as np
from puck import data_pipeline, features as feature_util

model = joblib.load('analysis/xgs/xg_model_nested.joblib')
xs = np.linspace(0, 100, 20)
ys = np.linspace(-42.5, 42.5, 10)
xx, yy = np.meshgrid(xs, ys)

df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'shooter_role': 'F', 'event': 'shot-on-goal', 'game_state': '5v5', 'shot_type': 'Wrist Shot'})
df = data_pipeline.preprocess_features(df, is_training=False, apply_imputation=False)
p = model.predict_proba_layer(df, 'block')

df['p'] = p
# Sort by p to find highest confidence areas
print(df.sort_values('p', ascending=False).head(20))
