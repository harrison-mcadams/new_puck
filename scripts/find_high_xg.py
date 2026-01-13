import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgs, data_pipeline, analyze

# Load Model
model_path = 'analysis/xgs/xg_model_nested.joblib'
clf = joblib.load(model_path)

# Load Data and Preprocess exactly as training
print("Loading data...")
df = fit_xgs.load_all_seasons_data()
df = data_pipeline.preprocess_features(df, is_training=False, apply_filtering=True)
_, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Predict
print("Predicting...")
df_test['xG'] = clf.predict_proba(df_test)[:, 1]

# Top 10
top_10 = df_test.sort_values('xG', ascending=False).head(10)
print("\n--- TOP 10 SHOTS BY xG ---")
pd.set_option('display.max_columns', None)
print(top_10)

# Layer breakdown for top shot
top_shot = top_10.iloc[0]
print(f"\nBreakdown for top shot (Distance={top_shot['distance']:.1f}):")
top_df = pd.DataFrame([top_shot])
for f in clf.features:
    print(f"  {f}: {top_shot.get(f, 'MISSING')}")
print(f"P(Block):   {clf.predict_proba_layer(top_df, 'block')[0]:.4f}")
print(f"P(Acc):     {clf.predict_proba_layer(top_df, 'accuracy')[0]:.4f}")
print(f"P(Finish):  {clf.predict_proba_layer(top_df, 'finish')[0]:.4f}")
