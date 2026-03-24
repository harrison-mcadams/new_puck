
import pandas as pd
import numpy as np
import os
import sys
import joblib

# Add project root to path
sys.path.append(os.getcwd())

from puck import data_pipeline, fit_xgboost_nested

def benchmark():
    print("--- Direct Historical Benchmark (2023-2024) ---")
    
    csv_path = 'data/20232024.csv'
    if not os.path.exists(csv_path):
        print(f"Data file {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    # Get 1000 shots for better stats
    shot_events = ['shot-on-goal', 'goal', 'missed-shot', 'blocked-shot']
    df_shots = df[df['event'].isin(shot_events)].head(1000).copy()
    
    model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
    print(f"Loading model: {model_path}")
    clf = joblib.load(model_path)
    
    # Preprocess
    print("Preprocessing shots...")
    df_proc = data_pipeline.preprocess_features(df_shots, is_training=False)
    
    # Predict
    print("Predicting...")
    # predict_proba returns [P(no goal), P(goal)]
    p_goal = clf.predict_proba(df_proc)[:, 1]
    
    # Predict layers
    p_block = clf.predict_proba_layer(df_proc, 'block')
    p_acc = clf.predict_proba_layer(df_proc, 'accuracy')
    p_fin = clf.predict_proba_layer(df_proc, 'finish')
    
    df_proc['xgs_new'] = p_goal
    df_proc['p_block_new'] = p_block
    df_proc['p_acc_new'] = p_acc
    df_proc['p_fin_new'] = p_fin
    
    print("\nResults for 1000 historical shots:")
    print(f"  Mean xG: {df_proc['xgs_new'].mean():.4f}")
    print(f"  Goal Rate: {(df_proc['event'] == 'goal').mean():.4f}")
    print(f"  Ratio: {df_proc['xgs_new'].mean() / (df_proc['event'] == 'goal').mean():.2f}")
    
    print("\nLayer Averages:")
    print(f"  P(Block): {p_block.mean():.4f}")
    print(f"  P(Acc):   {p_acc.mean():.4f}")
    print(f"  P(Fin):   {p_fin.mean():.4f}")

if __name__ == "__main__":
    benchmark()
