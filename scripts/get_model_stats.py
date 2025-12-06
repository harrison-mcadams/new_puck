import sys
import os
import pandas as pd
import joblib
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import fit_xgs

def get_stats():
    model_path = 'analysis/xgs/xg_model.joblib'
    meta_path = model_path + '.meta.json'
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    clf = joblib.load(model_path)
    
    # Always use the raw input features that generate the final model features
    features = ['distance', 'angle_deg', 'game_state', 'is_net_empty']
    
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            # We don't use final_features here because clean_df_for_model expects input cols
            cat_map = meta.get('categorical_levels_map', {})
    else:
        print("Meta file not found, assuming default features.")
        cat_map = None

    # Re-load all data to evaluate on the full set (or you could just evaluate on recent seasons)
    SEASONS = [f"{y}{y+1}" for y in range(2025, 2013, -1)]
    all_dfs = []
    
    print("Loading data...")
    for season in SEASONS:
        csv_path = os.path.join('data', season, f"{season}_df.csv")
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                all_dfs.append(df)
            except Exception:
                pass
    
    if not all_dfs:
        print("No data found!")
        return

    full_df = pd.concat(all_dfs, ignore_index=True)
    print(f"Evaluating on {len(full_df)} rows...")
    
    # Clean/Prepare data
    # IMPORTANT: Use the same features and categorical mappings as training!
    model_df, final_feats, _ = fit_xgs.clean_df_for_model(
        full_df, 
        features, 
        fixed_categorical_levels=cat_map
    )
    
    X = model_df[final_feats].values
    y = model_df['is_goal'].values
    
    # Evaluate
    print("Calculating metrics and generating plot...")
    # This will also save analysis/xgs/xg_likelihood.png
    y_prob, y_pred, metrics = fit_xgs.evaluate_model(clf, X, y)
    
    print("\nModel Metrics:")
    print(json.dumps(metrics, indent=2))
    
    # Save metrics to disk too
    with open('analysis/xgs/xg_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nMetrics saved to analysis/xgs/xg_model_metrics.json")
    print("Calibration plot saved to analysis/xgs/xg_likelihood.png")

if __name__ == "__main__":
    get_stats()
