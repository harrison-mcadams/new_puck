
import os
import sys
import pandas as pd
import numpy as np
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import analyze, data_pipeline

logging.basicConfig(level=logging.WARNING) # Suppress noise

def check_calibration_fleet():
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    results = []

    for season in seasons:
        try:
            csv_path = analyze.locate_season_csv(season)
            df = pd.read_csv(csv_path)
            
            # Align with eval script settings
            df = data_pipeline.preprocess_features(
                df, is_training=False, apply_imputation=True,
                apply_arena_adjustments=True,
                apply_bio_enrichment=True, apply_filtering=True,
                impute_alpha=0.2
            )
            
            # Load global model
            from joblib import load
            model_path = os.path.join('analysis', 'xgs', 'xg_model_xgboost_nested_20202021.joblib')
            model = load(model_path)
            
            # Predict
            df['eval_xg'] = model.predict_proba(df)[:, 1]
            df['is_goal'] = (df['event'].str.lower() == 'goal').astype(float)
            
            total_xg = df['eval_xg'].sum()
            total_goals = df['is_goal'].sum()
            ratio = total_xg / total_goals if total_goals > 0 else 1.0
            
            results.append({
                'Season': season,
                'Total_xG': total_xg,
                'Total_Goals': total_goals,
                'Ratio': ratio,
                'xG_per_Game': total_xg / df['game_id'].nunique(),
                'GF_per_Game': total_goals / df['game_id'].nunique()
            })
        except Exception as e:
            print(f"Error processing {season}: {e}")

    final_df = pd.DataFrame(results)
    print("\n--- Fleet-Wide xG Calibration (nested_xg) ---")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    check_calibration_fleet()
