
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import config, analyze, data_pipeline

def diagnostic():
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_full_history.joblib'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = joblib.load(model_path)
    seasons = [int(f"{y}{y+1}") for y in range(2010, 2026)]
    
    results = []
    
    # We want a representative "Standard Shot" to compare across seasons
    # Wrist shot, 5v5, distance 25ft, angle 0
    standard_shot = pd.DataFrame([{
        'distance': 25.0,
        'angle_deg': 0.0,
        'x': 64.0, # (89 - 25)
        'y': 0.0,
        'shot_type': 'wrist',
        'game_state': '5v5',
        'relative_game_state': '5v5',
        'is_rush': 0,
        'is_rebound': 0,
        'shooter_role': 'F',
        'shoots_catches': 'L',
        'is_home': 1,
        'score_diff': 0,
        'period_number': 2,
        'speed_from_last_event': 10.0,
        'last_event_type': 'faceoff',
        'dist_from_last_event': 30.0,
        'last_event_time_diff': 2.0
    }])

    print("Checking Model Predictions vs Empiric Rates by Season...")
    print(f"{'Season':<10} | {'Model xG':<10} | {'Empiric Goal Rate':<18} | {'Block Rate':<10}")
    print("-" * 60)

    for s in seasons:
        s_str = str(s)
        # Model Prediction
        shot_s = standard_shot.copy()
        shot_s['season'] = s # The model now expects ints
        
        # We need to manually prepare the categorical features for the model call
        # since predict_proba calls _prepare_inference_df
        try:
            prob = model.predict_proba(shot_s)[0, 1]
            p_block = model.predict_proba_layer(shot_s, 'block')[0]
        except Exception as e:
            print(f"Error predicting for {s}: {e}")
            prob = np.nan
            p_block = np.nan

        # Empiric Data
        try:
            csv_path = analyze.locate_season_csv(s_str)
            df_s = pd.read_csv(csv_path, low_memory=False)
            # Use raw rates (no imputation) to see what the model is learning from
            df_s = data_pipeline.preprocess_features(df_s, apply_filtering=True, apply_imputation=False)
            
            # Filter for slot shots to match our standard shot context
            # Slot: dist < 35, angle < 45
            df_slot = df_s[(df_s['distance'] < 35) & (df_s['angle_deg'].abs() < 45)].copy()
            if len(df_slot) > 0:
                emp_rate = (df_slot['event'] == 'goal').mean()
                block_rate = (df_slot['event'] == 'blocked-shot').mean()
            else:
                emp_rate = 0
                block_rate = 0
        except Exception as e:
            print(f"Error loading data for {s}: {e}")
            import traceback
            traceback.print_exc()
            emp_rate = np.nan
            block_rate = np.nan

        print(f"{s:<10} | {prob:<10.4f} | {emp_rate:<18.4f} | {block_rate:<10.4f}")
        results.append({
            'season': s,
            'model_xg': prob,
            'model_block': p_block,
            'emp_rate': emp_rate,
            'block_rate': block_rate
        })

    # Plot
    df_res = pd.DataFrame(results)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df_res['season'].astype(str), df_res['model_xg'], 'b-o', label='Model xG (25ft Wrist)')
    ax1.plot(df_res['season'].astype(str), df_res['emp_rate'], 'g-s', label='Empiric Goal Rate (Slot)')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('Goal Probability / Rate', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    ax2.plot(df_res['season'].astype(str), df_res['block_rate'], 'r--x', label='Empiric Block Rate (Slot)')
    ax2.set_ylabel('Block Rate', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.legend(loc='upper right')
    
    plt.title('Seasonal Trends: Model vs Empiric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analysis/seasonal_trends_diagnostic.png')
    print("\nDiagnostic plot saved to analysis/seasonal_trends_diagnostic.png")

if __name__ == "__main__":
    diagnostic()
