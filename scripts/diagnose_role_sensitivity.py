
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import features as feature_util, data_pipeline

def diagnose():
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    if not model_path.exists():
        print("Model not found.")
        return
    
    model = joblib.load(model_path)
    
    # Create test points: Distances 0 to 100ft
    dists = np.linspace(5, 100, 20)
    
    results = []
    for role in ['F', 'D']:
        test_df = pd.DataFrame({
            'distance': dists,
            'angle_deg': 0.0,
            'shooter_role': role,
            'event': 'shot-on-goal',
            'game_state': '5v5',
            'shot_type': 'Wrist Shot',
            'is_rebound': 0,
            'is_rush': 0,
            'period_number': 2,
            'time_elapsed_in_period_s': 600.0,
            'total_time_elapsed_s': 1800.0,
            'score_diff': 0,
            'last_event_type': 'Faceoff',
            'last_event_time_diff': 10.0,
            'shoots_catches': 'L'
        })
        
        # Predict using the layer diagnostic
        # Note: We need to use predict_proba_layer('block')
        p_block = model.predict_proba_layer(test_df, layer='block')
        
        for d, p in zip(dists, p_block):
            results.append({'Role': role, 'Distance': d, 'P_Blocked': p})
            
    df_res = pd.DataFrame(results)
    
    # Plot
    plt.figure(figsize=(10, 6))
    for role in ['F', 'D']:
        subset = df_res[df_res['Role'] == role]
        plt.plot(subset['Distance'], subset['P_Blocked'], marker='o', label=f'Role: {role}')
        
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% Threshold')
    plt.title('P(Blocked) vs Distance by Role')
    plt.xlabel('Distance to Net (ft)')
    plt.ylabel('P(Blocked)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('analysis/diagnostic_block_role_sensitivity.png')
    print("Saved plot to analysis/diagnostic_block_role_sensitivity.png")
    
    # Print table
    print("\nTable of P(Blocked):")
    df_pivot = df_res.pivot(index='Distance', columns='Role', values='P_Blocked')
    print(df_pivot)

if __name__ == "__main__":
    diagnose()
