
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import data_pipeline, rink, features
from puck.fit_xgboost_nested import XGBNestedXGClassifier 

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    output_path = "analysis/block_prob_and_xg_heatmap.png"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    # Generate Grid
    print("Generating Grid...")
    xs = np.linspace(0, 100, 200)
    ys = np.linspace(-42.5, 42.5, 170)
    xx, yy = np.meshgrid(xs, ys)
    
    # Flatten
    grid_df_base = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel()
    })
    
    # Apply defaults
    grid_df_base['event'] = 'shot-on-goal' 
    grid_df_base['shot_type'] = 'Wrist Shot' # Default
    grid_df_base['game_state'] = '5v5'
    grid_df_base['is_rebound'] = 0
    grid_df_base['is_rush'] = 0
    grid_df_base['period_number'] = 2
    grid_df_base['time_elapsed_in_period_s'] = 600.0
    grid_df_base['total_time_elapsed_s'] = 1200.0 + 600.0
    grid_df_base['score_diff'] = 0
    grid_df_base['last_event_type'] = 'Faceoff'
    grid_df_base['last_event_time_diff'] = 10.0
    grid_df_base['shoots_catches'] = 'L'
    grid_df_base['period_time_type'] = 'elapsed'
    grid_df_base['home_team_defending_side'] = 'left' 
    grid_df_base['player_name'] = 'Average Joe'
    grid_df_base['team_abbrev'] = 'AVG'
    grid_df_base['home_abb'] = 'AVG'
    grid_df_base['away_abb'] = 'OPP'
    
    # Setup Figure: 3 Columns
    # 1. Block Prob (F)
    # 2. Block Prob (D)
    # 3. xG Prob (F) - Control
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    configs = [
        {'role': 'F', 'type': 'block', 'title': 'Block Prob (Forward)', 'cmap': 'magma', 'vmax': 1.0},
        {'role': 'D', 'type': 'block', 'title': 'Block Prob (Defenseman)', 'cmap': 'magma', 'vmax': 1.0},
        {'role': 'F', 'type': 'xg',    'title': 'xG Prob (Forward, Control)', 'cmap': 'plasma', 'vmax': 0.3} # xG is lower
    ]
    
    for ax, config in zip(axes, configs):
        role = config['role']
        pred_type = config['type']
        print(f"Processing {config['title']}...")
        
        # Clone DF
        grid_df = grid_df_base.copy()
        grid_df['shooter_role'] = role
        
        # Preprocess
        grid_df = data_pipeline.preprocess_features(
            grid_df,
            is_training=False,
            apply_imputation=False,
            apply_arena_adjustments=False,
            apply_dithering=False, 
            apply_filtering=False
        )
        
        # Predict 
        if pred_type == 'block':
            if hasattr(model, 'predict_proba_layer'):
                probs = model.predict_proba_layer(grid_df, layer='block')
            else:
                print("Error: Model logic missing 'predict_proba_layer'")
                return
        elif pred_type == 'xg':
            # Full xG probability
            probs = model.predict_proba(grid_df)[:, 1]
            
        # Reshape & Mask
        zz = probs.reshape(xx.shape)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                xi = xx[i, j]
                if not (abs(yy[i, j]) <= rink.rink_half_height_at_x(xi)):
                    zz[i, j] = np.nan
        
        # Plot
        rink.draw_rink(ax)
        contour = ax.contourf(xx, yy, zz, levels=20, cmap=config['cmap'], alpha=0.9, vmin=0, vmax=config['vmax'])
        ax.set_title(config['title'])
        
        # Individual colorbar per plot because scales differ significantly (Block vs xG)
        plt.colorbar(contour, ax=ax, fraction=0.03, pad=0.04)
        
    plt.suptitle("Probability Maps: Blocked Shots vs xG (5v5 Wrist Shot)", fontsize=16)
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
