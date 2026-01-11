
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
from puck.fit_xgboost_nested import XGBNestedXGClassifier # Required for joblib loading

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    output_path = "analysis/xg_heatmap_5v5.png"
    
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = joblib.load(model_path)
    
    # Generate Grid
    print("Generating Grid...")
    # X: 0 to 100 (Offensive Zone)
    # Y: -42.5 to 42.5
    xs = np.linspace(0, 100, 200)
    ys = np.linspace(-42.5, 42.5, 170)
    xx, yy = np.meshgrid(xs, ys)
    
    # Flatten
    grid_df = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel()
    })
    
    # Filter points outside rink
    # Use rink_half_height_at_x to mask (though simple rect is mostly fine, corners matter)
    mask_in_rink = []
    for idx, row in grid_df.iterrows():
        max_y = rink.rink_half_height_at_x(row['x'])
        mask_in_rink.append(abs(row['y']) <= max_y)
    
    # Apply defaults
    # Sensible defaults for "Average Shot"
    grid_df['event'] = 'shot-on-goal'
    grid_df['shot_type'] = 'Wrist Shot'
    grid_df['game_state'] = '5v5'
    grid_df['is_rebound'] = 0
    grid_df['is_rush'] = 0
    grid_df['period_number'] = 2
    grid_df['time_elapsed_in_period_s'] = 600.0
    grid_df['total_time_elapsed_s'] = 1200.0 + 600.0
    grid_df['score_diff'] = 0
    grid_df['last_event_type'] = 'Faceoff'
    grid_df['last_event_time_diff'] = 10.0
    grid_df['shoots_catches'] = 'L'
    grid_df['shooter_role'] = 'F'
    grid_df['period_time_type'] = 'elapsed'
    grid_df['home_team_defending_side'] = 'left' # arbitrary
    grid_df['player_name'] = 'Average Joe'
    grid_df['team_abbrev'] = 'AVG'
    grid_df['home_abb'] = 'AVG'
    grid_df['away_abb'] = 'OPP'
    
    # Calculate Dist/Angle (Pipeline handles this, but preprocess expects raw input)
    # We can just call preprocess_features? 
    # Yes, let's use the pipeline to ensure consistency.
    # But filtering might remove things? Disable filtering.
    # Imputation? Off (no blocked shots).
    # Adjustments? Off (ideal rink).
    # Dithering? Off (we want clean map).
    
    print("Preprocessing Features...")
    # NOTE: Preprocess usually expects 'x' and 'y' to be raw stats. 
    # It calculates distance/angle for us.
    grid_df = data_pipeline.preprocess_features(
        grid_df,
        is_training=False,
        apply_imputation=False,
        apply_arena_adjustments=False,
        apply_dithering=False, # We want clean map
        apply_filtering=False
    )
    
    # Format features just in case pipeline missed something? Pipeline calls format_features at end.
    
    # Predict
    print("Predicting xG...")
    # Extract only the features the model expects?
    # Usually model works on DataFrame if columns match.
    # Depending on how the model was trained (Pipeline vs XGB directly), it might need specific cols?
    # If it's a generic sklearn/xgb pipeline object, it handles selection.
    # If it's just the booster, we need to match columns.
    # joblib usually saves the full sklearn pipeline.
    
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(grid_df)[:, 1]
        else:
            # Maybe Raw booster?
            # Requires DMatrix?
            # Let's assume Sklearn API wrapper
            probs = model.predict(grid_df) # if regressor? But it's xG (prob)
            # xG models are often classifiers or regressors. 
            # If classifier: predict_proba
            # If regressor: predict
            pass
    except Exception as e:
        # Fallback for "Nested" model which might be a custom class or dict?
        # Let's inspect type of model if this is risky.
        # Assuming standard sklearn-compatible.
        print(f"Warning: Standard predict failed ({e}). Checking if custom Nested model...")
        # If it's the dict from fit_nested_xgs? 
        # Actually usually we save the final wrapper.
        pass
        
    # Re-try safely
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(grid_df)[:, 1]
    elif hasattr(model, 'predict'):
        probs = model.predict(grid_df)
    else:
        print("Error: Model object has no predict method.")
        return

    # Mask out-of-rink
    grid_df['xg'] = probs
    # Set out-of-rink to NaN for clean plotting
    grid_df.loc[~np.array(mask_in_rink), 'xg'] = np.nan
    
    # Plot
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(12, 10))
    rink.draw_rink(ax)
    
    # Reshape for contour/pcolormesh using the original grid shape
    # xx and yy were 170x200
    zz = probs.reshape(xx.shape)
    
    # Mask again in the 2D array
    # Rink mask logic repeated for 2D?
    # Actually just scattering or pcolormesh with the DataFrame is easier if we have NaNs.
    # But contourf is prettier.
    
    # Mask zz based on rink bounds
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            xi = xx[i, j]
            if not (abs(yy[i, j]) <= rink.rink_half_height_at_x(xi)):
                zz[i, j] = np.nan
                
    # Heatmap
    # xG is usually 0.0 to 0.3+. Let's set vmin/vmax sensibly.
    contour = ax.contourf(xx, yy, zz, levels=20, cmap='magma', alpha=0.8, vmin=0, vmax=0.3)
    cbar = plt.colorbar(contour, ax=ax, label='Expected Goal Probability (xG)')
    
    ax.set_title("xG Probability Map (5v5, Wrist Shot)")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
