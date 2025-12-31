
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') # Fix for thread/GUI errors
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from puck import config, fit_nested_xgs, fit_xgboost_nested, features

# --- HACK FOR JOBLIB LOADING ---
# If the model was trained by running `python puck/fit_nested_xgs.py`, the class 
# might be saved as `__main__.NestedXGClassifier`. We alias it here to allow loading.
if not hasattr(sys.modules['__main__'], 'NestedXGClassifier'):
    setattr(sys.modules['__main__'], 'NestedXGClassifier', fit_nested_xgs.NestedXGClassifier)
    
if not hasattr(sys.modules['__main__'], 'LayerConfig'):
    setattr(sys.modules['__main__'], 'LayerConfig', fit_nested_xgs.LayerConfig)
    
# Also for XGBoost version if that's what was used
# Check both modules just in case
if hasattr(fit_xgboost_nested, 'XGBNestedXGClassifier'):
    if not hasattr(sys.modules['__main__'], 'XGBNestedXGClassifier'):
        setattr(sys.modules['__main__'], 'XGBNestedXGClassifier', fit_xgboost_nested.XGBNestedXGClassifier)
    # Check if XGB version has LayerConfig? It does.
    if hasattr(fit_xgboost_nested, 'LayerConfig'):
         if not hasattr(sys.modules['__main__'], 'LayerConfig'): # Might conflict if both define it?
             # If they are identical dataclasses it might be okay. 
             # But usually trained with one or other. 
             # Let's hope RF version is predominant or they are compatible.
             pass 

def main():
    print("Starting Model Deep Dive...")
    
    # 1. Load Model
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested_all.joblib'
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        err_msg = f"CRITICAL ERROR LOADING MODEL: {e}"
        print(err_msg)
        with open("debug_load_error.txt", "w") as f:
            f.write(err_msg)
        return
    
    print(f"Model type: {type(model).__name__}")
    
    # 2. Load Data (Validation Set)
    # We'll load all data and split, similar to training, to get a valid test set
    print("Loading data...")
    df = fit_nested_xgs.load_data()
    df = fit_xgboost_nested.preprocess_data(df)
    
    # Impute (using point_pull as standard)
    from puck import impute
    df = impute.impute_blocked_shot_origins(df, method='point_pull')
    
    # Split
    from sklearn.model_selection import train_test_split
    _, df_test = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Validation set size: {len(df_test)}")
    
    # --- CRITICAL: Ensure df_test has all columns expected by the model ---
    # The model was trained with specific OHE columns (e.g. game_state_6v5).
    # If the validation set lacks them, direct access (like in PDPs) will fail.
    
    all_model_feats = set()
    if hasattr(model, 'final_features'): all_model_feats.update(model.final_features)
    if hasattr(model, 'config_block'): all_model_feats.update(model.config_block.feature_cols)
    if hasattr(model, 'config_accuracy'): all_model_feats.update(model.config_accuracy.feature_cols)
    if hasattr(model, 'config_finish'): all_model_feats.update(model.config_finish.feature_cols)
    
    for c in all_model_feats:
        if c not in df_test.columns:
            df_test[c] = 0
            
    print(f"Aligned validation data. Total columns: {len(df_test.columns)}")
    
    # Output Dir
    out_dir = Path(config.ANALYSIS_DIR) / 'deep_dive'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Feature Importance
    plot_all_feature_importances(model, out_dir)
    
    # 4. Calibration
    plot_calibration(model, df_test, out_dir)
    
    # 5. Partial Dependence Plots
    plot_pdps(model, df_test, out_dir)
    
    # 6. Spatial Maps
    plot_spatial_maps(model, out_dir)
    
    # 7. Categorical Impact
    plot_categorical_impact(model, df_test, out_dir)
    
    print(f"\nDeep Dive Complete! check {out_dir}")

def plot_all_feature_importances(model, out_dir):
    print("\nGenerating Feature Importance Plots...")
    
    # Helper to extract and plot
    def plot_imp(sub_model, feature_names, title, filename):
        if hasattr(sub_model, 'feature_importances_'):
            imps = sub_model.feature_importances_
            # If XGBoost, it might be a property, if RF it is valid too.
            
            # Create DF
            fi_df = pd.DataFrame({'feature': feature_names, 'importance': imps})
            fi_df = fi_df.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=fi_df.head(15), x='importance', y='feature', palette='viridis')
            plt.title(f'Feature Importance: {title}')
            plt.tight_layout()
            plt.savefig(out_dir / filename)
            plt.close()
            print(f"  Saved {filename}")
        else:
            print(f"  Skipping {title} (no feature_importances_ found)")

    # Block Model
    if hasattr(model, 'model_block'):
        # Get features. For NestedXGClassifier (RF), it's in config_block.feature_cols
        # For XGB, it's similar.
        feats = model.config_block.feature_cols
        plot_imp(model.model_block, feats, "Block Model", "importance_block.png")
        
    # Accuracy Model
    if hasattr(model, 'model_accuracy'):
        feats = model.config_accuracy.feature_cols
        plot_imp(model.model_accuracy, feats, "Accuracy Model", "importance_accuracy.png")
        
    # Finish Model
    if hasattr(model, 'model_finish'):
        feats = model.config_finish.feature_cols
        plot_imp(model.model_finish, feats, "Finish Model", "importance_finish.png")

def plot_calibration(model, df_test, out_dir):
    print("\nGenerating Calibration Curves...")
    from sklearn.calibration import calibration_curve
    
    # Overall xG
    probs = model.predict_proba(df_test)[:, 1]
    # Target: is it a goal?
    y_true = (df_test['event'] == 'goal').astype(int)
    
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=15, strategy='uniform')
    
    plt.figure(figsize=(8, 8))
    plt.plot(prob_pred, prob_true, marker='o', label='Nested Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted xG')
    plt.ylabel('Actual Goal Rate')
    plt.title('Calibration Curve: Overall xG')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(out_dir / 'calibration_overall_detailed.png')
    plt.close()
    print("  Saved calibration_overall_detailed.png")

def plot_pdps(model, df_test, out_dir):
    print("\nGenerating Partial Dependence Plots...")
    # We'll use sklearn's PartialDependenceDisplay
    # But we need to pass the specific sub-model and compatible data
    
    # 1. Block Model PDP (Distance, Angle)
    # Block model uses specific features.
    
    # Prepare data for block model
    # We need to ensure columns match what the model expects
    
    common_features = ['distance', 'angle_deg', 'time_elapsed_in_period_s', 'score_diff']
    # Filter only those that exist in the model features
    block_feats = [f for f in common_features if f in model.config_block.feature_cols]
    
    if block_feats:
        # Sub-sample for speed
        X_eval = df_test[model.config_block.feature_cols].sample(1000, random_state=42).copy()
        # Handle Nans? RF doesn't like them. But impute should have handled it.
        X_eval = X_eval.fillna(0) # Safety
        
        print(f"  Computing Block PDPs for {block_feats}...")
        fig, ax = plt.subplots(figsize=(12, 4))
        PartialDependenceDisplay.from_estimator(
            model.model_block, X_eval, block_feats, ax=ax, n_cols=len(block_feats)
        )
        plt.suptitle("PDP: Block Probability (Higher = More Likely Blocked)")
        plt.tight_layout()
        plt.savefig(out_dir / 'pdp_block.png')
        plt.close()
    
    # 2. Finish Model PDP (Goal | On Net)
    # Features often include shot type. We'll stick to continuous for PDP lines.
    finish_feats = [f for f in common_features if f in model.config_finish.feature_cols]
    
    if finish_feats:
        # Restrict to On Net shots for realistic distribution
        df_on_net = df_test[df_test['is_on_net'] == 1]
        if len(df_on_net) > 1000:
            X_eval = df_on_net[model.config_finish.feature_cols].sample(1000, random_state=42).fillna(0)
        else:
            X_eval = df_on_net[model.config_finish.feature_cols].fillna(0)
            
        print(f"  Computing Finish PDPs for {finish_feats}...")
        fig, ax = plt.subplots(figsize=(12, 4))
        PartialDependenceDisplay.from_estimator(
            model.model_finish, X_eval, finish_feats, ax=ax, n_cols=len(finish_feats)
        )
        plt.suptitle("PDP: Finish Probability (Goal | On Net)")
        plt.tight_layout()
        plt.savefig(out_dir / 'pdp_finish.png')
        plt.close()

def plot_spatial_maps(model, out_dir):
    print("\nGenerating Spatial Heatmaps...")
    
    # Grid of x, y coordinates
    # NHL zone is roughly x=[25, 100], y=[-42.5, 42.5]
    # Standardize: x=0 is center ice?
    # Our data usually: x in [0, 100] for offensive zone? Or -100 to 100?
    # Let's check 'distance' calculation. usually dist = sqrt((x-89)^2 + y^2) or similar.
    # We'll generate dist/angle from x,y.
    
    # Assume offensive zone x: 25 to 100 (net at 89), y: -42.5 to 42.5
    xs = np.linspace(25, 100, 100)
    ys = np.linspace(-42.5, 42.5, 100)
    xx, yy = np.meshgrid(xs, ys)
    
    # Flatten
    grid_df = pd.DataFrame({
        'x': xx.ravel(),
        'y': yy.ravel()
    })
    
    # Calculate Dist/Angle (assuming net at 89,0)
    NET_X = 89.0
    grid_df['distance'] = np.sqrt((grid_df['x'] - NET_X)**2 + grid_df['y']**2)
    
    # Angle calculation
    # We want to match puck.rink.calculate_distance_and_angle logic approx
    # But vectorized for pandas/numpy
    # Vector from Goal to Shot: (x-89, y-0)
    # Reference vector: (-1, 0) since goal is at right end, facing left? 
    # Actually usually standard is 0 degrees = straight on.
    # checking rink.py: 
    # if goal_x > 0 (right goal), goalie faces -x. ref is (-1, 0)? No, ref is goal line?
    # standard in this repo seems to be: 0 is facing goal center.
    # Let's perform a rigorous calculation matching the provided vectors.
    
    # Simple approach that usually works for "Angle to Center":
    # angle = arctan2(y_diff, x_diff)
    # But usually we map 0 to center.
    # Let's trust that the model learns from the values provided by rink.py
    # which returns 0-360.
    
    # Vectorized implementation of rink.py logic for RIGHT goal (x=89):
    # goal_x = 89, goal_y = 0.
    # rx, ry = 0.0, -1.0 (from rink.py)
    # vx = x - 89
    # vy = y - 0
    # cross = 0*vy - (-1)*vx = vx
    # dot = 0*vx + (-1)*vy = -vy
    # angle_rad = atan2(vx, -vy)
    # degrees = -deg(angle_rad) % 360
    
    vx = grid_df['x'] - NET_X
    vy = grid_df['y']
    
    # Note: validation of rink.py logic:
    # Front (88,0): vx=-1, vy=0. cross=-1, dot=0. atan2(-1,0)=-1.57 (-90). -(-90)=90. Correct.
    # Behind (90,0): vx=1, vy=0. cross=1, dot=0. atan2(1,0)=1.57 (90). -(90)=-90 -> 270. Correct.
    
    cross = vx # since rx=0, ry=-1. cross = 0*y - (-1)*x_diff = x_diff
    dot = -vy  # dot = 0*x_diff + (-1)*y = -y
    
    angle_rad_ccw = np.arctan2(cross, dot)
    grid_df['angle_deg'] = (-np.degrees(angle_rad_ccw)) % 360.0
    
    # Add dummy variables for other features
    grid_df['game_state'] = '5v5'
    grid_df['time_elapsed_in_period_s'] = 600
    grid_df['period_number'] = 2
    grid_df['score_diff'] = 0
    grid_df['shot_type'] = 'wrist' # Using lowercase 'wrist' to match VOCAB
    grid_df['shoots_catches'] = 'L'
    grid_df['is_rebound'] = 0
    grid_df['is_rush'] = 0
    grid_df['last_event_type'] = 'Faceoff'
    grid_df['last_event_time_diff'] = 10
    grid_df['rebound_angle_change'] = 0
    grid_df['rebound_time_diff'] = 0
    
    # Predict
    # Need to handle OHE if model expects it inside predict_proba
    # The Nested class handles it usually.
    
    probs = model.predict_proba(grid_df)[:, 1]
    
    # Plot
    plt.figure(figsize=(10, 8))
    # Reshape
    p_grid = probs.reshape(xx.shape)
    
    plt.imshow(p_grid, extent=[25, 100, -42.5, 42.5], origin='lower', cmap='plasma', alpha=0.9, vmin=0, vmax=0.3)
    plt.colorbar(label='Expected Goals (xG)')
    plt.title('Spatial xG Map (Wrist Shot, 5v5)')
    plt.xlabel('X Coordinate (ft)')
    plt.ylabel('Y Coordinate (ft)')
    
    # Add Rink Lines (approx)
    plt.axvline(x=89, color='white', linestyle='-', alpha=0.5) # Goal Line
    plt.axvline(x=25, color='blue', linestyle='-', alpha=0.3) # Blue Line
    
    plt.savefig(out_dir / 'spatial_xg_wrist.png')
    plt.close()
    print("  Saved spatial_xg_wrist.png")
    
    # Block Probability Map
    # Requires calling sub-model or predict_proba_layer
    if hasattr(model, 'predict_proba_layer'):
        p_block = model.predict_proba_layer(grid_df, 'block')
        p_block_grid = p_block.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(p_block_grid, extent=[25, 100, -42.5, 42.5], origin='lower', cmap='Reds', alpha=0.9, vmin=0, vmax=1.0)
        plt.colorbar(label='Block Probability')
        plt.title('Spatial Block Probability Map')
        plt.axvline(x=89, color='black', linestyle='-', alpha=0.5)
        plt.savefig(out_dir / 'spatial_block.png')
        plt.close()
        print("  Saved spatial_block.png")


def plot_categorical_impact(model, df_test, out_dir):
    print("\nGenerating Categorical Impact Plots...")
    
    # Create synthetic dataset with native categorical support
    base_df = pd.DataFrame([
        {'distance': 30, 'angle_deg': 25, 'game_state': '5v5', 'is_rebound': 0, 'is_rush': 0}
    ])
    
    # Types to test (clean names) -> VOCAB value
    type_map = {
        'Wrist Shot': 'wrist',
        'Snap Shot': 'snap',
        'Slap Shot': 'slap',
        'Backhand': 'backhand',
        'Tip-In': 'tip-in',
        'Deflected': 'deflected',
        'Wrap-around': 'wrap-around'
    }
    
    recs = []
    
    for display_name, vocab_val in type_map.items():
        r = base_df.copy()
        r['shot_type_display'] = display_name
        r['shot_type'] = vocab_val
        
        # Fill other numeric features
        needed_cols = set()
        if hasattr(model, 'config_finish'): needed_cols.update(model.config_finish.feature_cols)
        if hasattr(model, 'config_accuracy'): needed_cols.update(model.config_accuracy.feature_cols)
        
        for c in needed_cols:
             if c not in r.columns and c != 'shot_type':
                 r[c] = 0
                 
        recs.append(r)
        
    syn_df = pd.concat(recs, ignore_index=True)
    
    # Ensure dtypes match what model expects (preprocess will handle it in pipeline, but predict_proba calls _prepare_df)
    # _prepare_df casts to category.
    
    try:
        probs = model.predict_proba(syn_df)[:, 1]
    except Exception as e:
        print(f"  Error predicting for categorical impact: {e}")
        import traceback
        traceback.print_exc()
        return

    syn_df['xG'] = probs
    
    print("  Synthetic Predictions (FINAL CHECK):")
    print(syn_df[['shot_type_display', 'xG']].to_string())
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=syn_df, x='xG', y='shot_type_display', palette='magma')
    plt.title('Model xG by Shot Type (Controlled: 30ft, 25deg, 5v5)')
    plt.xlabel('Predicted xG')
    plt.ylabel('Shot Type')
    plt.tight_layout()
    plt.savefig(out_dir / 'impact_shot_type_controlled.png')
    plt.close()
    print("  Saved impact_shot_type_controlled.png")

if __name__ == "__main__":
    main()
