
import sys
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck.fit_glm_nested import NestedGLM

def diagnoses_model_bias():
    print("--- Diagnosing Nested Model Predictions (Blocked Probability) ---")
    
    # Load Model
    model_path = os.path.join('analysis', 'xgs', 'xg_model_nested.joblib')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    print(f"Loading {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Check if we have the Block Model
    if not hasattr(model, 'model_block'):
        print("Model does not appear to be a NestedGLM (no model_block).")
        return
        
    print(f"Model Type: {type(model)}")
    print(f"Features: {model.features}")
    
    # Define Grid
    x_range = np.linspace(25, 89, 20)
    y_range = np.linspace(-40, 40, 20)
    X, Y = np.meshgrid(x_range, y_range)
    
    flat_x = X.flatten()
    flat_y = Y.flatten()
    
    # Calculate Dist/Angle
    # Standard Net: (89, 0)
    # Angle physics? 
    # Usually: 0 deg = straight on. 90 deg = goal line.
    
    def get_geom(x, y):
        dx = 89 - x
        dy = y # or 0 - y
        dist = np.hypot(dx, dy)
        angle = np.degrees(np.arctan2(np.abs(dy), dx))
        return dist, angle

    dists, angles = [], []
    for x, y in zip(flat_x, flat_y):
        d, a = get_geom(x, y)
        dists.append(d)
        angles.append(a)
        
    # Construct DF
    # We need dummy values for other features
    df = pd.DataFrame({
        'distance': dists,
        'angle_deg': angles,
        # Dummy cols
        'shot_type': ['wrist'] * len(flat_x),
        'shooter_role': ['F'] * len(flat_x), 
        'shoots_catches': ['L'] * len(flat_x),
        'last_event_type': ['Faceoff'] * len(flat_x),
        'game_state': ['5v5'] * len(flat_x),
        'speed_from_last_event': [0.0] * len(flat_x), # Ensure numeric
        'distance_from_last_event': [0.0] * len(flat_x),
        'time_since_last_event': [5.0] * len(flat_x)
    })
    
    # Ensure all required features exist
    for f in model.features:
        if f not in df.columns:
            if f == 'dist_angle': continue # auto-generated
            df[f] = 0 # default fill
            
    print("\nPredicting Block Probability...")
    # model.predict_proba_layer(df, 'block') returns P(Blocked)
    try:
        p_blocked = model.predict_proba_layer(df, 'block')
    except Exception as e:
        print(f"Prediction failed: {e}")
        # Try manual block model access
        try:
            feats = [f for f in model.features if f != 'shot_type']
            # Enrich
            if model.use_splines:
                df = model._enrich_interaction(df)
            p_blocked = model.model_block.predict_proba(df[feats])[:, 1]
        except Exception as e2:
            print(f"Manual block prediction failed: {e2}")
            return

    df['x'] = flat_x
    df['y'] = flat_y
    df['p_blocked'] = p_blocked
    
    # INSPECT POINT ZONE (X < 53)
    # User specified x < 53.
    mask_high = (df['x'].between(30, 53)) & (df['y'].abs() < 20)
    high_slot = df[mask_high]
    
    print("\n--- Point Zone Predictions (X=30-53) ---")
    print(high_slot[['x', 'y', 'distance', 'p_blocked']].describe())
    
    # Sample points
    print("\nSample Points:")
    print(high_slot[['x', 'y', 'distance', 'p_blocked']].head(10))
    
    avg_p = high_slot['p_blocked'].mean()
    print(f"\nAverage P(Blocked) in Point Zone: {avg_p:.3f}")
    
    if avg_p > 0.8:
        print("CONFIRMED: Model predicts >80% Blocked in Point Zone.")
    else:
        print("DISPROVED: Model predicts low Blocked probability.")

    # --- PASS 2: Defenceman Slapshot ---
    print("\n--- PASS 2: Defenceman Slapshot (Role=D, Type=Slap) ---")
    df['shooter_role'] = 'D'
    df['shot_type'] = 'slap'
    
    # Re-predict
    try:
        p_blocked_2 = model.predict_proba_layer(df, 'block')
    except:
        # Fallback manual
        if model.use_splines: df = model._enrich_interaction(df)
        p_blocked_2 = model.model_block.predict_proba(df[feats])[:, 1]
    
    df['p_blocked'] = p_blocked_2
    
    mask_high = (df['x'].between(30, 53)) & (df['y'].abs() < 20)
    high_slot = df[mask_high]
    
    print(high_slot[['x', 'y', 'distance', 'p_blocked']].describe())
    avg_p_2 = high_slot['p_blocked'].mean()
    print(f"\nAverage P(Blocked) [D-Man/Slap] in Point Zone: {avg_p_2:.3f}")
    
    if avg_p_2 > 0.8:
        print("CONFIRMED: Model predicts >80% Blocked for D-Men Slapshots in Point Zone.")


if __name__ == "__main__":
    diagnoses_model_bias()
