"""Debug: Compare Python model predictions with dashboard spatial grid values.
Inspects whether the spatial grid and non-spatial contributions are reasonable.
"""
import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'puck'))

import joblib
import numpy as np
import pandas as pd

model_path = "analysis/xgs/xg_model_nested_tensor.joblib"
print(f"Loading model from {model_path}...")
model = joblib.load(model_path)

print(f"Model type: {type(model).__name__}")
print(f"Features: {model.features}")
print(f"use_splines: {model.use_splines}")

# Test points: slot, point, behind net
test_points = [
    {"label": "Slot (high danger)", "distance": 15, "angle_deg": 90},
    {"label": "Circle (medium)", "distance": 30, "angle_deg": 70},
    {"label": "Point (low danger)", "distance": 60, "angle_deg": 85},
    {"label": "Behind net", "distance": 15, "angle_deg": 180},
]

# Default values for non-spatial features
defaults = {
    'game_state': '5v5',
    'score_diff': 0,
    'period_number': 2,
    'time_elapsed_in_period_s': 600,
    'total_time_elapsed_s': 1800,
    'is_home': 1,
    'shot_type': 'wrist',
    'shoots_catches': 'L',
    'is_rebound': 0,
    'rebound_angle_change': 0,
    'rebound_time_diff': 0,
    'is_rush': 0,
    'last_event_type': 'faceoff',
    'last_event_time_diff': 10,
    'dist_from_last_event': 20,
    'speed_from_last_event': 2,
    'angle_change_last_event': 0,
    'shooter_role': 'F',
}

print("\n" + "="*80)
print("PYTHON MODEL PREDICTIONS AT KEY LOCATIONS")
print("="*80)

for pt in test_points:
    row = {**defaults, 'distance': pt['distance'], 'angle_deg': pt['angle_deg']}
    df = pd.DataFrame([row])
    
    # Block
    block_feats = [f for f in model.features if f != 'shot_type']
    p_block = model.model_block.predict_proba(df[block_feats])[:, 1][0]
    
    # Accuracy
    p_acc = model.model_acc.predict_proba(df[model.features])[:, 1][0]
    
    # Finish
    p_fin = model.model_finish.predict_proba(df[model.features])[:, 1][0]
    
    # xG
    p_xg = (1 - p_block) * p_acc * p_fin
    
    print(f"\n{pt['label']} (d={pt['distance']}, a={pt['angle_deg']}°):")
    print(f"  P(blocked) = {p_block:.4f}")
    print(f"  P(on net | unblocked) = {p_acc:.4f}")
    print(f"  P(goal | on net) = {p_fin:.4f}")
    print(f"  xG = {p_xg:.4f}")

# Now inspect the pipeline internals for each layer
print("\n\n" + "="*80)
print("PIPELINE STRUCTURE AND COEFFICIENT RANGES")
print("="*80)

for layer_name, pipeline in [("block", model.model_block), ("accuracy", model.model_acc), ("finish", model.model_finish)]:
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    
    print(f"\n--- {layer_name.upper()} ---")
    print(f"  Intercept: {clf.intercept_[0]:.4f}")
    print(f"  Total coefs: {len(clf.coef_[0])}")
    
    idx = 0
    for name, trans, cols in preprocessor.transformers_:
        if name == 'remainder': continue
        try:
            out_names = trans.get_feature_names_out()
        except:
            out_names = [f"x{i}" for i in range(10)]
        n_out = len(out_names)
        
        coefs = clf.coef_[0][idx:idx+n_out]
        print(f"  '{name}' ({list(cols)}): {n_out} outputs, coef range [{coefs.min():.4f}, {coefs.max():.4f}], abs mean {np.abs(coefs).mean():.4f}")
        idx += n_out

# Now check what the spatial grid looks like
print("\n\n" + "="*80)
print("SPATIAL GRID VALUE RANGES (from compute_spatial_grid)")
print("="*80)

from puck.rink import calculate_distance_and_angle

for layer_name, pipeline in [("block", model.model_block), ("accuracy", model.model_acc), ("finish", model.model_finish)]:
    preprocessor = pipeline.named_steps['preprocessor']
    clf = pipeline.named_steps['clf']
    
    # Find spatial_tensor
    tensor_transformer = None
    for name, trans, cols in preprocessor.transformers_:
        if name == 'spatial_tensor':
            tensor_transformer = trans
            break
    
    if tensor_transformer is None:
        print(f"  {layer_name}: No spatial_tensor transformer found!")
        continue
    
    all_out_feats = preprocessor.get_feature_names_out()
    spatial_indices = [i for i, f in enumerate(all_out_feats) if f.startswith('spatial_tensor__')]
    spatial_coefs = clf.coef_[0][spatial_indices]
    
    # Generate grid
    X_POINTS, Y_POINTS = 50, 43
    xs = np.linspace(0, 100, X_POINTS)
    ys = np.linspace(-42.5, 42.5, Y_POINTS)
    
    grid_rows = []
    for y in ys:
        for x in xs:
            dist, ang = calculate_distance_and_angle(x, y, 89.0, 0.0)
            grid_rows.append({'distance': dist, 'angle_deg': ang})
    
    grid_df = pd.DataFrame(grid_rows)
    X_trans = tensor_transformer.transform(grid_df)
    scores = X_trans @ spatial_coefs
    
    print(f"  {layer_name}: spatial scores range [{scores.min():.4f}, {scores.max():.4f}], mean={scores.mean():.4f}")
    
    # Now compute log-odds at key points (spatial + intercept)
    # Also compute the NON-spatial contribution for defaults
    
    # Get non-spatial coefs
    non_spatial_idx = [i for i in range(len(clf.coef_[0])) if i not in spatial_indices]
    non_spatial_coefs = clf.coef_[0][non_spatial_idx]
    
    # Transform defaults through non-spatial pipeline
    row = {**defaults}
    df = pd.DataFrame([row])
    
    if layer_name == 'block':
        feats = [f for f in model.features if f != 'shot_type']
    else:
        feats = model.features
    
    full_transformed = preprocessor.transform(df[feats])
    non_spatial_features = full_transformed[0, non_spatial_idx]
    non_spatial_score = np.dot(non_spatial_features, non_spatial_coefs)
    
    print(f"  {layer_name}: non-spatial score for defaults = {non_spatial_score:.4f}")
    print(f"  {layer_name}: intercept = {clf.intercept_[0]:.4f}")
    print(f"  {layer_name}: total log-odds range = [{clf.intercept_[0] + non_spatial_score + scores.min():.4f}, {clf.intercept_[0] + non_spatial_score + scores.max():.4f}]")
    
    # Convert to probabilities
    lo_min = clf.intercept_[0] + non_spatial_score + scores.min()
    lo_max = clf.intercept_[0] + non_spatial_score + scores.max()
    from scipy.special import expit
    print(f"  {layer_name}: prob range = [{expit(lo_min):.4f}, {expit(lo_max):.4f}]")
