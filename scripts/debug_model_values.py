
import sys
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import fit_glm_nested

np.set_printoptions(threshold=sys.maxsize, linewidth=200, suppress=True)

def debug_pipeline_steps(pipeline, X, layer_name):
    print(f"\n--- Debugging Layer: {layer_name} ---")
    
    # 1. Transform
    pre = pipeline.named_steps['preprocessor']
    num_trans = pre.named_transformers_['num']
    
    # Get indices
    feature_names = pre.transformers_[0][2] # List of num features
    
    target_feats = ['distance', 'angle_deg']
    
    X_num = X[feature_names].copy()
    
    spline_step = num_trans.named_steps['spline']
    scaler_step = num_trans.named_steps['scaler']
    
    # Output File
    out_file = "debug_output.txt"
    mode = "a" if os.path.exists(out_file) else "w"
    
    with open(out_file, mode) as f:
        f.write(f"\n\n--- Layer: {layer_name} ---\n")
        
        # Calculate full pipeline transform first to get scaler steps easily?
        # Manually step through for target features
        
        X_numpy = X_num.to_numpy()
        X_spline = spline_step.transform(X_numpy)
        X_scaled = scaler_step.transform(X_spline)
        
        clf = pipeline.named_steps['clf']
        coefs_all = clf.coef_[0]
        
        n_out_total = spline_step.n_features_out_
        n_feats = len(feature_names)
        n_out_per_feat = n_out_total // n_feats
        
        for feat in target_feats:
            if feat not in feature_names:
                continue
                
            idx = feature_names.index(feat)
            
            # Knots
            knots = spline_step.bsplines_[idx].t
            degree = spline_step.bsplines_[idx].k
            
            f.write(f"\nFeature: {feat}\n")
            f.write(f"Knots ({len(knots)}): \n{knots}\n")
            f.write(f"Degree: {degree}\n")
            
            # Values
            start = idx * n_out_per_feat
            end = start + n_out_per_feat
            
            basis = X_spline[0, start:end]
            scaled = X_scaled[0, start:end]
            mean = scaler_step.mean_[start:end]
            scale = scaler_step.scale_[start:end]
            coefs = coefs_all[start:end]
            
            f.write(f"Input Val: {X_num[feat].iloc[0]}\n")
            f.write(f"Basis (cols {start}-{end}):\n{basis}\n")
            f.write(f"Scaled:\n{scaled}\n")
            f.write(f"Scaler Mean:\n{mean}\n")
            f.write(f"Scaler Scale:\n{scale}\n")
            f.write(f"Coefs:\n{coefs}\n")
            f.write(f"Contrib: {np.dot(scaled, coefs)}\n")
            
        f.write(f"Layer Intercept: {clf.intercept_[0]}\n")

def main():
    model_path = "analysis/xgs/xg_model_nested.joblib"
    model = joblib.load(model_path)
    
    # Clear file
    if os.path.exists("debug_output.txt"):
        os.remove("debug_output.txt")
        
    row = {
        'distance': 10.0,
        'angle_deg': 0.0,
        'dist_angle': 0.0,
        'game_state': '5v5',
        'score_diff': 0,
        'period_number': 2,
        'time_elapsed_in_period_s': 600,
        'total_time_elapsed_s': 1800,
        'shot_type': 'wrist',
        'shoots_catches': 'L',
        'shooter_role': 'F',
        'is_rush': 0,
        'is_rebound': 0,
        'rebound_angle_change': 0,
        'rebound_time_diff': 0,
        'last_event_type': 'faceoff',
        'last_event_time_diff': 10,
        'dist_from_last_event': 20,
        'speed_from_last_event': 2.0,
        'angle_change_last_event': 0
    }
    
    df = pd.DataFrame([row])
    
    # Debug Layers
    feats_block = [f for f in model.features if f != 'shot_type']
    debug_pipeline_steps(model.model_block, df[feats_block], "Block")
    debug_pipeline_steps(model.model_acc, df[model.features], "Accuracy")
    debug_pipeline_steps(model.model_finish, df[model.features], "Finish")
    
    # Full Model Predict
    pred = model.predict_proba(df)
    with open("debug_output.txt", "a") as f:
        f.write(f"\nFull Nested xG: {pred[:, 1][0]}\n")

if __name__ == "__main__":
    main()
