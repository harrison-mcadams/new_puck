"""debug_prototype_shot.py

Probes the NestedGLM model with synthetic "High Danger" shots to diagnose 
why stats like Wrist Shots from 5ft are not receiving High xG.
"""

import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import features as feature_util

def main():
    print("--- Probing Nested GLM with Synthetic Data ---")
    
    # Load Model
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    try:
        clf = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
        print(f"Model type: {type(clf).__name__}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create Synthetic Data
    # varying distance, fixed angle (0 = center), shot_type='wrist'
    distances = [2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]
    
    records = []
    for is_reb in [0, 1]:  # Test both regular and rebound
        for d in distances:
            rec = {
                'distance': d,
                'angle_deg': 90.0, # 90 deg seemed like the top shot's angle
                'shot_type': 'wrist',          # Standardized lowercase
                'shoots_catches': 'L',
                'shooter_role': 'F',
                'game_state': '5v5',
                'score_diff': 0,
                'period_number': 2,
                'time_elapsed_in_period_s': 600,
                'total_time_elapsed_s': 1800,
                'is_rebound': is_reb,
                'rebound_angle_change': 0.0 if not is_reb else 45.0,
                'rebound_time_diff': 0.0 if not is_reb else 1.0,
                'is_rush': 0,
                'last_event_type': 'Faceoff' if not is_reb else 'shot-on-goal',
                'last_event_time_diff': 10.0 if not is_reb else 1.0,
                # Features that must exist for pipeline (will be filled if missing logic inside pipeline, but we provide explicitly)
                'event': 'shot-on-goal' # Dummy for logic that checks event type? 
                                        # NestedGLM predict_proba doesn't check event type, just features.
            }
            records.append(rec)
        
    df = pd.DataFrame(records)
    
    # Ensure all expected features are present (using logic from data_pipeline formatting if needed)
    # The models use `clf.features`.
    # Let's check what features the model expects.
    expected_feats = clf.features
    print(f"\nModel expects {len(expected_feats)} features.")
    
    # Fill missing columns with defaults to match training
    for f in expected_feats:
        if f not in df.columns:
            # Sane defaults
            if 'diff' in f: df[f] = 0.0
            else: df[f] = 0
            
    # Predict
    # Overall xG
    probs = clf.predict_proba(df)[:, 1]
    
    # Layer Predictions
    p_block = clf.predict_proba_layer(df, 'block')
    p_acc = clf.predict_proba_layer(df, 'accuracy')
    p_finish = clf.predict_proba_layer(df, 'finish')
    
    import os
    print(f"Current CWD: {os.getcwd()}")
    
    # Display
    out_file = 'analysis/probe_output_spline_REAL.txt'
    with open(out_file, 'w') as f:
        f.write("Rec | Dist | Reb | xG | P(Block) | P(Acc) | P(Fin)\n")
        for i in range(len(df)):
            row = df.iloc[i]
            p_unblocked = 1.0 - p_block[i]
            f.write(f"{i} | {row['distance']} | {row['is_rebound']} | {probs[i]:.4f} | {p_block[i]:.4f} | {p_acc[i]:.4f} | {p_finish[i]:.4f}\n")
    print(f"Wrote results to {out_file}")

    print("\n--- Finish Model Coefficients ---")
    try:
        # Access the underlying LogisticRegression model
        # Structure: clf.model_finish is a Pipeline. Step 'clf' is the model.
        # But wait, NestedGLM implementation details in fit_glm_nested.py:
        # self.model_finish = Pipeline([ ('preprocessor', ...), ('clf', LogisticRegression) ])
        model_fin = clf.model_finish.named_steps['clf']
        preproc = clf.model_finish.named_steps['preprocessor']
        
        # Get feature names from preprocessor
        # This is tricky with ColumnTransformer + Pipelines
        feature_names = []
        try:
            # Fallback for simpler inspections
            if hasattr(preproc, 'get_feature_names_out'):
                feature_names = preproc.get_feature_names_out()
        except:
            feature_names = [f"Feature_{i}" for i in range(model_fin.coef_.shape[1])]
            
        coefs = model_fin.coef_[0]
        intercept = model_fin.intercept_[0]
        
        print(f"Intercept: {intercept:.4f}")
        
        # Print top/bottom coefficients
        coef_dict = dict(zip(feature_names, coefs))
        sorted_coefs = sorted(coef_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nTop 5 Positives (Increases Goal Prob):")
        for k, v in sorted_coefs[:5]:
            print(f"  {k}: {v:.4f}")
            
        print("\nTop 5 Negatives (Decreases Goal Prob):")
        for k, v in sorted_coefs[-5:]:
            print(f"  {k}: {v:.4f}")
            
        # Look for Distance specifically
        print("\nDistance-related Coefs:")
        for k, v in coef_dict.items():
            if 'distance' in k.lower():
                print(f"  {k}: {v:.4f}")
                
    except Exception as e:
        print(f"Failed to inspect coefficients: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
