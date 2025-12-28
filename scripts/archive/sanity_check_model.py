
import sys
import os
import pandas as pd
import numpy as np
import joblib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs

def sanity_check():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 1. Feature Check: Unknown Shot Type in Unblocked Shots (Model Internal)
    # Ideally, we check the model's learned priors or something, but we can't easily.
    # Instead, let's predict on a synthetic 'Unknown' shot and see if it's 0.0 like before.
    print("\n[Sanity Check 1] Prediction on Synthetic Events")
    
    # Create synthetic dataframe
    # Case A: Standard Wrist Shot from slot (High Quality)
    # Case B: Slap Shot from point (Med Quality)
    # Case C: Unknown type (Should NOT be 0.0 accuracy anymore, or at least handled gracefully)
    # Case D: Blocked Shot (Should have high Block prob, low xG)
    
    synthetic_data = [
        {'distance': 10, 'angle_deg': 0, 'game_state': '5v5', 'is_net_empty': 0, 'shot_type': 'wrist', 'desc': 'Close Wrist Shot'},
        {'distance': 60, 'angle_deg': 0, 'game_state': '5v5', 'is_net_empty': 0, 'shot_type': 'slap', 'desc': 'Point Slap Shot'},
        {'distance': 15, 'angle_deg': 45, 'game_state': '5v5', 'is_net_empty': 0, 'shot_type': 'Unknown', 'desc': 'Unknown Type (e.g. missing data)'},
        {'distance': 25, 'angle_deg': 10, 'game_state': '5v5', 'is_net_empty': 0, 'shot_type': 'wrist', 'desc': 'Blocked Shot Simulation (features same as wrist)'} 
        # Note: 'is_blocked' target depends on model layer, for prediction we pass features.
        # But 'shot_type' of blocked shots is usually Unknown during training imputation. 
        # But here we pass 'wrist' to see if it predicts block prob based on location? 
        # Actually Block Model uses location info primarily.
    ]
    
    df_syn = pd.DataFrame(synthetic_data)
    
    # Predict Full xG
    print("Running Full Prediction...")
    probs = clf.predict_proba(df_syn)[:, 1]
    df_syn['xG'] = probs
    
    # Predict Layers if possible
    print("Running Layer Predictions...")
    try:
        df_syn['P_Block'] = clf.predict_proba_layer(df_syn, 'block')
        # Accuracy is P(On Net | Unblocked)
        df_syn['P_Acc'] = clf.predict_proba_layer(df_syn, 'accuracy') 
        # Finish is P(Goal | On Net)
        df_syn['P_Finish'] = clf.predict_proba_layer(df_syn, 'finish')
    except Exception as e:
        print(f"Layer prediction failed: {e}")

    print(df_syn[['desc', 'shot_type', 'distance', 'P_Block', 'P_Acc', 'P_Finish', 'xG']].to_string())

    # Check 1: Close Wrist > Point Slap
    xg_close = df_syn.loc[0, 'xG']
    xg_far = df_syn.loc[1, 'xG']
    print(f"\ncheck: Low Dist xG ({xg_close:.3f}) > High Dist xG ({xg_far:.3f})? {'PASS' if xg_close > xg_far else 'FAIL'}")

    # Check 2: Unknown Type shouldn't be 0.0 Accuracy (unless truly terrible)
    # In leakage model, Unknown Accuracy was ~0. In clean model, should be average-ish?
    acc_unknown = df_syn.loc[2, 'P_Acc']
    print(f"check: Unknown Accuracy ({acc_unknown:.3f}) > 0.1? {'PASS' if acc_unknown > 0.1 else 'FAIL (Leakage might still exist?)'}")

    # 3. Layer Logic
    # xG ~= (1 - P_Block) * P_Acc * P_Finish
    # Let's verify calculation for row 0
    row0 = df_syn.iloc[0]
    calc_xg = (1 - row0['P_Block']) * row0['P_Acc'] * row0['P_Finish']
    print(f"check: xG Calculation Consistency ({row0['xG']:.4f} vs {calc_xg:.4f})? {'PASS' if abs(row0['xG'] - calc_xg) < 0.001 else 'FAIL'}")

if __name__ == "__main__":
    sanity_check()
