
import sys
import os
import pandas as pd
import numpy as np
import joblib
import math

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import impute
from puck.fit_nested_xgs import NestedXGClassifier

def main():
    print("--- Debugging Imputation & Model Behavior (Attempt 2) ---")
    
    # Target Case:
    # 99911  2025020337  blocked-shot       NaN  178.241409   96.766175  0.667073
    
    dist = 178.241409
    angle_deg = 96.766175
    angle_rad = math.radians(angle_deg)
    
    # Net at (89, 0)
    # Angle usually defined from center line? Or from net?
    # In calculate_geometry: abs(degrees(arctan(abs(dy/dx))))
    # Wait. calculate_geometry returns 0-90 usually if using absolute dy/dx.
    # But fillna(90.0) covers dx=0.
    # If angle is 96, it must imply behind the net? Or raw angle calc allows >90?
    # puck.parse usually parses angle 0-180?
    
    # Let's reverse engineer X/Y from Dist/Angle.
    # Logic in standard NHL data:
    # x is distance from center (0) to end (100). Net at 89.
    # This implies coordinates.
    # If we assume generic X/Y.
    
    # Let's try to pass X/Y that yields dist=178, angle=96.
    # This is tricky without knowing the exact angle convention.
    # BUT, impute.py imputes based on X/Y.
    # IF the dataframe has X/Y.
    
    # If the dataframe passed to _predict_xgs has 'distance' and 'angle_deg' but NO X/Y?
    # impute.py checks:
    # bx = df_out.loc[mask_blocked, x_col]
    # If x_col raises Key Error?
    
    # Let's create a DF with X/Y approximately correct.
    # If angle=96, it's > 90.
    # dy/dx relation: tan(96) is very large negative.
    # This implies dx is small? NO.
    # If angle is calculated as `degrees(arctan2(y, x_dist))`, then 96 means slightly behind net?
    # If net is at 89.
    # x = 89 + small?
    # But distance is 178.
    # So y must be large.
    
    # Let's try passing 'distance' and 'angle' directly to model, SKIPPING imputation, first.
    # To see if the MODEL produces 0.66 for these features.
    
    df_feat = pd.DataFrame({
        'distance': [dist],
        'angle_deg': [angle_deg],
        'game_state': ['5v5'],
        'is_net_empty': [0],
        'shot_type': ['Unknown']
    })
    
    print("\n[Input Features]")
    print(df_feat)
    
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    print(f"\n[Loading Model: {model_path}]")
    clf = joblib.load(model_path)
    
    # Predict directly
    try:
        probs = clf.predict_proba(df_feat)[:, 1]
        print(f"Predicted xG (Direct Features): {probs[0]}")
    except Exception as e:
        print(f"Direct prediction failed: {e}")
        
    
    # Now try full pipeline with Imputation simulation
    # Assume X/Y such that dist=178.
    # Case A: Defensive Zone (-89, 0). Dist=178. Angle=0.
    # Case B: Behind Net? (89+small, 178). Dist=178. Angle=90.
    
    # If existing xG was 0.66, and features were 178/96.
    # The printed features in debug_xg_values were from the RESULT dataframe.
    # If imputation ran, the result dataframe would show IMPUTED features?
    # No, `impute.py` updates 'distance' and 'angle_deg' IN PLACE.
    # So 178/96 ARE the imputed features used for prediction?
    # OR `debug_xg_values` printed the `attempts` df which is `df_pred`.
    # `df_pred` is the ORIGINAL DF. `_predict_xgs` modifies a COPY.
    # So 178/96 are the ORIGINAL features.
    
    # So we need to know what they were IMPUTED to.
    # We can guess based on X/Y.
    # If Dist=178, Angle=96.
    # X/Y?
    
    # Let's assume input X/Y matches 178/96.
    # Net X=89.
    # dx = dist * cos(angle). dy = dist * sin(angle).
    # (Assuming angle is from x-axis heading towards net?)
    # NHL API angle: 0 is center line? No. usually 0 is straight at net.
    # Let's try to construct X/Y.
    
    # If we can't reconstruct, we can just check if Model(178, 96) = 0.66.
    # If direct prediction gives 0.66, then the model is broken for 178/96.
    
    pass

if __name__ == "__main__":
    main()
