
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer

sys.path.append(str(Path(__file__).resolve().parent.parent))

def calc_feats(x, y):
    goal_x = 89.0
    dx = x - goal_x
    dy = y
    dist = np.sqrt(dx**2 + dy**2)
    rx, ry = 0.0, -1.0 
    cross = rx * dy - ry * dx
    dot = rx * dx + ry * dy
    angle_rad_ccw = np.arctan2(cross, dot)
    angle_deg = (-np.degrees(angle_rad_ccw)) % 360.0
    return dist, angle_deg

def debug_values():
    print("DEBUG START")
    model_path = Path('analysis/xgs/xg_model_nested_tensor.joblib')
    try:
        clf = joblib.load(model_path)
    except:
        return

    preprocessor = clf.model_block.named_steps['preprocessor']
    tensor_transformer = None
    
    for name, trans, cols in preprocessor.transformers_:
        if 'TensorSpline' in type(trans).__name__:
            tensor_transformer = trans
            break
        if isinstance(trans, Pipeline):
             for step_name, step_trans in trans.steps:
                 if 'TensorSpline' in type(step_trans).__name__:
                     tensor_transformer = step_trans
                     break
        if tensor_transformer: break
            
    if tensor_transformer:
        x_val = 50.0
        y_val = 0.0
        d, a = calc_feats(x_val, y_val)
        X_df = pd.DataFrame({'distance': [d], 'angle_deg': [a]})
        
        print(f"Point: {d}, {a}")
        
        for i, st in enumerate(tensor_transformer.splines_):
            val = X_df.iloc[:, [i]]
            b_model = st.transform(val)[0]
            
            # Reference with Bias
            knots = st.bsplines_[0].t
            deg = st.bsplines_[0].k
            # Recreate transformer manually
            # We can't easily recreate exact state without fitting, but we have knots.
            # We can use scipy or just logic.
            # Let's use clean SplineTransformer with SAME knots if possible, or just observe.
            
            print(f"Feature {i} Model Output (len={len(b_model)}): {b_model}")
            
            # Hypothesis: If drop last, then b_model corresponds to bases 0, 1, ... N-2
            # Check sum. If include_bias=False, sum < 1 (unless dropped one is 0)
            print(f"Sum: {b_model.sum()}")
            
            # Check against full basis sum (should be 1)
            # If sum is 1, then we didn't drop a non-zero basis? 
            # (Test point might be where dropped basis is 0).
            
    else:
        print("No TensorSpline found")

if __name__ == "__main__":
    debug_values()
