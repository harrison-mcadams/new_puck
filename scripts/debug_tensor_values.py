
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

# Add project root
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
    model_path = Path('analysis/xgs/xg_model_nested_tensor.joblib')
    print(f"Loading model from {model_path}...")
    try:
        clf = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    preprocessor = clf.model_block.named_steps['preprocessor']
    
    tensor_transformer = None
    
    for name, trans, cols in preprocessor.transformers_:
        print(f"Checking transformer: {name} ({type(trans).__name__})")
        
        # Check if it is the TensorSpline directly
        if 'TensorSpline' in type(trans).__name__:
            tensor_transformer = trans
            break
            
        # Check if it's a pipeline containing TensorSpline
        if isinstance(trans, Pipeline):
             for step_name, step_trans in trans.steps:
                 if 'TensorSpline' in type(step_trans).__name__:
                     print(f"  Found TensorSpline in sub-pipeline step: {step_name}")
                     tensor_transformer = step_trans
                     break
        if tensor_transformer: break
            
    if tensor_transformer:
        print("\nFOUND TensorSpline!")
        
        # Test Point
        x_test = 50.0
        y_test = 0.0
        dist, angle = calc_feats(x_test, y_test)
        X_df = pd.DataFrame({'distance': [dist], 'angle_deg': [angle]})
        
        print(f"\nTest Point: x={x_test}, y={y_test} -> Dist={dist:.4f}, Angle={angle:.4f}")
        
        # Inspect Individual Splines
        print("\n--- Individual Splines ---")
        bases = []
        for i, st in enumerate(tensor_transformer.splines_):
            feat_name = X_df.columns[i] if i < len(X_df.columns) else f"Input_{i}"
            if i < X_df.shape[1]:
                val = X_df.iloc[:, [i]]
                b = st.transform(val)
                bases.append(b)
                # Print output vector
                print(f"Feature {i} ({feat_name}):")
                print(f"  Knots: {st.bsplines_[0].t}")
                print(f"  Output Vector (len={b.shape[1]}): {b[0]}")
                
        # Tensor Out
        print("\n--- Tensor Product ---")
        t_out = tensor_transformer.transform(X_df)
        print(f"Tensor Out Shape: {t_out.shape}")
        print(f"Tensor Out Vector (first 10): {t_out[0][:10]}...")
        
        # Manual Check
        if len(bases) >= 2:
            b0 = bases[0][0]
            b1 = bases[1][0]
            manual = []
            
            # Check Order: B0 outer B1 vs B1 outer B0
            # Implementation says: result (B0) outer next (B1)
            # Flattened: B0[0]*B1[0], B0[0]*B1[1] ...
            
            for v0 in b0:
                for v1 in b1:
                    manual.append(v0*v1)
            
            manual = np.array(manual)
            diff = np.abs(manual - t_out[0]).sum()
            print(f"Manual Check Diff (B0 x B1): {diff:.6f}")
            
    else:
        print("Could not find TensorSpline transformer.")

if __name__ == "__main__":
    debug_values()
