
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))

def debug_knots():
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
        print("FOUND TensorSpline")
        if hasattr(tensor_transformer, 'feature_names_in_'):
            print(f"Features In: {tensor_transformer.feature_names_in_}")
        else:
            print("No feature names in.")
            
        for i, st in enumerate(tensor_transformer.splines_):
            knots = st.bsplines_[0].t
            print(f"Spline {i} Knots (range): {knots.min()} to {knots.max()}")
            print(f"Spline {i} Knots (full): {knots}")
            
if __name__ == "__main__":
    debug_knots()
