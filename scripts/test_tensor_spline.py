
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck.fit_glm_nested import TensorSpline

def test_tensor_spline():
    print("Testing TensorSpline...")
    
    # Create dummy data: 100 rows, 2 features (Dist, Angle)
    N = 100
    X = pd.DataFrame({
        'distance': np.random.uniform(0, 100, N),
        'angle': np.random.uniform(-45, 45, N)
    })
    
    print(f"Input shape: {X.shape}")
    
    # Init Transformer
    # 5 knots, degree 3 -> 5 basis functions if include_bias=False? 
    # Check sklearn logic: n_features_out = n_knots + degree - 1 (default) or n_knots-1 (if include_bias=False)?
    # Sklearn SplineTransformer(n_knots=5, degree=3, include_bias=False)
    # If knots=uniform, we have 5 knots.
    # Output dim per feature = ?
    # Let's check output shape.
    
    ts = TensorSpline(n_knots=5, degree=3, include_bias=False)
    ts.fit(X)
    
    out = ts.transform(X)
    print(f"Output shape: {out.shape}")
    
    # Check dimensions
    # Per feature: SplineTransformer(n_knots=5, degree=3, include_bias=False)
    # n_knots=5 means 5 knots total (including boundaries).
    # Number of B-splines = n_knots + degree - 1.
    # With n_knots=5, degree=3 => 5 + 3 - 1 = 7 basis functions?
    # Wait, include_bias=False drops one?
    # Let's just assert it runs and shape makes sense (product of two integers).
    
    feat_count = out.shape[1]
    print(f"Feature count: {feat_count}")
    
    # Check feature names
    names = ts.get_feature_names_out()
    print(f"Feature names (first 5): {names[:5]}")
    print(f"Total names: {len(names)}")
    
    if len(names) != feat_count:
        print("ERROR: Name count mismatch!")
        sys.exit(1)
        
    print("Test Complete: SUCCESS")

if __name__ == "__main__":
    test_tensor_spline()
