
from sklearn.preprocessing import SplineTransformer
import numpy as np

def check_bias():
    X = np.arange(10).reshape(-1, 1)
    
    # With bias
    st_bias = SplineTransformer(n_knots=3, degree=3, include_bias=True)
    st_bias.fit(X)
    out_bias = st_bias.transform(X)
    print(f"With Bias: {out_bias.shape[1]} features")
    print(f"Full feature names: {st_bias.get_feature_names_out()}")
    
    # Without bias
    st_no_bias = SplineTransformer(n_knots=3, degree=3, include_bias=False)
    st_no_bias.fit(X)
    out_no_bias = st_no_bias.transform(X)
    print(f"No Bias: {out_no_bias.shape[1]} features")
    print(f"No Bias names: {st_no_bias.get_feature_names_out()}")
    
    # Compare
    # Check if first or last column is missing
    # We can check by values or logic.
    # Usually it drops the first one (index 0) to avoid collinearity with global intercept.
    
    # Let's check overlap
    # We expect out_no_bias output to match columns 1..N of out_bias? OR 0..N-1?
    
    # Check first column of no_bias vs first column of bias
    diff_start = np.abs(out_no_bias[:,0] - out_bias[:,0]).sum()
    print(f"Diff (NoBias[0] vs Bias[0]): {diff_start}")
    
    diff_shifted = np.abs(out_no_bias[:,0] - out_bias[:,1]).sum()
    print(f"Diff (NoBias[0] vs Bias[1]): {diff_shifted}")
    
    if diff_start < 1e-9:
        print("CONCLUSION: include_bias=False keeps the FIRST basis (drops LAST? Or no intercept in bias mode?)")
        # Wait, if include_bias=True produces N features.
        # If no_bias[0] == bias[0], then it kept index 0.
    elif diff_shifted < 1e-9:
        print("CONCLUSION: include_bias=False drops the FIRST basis (keeps 1..N).")
    else:
        print("CONCLUSION: Something else.")

if __name__ == "__main__":
    check_bias()
