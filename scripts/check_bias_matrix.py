
from sklearn.preprocessing import SplineTransformer
import numpy as np
import pandas as pd

def check():
    X = np.arange(5).reshape(-1, 1)
    
    st_true = SplineTransformer(n_knots=3, degree=3, include_bias=True)
    st_true.fit(X)
    mat_true = st_true.transform(X)
    
    st_false = SplineTransformer(n_knots=3, degree=3, include_bias=False)
    st_false.fit(X)
    mat_false = st_false.transform(X)
    
    print("Include Bias = True (Shape: {})".format(mat_true.shape))
    print(mat_true)
    
    print("\nInclude Bias = False (Shape: {})".format(mat_false.shape))
    print(mat_false)
    
    # Check strict equality of columns
    # We expect mat_false columns to be a subset of mat_true
    
    print("\nComparison:")
    n_cols_true = mat_true.shape[1]
    n_cols_false = mat_false.shape[1]
    
    for i in range(n_cols_false):
        col_f = mat_false[:, i]
        # Find match in true
        match = -1
        for j in range(n_cols_true):
            col_t = mat_true[:, j]
            if np.allclose(col_f, col_t):
                match = j
                break
        print(f"False Column {i} matches True Column {match}")

if __name__ == "__main__":
    check()
