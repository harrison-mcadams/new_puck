
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer
from sklearn.utils.validation import check_is_fitted

class TensorSpline(BaseEstimator, TransformerMixin):
    """
    Tensor Product Spline Transformer.
    
    Takes N input features, applies independent SplineTransformers to each,
    and computes the row-wise outer product (tensor product) of the resulting basis functions.
    
    Args:
        n_knots (int): Number of knots for splines.
        degree (int): Degree of splines.
        include_bias (bool): Whether to include bias term (usually False for tensor logic if intercept handled else).
                             NOTE: if True, the tensor product will include the interaction of biases (1*1=1).
    """
    def __init__(self, n_knots=7, degree=3, include_bias=False):
        self.n_knots = n_knots
        self.degree = degree
        self.include_bias = include_bias
        self.splines_ = []
        self.n_features_in_ = 0
        self.feature_names_in_ = None
        
    def fit(self, X, y=None):
        X = pd.DataFrame(X) if isinstance(X, (pd.DataFrame, pd.Series)) else pd.DataFrame(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns.tolist() if hasattr(X, 'columns') else [f"x{i}" for i in range(self.n_features_in_)]
        
        self.splines_ = []
        for i in range(self.n_features_in_):
            # Fit independent spline for each dimension
            st = SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=self.include_bias)
            st.fit(X.iloc[:, [i]])
            self.splines_.append(st)
        
        return self
    
    def transform(self, X):
        check_is_fitted(self)
        X = pd.DataFrame(X) if isinstance(X, (pd.DataFrame, pd.Series)) else pd.DataFrame(X)
        
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
            
        # 1. Transform each dimension independently
        bases = []
        for i, st in enumerate(self.splines_):
            # shape (n_samples, n_basis)
            b = st.transform(X.iloc[:, [i]])
            bases.append(b)
            
        # 2. Compute Tensor Product
        # Currently optimized for 2D (Distance, Angle). 
        # For general N-D, we need recursive Kronecker product.
        
        if len(bases) == 1:
            return bases[0]
        
        # Start with first
        result = bases[0]
        
        for i in range(1, len(bases)):
            b_next = bases[i]
            # Row-wise Kronecker product (Outer product)
            # Result size = result.shape[1] * b_next.shape[1]
            
            # Efficient way using einsum
            # result: (N, A), b_next: (N, B) -> (N, A, B) -> flatten to (N, A*B)
            # Using broadcasting
            N = result.shape[0]
            A = result.shape[1]
            B = b_next.shape[1]
            
            # (N, A, 1) * (N, 1, B) = (N, A, B)
            tensor = result[:, :, np.newaxis] * b_next[:, np.newaxis, :]
            result = tensor.reshape(N, A * B)
            
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_in_
            
        # Generate names recursively
        names = self.splines_[0].get_feature_names_out([input_features[0]])
        
        for i in range(1, len(self.splines_)):
            next_names = self.splines_[i].get_feature_names_out([input_features[i]])
            
            new_names = []
            for n1 in names:
                for n2 in next_names:
                    new_names.append(f"{n1}_{n2}")
            names = new_names
            
        return names
