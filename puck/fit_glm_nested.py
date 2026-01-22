
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import logging

from . import features as feature_util
from . import config as puck_config

logger = logging.getLogger(__name__)

VOCAB_SHOT_TYPE = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'wrap-around', 'deflected']

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


class NestedGLM(BaseEstimator, ClassifierMixin):
    """
    Nested Expected Goals Model using Polynomial Logistic Regression (GLM) or Tensor Splines.
    
    Structure:
    1. Block Model: P(Unblocked | Shot)
    2. Accuracy Model: P(On Net | Unblocked)
    3. Finish Model: P(Goal | On Net)
    
    P(Goal) = P(Unblocked) * P(On Net) * P(Goal | On Net)
    
    Features:
    - Uses PolynomialFeatures(degree=3) or TensorSplines for numeric columns.
    - Uses OneHotEncoder for categorical columns.
    - Handles missing Shot Types via Marginalization (Weighted Average).
    """
    
    def __init__(self, features=None, poly_degree=3, use_splines=False, enable_marginalization=True):
        self.features = features or feature_util.get_features('all_inclusive')
        self.poly_degree = poly_degree
        self.use_splines = use_splines
        
        self.enable_marginalization = enable_marginalization
        
        # Sub-models
        self.model_block = None
        self.model_acc = None
        self.model_finish = None
        
        # Priors for marginalization
        self.shot_type_priors_ = {}
        
    def fit(self, X, y=None):
        logger.info(f"Fitting NestedGLM on {len(X)} rows. Poly Degree={self.poly_degree}, Splines={self.use_splines}")

        df = X.copy()
        # Note: TensorSpline handles interaction internally, no need for _enrich_interaction
        
        logger.info(f"Final Feature Set ({len(self.features)}): {self.features}")
        
        # 0. Learn Priors for Marginalization
        if self.enable_marginalization and 'shot_type' in df.columns:
            # Normalize to lowercase for counting
            st_lower = df['shot_type'].astype(str).str.lower()
            vc = st_lower.value_counts(normalize=True)
            # Filter to known vocabulary
            self.shot_type_priors_ = {k: v for k, v in vc.items() if k in VOCAB_SHOT_TYPE}
            # Renormalize
            total_prob = sum(self.shot_type_priors_.values())
            if total_prob > 0:
                self.shot_type_priors_ = {k: v/total_prob for k, v in self.shot_type_priors_.items()}
            logger.info(f"Learned Shot Type Priors: {self.shot_type_priors_}")

        # 1. Block Model (Trained on ALL shots)
        # Exclude 'shot_type'
        block_features = [f for f in self.features if f != 'shot_type']
        logger.info(f"  Fitting Block Model (Features: {len(block_features)})...")
        
        self.model_block = self._build_pipeline(features=block_features)
        y_block = (df['event'] == 'blocked-shot').astype(int)
        self.model_block.fit(df[block_features], y_block)
        
        # 2. Accuracy Model (Trained on Unblocked shots)
        logger.info("  Fitting Accuracy Model...")
        self.model_acc = self._build_pipeline(features=self.features)
        mask_unblocked = df['event'] != 'blocked-shot'
        X_unblocked = df[mask_unblocked]
        y_acc = X_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        if len(X_unblocked) > 0:
            self.model_acc.fit(X_unblocked[self.features], y_acc)
        else:
            logger.warning("No unblocked shots found! Accuracy model will be untrained.")
        
        # 3. Finish Model (Trained on Shots On Net)
        logger.info("  Fitting Finish Model...")
        self.model_finish = self._build_pipeline(features=self.features)
        mask_on_net = df['event'].isin(['shot-on-goal', 'goal'])
        X_on_net = df[mask_on_net]
        y_finish = (X_on_net['event'] == 'goal').astype(int)
        if len(X_on_net) > 0:
            self.model_finish.fit(X_on_net[self.features], y_finish)
        else:
            logger.warning("No shots on net found! Finish model will be untrained.")
            
        logger.info("NestedGLM Fit Complete.")
        return self

    def _build_pipeline(self, features=None):
        """Builds a standardized Sklearn pipeline for a single layer."""
        features = features or self.features
        
        # 1. Categorical Features
        cat_features = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state']
        cat_features = [f for f in cat_features if f in features]
        
        cat_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 2. Numeric Features
        num_features = [f for f in features if f not in cat_features]
        
        transformers = []
        
        if self.use_splines:
            # Spline Tensor Product Logic
            # Treat ('distance', 'angle_deg') as a unit for TensorSpline
            spatial_cols = [f for f in ['distance', 'angle_deg'] if f in num_features]
            other_num_cols = [f for f in num_features if f not in spatial_cols]
            
            # Tensor Spline for Spatial
            if len(spatial_cols) == 2:
                tensor_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('tensor', TensorSpline(n_knots=7, degree=3, include_bias=False)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('spatial_tensor', tensor_pipe, spatial_cols))
            else:
                # If we don't have both, just treat them as generic numeric
                other_num_cols.extend(spatial_cols)
            
            # Independent Splines for others
            if other_num_cols:
                other_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)), # fewer knots for non-spatial?
                    ('scaler', StandardScaler())
                ])
                transformers.append(('other_num', other_pipe, other_num_cols))
                
        else:
            # Polynomial Logic (Global curve)
            poly_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num_poly', poly_pipe, num_features))
            
        # Add Categorical
        if cat_features:
            transformers.append(('cat', cat_trans, cat_features))
            
        preprocessor = ColumnTransformer(transformers)
        
        # Classifier: Logistic Regression (Ridge by default, l2 penalty)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)) 
        ])
        
        return pipeline

    def predict_proba_layer(self, X, layer):
        """Returns probability of success (1) for a specific layer."""
        if layer == 'block':
            feats = [f for f in self.features if f != 'shot_type']
            return self.model_block.predict_proba(X[feats])[:, 1]
            
        # Accuracy & Finish need marginalization
        df = X[self.features].copy()
        
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
            
        if layer == 'accuracy':
            model = self.model_acc
        elif layer == 'finish':
            model = self.model_finish
        else:
            raise ValueError(f"Unknown layer: {layer}")
            
        p_final = model.predict_proba(df[self.features])[:, 1]
        
        if self.enable_marginalization and mask_nan.any() and self.shot_type_priors_:
            df_nan = df[mask_nan].copy()
            accumulated_prob = np.zeros(len(df_nan))
            
            for st, weight in self.shot_type_priors_.items():
                df_nan_imputed = df_nan.copy()
                df_nan_imputed['shot_type'] = st
                prob_st = model.predict_proba(df_nan_imputed[self.features])[:, 1]
                accumulated_prob += prob_st * weight
                
            p_final[mask_nan] = accumulated_prob
            
        return p_final

    def predict_proba(self, X):
        """
        Predicts P(Goal) using the nested chain.
        Applies Marginalization for rows with missing shot_type.
        """
        df = X[self.features].copy()
        
        # 1. Identify rows needing marginalization
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
        
        # 2. Main Prediction Path
        p_final = self._predict_single_pass(df)
        
        # 3. Marginalization Path
        if self.enable_marginalization and mask_nan.any() and self.shot_type_priors_:
            df_nan = df[mask_nan].copy()
            accumulated_prob = np.zeros(len(df_nan))
            
            for st, weight in self.shot_type_priors_.items():
                df_nan_imputed = df_nan.copy()
                df_nan_imputed['shot_type'] = st
                prob_st = self._predict_single_pass(df_nan_imputed)
                accumulated_prob += prob_st * weight
            
            p_final[mask_nan] = accumulated_prob
            
        return np.column_stack((1 - p_final, p_final))

    def _predict_single_pass(self, df):
        """Helper to run the P(Unblocked)*P(OnNet)*P(Finish) chain without marginalization logic."""
        feats_block = [f for f in self.features if f != 'shot_type']
        p_blocked = self.model_block.predict_proba(df[feats_block])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        p_on_net = self.model_acc.predict_proba(df[self.features])[:, 1]
        p_finish = self.model_finish.predict_proba(df[self.features])[:, 1]
        
        return p_unblocked * p_on_net * p_finish

