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
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import joblib
import json
import time
import logging
from pathlib import Path

from . import features as feature_util
from . import config as puck_config
# These are imported inside methods to avoid circular dependencies if any
# from . import data_pipeline, moneypuck, model_summary

logger = logging.getLogger(__name__)

VOCAB_SHOT_TYPE = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'wrap-around', 'deflected']

from .spline_transformer import TensorSpline

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class NonNestedGLM(BaseEstimator, ClassifierMixin):
    """
    Standard Expected Goals Model using Polynomial Logistic Regression (GLM) or Tensor Splines.
    
    Structure:
    1. Single Model: P(Goal | Shot)
    
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
        self.model = None
        
        # Priors for marginalization
        self.shot_type_priors_ = {}
        
    def fit(self, X, y=None):
        logger.info(f"Fitting NonNestedGLM on {len(X)} rows. Poly Degree={self.poly_degree}, Splines={self.use_splines}")

        df = X.copy()
        
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

        # 1. Prediction Model (Trained on valid shots, which should have blocked excluded)
        logger.info("  Fitting Model...")
        self.model = self._build_pipeline(features=self.features)
        
        # We assume blocked shots are already filtered out if exclude_blocked was True in pipeline
        # But we can be safe:
        mask_valid = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
        X_valid = df[mask_valid]
        y_valid = (X_valid['event'] == 'goal').astype(int)
        
        if len(X_valid) > 0:
            self.model.fit(X_valid[self.features], y_valid)
        else:
            logger.warning("No valid shots found! Model will be untrained.")
            
        logger.info("NonNestedGLM Fit Complete.")
        return self

    @classmethod
    def train(cls, df_raw: pd.DataFrame, save_path: str = None, verbose: bool = True):
        """
        High-level training routine.
        1. Preprocess
        2. Split
        3. Fit
        4. Evaluate
        5. Save
        """
        from . import data_pipeline, moneypuck, model_summary
        
        def vprint(*args):
            if verbose: print(*args)

        vprint("--- Training Non-Nested GLM (Poly/Tensor) Model ---")
        
        # 1. Preprocess
        vprint("Applying Preprocessing Pipeline (excluding blocked shots)...")
        df = data_pipeline.preprocess_features(
            df_raw, 
            is_training=True, 
            verbose=verbose, 
            apply_arena_adjustments=True,
            apply_imputation=False,
            apply_dithering=True,
            apply_filtering=True,
            exclude_blocked=True
        )

        # 2. Split
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        # 3. Initialize & Fit
        feature_list = feature_util.get_features('all_inclusive')
        clf = cls(
            features=feature_list,
            use_splines=True,
            enable_marginalization=True
        )

        vprint(f"Training on {len(df_train)} rows with {len(feature_list)} features...")
        start_t = time.time()
        clf.fit(df_train)
        vprint(f"Training took {time.time() - start_t:.1f}s.")

        # 4. Evaluate
        vprint("\n--- Evaluation (Test Set) ---")
        y_test_goal = (df_test['event'] == 'goal').astype(int)
        probs = clf.predict_proba(df_test)[:, 1]
        
        auc = roc_auc_score(y_test_goal, probs)
        ll = log_loss(y_test_goal, probs)
        vprint(f"AUC: {auc:.4f}, LogLoss: {ll:.4f}")

        # 5. Save Model & Metadata
        if save_path is None:
            save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_non_nested_tensor.joblib')
        
        vprint(f"Saving model to {save_path}...")
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
        
        meta = {
            'final_features': clf.features,
            'categorical_levels_map': {}, 
            'feature_set_name': 'non_nested_glm_tensor',
            'model_type': 'non_nested_tensor',
            'raw_features': clf.features
        }
        with open(save_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        # 6. Diagnostics & Summary
        out_dir = Path(puck_config.ANALYSIS_DIR) / 'non_nested_xgs'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Calibration Plot
        if plt:
            fig, ax = plt.subplots(figsize=(6, 5))
            prob_true, prob_pred = calibration_curve(y_test_goal, probs, n_bins=10, strategy='uniform')
            ax.plot(prob_pred, prob_true, marker='o', label="Non-Nested Model")
            ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
            ax.set_title("Non-Nested GLM Calibration")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(out_dir / 'glm_calibration.png')
            plt.close()

        # Model Summary
        vprint("Generating model summary...")
        model_summary.generate_model_summary(model_path=save_path, test_df=df_test, output_dir=str(out_dir), verbose=verbose)

        return clf

    def _build_pipeline(self, features=None):
        """Builds a standardized Sklearn pipeline."""
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
        
        # Binary features should NOT get spline treatment — just impute + scale
        binary_feature_names = ['is_home', 'is_rush', 'is_rebound']
        binary_cols = [f for f in binary_feature_names if f in num_features]
        
        transformers = []
        
        if self.use_splines:
            # Spline Tensor Product Logic
            # Treat ('distance', 'angle_deg') as a unit for TensorSpline
            spatial_cols = [f for f in ['distance', 'angle_deg'] if f in num_features]
            other_num_cols = [f for f in num_features if f not in spatial_cols and f not in binary_cols]
            
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
            
            # Independent Splines for continuous numeric features
            if other_num_cols:
                other_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('other_num', other_pipe, other_num_cols))
            
            # Simple passthrough for binary features (no spline expansion)
            if binary_cols:
                binary_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('binary', binary_pipe, binary_cols))
                
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

    def predict_proba(self, X):
        """
        Predicts P(Goal).
        Applies Marginalization for rows with missing shot_type.
        """
        df = X[self.features].copy()
        
        # 1. Identify rows needing marginalization
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
        
        # 2. Main Prediction Path
        p_final = self.model.predict_proba(df[self.features])[:, 1]
        
        # 3. Marginalization Path
        if self.enable_marginalization and mask_nan.any() and self.shot_type_priors_:
            df_nan = df[mask_nan].copy()
            accumulated_prob = np.zeros(len(df_nan))
            
            for st, weight in self.shot_type_priors_.items():
                df_nan_imputed = df_nan.copy()
                df_nan_imputed['shot_type'] = st
                prob_st = self.model.predict_proba(df_nan_imputed[self.features])[:, 1]
                accumulated_prob += prob_st * weight
            
            p_final[mask_nan] = accumulated_prob
            
        return np.column_stack((1 - p_final, p_final))

def train_glm(df_raw, **kwargs):
    """Functional wrapper for NonNestedGLM.train"""
    return NonNestedGLM.train(df_raw, **kwargs)

def fit_glm(X, y=None, **kwargs):
    """Functional wrapper for NonNestedGLM.fit"""
    model = NonNestedGLM(**kwargs)
    return model.fit(X, y)
