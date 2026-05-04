
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
from typing import List, Dict, Any, Optional, Union
import joblib
import json
import time
import logging
from pathlib import Path

from . import features as feature_util
from . import config as puck_config
from .verify import verify_df

logger = logging.getLogger(__name__)

VOCAB_SHOT_TYPE = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'wrap-around', 'deflected']

from .spline_transformer import TensorSpline

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


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
        
        logger.info(f"Final Feature Set ({len(self.features)}): {self.features}")
        
        # Run verification prior to fitting
        verify_df(df, self.features, verify_blocked=True)

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
        # We now INCLUDE shot_type in the block model as requested.
        block_features = self.features
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

    @classmethod
    def train(cls, df_raw: pd.DataFrame, save_path: Optional[str] = None, out_dir: Optional[str] = None, verbose: bool = True, **kwargs):
        """
        High-level training routine for NestedGLM.
        """
        from . import data_pipeline, model_summary
        
        def vprint(*args):
            if verbose: print(*args)

        vprint("--- Training Nested GLM (Nested Poly/Tensor) Model ---")
        
        # 1. Preprocess
        vprint("Applying Preprocessing Pipeline (including imputation)...")
        df = data_pipeline.preprocess_features(
            df_raw, 
            is_training=True, 
            verbose=verbose, 
            apply_arena_adjustments=kwargs.get('apply_arena_adjustments', True),
            apply_imputation=kwargs.get('apply_imputation', True),
            apply_dithering=kwargs.get('apply_dithering', True),
            apply_filtering=kwargs.get('apply_filtering', True),
            apply_attribution_fix=kwargs.get('apply_attribution_fix', True),
            apply_html_enrichment=kwargs.get('apply_html_enrichment', False),
            impute_alpha=kwargs.get('impute_alpha', 0.2),
            exclude_blocked=kwargs.get('exclude_blocked', False)
        )

        # 2. Split
        df_train, df_test = train_test_split(
            df, 
            test_size=kwargs.get('test_size', 0.2), 
            random_state=kwargs.get('random_state', 42)
        )

        # 3. Initialize & Fit
        feature_list = feature_util.get_features('all_inclusive')
        clf = cls(
            features=feature_list,
            use_splines=True,
            enable_marginalization=True
        )

        vprint(f"Training Nested Model on {len(df_train)} rows with {len(feature_list)} features...")
        start_t = time.time()
        clf.fit(df_train)
        vprint(f"Training took {time.time() - start_t:.1f}s.")

        # 4. Evaluate
        vprint("\n--- Evaluation (Test Set) ---")
        y_test_goal = (df_test['event'] == 'goal').astype(int)
        probs = clf.predict_proba(df_test)[:, 1]
        
        auc = roc_auc_score(y_test_goal, probs)
        ll = log_loss(y_test_goal, probs)
        try:
            brier = brier_score_loss(y_test_goal, probs)
        except Exception:
            brier = np.nan
        vprint(f"Overall xG AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {brier:.6f}")
        
        clf.test_metrics_ = {'auc': auc, 'logloss': ll, 'brier': brier}

        # 5. Save Model & Metadata
        if save_path is None:
            save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested_tensor_20202021.joblib')
        
        vprint(f"Saving model to {save_path}...")
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
        
        meta = {
            'final_features': clf.features,
            'categorical_levels_map': {}, 
            'feature_set_name': 'nested_glm_tensor',
            'model_type': 'nested_tensor',
            'raw_features': clf.features
        }
        with open(save_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        # 6. Diagnostics & Summary
        if out_dir is None:
            diag_dir = Path(puck_config.ANALYSIS_DIR) / 'nested_xgs'
        else:
            diag_dir = Path(out_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        # Layer Diagnostics plots
        if plt:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Block
            df_test_loc = df_test.copy()
            df_test_loc['is_blocked'] = (df_test_loc['event'] == 'blocked-shot').astype(int)
            p_block = clf.predict_proba_layer(df_test_loc, 'block')
            prob_true, prob_pred = calibration_curve(df_test_loc['is_blocked'], p_block, n_bins=10, strategy='uniform')
            axes[0].plot(prob_pred, prob_true, marker='o', label="Block Layer")
            axes[0].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
            axes[0].set_title("Block Layer")
            
            # Accuracy (Unblocked)
            mask_unblocked = df_test_loc['is_blocked'] == 0
            if mask_unblocked.any():
                df_acc = df_test_loc[mask_unblocked]
                a_targets = df_acc['event'].isin(['shot-on-goal', 'goal']).astype(int)
                p_acc = clf.predict_proba_layer(df_acc, 'accuracy')
                prob_true, prob_pred = calibration_curve(a_targets, p_acc, n_bins=10, strategy='uniform')
                axes[1].plot(prob_pred, prob_true, marker='o', label="Accuracy Layer")
                axes[1].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
                axes[1].set_title("Accuracy Layer")
                
            # Finish (On Net)
            mask_on_net = (df_test_loc['is_blocked'] == 0) & (df_test_loc['event'].isin(['shot-on-goal', 'goal']))
            if mask_on_net.any():
                df_fin = df_test_loc[mask_on_net]
                f_targets = (df_fin['event'] == 'goal').astype(int)
                p_fin = clf.predict_proba_layer(df_fin, 'finish')
                prob_true, prob_pred = calibration_curve(f_targets, p_fin, n_bins=10, strategy='uniform')
                axes[2].plot(prob_pred, prob_true, marker='o', label="Finish Layer")
                axes[2].plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
                axes[2].set_title("Finish Layer")
                
            plt.savefig(diag_dir / 'glm_calibration.png')
            plt.close()

        # Model Summary
        vprint("Generating model summary...")
        model_summary.generate_model_summary(model_path=save_path, test_df=df_test, output_dir=str(diag_dir), verbose=verbose)

        return clf

    def _build_pipeline(self, features=None):
        """Builds a standardized Sklearn pipeline for a single layer."""
        features = features or self.features
        
        # 1. Categorical Features
        cat_features = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state', 'relative_game_state', 'rebound_source']
        cat_features = [f for f in cat_features if f in features]
        
        cat_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # 2. Numeric Features
        num_features = [f for f in features if f not in cat_features]
        
        # Binary features should NOT get spline treatment 
        binary_feature_names = ['is_home', 'is_rush', 'is_rebound']
        binary_cols = [f for f in binary_feature_names if f in num_features]
        
        transformers = []
        
        if self.use_splines:
            spatial_cols = [f for f in ['distance', 'angle_deg'] if f in num_features]
            other_num_cols = [f for f in num_features if f not in spatial_cols and f not in binary_cols]
            
            if len(spatial_cols) == 2:
                tensor_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('tensor', TensorSpline(n_knots=7, degree=3, include_bias=False)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('spatial_tensor', tensor_pipe, spatial_cols))
            else:
                other_num_cols.extend(spatial_cols)
            
            if other_num_cols:
                other_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('other_num', other_pipe, other_num_cols))
            
            if binary_cols:
                binary_pipe = Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler())
                ])
                transformers.append(('binary', binary_pipe, binary_cols))
                
        else:
            poly_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num_poly', poly_pipe, num_features))
            
        if cat_features:
            transformers.append(('cat', cat_trans, cat_features))
            
        preprocessor = ColumnTransformer(transformers)
        
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)) 
        ])
        
        return pipeline

    def predict_proba_layer(self, X, layer):
        """Returns probability of success (1) for a specific layer."""
        df = X[self.features].copy()
        
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)

        if layer == 'block':
            # Block layer now includes shot_type
            p_final = self.model_block.predict_proba(df[self.features])[:, 1]
            if self.enable_marginalization and mask_nan.any() and self.shot_type_priors_:
                df_nan = df[mask_nan].copy()
                accumulated_prob = np.zeros(len(df_nan))
                for st, weight in self.shot_type_priors_.items():
                    df_nan_imputed = df_nan.copy()
                    df_nan_imputed['shot_type'] = st
                    prob_st = self.model_block.predict_proba(df_nan_imputed[self.features])[:, 1]
                    accumulated_prob += prob_st * weight
                p_final[mask_nan] = accumulated_prob
            return p_final
            
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
        df = X[self.features].copy()
        
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
        
        p_final = self._predict_single_pass(df)
        
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
        # Now uses full features for all layers
        p_blocked = self.model_block.predict_proba(df[self.features])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        p_on_net = self.model_acc.predict_proba(df[self.features])[:, 1]
        p_finish = self.model_finish.predict_proba(df[self.features])[:, 1]
        
        return p_unblocked * p_on_net * p_finish

def train_nested_glm(df_raw, **kwargs):
    return NestedGLM.train(df_raw, **kwargs)
