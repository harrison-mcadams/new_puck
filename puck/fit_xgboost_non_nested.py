"""fit_xgboost_non_nested.py

XGBOOST NON-NESTED EXPECTED GOALS MODEL
=======================================
This module implements a standard single-pass xG model using XGBoost.
It leverages XGBoost's native capabilities for categorical and missing data.

Parity: Mirrors NonNestedGLM structure.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import json
import time
from typing import List, Dict, Optional, Any
from pathlib import Path

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from . import features as feature_util
from . import config as puck_config
from .spline_transformer import TensorSpline

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# --- STANDARD VOCABULARIES ---
VOCAB_GAME_STATE = [
    '5v5', '5v4', '4v5', '4v4', '6v5', '5v6', '3v3', '5v3', '4v3', 
    '6v4', '3v5', '4v6', '3v4', '6v3', '3v6', '6v6', '1v0', '0v1'
]
VOCAB_SHOT_TYPE = [
    'wrist', 'snap', 'slap', 'backhand', 'tip-in', 'deflected', 'wrap-around'
]
VOCAB_SHOOTER_ROLE = ['F', 'D']
VOCAB_SHOOTS_CATCHES = ['L', 'R']

CATEGORICAL_VOCABS = {
    'shot_type': VOCAB_SHOT_TYPE,
    'shooter_role': VOCAB_SHOOTER_ROLE,
    'shoots_catches': VOCAB_SHOOTS_CATCHES,
    'game_state': VOCAB_GAME_STATE,
    'relative_game_state': VOCAB_GAME_STATE,
    'last_event_type': [
        'faceoff', 'hit', 'giveaway', 'takeaway', 'missed-shot', 'blocked-shot', 'shot-on-goal', 'goal', 'penalty'
    ]
}

logger = logging.getLogger(__name__)

class XGBNonNestedXGClassifier(BaseEstimator, ClassifierMixin):
    """
    Standard Expected Goals Model using XGBoost.
    
    Structure:
    1. Single Model: P(Goal | Shot)
    """
    
    def __init__(self, 
                 features: Optional[List[str]] = None,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 enable_categorical: bool = True,
                 use_calibration: bool = True,
                 use_splines: bool = True):
        
        self.features = features or feature_util.get_features('all_inclusive')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        self.use_calibration = use_calibration
        self.use_splines = use_splines
        
        # Spatial Base Model (GLM)
        self.spatial_glm_ = None
        self.use_splines = use_splines
        
        # Models & Calibrators
        self.model = None
        self.calibrator = None
        
        # Consistent Dtypes
        self.feature_dtypes = {}
        
        # Marginalization Support
        self.categorical_priors_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Fitting XGBNonNestedXGClassifier on {len(X)} rows.")

        if self.use_calibration:
            df_train, df_calib = train_test_split(X, test_size=0.2, random_state=self.random_state)
        else:
            df_train = X
            df_calib = None

        df = self._prepare_training_df(df_train)
        
        # Learn Priors
        self.categorical_priors_ = {}
        for col, vocab in CATEGORICAL_VOCABS.items():
            if col in df.columns:
                counts = df[col].value_counts(normalize=True, dropna=True)
                priors = {k: v for k, v in counts.items() if k in vocab}
                total_prob = sum(priors.values())
                if total_prob > 0:
                    self.categorical_priors_[col] = {k: v/total_prob for k, v in priors.items()}

        # Predict Goal directly (trained on non-blocked shots usually, or all depending on pipeline)
        # Standard non-nested pipeline excludes blocked shots.
        mask_valid = df['event'].isin(['shot-on-goal', 'missed-shot', 'goal'])
        df_valid = df[mask_valid].copy()
        y_valid = (df_valid['event'] == 'goal').astype(int)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
            enable_categorical=self.enable_categorical,
            objective='binary:logistic',
            tree_method='hist',
            device='cpu',
            eval_metric='logloss',
            base_score=0.5
        )
        
        if len(df_valid) > 0:
            self.model.fit(df_valid[self.features], y_valid)
            self.feature_dtypes = df_valid[self.features].dtypes.to_dict()
        else:
            logger.warning("No valid shots for training!")

        if self.use_calibration and df_calib is not None:
            self._fit_calibrator(df_calib)

        return self

    @classmethod
    def train(cls, df_raw: pd.DataFrame, save_path: Optional[str] = None, verbose: bool = True):
        from . import data_pipeline, model_summary
        def vprint(*args):
            if verbose: print(*args)

        vprint("--- Training XGBoost (Non-Nested) Model ---")
        
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

        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        feature_list = feature_util.get_features('all_inclusive')
        clf = cls(
            features=feature_list,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05
        )

        vprint(f"Training on {len(df_train)} rows...")
        clf.fit(df_train)

        vprint("\n--- Evaluation ---")
        y_test = (df_test['event'] == 'goal').astype(int)
        probs = clf.predict_proba(df_test)[:, 1]
        vprint(f"AUC: {roc_auc_score(y_test, probs):.4f}, LogLoss: {log_loss(y_test, probs):.4f}")

        if save_path is None:
            save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_non_nested.joblib')
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
        
        meta = {'features': clf.features, 'model_type': 'xgboost_non_nested'}
        with open(save_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        out_dir = Path(puck_config.ANALYSIS_DIR) / 'xgboost_non_nested_xgs'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if plt:
            fig, ax = plt.subplots(figsize=(6, 5))
            prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
            ax.plot(prob_pred, prob_true, marker='o')
            ax.plot([0, 1], [0, 1], '--k', alpha=0.3)
            ax.set_title("XGBoost Non-Nested Calibration")
            plt.savefig(out_dir / 'xgboost_calibration.png')
            plt.close()

        vprint("Generating summary...")
        model_summary.generate_model_summary(model_path=save_path, test_df=df_test, output_dir=str(out_dir), verbose=verbose)

        return clf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = self._prepare_inference_df(X)
        p = self._predict_marginalized(self.model, df, self.features)
        
        if self.calibrator:
            p = self.calibrator.predict(p)
            
        return np.column_stack((1 - p, p))

    def _prepare_training_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Spatial Base Model (Stacked GLM)
        if self.use_splines:
            logger.info("  Training Spatial GLM Base...")
            # We train a smooth GLM on splines first
            spatial_pipe = make_pipeline(
                TensorSpline(n_knots=7, degree=3),
                LogisticRegression(C=1.0)
            )
            
            # Predict 'goal' outcome for spatial baseline
            y_spatial = (df['event'] == 'goal').astype(int)
            spatial_pipe.fit(df[['x', 'y']], y_spatial)
            self.spatial_glm_ = spatial_pipe
            
            # Add spatial_xg as a feature
            df['spatial_xg'] = self.spatial_glm_.predict_proba(df[['x', 'y']])[:, 1]
            
            # Update internal features list: 
            # - Remove individual splines (though they aren't there yet)
            # - MUST keep distance and angle_deg as secondary features
            # - Add spatial_xg
            if 'spatial_xg' not in self.features:
                self.features.append('spatial_xg')
            
            # Ensure distance and angle are present (they usually are in 'all_inclusive')
            for f in ['distance', 'angle_deg']:
                if f not in self.features:
                    self.features.append(f)

        # 2. Categoricals
        for col in self.features:
            if col in df.columns:
                if df[col].dtype == 'object' or col in CATEGORICAL_VOCABS:
                    vocab = CATEGORICAL_VOCABS.get(col)
                    df[col] = pd.Categorical(df[col], categories=vocab) if vocab else df[col].astype('category')
        return df

    def _prepare_inference_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Spatial Base Model
        if self.use_splines and self.spatial_glm_:
            df['spatial_xg'] = self.spatial_glm_.predict_proba(df[['x', 'y']])[:, 1]

        # 2. Base Features & Dtypes
        for col, dt in self.feature_dtypes.items():
            if col not in df.columns:
                df[col] = np.nan
            
            if isinstance(dt, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col], categories=dt.categories)
            elif col in CATEGORICAL_VOCABS:
                 df[col] = pd.Categorical(df[col], categories=CATEGORICAL_VOCABS[col])
            else:
                try:
                    df[col] = df[col].astype(float)
                except:
                    pass
        return df

    def _predict_marginalized(self, model, df, features):
        p_base = model.predict_proba(df[features])[:, 1]
        col = 'shot_type'
        if col not in df.columns or col not in self.categorical_priors_:
            return p_base
        mask_nan = df[col].isna()
        if not mask_nan.any():
            return p_base
        priors = self.categorical_priors_[col]
        df_nan = df[mask_nan].copy()
        weighted_prob = np.zeros(len(df_nan))
        for val, weight in priors.items():
            df_nan[col] = pd.Categorical([val]*len(df_nan), categories=CATEGORICAL_VOCABS[col])
            weighted_prob += model.predict_proba(df_nan[features])[:, 1] * weight
        p_base[mask_nan] = weighted_prob
        return p_base

    def _fit_calibrator(self, df_calib_raw: pd.DataFrame):
        df_c = self._prepare_inference_df(df_calib_raw)
        p_raw = self.predict_proba(df_c)[:, 1]
        y_goal = (df_c['event'] == 'goal').astype(int)
        if len(y_goal.unique()) > 1:
            self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            self.calibrator.fit(p_raw, y_goal)

def train_xgboost(df_raw, **kwargs):
    return XGBNonNestedXGClassifier.train(df_raw, **kwargs)
