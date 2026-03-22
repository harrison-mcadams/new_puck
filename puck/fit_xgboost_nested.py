"""fit_xgboost_nested.py

XGBOOST NESTED EXPECTED GOALS MODEL
===================================
This module implements the "Layered" or "Nested" xG model using XGBoost.
It leverages XGBoost's native capabilities for:
1.  Handling Missing Data (NaN): No distinct "Unknown" category needed.
2.  Categorical Support: Native 'enable_categorical=True' ensures optimal splits.

Parity: Mirrors NestedGLM structure for consistent training and evaluation.
"""

import numpy as np
import pandas as pd
import joblib
import logging
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
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

# Map feature names to their vocabulary
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

# --- LOGGING SETUP ---
logger = logging.getLogger(__name__)

@dataclass
class LayerConfig:
    name: str
    target_col: str
    feature_cols: List[str]
    # XGBoost Params
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1

class XGBNestedXGClassifier(BaseEstimator, ClassifierMixin):
    """
    Nested Expected Goals Model using XGBoost.
    
    Structure:
    1. Block Model: P(Unblocked | Shot)
    2. Accuracy Model: P(On Net | Unblocked)
    3. Finish Model: P(Goal | On Net)
    
    P(Goal) = P(Unblocked) * P(On Net) * P(Goal | On Net)
    
    Features:
    - Utilizes native XGBoost categorical support.
    - Handles missing values via marginalization.
    """
    
    def __init__(self, 
                 features: Optional[List[str]] = None,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 enable_categorical: bool = True,
                 enable_marginalization: bool = True,
                 use_balancing: bool = True,
                 use_calibration: bool = True,
                 layer_params: Optional[Dict[str, Any]] = None,
                 use_splines: bool = True):
        
        self.features = features or feature_util.get_features('all_inclusive')
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        self.enable_marginalization = enable_marginalization
        self.use_balancing = use_balancing
        self.use_calibration = use_calibration
        self.layer_params = layer_params or {}
        self.use_splines = use_splines
        
        # Spatial Base Models (GLMs)
        self.spatial_glm_block_ = None
        self.spatial_glm_acc_ = None
        self.spatial_glm_fin_ = None
        self.use_splines = use_splines
        
        # Sub-models
        self.model_block = None
        self.model_acc = None
        self.model_finish = None
        
        # Calibrators
        self.calibrator_goal = None
        self.calibrator_block = None
        
        # Consistent Dtypes for Inference
        self.feature_dtypes = {}
        
        # Marginalization Support
        self.categorical_priors_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Fitting XGBNestedXGClassifier on {len(X)} rows. Calib={self.use_calibration}")

        # 0. Split for Internal Calibration if requested
        if self.use_calibration:
            df_train, df_calib = train_test_split(X, test_size=0.2, random_state=self.random_state)
            logger.info(f"  Internal Split: Train={len(df_train)}, Calib={len(df_calib)}")
        else:
            df_train = X
            df_calib = None

        df = self._prepare_training_df(df_train)
        
        # 1. Learn Priors for Marginalization
        self.categorical_priors_ = {}
        for col, vocab in CATEGORICAL_VOCABS.items():
            if col in df.columns:
                counts = df[col].value_counts(normalize=True, dropna=True)
                priors = {k: v for k, v in counts.items() if k in vocab}
                total_prob = sum(priors.values())
                if total_prob > 0:
                    priors = {k: v/total_prob for k, v in priors.items()}
                    self.categorical_priors_[col] = priors
        logger.info(f"Learned Categorical Priors: {list(self.categorical_priors_.keys())}")

        # 2. Block Model (Trained on ALL shots)
        feat_block = [f for f in self.features if f != 'shot_type']
        if 'spatial_block' not in feat_block:
             feat_block.append('spatial_block')
             
        y_block = (df['event'] == 'blocked-shot').astype(int)
        
        p_block = self._get_xgb_params('block')
        self.model_block = XGBClassifier(**p_block)
        self.model_block.fit(df[feat_block], y_block)
        
        # 3. Accuracy Model (Trained on Unblocked shots)
        mask_unblocked = df['event'] != 'blocked-shot'
        df_unblocked = df[mask_unblocked].copy()
        y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        
        p_acc = self._get_xgb_params('accuracy')
        self.model_acc = XGBClassifier(**p_acc)
        self.model_acc.fit(df_unblocked[self.features], y_acc)
        
        # 4. Finish Model (Trained on Shots On Net)
        mask_on_net = df['event'].isin(['shot-on-goal', 'goal'])
        df_on_net = df[mask_on_net].copy()
        y_finish = (df_on_net['event'] == 'goal').astype(int)
        
        p_finish = self._get_xgb_params('finish')
        if self.use_balancing and 'scale_pos_weight' not in p_finish:
            pos = y_finish.sum()
            neg = len(y_finish) - pos
            if pos > 0:
                p_finish['scale_pos_weight'] = neg / pos
        
        self.model_finish = XGBClassifier(**p_finish)
        self.model_finish.fit(df_on_net[self.features], y_finish)
        
        # Record Dtypes for consistency
        self.feature_dtypes = df[self.features].dtypes.to_dict()

        # 5. Internal Calibration
        if self.use_calibration and df_calib is not None:
            self._fit_calibrators(df_calib)

        logger.info("Fit Complete.")
        return self

    @classmethod
    def train(cls, df_raw: pd.DataFrame, save_path: Optional[str] = None, out_dir: Optional[str] = None, verbose: bool = True):
        """
        High-level training routine for XGBoost Nested Model.
        """
        from . import data_pipeline, model_summary
        
        def vprint(*args):
            if verbose: print(*args)

        vprint("--- Training XGBoost (Nested) Model ---")
        
        # 1. Preprocess
        vprint("Applying Preprocessing Pipeline...")
        df = data_pipeline.preprocess_features(
            df_raw, 
            is_training=True, 
            verbose=verbose, 
            apply_arena_adjustments=True,
            apply_imputation=True,
            apply_dithering=True,
            apply_filtering=True,
            impute_alpha=0.2
        )

        # 2. Split
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        # 3. Initialize & Fit
        feature_list = feature_util.get_features('all_inclusive')
        clf = cls(
            features=feature_list,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            use_calibration=True,
            use_balancing=True
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
        brier = brier_score_loss(y_test_goal, probs)
        vprint(f"Overall xG AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {brier:.6f}")

        # 5. Save Model & Metadata
        if save_path is None:
            save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_nested.joblib')
        
        vprint(f"Saving model to {save_path}...")
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
        
        meta = {
            'final_features': clf.features,
            'model_type': 'xgboost_nested',
            'train_params': {
                'n_estimators': clf.n_estimators,
                'max_depth': clf.max_depth,
                'learning_rate': clf.learning_rate
            }
        }
        with open(save_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        # 6. Diagnostics
        if out_dir is None:
            diag_dir = Path(puck_config.ANALYSIS_DIR) / 'xgboost_nested_xgs'
        else:
            diag_dir = Path(out_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        if plt:
            cls._plot_calibration(clf, df_test, diag_dir)

        vprint("Generating model summary...")
        model_summary.generate_model_summary(model_path=save_path, test_df=df_test, output_dir=str(diag_dir), verbose=verbose)

        return clf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = self._prepare_inference_df(X)
        
        # 1. P(Blocked)
        feat_block = [f for f in self.features if f != 'shot_type']
        if self.model_block is None:
            raise NotFittedError("Model not fitted.")
        p_blocked = self.model_block.predict_proba(df[feat_block])[:, 1]
        if self.calibrator_block:
            p_blocked = self.calibrator_block.predict_proba(p_blocked.reshape(-1, 1))[:, 1]
        
        p_unblocked = 1.0 - p_blocked
        
        # 2. P(On Net) and P(Finish) with Marginalization
        p_acc = self._predict_marginalized(self.model_acc, df, self.features)
        p_finish = self._predict_marginalized(self.model_finish, df, self.features)
        
        p_goal = p_unblocked * p_acc * p_finish
        
        # 3. Final Calibration
        if self.calibrator_goal:
            p_goal = self.calibrator_goal.predict(p_goal)
            
        return np.column_stack((1 - p_goal, p_goal))

    def predict_proba_layer(self, X: pd.DataFrame, layer: str) -> np.ndarray:
        df = self._prepare_inference_df(X)
        if layer == 'block':
            feat_block = [f for f in self.features if f != 'shot_type']
            p = self.model_block.predict_proba(df[feat_block])[:, 1]
            if self.calibrator_block:
                p = self.calibrator_block.predict_proba(p.reshape(-1, 1))[:, 1]
            return p
        elif layer == 'accuracy':
            return self._predict_marginalized(self.model_acc, df, self.features)
        elif layer == 'finish':
            return self._predict_marginalized(self.model_finish, df, self.features)
        raise ValueError(f"Unknown layer: {layer}")

    # --- INTERNAL HELPERS ---

    def _get_xgb_params(self, layer_name: str) -> Dict[str, Any]:
        params = {
            'n_estimators': int(self.n_estimators),
            'max_depth': int(self.max_depth),
            'learning_rate': float(self.learning_rate),
            'random_state': int(self.random_state),
            'enable_categorical': bool(self.enable_categorical),
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': 'logloss',
            'base_score': 0.5
        }
        if layer_name in self.layer_params:
            params.update(self.layer_params[layer_name])
        return params

    def _prepare_training_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Spatial Base Models (Stacked GLMs)
        if self.use_splines:
            logger.info("  Training Triple Spatial GLM Baselines...")
            
            # Layer 1: Block
            logger.info("    Fitting Spatial Block GLM...")
            pipe_block = make_pipeline(TensorSpline(n_knots=7, degree=3), LogisticRegression(C=1.0))
            y_block = (df['event'] == 'blocked-shot').astype(int)
            pipe_block.fit(df[['x', 'y']], y_block)
            self.spatial_glm_block_ = pipe_block
            df['spatial_block'] = self.spatial_glm_block_.predict_proba(df[['x', 'y']])[:, 1]
            
            # Layer 2: Accuracy (Unblocked shots)
            logger.info("    Fitting Spatial Accuracy GLM...")
            mask_unblocked = df['event'] != 'blocked-shot'
            df_unblocked = df[mask_unblocked]
            pipe_acc = make_pipeline(TensorSpline(n_knots=7, degree=3), LogisticRegression(C=1.0))
            y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
            pipe_acc.fit(df_unblocked[['x', 'y']], y_acc)
            self.spatial_glm_acc_ = pipe_acc
            df['spatial_acc'] = self.spatial_glm_acc_.predict_proba(df[['x', 'y']])[:, 1]
            
            # Layer 3: Finish (On Net shots)
            logger.info("    Fitting Spatial Finish GLM...")
            mask_on_net = df['event'].isin(['shot-on-goal', 'goal'])
            df_on_net = df[mask_on_net]
            pipe_fin = make_pipeline(TensorSpline(n_knots=7, degree=3), LogisticRegression(C=1.0))
            y_fin = (df_on_net['event'] == 'goal').astype(int)
            pipe_fin.fit(df_on_net[['x', 'y']], y_fin)
            self.spatial_glm_fin_ = pipe_fin
            df['spatial_fin'] = self.spatial_glm_fin_.predict_proba(df[['x', 'y']])[:, 1]

            # Update features list: 
            # - Ensure distance and angle_deg are kept/added
            # - Add spatial features
            for f in ['distance', 'angle_deg', 'spatial_block', 'spatial_acc', 'spatial_fin']:
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
        
        # 1. Spatial Base Models
        if self.use_splines:
            if self.spatial_glm_block_:
                df['spatial_block'] = self.spatial_glm_block_.predict_proba(df[['x', 'y']])[:, 1]
            if self.spatial_glm_acc_:
                df['spatial_acc'] = self.spatial_glm_acc_.predict_proba(df[['x', 'y']])[:, 1]
            if self.spatial_glm_fin_:
                df['spatial_fin'] = self.spatial_glm_fin_.predict_proba(df[['x', 'y']])[:, 1]

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
        """Predict with weighted average over categorical priors for NaN values."""
        p_base = model.predict_proba(df[features])[:, 1]
        
        # Identify rows with NaN in critical categoricals
        # For XGBoost parity, we usually just care about 'shot_type' missing
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

    def _fit_calibrators(self, df_calib_raw: pd.DataFrame):
        df_c = self._prepare_inference_df(df_calib_raw)
        
        # 1. Block Calibrator
        feat_block = [f for f in self.features if f != 'shot_type']
        p_block_raw = self.model_block.predict_proba(df_c[feat_block])[:, 1]
        y_block = (df_c['event'] == 'blocked-shot').astype(int)
        
        if len(y_block.unique()) > 1:
            self.calibrator_block = LogisticRegression(C=1.0)
            self.calibrator_block.fit(p_block_raw.reshape(-1, 1), y_block)
        
        # 2. Goal Calibrator
        p_goal_est = self.predict_proba(df_c)[:, 1]
        y_goal = (df_c['event'] == 'goal').astype(int)
        
        if len(y_goal.unique()) > 1:
            self.calibrator_goal = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            self.calibrator_goal.fit(p_goal_est, y_goal)

    @staticmethod
    def _plot_calibration(clf, df_test, diag_dir):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Block
        y_block = (df_test['event'] == 'blocked-shot').astype(int)
        p_block = clf.predict_proba_layer(df_test, 'block')
        prob_true, prob_pred = calibration_curve(y_block, p_block, n_bins=10)
        axes[0].plot(prob_pred, prob_true, marker='o')
        axes[0].plot([0, 1], [0, 1], '--k', alpha=0.3)
        axes[0].set_title("Block Layer")
        
        # Accuracy
        mask_unblocked = df_test['event'] != 'blocked-shot'
        y_acc = df_test.loc[mask_unblocked, 'event'].isin(['shot-on-goal', 'goal']).astype(int)
        p_acc = clf.predict_proba_layer(df_test[mask_unblocked], 'accuracy')
        prob_true, prob_pred = calibration_curve(y_acc, p_acc, n_bins=10)
        axes[1].plot(prob_pred, prob_true, marker='o')
        axes[1].plot([0, 1], [0, 1], '--k', alpha=0.3)
        axes[1].set_title("Accuracy (Unblocked)")
        
        # Finish
        mask_on_net = df_test['event'].isin(['shot-on-goal', 'goal'])
        y_fin = (df_test.loc[mask_on_net, 'event'] == 'goal').astype(int)
        p_fin = clf.predict_proba_layer(df_test[mask_on_net], 'finish')
        prob_true, prob_pred = calibration_curve(y_fin, p_fin, n_bins=10)
        axes[2].plot(prob_pred, prob_true, marker='o')
        axes[2].plot([0, 1], [0, 1], '--k', alpha=0.3)
        axes[2].set_title("Finish (On Net)")
        
        plt.tight_layout()
        plt.savefig(diag_dir / 'xgboost_calibration.png')
        plt.close()

def train_xgboost_nested(df_raw, **kwargs):
    return XGBNestedXGClassifier.train(df_raw, **kwargs)
