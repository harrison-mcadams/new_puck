"""fit_xgboost_tensor.py

XGBOOST TENSOR EXPECTED GOALS MODEL
===================================
This module implements the "Layered" or "Nested" xG model using pure XGBoost
for spatial representation (No GLM pre-processing).

It leverages XGBoost's native capabilities for:
1. Handling Missing Data (NaN): No distinct "Unknown" category needed.
2. Categorical Support: Native 'enable_categorical=True' ensures optimal splits.
3. Complex Non-Linear Interactions (Space): Learns spatial geometry purely from x, y, distance, and angle.

Parity: Mirrors Nested GLM structure but delegates coordinate space fully to the trees.
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
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

from . import features as feature_util
from . import config as puck_config
from .spline_transformer import TensorSpline

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from .verify import verify_df

# --- STANDARD VOCABULARIES ---
VOCAB_GAME_STATE = [
    '5v5', '5v4', '4v5', '4v4', '6v5', '5v6', '3v3', '5v3', '4v3', 
    '6v4', '3v5', '4v6', '3v4', '6v3', '3v6', '6v6', '1v0', '0v1'
]
VOCAB_SHOT_TYPE = [
    'wrist', 'snap', 'slap', 'backhand', 'tip-in', 'deflected', 'wrap-around', 
    'bat', 'poke', 'between-legs', 'cradle', 'Unknown'
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
        'faceoff', 'hit', 'giveaway', 'takeaway', 'missed-shot', 'blocked-shot', 'shot-on-goal', 'goal', 'penalty', 'stoppage', 'period-start', 'period-end'
    ],
    'rebound_source': ['none', 'shot-on-goal', 'missed-shot', 'blocked-shot', 'goal'],
    'season': [int(f"{y}{y+1}") for y in range(2009, 2026)]
}

logger = logging.getLogger(__name__)

class XGBTensorXGClassifier(BaseEstimator, ClassifierMixin):

    """
    Nested Expected Goals Model using Pure XGBoost (No GLM Base).
    
    Structure:
    1. Block Model: P(Unblocked | Shot)
    2. Accuracy Model: P(On Net | Unblocked)
    3. Finish Model: P(Goal | On Net)
    
    P(Goal) = P(Unblocked) * P(On Net) * P(Goal | On Net)
    """
    
    def __init__(self, 
                 features: Optional[List[str]] = None,
                 n_estimators: int = 3000,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 random_state: int = 42,
                 enable_categorical: bool = True,
                 enable_marginalization: bool = False,
                 use_balancing: bool = False,
                 use_calibration: bool = False,
                 layer_params: Optional[Dict[str, Any]] = None,
                 use_splines: bool = True,
                 predict_mode: str = 'nested',
                 enable_verification: bool = True):
        
        # Defensive copy to prevent bleeding from other model's modifications to the global feature set
        base_feats = features.copy() if features else feature_util.get_features('all_inclusive').copy()
        self.features = [f for f in base_feats if f not in ['spatial_block', 'spatial_acc', 'spatial_fin']]
        
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
        self.n_knots = 5
        self.spline_transformer_ = None
        self.spline_feature_names_ = []
        self.predict_mode = predict_mode
        self.enable_verification = enable_verification
        
        # Sub-models
        
        # Sub-models
        self.model_block = None
        self.model_acc = None
        self.model_finish = None
        self.model_overall = None
        
        # Consistent Dtypes for Inference
        self.feature_dtypes = {}
        
        # Marginalization Support
        self.categorical_priors_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        logger.info(f"Fitting XGBTensorXGClassifier on {len(X)} rows.")

        if self.use_calibration:
            df_pool, df_calib = train_test_split(X, test_size=0.2, random_state=self.random_state)
        else:
            df_pool = X
            df_calib = None

        # Reserve 10% for early stopping from the training pool
        df_train_raw, df_val_raw = train_test_split(df_pool, test_size=0.1, random_state=self.random_state)
        
        # Feature parity across all sub-models as requested by USER.
        core_feats = self.features.copy()
        # Previously restricted leaky features (spatial sequence inconsistencies) are now included.
        self.features_block = core_feats.copy()
        self.features_acc = core_feats.copy()
        self.features_fin = core_feats.copy()
        
        # Prepare DFs
        df = self._prepare_training_df(df_train_raw)
        
        # Record Dtypes so _prepare_inference_df can apply them to the validation set
        self.feature_dtypes = df[self.features].dtypes.to_dict()
        
        df_val = self._prepare_inference_df(df_val_raw)
        
        # If splines were added, update the layer-specific lists
        if self.use_splines:
            for flist in [self.features_block, self.features_acc, self.features_fin]:
                for bname in self.spline_feature_names_:
                    if bname not in flist:
                        flist.append(bname)

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

        # Run verification prior to fitting
        verify_df(df, self.features, verify_blocked=True)

        # 2. Block Model
        y_block = (df['event'] == 'blocked-shot').astype(int)
        
        # [DIAGNOSTIC] Check alignment
        if 'distance' in df.columns:
            corr = df['distance'].corr(y_block)
            logger.info(f"  [ALIGNMENT CHECK] Correlation(distance, y_block): {corr:.4f}")
            logger.info(f"  [ALIGNMENT CHECK] Block Rate in Training DF: {y_block.mean():.4f}")

        p_block = self._get_xgb_params('block')
        
        if self.use_balancing and 'scale_pos_weight' not in p_block:
            pos = y_block.sum()
            neg = len(y_block) - pos
            if pos > 0:
                p_block['scale_pos_weight'] = neg / pos
                logger.info(f"  Block Model Balance (scale_pos_weight): {p_block['scale_pos_weight']:.2f}")

        self.model_block = XGBClassifier(**p_block)
        
        X_val_block = df_val[self.features_block]
        y_val_block = (df_val['event'] == 'blocked-shot').astype(int)
        
        self.model_block.fit(
            df[self.features_block], y_block,
            eval_set=[(X_val_block, y_val_block)],
            verbose=False
        )
        
        # 3. Accuracy Model
        mask_unblocked = df['event'] != 'blocked-shot'
        df_unblocked = df[mask_unblocked].copy()
        y_acc = df_unblocked['event'].isin(['shot-on-goal', 'goal']).astype(int)
        
        p_acc = self._get_xgb_params('accuracy')
        self.model_acc = XGBClassifier(**p_acc)
        
        X_val_acc = df_val[df_val['event'] != 'blocked-shot'][self.features_acc]
        y_val_acc = df_val[df_val['event'] != 'blocked-shot']['event'].isin(['shot-on-goal', 'goal']).astype(int)
        
        self.model_acc.fit(
            df_unblocked[self.features_acc], y_acc,
            eval_set=[(X_val_acc, y_val_acc)],
            verbose=False
        )
        
        # 4. Finish Model
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
        
        mask_val_on_net = df_val['event'].isin(['shot-on-goal', 'goal'])
        X_val_finish = df_val[mask_val_on_net][self.features_fin]
        y_val_finish = (df_val[mask_val_on_net]['event'] == 'goal').astype(int)
        
        self.model_finish.fit(
            df_on_net[self.features_fin], y_finish,
            eval_set=[(X_val_finish, y_val_finish)],
            verbose=False
        )

        # 5. Overall Model (Direct Goal Prediction for calibration)
        logger.info("Fitting Overall Model (P(Goal | Shot))...")
        y_goal = (df['event'] == 'goal').astype(int)
        p_overall = self._get_xgb_params('overall')
        # Never balance overall model, we want raw calibration
        p_overall['scale_pos_weight'] = 1.0 
        self.model_overall = XGBClassifier(**p_overall)
        
        y_val_overall = (df_val['event'] == 'goal').astype(int)
        
        self.model_overall.fit(
            df[self.features], y_goal,
            eval_set=[(df_val[self.features], y_val_overall)],
            verbose=False
        )
        
        # Note: self.feature_dtypes already recorded earlier in fit()
        
        if self.use_calibration and df_calib is not None:
            self._fit_calibrators(df_calib)

        logger.info("Fit Complete.")
        return self

    @classmethod
    def train(cls, df_raw: pd.DataFrame, save_path: Optional[str] = None, out_dir: Optional[str] = None, verbose: bool = True, **kwargs):
        from . import data_pipeline, model_summary
        
        def vprint(*args):
            if verbose: print(*args)

        vprint("--- Training XGBoost (Tensor - No GLM) Model ---")
        
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

        df_train, df_test = train_test_split(
            df, 
            test_size=kwargs.get('test_size', 0.2), 
            random_state=kwargs.get('random_state', 42)
        )

        feature_list = feature_util.get_features('all_inclusive')
        clf = cls(
            features=feature_list,
            n_estimators=2000,
            max_depth=6,
            learning_rate=0.05,
            use_calibration=kwargs.get('use_calibration', False),
            use_balancing=kwargs.get('use_balancing', False),
            use_splines=kwargs.get('use_splines', True)
        )

        vprint(f"Training on {len(df_train)} rows with {len(clf.features)} features...")
        start_t = time.time()
        clf.fit(df_train)
        vprint(f"Training took {time.time() - start_t:.1f}s.")

        vprint("\n--- Evaluation (Test Set) ---")
        y_test_goal = (df_test['event'] == 'goal').astype(int)
        probs = clf.predict_proba(df_test)[:, 1]
        
        auc = roc_auc_score(y_test_goal, probs)
        ll = log_loss(y_test_goal, probs)
        brier = brier_score_loss(y_test_goal, probs)
        vprint(f"Overall xG AUC: {auc:.4f}, LogLoss: {ll:.4f}, Brier: {brier:.6f}")
        
        clf.test_metrics_ = {'auc': auc, 'logloss': ll, 'brier': brier}

        if save_path is None:
            save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor.joblib')
        
        vprint(f"Saving model to {save_path}...")
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, save_path)
        
        meta = {
            'final_features': clf.features,
            'model_type': 'xgboost_tensor',
            'train_params': {
                'n_estimators': clf.n_estimators,
                'max_depth': clf.max_depth,
                'learning_rate': clf.learning_rate
            }
        }
        with open(save_path + '.meta.json', 'w') as f:
            json.dump(meta, f)

        if out_dir is None:
            diag_dir = Path(puck_config.ANALYSIS_DIR) / 'xgboost_tensor_xgs'
        else:
            diag_dir = Path(out_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        vprint("Generating model summary...")
        model_summary.generate_model_summary(model_path=save_path, test_df=df_test, output_dir=str(diag_dir), verbose=verbose)

        return clf

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = self._prepare_inference_df(X)
        
        # [DIAGNOSTIC] Deep Verification (Core Model Logic)
        if getattr(self, 'enable_verification', True):
            # We use verify_blocked=True to ensure blocks have sensible distance/orientation
            verify_df(df, self.features, verify_blocked=True, mode='inference')

        if self.model_block is None:
            raise NotFittedError("Model not fitted.")
            
        p_blocked = self._predict_marginalized(self.model_block, df, self.features_block)
        p_unblocked = 1.0 - p_blocked
        p_acc = self._predict_marginalized(self.model_acc, df, self.features_acc)
        p_finish = self._predict_marginalized(self.model_finish, df, self.features_fin)
        p_nested = p_unblocked * p_acc * p_finish

        if getattr(self, 'predict_mode', 'nested') == 'nested':
            p_goal = p_nested
        else:
            # Fallback to overall model if explicitly requested
            p_goal = self.model_overall.predict_proba(df[self.features])[:, 1]
            
        # [DIAGNOSTIC] Compare with nested product (for dashboard/breakdown awareness)
        if len(df) > 1000:
            logger.info(f"  [CALIBRATION] Overall Mean xG (from flat model): {self.model_overall.predict_proba(df[self.features])[:, 1].mean():.4f}")
            logger.info(f"  [CALIBRATION] Nested Mean xG:  {p_nested.mean():.4f}")

        return np.column_stack((1 - p_goal, p_goal))

    def predict_proba_layer(self, X: pd.DataFrame, layer: str) -> np.ndarray:
        df = self._prepare_inference_df(X)
        if layer == 'block':
            return self._predict_marginalized(self.model_block, df, self.features_block)
        elif layer == 'accuracy':
            return self._predict_marginalized(self.model_acc, df, self.features_acc)
        elif layer == 'finish':
            return self._predict_marginalized(self.model_finish, df, self.features_fin)
        raise ValueError(f"Unknown layer: {layer}")

    def _get_xgb_params(self, layer_name: str) -> Dict[str, Any]:
        # Determine base_score from historical averages to prevent OOD inflation
        # (XGBoost defaults to 0.5, which is way too high for NHL goal rates)
        base_scores = {
            'block': 0.26,
            'accuracy': 0.70,
            'finish': 0.10,
            'overall': 0.05
        }
        b_score = base_scores.get(layer_name, 0.5)

        params = {
            'n_estimators': 2000,
            'max_depth': int(self.max_depth),
            'learning_rate': float(self.learning_rate),
            'random_state': int(self.random_state),
            'enable_categorical': bool(self.enable_categorical),
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'device': 'cpu',
            'eval_metric': 'logloss',
            'base_score': b_score,
            'min_child_weight': 500,
            'gamma': 5.0,
            'reg_lambda': 15.0,
            'reg_alpha': 2.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 50
        }
        if layer_name in self.layer_params:
            params.update(self.layer_params[layer_name])
        return params

    def _prepare_training_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        if self.use_splines:
            logger.info(f"  Generating Tensor Product Spline Basis ({self.n_knots}x{self.n_knots})...")
            self.spline_transformer_ = TensorSpline(n_knots=self.n_knots, degree=3)
            # Use raw coordinates for basis
            coords = df[['x', 'y']].astype(float)
            self.spline_transformer_.fit(coords)
            self.spline_feature_names_ = self.spline_transformer_.get_feature_names_out(['x', 'y'])
            
            basis = self.spline_transformer_.transform(coords)
            df_basis = pd.DataFrame(basis, columns=self.spline_feature_names_, index=df.index)
            # Ensure indices are perfectly aligned before joining
            df = pd.merge(df, df_basis, left_index=True, right_index=True, how='left')
                        
        for col in self.features:
            if col in df.columns:
                if df[col].dtype == 'object' or col in CATEGORICAL_VOCABS:
                    vocab = CATEGORICAL_VOCABS.get(col)
                    df[col] = pd.Categorical(df[col], categories=vocab) if vocab else df[col].astype('category')
        return df

    def _prepare_inference_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        use_splines = getattr(self, 'use_splines', True)
        if use_splines and getattr(self, 'spline_transformer_', None):
            coords = df[['x', 'y']].astype(float).fillna(0)
            basis = self.spline_transformer_.transform(coords)
            df_basis = pd.DataFrame(basis, columns=self.spline_feature_names_, index=df.index)
            df = pd.concat([df, df_basis], axis=1)
            
        feature_dtypes = getattr(self, 'feature_dtypes', {})
        for col, dt in feature_dtypes.items():
            if col not in df.columns:
                df[col] = np.nan
            
            if isinstance(dt, pd.CategoricalDtype):
                df[col] = pd.Categorical(df[col], categories=dt.categories)
            else:
                # If it's not categorical in the model, keep it as numeric (even if in CATEGORICAL_VOCABS)
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        return df

    def _predict_marginalized(self, model, df, features):
        p_base = model.predict_proba(df[features])[:, 1]
        col = 'shot_type'
        priors_map = getattr(self, 'categorical_priors_', {})
        if col not in df.columns or col not in priors_map:
            return p_base
        mask_nan = df[col].isna()
        if not mask_nan.any():
            return p_base
        priors = priors_map[col]
        df_nan = df[mask_nan].copy()
        weighted_prob = np.zeros(len(df_nan))
        for val, weight in priors.items():
            df_nan[col] = pd.Categorical([val]*len(df_nan), categories=CATEGORICAL_VOCABS[col])
            weighted_prob += model.predict_proba(df_nan[features])[:, 1] * weight
        p_base[mask_nan] = weighted_prob
        return p_base

    def _fit_calibrators(self, df_calib_raw: pd.DataFrame):
        pass

def train_xgboost_tensor(df_raw, **kwargs):
    return XGBTensorXGClassifier.train(df_raw, **kwargs)

# Alias for backward compatibility with pickled models
XGBAlternateXGClassifier = XGBTensorXGClassifier

