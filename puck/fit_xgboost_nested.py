"""fit_xgboost_nested.py

XGBOOST NESTED EXPECTED GOALS MODEL
===================================
This module implements the "Layered" or "Nested" xG model using XGBoost.
It leverages XGBoost's native capabilities for:
1.  Handling Missing Data (NaN): No distinct "Unknown" category needed.
2.  Categorical Support: Native 'enable_categorical=True' ensures optimal splits.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from . import features as feature_util

# --- STANDARD VOCABULARIES ---
VOCAB_GAME_STATE = [
    '5v5', '5v4', '4v5', '4v4', '6v5', '5v6', '3v3', '5v3', '4v3', 
    '6v4', '3v5', '4v6', '3v4', '6v3', '3v6', '6v6', '1v0', '0v1'
]
VOCAB_SHOT_TYPE = [
    'wrist', 'snap', 'slap', 'backhand', 'tip-in', 'deflected', 'wrap-around', 'Unknown'
]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("XGBNestedxG")

@dataclass
class LayerConfig:
    name: str
    target_col: str
    feature_cols: List[str]
    # XGBoost Params
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1

def preprocess_data(df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Clean and prepare data for the XGBoost Nested Model."""
    df = df.copy()
    
    # Fast metadata wipe for categoricals to avoid code mismatches
    # We MUST reset the index and clear any existing category mapping
    df = df.reset_index(drop=True)
    for col in (df.columns):
        if hasattr(df[col], 'cat'):
            df[col] = df[col].astype(object)
    
    # Create Targets if event exists
    if 'event' in df.columns:
        # Standard filtering (optional but good for training)
        valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
        df = df[df['event'].isin(valid_events)].copy()
        
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
        df['is_goal_layer'] = (df['event'] == 'goal').astype(int)
    
    # Standardize Categoricals using fixed VOCABs
    if 'game_state' in df.columns:
        df['game_state'] = pd.Categorical(df['game_state'], categories=VOCAB_GAME_STATE)
    
    if 'shot_type' in df.columns:
        df['shot_type'] = df['shot_type'].fillna('Unknown')
        df['shot_type'] = pd.Categorical(df['shot_type'], categories=VOCAB_SHOT_TYPE)

    for col in (df.columns):
        if col not in ['game_state', 'shot_type']:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype('category')
            elif features and col in features:
                try:
                    df[col] = df[col].astype(float)
                except (TypeError, ValueError):
                    pass
            
    # Final clean up: explicitly cast to RangeIndex and ensure no weird index metadata
    df.index = pd.RangeIndex(len(df))
    return df

class XGBNestedXGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 features: List[str] = None,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 enable_categorical: bool = True,
                 use_calibration: bool = True,
                 use_balancing: bool = True,
                 layer_params: Dict[str, Any] = None):
        
        if features is None:
            self.features = feature_util.get_features('all_inclusive')
        else:
            self.features = features
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        self.nan_mask_rate = 0.05
        self.use_calibration = use_calibration
        self.use_balancing = use_balancing
        self.layer_params = layer_params or {}
        self.calibrator = None
        
        self.model_block = None
        self.model_accuracy = None
        self.model_finish = None
        self.feature_dtypes = {} # To store dtypes for inference consistency
        
        # Introspection Configs (for diagnostics)
        feat_block = [f for f in self.features if 'shot_type' not in f]
        self.config_block = LayerConfig(name='block', target_col='is_blocked', feature_cols=feat_block)
        self.config_accuracy = LayerConfig(name='accuracy', target_col='is_on_net', feature_cols=self.features)
        self.config_finish = LayerConfig(name='finish', target_col='is_goal_layer', feature_cols=self.features)
        
    def _get_xgb_params(self, layer_name: str) -> Dict[str, Any]:
        params = {
            'n_estimators': int(self.n_estimators),
            'max_depth': int(self.max_depth),
            'learning_rate': float(self.learning_rate),
            'random_state': int(self.random_state),
            'enable_categorical': bool(self.enable_categorical),
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cpu',
            'objective': 'binary:logistic'
        }
        if layer_name in self.layer_params:
            overrides = self.layer_params[layer_name]
            params.update({k: v for k, v in overrides.items() if k != 'score'})
        return params

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for prediction (inference)."""
        df_out = df.copy()
        
        # Fast metadata wipe for categoricals to avoid code mismatches
        df_out = df_out.reset_index(drop=True)
        for col in (df_out.columns):
            if hasattr(df_out[col], 'cat'):
                df_out[col] = df_out[col].astype(object)
        
        try:
            # Standardize known categoricals
            if 'game_state' in df_out.columns:
                df_out['game_state'] = pd.Categorical(df_out['game_state'], categories=VOCAB_GAME_STATE)
            
            if 'shot_type' in df_out.columns:
                df_out['shot_type'] = df_out['shot_type'].fillna('Unknown')
                df_out['shot_type'] = pd.Categorical(df_out['shot_type'], categories=VOCAB_SHOT_TYPE)

            for col in (self.features or []):
                if col not in df_out.columns:
                    # If we have a recorded dtype (especially categorical), use it
                    if self.feature_dtypes and col in self.feature_dtypes:
                        dt = self.feature_dtypes[col]
                        if isinstance(dt, pd.CategoricalDtype):
                            df_out[col] = pd.Series([np.nan]*len(df_out), dtype=dt)
                        else:
                            df_out[col] = np.nan
                    else:
                        df_out[col] = np.nan
                
                # Apply recorded categories if they exist to ensure code mapping is identical
                if self.feature_dtypes and col in self.feature_dtypes:
                    dt = self.feature_dtypes[col]
                    if isinstance(dt, pd.CategoricalDtype):
                        # Wipe existing if necessary (safety)
                        if hasattr(df_out[col], 'cat'):
                            df_out[col] = df_out[col].astype(object)
                        df_out[col] = pd.Categorical(df_out[col], categories=dt.categories)
                    else:
                        try:
                            # Standardize numeric to float
                            if pd.api.types.is_numeric_dtype(dt):
                                df_out[col] = df_out[col].astype(float)
                        except:
                            pass
                else:
                    # FALLBACK: If we don't have recorded dtypes yet (e.g. during calibration fit),
                    # convert objects to category to satisfy XGBoost.
                    if col not in ['game_state', 'shot_type']:
                        if pd.api.types.is_object_dtype(df_out[col]) or pd.api.types.is_string_dtype(df_out[col]):
                            df_out[col] = df_out[col].astype('category')
                        else:
                            try:
                                if pd.api.types.is_numeric_dtype(df_out[col]):
                                    df_out[col] = df_out[col].astype(float)
                            except:
                                pass
        except Exception as e:
            logger.error(f"Error in _prepare_df: {e}")
            raise
                    
        # Final clean up: explicitly cast to RangeIndex and ensure no weird index metadata
        df_out.index = pd.RangeIndex(len(df_out))
        return df_out

    def fit(self, X: pd.DataFrame, y=None):
        if self.use_calibration:
            df_train_raw, df_calib_raw = train_test_split(X, test_size=0.2, random_state=self.random_state)
            logger.info(f"Calibration enabled. Training on {len(df_train_raw)} rows, Calibrating on {len(df_calib_raw)} rows.")
        else:
            df_train_raw = X
            df_calib_raw = None

        df = preprocess_data(df_train_raw, features=self.features)
        
        feat_block = [f for f in self.features if 'shot_type' not in f]
        feat_full = self.features
        
        # 1. Block Model
        logger.info(f"Training Block Model... Index: {df.index}")
        p_block = self._get_xgb_params('block')
        self.model_block = XGBClassifier(**p_block)
        self.model_block.fit(df[feat_block], df['is_blocked'])
        
        # 2. Accuracy Model
        df_unblocked = df[df['is_blocked'] == 0].copy().reset_index(drop=True)
        
        if self.nan_mask_rate > 0 and 'shot_type' in df_unblocked.columns:
            n_mask = int(len(df_unblocked) * self.nan_mask_rate)
            if n_mask > 0:
                rng = np.random.default_rng(self.random_state)
                # Mask to Unknown which is handled natively by NaN in most cases, 
                # OR we set it to 'Unknown' and let categorical handle it.
                # Since we use Native Categorical, we can check if 'Unknown' is in categories.
                # Our VOCAB has 'Unknown'.
                mask_idx = rng.choice(df_unblocked.index, size=n_mask, replace=False)
                df_unblocked.loc[mask_idx, 'shot_type'] = 'Unknown'
        
        logger.info(f"Training Accuracy Model (N={len(df_unblocked)})... Index: {df_unblocked.index}")
        p_acc = self._get_xgb_params('accuracy')
        self.model_accuracy = XGBClassifier(**p_acc)
        self.model_accuracy.fit(df_unblocked[feat_full], df_unblocked['is_on_net'])
        
        # 3. Finish Model
        df_on_net = df[df['is_on_net'] == 1].copy().reset_index(drop=True)
        
        if self.nan_mask_rate > 0 and 'shot_type' in df_on_net.columns:
            n_mask = int(len(df_on_net) * self.nan_mask_rate)
            if n_mask > 0:
                rng = np.random.default_rng(self.random_state)
                mask_idx = rng.choice(df_on_net.index, size=n_mask, replace=False)
                df_on_net.loc[mask_idx, 'shot_type'] = 'Unknown'
        
        logger.info(f"Training Finish Model (N={len(df_on_net)})... Index: {df_on_net.index}")
        p_finish = self._get_xgb_params('finish')
        if self.use_balancing and 'scale_pos_weight' not in p_finish:
            pos = df_on_net['is_goal_layer'].sum()
            neg = len(df_on_net) - pos
            if pos > 0:
                p_finish['scale_pos_weight'] = neg / pos
                logger.info(f"  Applied scale_pos_weight: {p_finish['scale_pos_weight']:.2f}")

        self.model_finish = XGBClassifier(**p_finish)
        self.model_finish.fit(df_on_net[feat_full], df_on_net['is_goal_layer'])
        
        # Record final dtypes for categorical consistency
        # CRITICAL: Do this BEFORE predict_proba call during calibration
        self.feature_dtypes = df[self.features].dtypes.to_dict()
        
        # 4. Calibration
        if self.use_calibration and df_calib_raw is not None:
            logger.info("Fitting Platt Scaling calibrator...")
            calib_preds = self.predict_proba(df_calib_raw)[:, 1]
            if 'event' in df_calib_raw.columns:
                df_c = preprocess_data(df_calib_raw, features=self.features)
                targets = (df_c['event'] == 'goal').astype(int)
            else:
                # If we only have subsets, we might need preprocessing
                # But usually df_calib_raw is just the raw split
                df_c = preprocess_data(df_calib_raw, features=self.features)
                targets = df_c['is_goal_layer']
            
            # Ensure alignment? predict_proba prepares df -> reset index
            # targets should match
            # If df_calib_raw has index, predict_proba resets it?
            # predict_proba returns array.
            # targets needs to be array of same length.
            
            self.calibrator = LogisticRegression(C=1e5) # Large C for Platt Scaling
            self.calibrator.fit(calib_preds.reshape(-1, 1), targets)
        
        self.final_features = self.features
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        df = self._prepare_df(X)
        feat_block = [f for f in self.features if 'shot_type' not in f]
        
        # 0. Debug Logging
        if len(df) < 100: # Only for small/synthetic checks to avoid log spam
            logger.info(f"Predict Proba Input: Index={type(df.index)}, Dtypes={df.dtypes.to_dict()}")

        # 1. P(Blocked)
        p_blocked = self.model_block.predict_proba(df[feat_block])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        # 2. P(On Net | Unblocked)
        p_on_net_cond = self.model_accuracy.predict_proba(df[self.features])[:, 1]
        
        # 3. P(Goal | On Net)
        p_goal_cond = self.model_finish.predict_proba(df[self.features])[:, 1]
        
        # 4. Combine
        p_goal = p_unblocked * p_on_net_cond * p_goal_cond
        
        # 5. Apply Calibration
        if self.use_calibration and self.calibrator:
            p_goal = self.calibrator.predict_proba(p_goal.reshape(-1, 1))[:, 1]
        
        return np.column_stack((1 - p_goal, p_goal))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        
    def predict_proba_layer(self, X: pd.DataFrame, layer: str) -> np.ndarray:
        """Helper for diagnostics."""
        df = self._prepare_df(X)
        if layer == 'block':
            feat_block = [f for f in self.features if 'shot_type' not in f]
            return self.model_block.predict_proba(df[feat_block])[:, 1]
        elif layer == 'accuracy':
            return self.model_accuracy.predict_proba(df[self.features])[:, 1]
        elif layer == 'finish':
            return self.model_finish.predict_proba(df[self.features])[:, 1]
        else:
            raise ValueError(f"Unknown layer: {layer}")
