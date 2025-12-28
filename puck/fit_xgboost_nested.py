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
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple

# We expect xgboost to be installed
import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

from . import features as feature_util

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

def preprocess_data(df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
    """
    Clean and prepare data for the XGBoost Nested Model.
    
    Steps:
    1. Filter to valid Shot Attempts.
    2. Remove Empty Net & Extreme Game States.
    3. Filter to Regular Season.
    4. Create Target Columns.
    5. Cast Categorical Features.
    """
    # 1. Scope: Shot Attempts Only
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    if 'event' in df.columns:
        df = df[df['event'].isin(valid_events)].copy()
    
    # 2. Exclude Empty Net & Extreme States
    if 'is_net_empty' in df.columns:
        # Filter 1, True, or '1'
        mask_empty = (df['is_net_empty'] == 1) | (df['is_net_empty'] == True)
        df = df[~mask_empty].copy()

    if 'game_state' in df.columns:
        mask_extreme = df['game_state'].isin(['1v0', '0v1'])
        df = df[~mask_extreme].copy()

    # 3. Regular Season Only
    if 'game_id' in df.columns:
        # Convert to str
        gid = df['game_id'].astype(str)
        # Check format YYYY02...
        mask_reg = (gid.str.len() >= 6) & (gid.str[4:6] == '02')
        # If any rows fail, filter
        if not mask_reg.all():
            df = df[mask_reg].copy()

    # 4. Create Targets (if 'event' exists)
    if 'event' in df.columns:
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
        df['is_goal_layer'] = (df['event'] == 'goal').astype(int)

    # 5. Handle Features & Categoricals
    if features is None:
        # Default features from central repo
        features = feature_util.get_features('all_inclusive')
        
    known_cats = ['game_state', 'shot_type', 'shoots_catches']
    
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
            
        # Ensure correct type for categoricals
        if col in known_cats:
            # Explicitly force category type
            df[col] = df[col].astype('category')
            
    # Explicitly ensure shot_type is NaN for Blocked Shots (redundant if checking later, but good for safety)
    # The user wants "appropriate shot attempts included... extra columns removed if needed"
    # We won't aggressively remove extra columns, but we ensure features are ready.
    
    # --- SUBSET TO MODEL COLUMNS ---
    # User requested removing unnecessary columns.
    # We keep: features + targets + identifiers + coordinates (needed for imputation)
    
    cols_to_keep = set(features)
    
    # Targets & IDs
    essential = ['game_id', 'event', 'is_blocked', 'is_on_net', 'is_goal_layer']
    for c in essential:
        if c in df.columns:
            cols_to_keep.add(c)
            
    # Coordinates (needed for subsequent imputation steps)
    potential_coords = ['x', 'y', 'coords_x', 'coords_y']
    for c in potential_coords:
        if c in df.columns:
            cols_to_keep.add(c)
            
    # Return subset
    final_cols = [c for c in df.columns if c in cols_to_keep]
    return df[final_cols].copy()


class XGBNestedXGClassifier(BaseEstimator, ClassifierMixin):
    """
    Nested xG Model using XGBoost.
    
    Architecture:
    1. Block Model: P(Unblocked)
       - Features: Distance, Angle, Game State, Handedness.
       - EXCLUDES: Shot Type (definitionally missing for blocks).
       
    2. Accuracy Model: P(On Net | Unblocked)
       - Features: All of the above + Shot Type.
       - Missing Shot Types are handled as NaN (XGBoost learns default path).
       
    3. Finish Model: P(Goal | On Net)
       - Features: All of the above + Shot Type.
    """
    
    def __init__(self, 
                 features: List[str] = None,
                 n_estimators: int = 200,
                 max_depth: int = 6,
                 learning_rate: float = 0.1,
                 random_state: int = 42,
                 enable_categorical: bool = True,
                 layer_params: Dict[str, Any] = None):
        
        # Default Features if None
        if features is None:
            self.features = feature_util.get_features('all_inclusive')
        else:
            self.features = features
            
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.enable_categorical = enable_categorical
        self.nan_mask_rate = 0.05 # Fraction of training examples to mask 'shot_type' to NaN
        self.layer_params = layer_params or {}
        
        # Internal Models
        self.model_block: Optional[XGBClassifier] = None
        self.model_accuracy: Optional[XGBClassifier] = None
        self.model_finish: Optional[XGBClassifier] = None
        
        # Configuration (set during fit)
        self.config_block: Optional[LayerConfig] = None
        self.config_accuracy: Optional[LayerConfig] = None
        self.config_finish: Optional[LayerConfig] = None
        
        self.final_features = [] # For compatibility with fit_xgs
        self.categorical_cols = []

    def _get_xgb_params(self) -> Dict[str, Any]:
        """Common parameters for all layers."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'enable_categorical': self.enable_categorical,
            'eval_metric': 'logloss',
            'tree_method': 'hist' # usually needed for enable_categorical
        }

    def _get_layer_params(self, layer_name: str) -> Dict[str, Any]:
        """Get params for a specific layer, merging defaults with overrides."""
        # Start with defaults
        params = self._get_xgb_params()
        
        # Override with specific layer params if available
        if layer_name in self.layer_params:
            overrides = self.layer_params[layer_name]
            # Remove 'score' if present (artifact from optimization script)
            if 'score' in overrides:
                overrides = {k: v for k, v in overrides.items() if k != 'score'}
            
            params.update(overrides)
            
        return params

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """
        Prepare DataFrame for XGBoost.
        - Casts categorical columns to 'category' dtype.
        - Ensures numeric columns are float.
        """
        df_out = df.copy()
        
        # Identify categorical columns based on known schema or dtype
        # We explicitly list common categoricals to ensure proper casting
        known_cats = ['game_state', 'shot_type', 'shoots_catches']
        
        for col in self.features:
            if col not in df_out.columns:
                if is_training:
                    raise ValueError(f"Missing feature '{col}' in training data.")
                else:
                    df_out[col] = np.nan # Predict time: allow missing
            
            # Cast known categoricals
            if col in known_cats:
                df_out[col] = df_out[col].astype('category')
            elif df_out[col].dtype == object and self.enable_categorical:
                 df_out[col] = df_out[col].astype('category')
                 
        return df_out

    def fit(self, X: pd.DataFrame, y=None):
        """Train the three layers."""
        df = self._prepare_data(X, is_training=True)
        
        # --- 1. Define Layer Features ---
        
        # BLOCK LAYER: Must EXCLUDE shot_type to prevent leakage
        feat_block = [f for f in self.features if 'shot_type' not in f]
        
        # ACCURACY / FINISH: Use all features
        feat_full = self.features
        
        self.config_block = LayerConfig("Block", "is_blocked", feat_block)
        self.config_accuracy = LayerConfig("Accuracy", "is_on_net", feat_full)
        self.config_finish = LayerConfig("Finish", "is_goal_layer", feat_full)
        
        # --- 2. Create Targets ---
        # (Assuming these columns exist or can be derived from 'event')
        if 'is_blocked' not in df.columns:
            # Fallback to deriving from 'event' if present
             if 'event' in df.columns:
                 df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
                 df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
                 df['is_goal_layer'] = (df['event'] == 'goal').astype(int)
             else:
                 raise ValueError("Targets (is_blocked, etc.) or 'event' column required.")

        # --- 3. Train Block Model (All Data) ---
        # --- 3. Train Block Model (All Data) ---
        logger.info(f"Training Block Model with features: {feat_block}")
        self.model_block = XGBClassifier(**self._get_layer_params('block'))
        self.model_block.fit(df[feat_block], df['is_blocked'])
        
        # --- 4. Train Accuracy Model (Unblocked Only) ---
        df_unblocked = df[df['is_blocked'] == 0].copy()
        
        # NaN INJECTION / DROPOUT for shot_type
        # To ensure the model learns a valid path for NaN shot_type (common in Blocked Shots),
        # we randomly mask some training examples to NaN.
        if self.nan_mask_rate > 0 and 'shot_type' in df_unblocked.columns:
            logger.info(f"Applying NaN injection to Accuracy Model (rate={self.nan_mask_rate})")
            n_mask = int(len(df_unblocked) * self.nan_mask_rate)
            if n_mask > 0:
                # Use a reliable random generator
                rng = np.random.default_rng(self.random_state)
                mask_idx = rng.choice(df_unblocked.index, size=n_mask, replace=False)
                # Need to use .loc to set.
                # Note: 'category' dtype allows NaN.
                df_unblocked.loc[mask_idx, 'shot_type'] = np.nan

        logger.info(f"Training Accuracy Model (N={len(df_unblocked)}) with features: {feat_full}")
        self.model_accuracy = XGBClassifier(**self._get_layer_params('accuracy'))
        self.model_accuracy.fit(df_unblocked[feat_full], df_unblocked['is_on_net'])
        
        # --- 5. Train Finish Model (On Net Only) ---
        df_on_net = df[df['is_on_net'] == 1].copy()
        
        # NaN INJECTION / DROPOUT for shot_type
        if self.nan_mask_rate > 0 and 'shot_type' in df_on_net.columns:
            logger.info(f"Applying NaN injection to Finish Model (rate={self.nan_mask_rate})")
            n_mask = int(len(df_on_net) * self.nan_mask_rate)
            if n_mask > 0:
                rng = np.random.default_rng(self.random_state)
                mask_idx = rng.choice(df_on_net.index, size=n_mask, replace=False)
                df_on_net.loc[mask_idx, 'shot_type'] = np.nan

        logger.info(f"Training Finish Model (N={len(df_on_net)}) with features: {feat_full}")
        self.model_finish = XGBClassifier(**self._get_layer_params('finish'))
        self.model_finish.fit(df_on_net[feat_full], df_on_net['is_goal_layer'])
        
        # Store metadata
        self.final_features = self.features
        self.categorical_cols = [c for c in self.features if df[c].dtype.name == 'category']
        
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return [P(NoGoal), P(Goal)]."""
        df = self._prepare_data(X, is_training=False)
        
        # 1. P(Blocked)
        # Note: Using only block features (no shot_type)
        p_blocked = self.model_block.predict_proba(df[self.config_block.feature_cols])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        # 2. P(On Net | Unblocked)
        # Note: Passes NaN for shot_type if missing (XGBoost handles it)
        p_on_net_cond = self.model_accuracy.predict_proba(df[self.config_accuracy.feature_cols])[:, 1]
        
        # 3. P(Goal | On Net)
        p_goal_cond = self.model_finish.predict_proba(df[self.config_finish.feature_cols])[:, 1]
        
        # 4. Combine: xG = P(Unblocked) * P(On Net | Unblocked) * P(Goal | On Net)
        p_goal = p_unblocked * p_on_net_cond * p_goal_cond
        
        return np.column_stack((1 - p_goal, p_goal))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.5).astype(int)
        
    def predict_proba_layer(self, X: pd.DataFrame, layer: str) -> np.ndarray:
        """Helper for diagnostics."""
        df = self._prepare_data(X)
        if layer == 'block':
            return self.model_block.predict_proba(df[self.config_block.feature_cols])[:, 1]
        elif layer == 'accuracy':
            return self.model_accuracy.predict_proba(df[self.config_accuracy.feature_cols])[:, 1]
        elif layer == 'finish':
            return self.model_finish.predict_proba(df[self.config_finish.feature_cols])[:, 1]
        else:
            raise ValueError(f"Unknown layer: {layer}")

