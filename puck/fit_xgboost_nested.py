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
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score

from . import features as feature_util

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
}

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
        
        # Filter out Empty Net shots
        if 'is_net_empty' in df.columns:
            df = df[df['is_net_empty'] == 0].copy()

        # Filter out Shootout/Penalty Shot states (1v0, 0v1)
        if 'game_state' in df.columns:
            df = df[~df['game_state'].isin(['1v0', '0v1'])].copy()
            
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
        self.use_calibration = use_calibration
        self.use_balancing = use_balancing
        self.layer_params = layer_params or {}
        self.calibrator = None
        self.calibrator_block = None
        
        self.model_block = None
        self.model_accuracy = None
        self.model_finish = None
        self.feature_dtypes = {} # To store dtypes for inference consistency
        
        # Introspection Configs (for diagnostics)
        feat_block = [f for f in self.features if 'shot_type' not in f]
        self.config_block = LayerConfig(name='block', target_col='is_blocked', feature_cols=feat_block)
        self.config_accuracy = LayerConfig(name='accuracy', target_col='is_on_net', feature_cols=self.features)
        self.config_finish = LayerConfig(name='finish', target_col='is_goal_layer', feature_cols=self.features)
        
        # Marginalization Support
        self.categorical_priors_ = {}  # Dict[str, Dict[str, float]] - priors for each categorical feature
        
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
            'objective': 'binary:logistic',
            'base_score': 0.5  # Explicitly set to avoid "must be in (0,1)" error
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
                # If 'Unknown' is passed, map to NaN (since we removed Unknown from VOCAB)
                df_out['shot_type'] = df_out['shot_type'].replace('Unknown', np.nan)
                df_out['shot_type'] = pd.Categorical(df_out['shot_type'], categories=VOCAB_SHOT_TYPE)

            for col in (self.features or []):
                pass # (Snipped for brevity in replacement search)
                if col not in df_out.columns:
                    # If we have a recorded dtype (especially categorical), use it
                    feature_dtypes = getattr(self, 'feature_dtypes', {})
                    if feature_dtypes and col in feature_dtypes:
                        dt = feature_dtypes[col]
                        if isinstance(dt, pd.CategoricalDtype):
                            df_out[col] = pd.Series([np.nan]*len(df_out), dtype=dt)
                        else:
                            df_out[col] = np.nan
                    else:
                        df_out[col] = np.nan
                
                # Apply recorded categories if they exist to ensure code mapping is identical
                feature_dtypes = getattr(self, 'feature_dtypes', {})
                if feature_dtypes and col in feature_dtypes:
                    dt = feature_dtypes[col]
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
        
        # Filter out 'Unknown' shot types from training data (for unblocked shots only)
        if 'shot_type' in df.columns:
            # We want to train only on valid shot types for accuracy/finish layers.
            # However, Blocked shots rarely have shot_type recorded (NaN). 
            # We MUST preserve them for the block layer.
            # We filter out rows where shot_type is NaN AND the shot was NOT blocked.
            valid_mask = ~df['shot_type'].isna() | (df['is_blocked'] == 1)
            
            if valid_mask.sum() < len(df):
                logger.info(f"Dropping {len(df) - valid_mask.sum()} rows with Unknown/NaN shot_type (unblocked shots only) from training.")
                df = df[valid_mask].reset_index(drop=True)


        # Calculate Priors for Marginalization (all categorical features)
        self.categorical_priors_ = {}
        for col, vocab in CATEGORICAL_VOCABS.items():
            if col in df.columns:
                counts = df[col].value_counts(normalize=True, dropna=True)
                priors = {k: v for k, v in counts.items() if k in vocab}
                # Re-normalize 
                total_prob = sum(priors.values())
                if total_prob > 0:
                    priors = {k: v/total_prob for k, v in priors.items()}
                    self.categorical_priors_[col] = priors
                    logger.info(f"Learned {col} priors: {priors}")
        
        # Backward compatibility
        self.shot_type_priors_ = self.categorical_priors_.get('shot_type')
        
        feat_block = [f for f in self.features if 'shot_type' not in f]
        feat_full = self.features
        
        # 1. Block Model
        logger.info(f"Training Block Model... Index: {df.index}")
        y_block = df['is_blocked']
        logger.info(f"Block Target Stats: Mean={y_block.mean():.4f}, Min={y_block.min()}, Max={y_block.max()}, Unique={y_block.unique()}")
        
        p_block = self._get_xgb_params('block')
        self.model_block = XGBClassifier(**p_block)
        self.model_block.fit(df[feat_block], y_block)
        
        # 2. Accuracy Model
        df_unblocked = df[df['is_blocked'] == 0].copy().reset_index(drop=True)
        
        logger.info(f"Training Accuracy Model (N={len(df_unblocked)})... Index: {df_unblocked.index}")
        p_acc = self._get_xgb_params('accuracy')
        self.model_accuracy = XGBClassifier(**p_acc)
        self.model_accuracy.fit(df_unblocked[feat_full], df_unblocked['is_on_net'])
        
        # 3. Finish Model
        df_on_net = df[df['is_on_net'] == 1].copy().reset_index(drop=True)
        
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
            logger.info("Fitting Platt Scaling calibrators...")
            
            # Prepare Calib Data
            df_c = preprocess_data(df_calib_raw, features=self.features)
            
            # A. Block Model Calibration
            p_block_raw = self.model_block.predict_proba(df_c[feat_block])[:, 1]
            self.calibrator_block = LogisticRegression(C=0.01) # Robust Regularization (Strategy 1)
            # check for single class edge case
            if len(df_c['is_blocked'].unique()) > 1:
                self.calibrator_block.fit(p_block_raw.reshape(-1, 1), df_c['is_blocked'])
                logger.info("  Block Model Calibrator FITTED.")
            else:
                logger.warning("  Block Model Calibration skipped (only 1 class in calibration set).")
                self.calibrator_block = None

            # B. Final Model Calibration
            # Recalculate full probability flow with newly calibrated block prob?
            # We should probably use the calibrated block prob in the chain.
            if self.calibrator_block:
                p_blocked_c = self.calibrator_block.predict_proba(p_block_raw.reshape(-1, 1))[:, 1]
            else:
                p_blocked_c = p_block_raw
            
            p_unblocked = 1.0 - p_blocked_c
            p_on_net_cond = self.model_accuracy.predict_proba(df_c[self.features])[:, 1]
            p_goal_cond = self.model_finish.predict_proba(df_c[self.features])[:, 1]
            p_goal_est = p_unblocked * p_on_net_cond * p_goal_cond

            if 'event' in df_calib_raw.columns:
                targets = (df_c['event'] == 'goal').astype(int)
            else:
                 targets = df_c['is_goal_layer'] # Fallback
            
            # Switch to Isotonic for aggressive upper-tail calibration
            # Logic: We prefer to uncap high-danger probabilities even if curve is step-function
            self.calibrator = IsotonicRegression(out_of_bounds='clip', y_min=0, y_max=1)
            
            # DEBUG: Inspect inputs to Isotonic Fit
            print(f"Isotonic Fit Debug: p_goal_est shape={p_goal_est.shape}, targets shape={targets.shape}")
            print(f"  p_goal_est stats: Min={p_goal_est.min():.4f}, Max={p_goal_est.max():.4f}, Mean={p_goal_est.mean():.4f}")
            print(f"  targets stats:    Sum={targets.sum()}, Mean={targets.mean():.4f}")
            
            if len(targets.unique()) > 1:
                try:
                    self.calibrator.fit(p_goal_est, targets) # Isotonic expects 1D input (n_samples,)
                    print("  Final Model Calibrator FITTED (Isotonic).")
                except Exception as e:
                    print(f"  Isotonic Fit FAILED: {e}")
            else:
                print("  Skipping calibration: Targets have only 1 unique value.")

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
        
        # Apply Block Calibration
        if getattr(self, 'calibrator_block', None):
            p_blocked = self.calibrator_block.predict_proba(p_blocked.reshape(-1, 1))[:, 1]

        p_unblocked = 1.0 - p_blocked
        
        # Standard Calculation (will be overwritten for NaNs)
        p_start_acc = self.model_accuracy.predict_proba(df[self.features])[:, 1]
        p_start_fin = self.model_finish.predict_proba(df[self.features])[:, 1]
        p_goal = p_unblocked * p_start_acc * p_start_fin
        
        # 4. Integrate Marginalization for Missing/Unknown Shot Types
        # Note: If no shot_types are missing, this loop is skipped or p_goal is returned directly.
        
        mask_nan = df['shot_type'].isna()
        if self.shot_type_priors_ and mask_nan.any():
            # Standard Calculation already done for NaNs (using default branch), 
            # BUT we want to replace it with weighted average.
            
            # Marginalize P(Goal | Unblocked) = E[P(Acc)*P(Fin)]
            # We assume p_unblocked is constant w.r.t shot_type.
            
            df_nan = df[mask_nan].copy()
            n_nan = len(df_nan)
            weighted_cond_prob = np.zeros(n_nan)
            
            for st_cat, weight in self.shot_type_priors_.items():
                df_nan['shot_type'] = st_cat
                df_nan['shot_type'] = pd.Categorical(df_nan['shot_type'], categories=VOCAB_SHOT_TYPE)
                
                p_acc_st = self.model_accuracy.predict_proba(df_nan[self.features])[:, 1]
                p_fin_st = self.model_finish.predict_proba(df_nan[self.features])[:, 1]
                
                weighted_cond_prob += (p_acc_st * p_fin_st) * weight
            
            # Reconstruct P(Goal) for NaNs
            p_goal[mask_nan] = p_unblocked[mask_nan] * weighted_cond_prob
        
        # 5. Apply Calibration
        if self.use_calibration and self.calibrator:
            # Isotonic .predict() takes 1D array and outputs probabilities directly
            p_goal = self.calibrator.predict(p_goal)
            
        return np.column_stack((1 - p_goal, p_goal))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        
    def _predict_marginalized(self, model, df, features):
        """
        Predict with marginalization over ALL categorical features with missing values.
        
        For each row with NaN in any categorical feature, computes weighted average
        over all possible values of that feature (or combination of features if multiple are missing).
        
        E[P(y|x, missing)] = Σ P(y|x, cat=k) × P(cat=k) for each missing categorical
        """
        from itertools import product
        
        # 1. Standard Prediction (baseline)
        p_base = model.predict_proba(df[features])[:, 1]
        
        if not self.categorical_priors_:
            return p_base
        
        # 2. Find which categorical features have NaNs in this data
        cat_cols_with_nan = []
        for col, priors in self.categorical_priors_.items():
            if col in df.columns and col in features:
                if df[col].isna().any():
                    cat_cols_with_nan.append(col)
        
        if not cat_cols_with_nan:
            return p_base
        
        # 3. For each row, determine which categoricals are NaN
        # We'll process rows that have ANY categorical NaN
        nan_masks = {}
        combined_nan_mask = pd.Series(False, index=df.index)
        for col in cat_cols_with_nan:
            nan_masks[col] = df[col].isna()
            combined_nan_mask |= nan_masks[col]
        
        if not combined_nan_mask.any():
            return p_base
        
        # 4. Marginalize for rows with NaN categoricals
        # For simplicity and performance, we marginalize one feature at a time
        # For rows with multiple NaN categoricals, we iterate through each
        
        df_work = df.copy()
        p_result = p_base.copy()
        
        for col in cat_cols_with_nan:
            mask_nan = df_work[col].isna()
            if not mask_nan.any():
                continue
            
            priors = self.categorical_priors_.get(col, {})
            if not priors:
                continue
            
            vocab = CATEGORICAL_VOCABS.get(col, list(priors.keys()))
            
            # Calculate weighted sum for rows with NaN in this column
            df_nan_rows = df_work[mask_nan].copy()
            weighted_sum = np.zeros(mask_nan.sum())
            
            for cat_val, weight in priors.items():
                # Set the categorical to this value
                df_nan_rows[col] = cat_val
                # Ensure proper categorical dtype
                df_nan_rows[col] = pd.Categorical(df_nan_rows[col], categories=vocab)
                
                # Predict
                p_cat = model.predict_proba(df_nan_rows[features])[:, 1]
                weighted_sum += p_cat * weight
            
            # Update result for these rows
            p_result[mask_nan] = weighted_sum
        
        return p_result

    def predict_proba_layer(self, X: pd.DataFrame, layer: str) -> np.ndarray:
        """Helper for diagnostics."""
        df = self._prepare_df(X)
        if layer == 'block':
            feat_block = [f for f in self.features if 'shot_type' not in f]
            p = self.model_block.predict_proba(df[feat_block])[:, 1]
            if getattr(self, 'calibrator_block', None):
                p = self.calibrator_block.predict_proba(p.reshape(-1, 1))[:, 1]
            return p
            
        elif layer == 'accuracy':
            return self._predict_marginalized(self.model_accuracy, df, self.features)
            
        elif layer == 'finish':
            # Use _predict_marginalized
             return self._predict_marginalized(self.model_finish, df, self.features)
             
        else:
            raise ValueError(f"Unknown layer: {layer}")
