"""mixed_effects.py

GAME-STATE AWARE MIXED EFFECTS MODEL (PARALLEL OFF/DEF)
======================================================
This module implements a mixed effects model that fits random slopes for BOTH
Offense (Team) and Defense (Opponent) simultaneously ("in parallel").

It also handles Game State splitting (5v5, 5v4, 4v5) by training separate
sub-models for each state.

Architecture:
-------------
1. Base Model: NestedGLM (Fixed Effect) -> Provides P_base / Base Margin.
2. Mixed Effect: XGBoost (gblinear)
   - Goal: Fit `margin = base_margin + Off_Effect + Def_Effect`
   - Off_Effect = Sum( Beta_Off_Team_i * Feature_j )
   - Def_Effect = Sum( Beta_Def_Team_i * Feature_j )
   
   To solve this simultaneously, we construct a sparse matrix where for each row:
   - The columns corresponding to Off_Team get the feature values.
   - The columns corresponding to Def_Team get the feature values.
   - All other columns are 0.
   - We fit one linear model on this large sparse matrix.

Usage:
------
    mixed = GameMixedEffectsXG(base_model_path="...")
    mixed.fit(df)
    mixed.predict_proba(df)
"""

import numpy as np
import pandas as pd
import joblib
import logging
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Dict, Optional, Any, Union
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import os
import scipy.sparse as sp

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from . import features as feature_util
from .spline_transformer import TensorSpline

# MonkeyPatch/Alias for backward compatibility with old pickles
class ParallelMixedEffectsModel:
    def predict_margin(self, X: pd.DataFrame, off_team_col: str, def_team_col: str) -> np.ndarray:
        """
        Predict margin for legacy Parallel model (Off + Def in one vector).
        Assumes coef_ structure: [Off_Team1 ... Off_TeamN, Def_Team1 ... Def_TeamN]
        """
        if not hasattr(self, 'coef_') or self.coef_ is None:
             return np.zeros(len(X))
             
        n_samples = len(X)
        # Use stored feature names
        feats = getattr(self, 'feature_names', [])
        if not feats:
             return np.zeros(n_samples)
             
        n_feats = len(feats)
        # Check explicit columns
        missing = [f for f in feats if f not in X.columns]
        if missing:
             # This might happen if 'intercept' is missing or other columns
             # But X should be prepared by GameMixedEffectsXG
             logger.warning(f"Legacy Model missing features: {missing[:5]}...")
             
        X_feats = X[feats].fillna(0).values.astype(np.float32)
        
        # Teams
        if not hasattr(self, 'team_idx_'):
             return np.zeros(n_samples)
             
        team_map = self.team_idx_
        n_teams = len(team_map)
        
        # We need to construct sparse matrix of shape (n_samples, 2 * n_teams * n_feats)
        # But actually we can just compute Off and Def separately and sum them.
        
        # Offense Part (Cols 0 .. N_teams*N_feats)
        off_indices = X[off_team_col].map(team_map)
        
        # Defense Part (Cols N_teams*N_feats .. End)
        def_indices = X[def_team_col].map(team_map)
        
        valid_mask = (~off_indices.isna()) & (~def_indices.isna())
        if not valid_mask.any():
             return np.zeros(n_samples)

        # We can do this efficiently by creating TWO sparse matrices (or one combined)
        # Let's do one combined to match the coef_ vector directly
        
        valid_rows = np.where(valid_mask)[0]
        off_idx_valid = off_indices.values[valid_rows].astype(int)
        def_idx_valid = def_indices.values[valid_rows].astype(int)
        X_valid = X_feats[valid_rows]
        n_valid = len(valid_rows)
        
        # Indices Construction
        row_indices = np.repeat(np.arange(n_valid), n_feats)
        feat_offsets = np.arange(n_feats)
        
        # Offense Cols: (Team_Idx * N_Feats) + Feat_Idx
        col_off = (off_idx_valid[:, None] * n_feats + feat_offsets).flatten()
        
        # Defense Cols: (Team_Idx * N_Feats) + Feat_Idx + (N_Teams * N_Feats)
        offset_def_block = n_teams * n_feats
        col_def = (def_idx_valid[:, None] * n_feats + feat_offsets).flatten() + offset_def_block
        
        # Combine
        # We repeat X values twice (once for off, once for def)
        data_rep = np.tile(X_valid.flatten(), 2) 
        # Wait, X_valid.flatten() groups by row then feature.
        # col_off corresponds to X_valid.flatten()
        # col_def corresponds to X_valid.flatten()
        # So we concatenate data, rows, cols
        
        data_all = np.concatenate([X_valid.flatten(), X_valid.flatten()])
        rows_all = np.concatenate([row_indices, row_indices])
        cols_all = np.concatenate([col_off, col_def])
        
        input_dim = getattr(self, 'input_dim_', 2 * n_teams * n_feats)
        
        X_sparse = sp.coo_matrix((data_all, (rows_all, cols_all)), 
                                 shape=(n_valid, input_dim)).tocsr()
                                 
        adj = X_sparse.dot(self.coef_)
        
        result = np.zeros(n_samples)
        result[valid_rows] = adj
        return result


class ComponentMixedEffectsModel(BaseEstimator):
    """
    Dummy class strictly for unpickling legacy models (e.g. from matchup.py).
    Do not use for training new models. Use StateMixedEffectsModel.
    """
    def __init__(self, *args, **kwargs):
        pass
    
    def predict_margin(self, X: pd.DataFrame, team_col: str) -> np.ndarray:
        if not hasattr(self, 'coef_') or self.coef_ is None:
             return np.zeros(len(X))
             
        # Extract features and multiply like legacy model did
        n_samples = len(X)
        team_indices_raw = X[team_col].map(getattr(self, 'team_idx_', {}))
        valid_mask = ~team_indices_raw.isna()
        
        if not valid_mask.any(): return np.zeros(n_samples)
        
        valid_rows = np.where(valid_mask)[0]
        team_idx_valid = team_indices_raw.values[valid_rows].astype(int)
        
        # Intercept Only for legacy Component intercepts
        adj = self.coef_[team_idx_valid]
        result = np.zeros(n_samples)
        result[valid_rows] = adj
        return result


class StateMixedEffectsModel(BaseEstimator):
    """
    Fits a joint mixed-effects model for a single game state (e.g., 5v5).
    Fits Random Intercepts for BOTH Offense and Defense simultaneously 
    to properly apportion credit/blame from the base marginal residual.
    """
    def __init__(self, 
                 l2_reg: float = 1.0):
        self.l2_reg = l2_reg
        self.coef_ = None # Store coefficients from Scipy solver (size: 2 * N_teams)
        self.teams_ = None
        self.team_idx_ = None # Map[team_name -> int]
        self.n_teams_ = 0
        
    def fit(self, df: pd.DataFrame, y: pd.Series, base_margin: np.ndarray, 
            off_col: str, def_col: str):
        """
        Fit the joint mixed effects model.
        
        Args:
            df: Feature dataframe
            y: Target (0/1)
            base_margin: Log-odds from base model
            off_col: Column name for shooting team
            def_col: Column name for defending team
        """
        # 1. Identify all unique teams across both columns to ensure alignment
        off_teams = set(df[off_col].dropna().unique())
        def_teams = set(df[def_col].dropna().unique())
        all_teams = off_teams.union(def_teams)
        
        # Filter valid strings
        teams = [t for t in all_teams if isinstance(t, str) and len(t) > 0]
        teams.sort()
        
        self.teams_ = teams
        self.team_idx_ = {t: i for i, t in enumerate(self.teams_)}
        self.n_teams_ = len(self.teams_)
        
        logger.info(f"State Model (Joint Intercepts): {len(df)} samples, {self.n_teams_} teams.")
        
        # 2. Map teams to indices
        off_indices = df[off_col].map(self.team_idx_)
        def_indices = df[def_col].map(self.team_idx_)
        
        # Filter invalid rows where team is missing
        valid_mask = (~off_indices.isna()) & (~def_indices.isna())
        if not valid_mask.all():
            logger.warning(f"Dropping {np.sum(~valid_mask)} rows with missing team info.")
            off_indices = off_indices[valid_mask]
            def_indices = def_indices[valid_mask]
            y = y[valid_mask]
            base_margin = base_margin[valid_mask]
            
        n_samples = len(valid_mask[valid_mask])
        off_idx_vals = off_indices.values.astype(int)
        def_idx_vals = def_indices.values.astype(int)
        
        # 3. Construct Unified Sparse Design Matrix
        # Size: (N_samples, 2 * N_teams)
        # Each row has exactly TWO 1.0s:
        # Col A = off_team_idx (Offense Intercept)
        # Col B = def_team_idx + n_teams (Defense Intercept)
        
        # Row indices (each row gets two entries)
        row_indices = np.repeat(np.arange(n_samples), 2)
        
        # Column indices
        # Interleave off and def indices
        col_indices = np.empty(2 * n_samples, dtype=int)
        col_indices[0::2] = off_idx_vals # Offense features (0 to N-1)
        col_indices[1::2] = def_idx_vals + self.n_teams_ # Defense features (N to 2N-1)
        
        # Data values (all 1.0 for intercepts)
        data = np.ones(2 * n_samples, dtype=np.float32)
        
        input_dim = 2 * self.n_teams_
        X_sparse = sp.coo_matrix((data, (row_indices, col_indices)), 
                                 shape=(n_samples, input_dim)).tocsr()
                                 
        # 4. Fit using logistic solver
        y_float = y.astype(np.float64)
        if hasattr(base_margin, 'values'):
             base_margin = base_margin.values
        base_margin = base_margin.reshape(-1).astype(np.float64)
        
        from puck.logistic_solver import fit_logistic_offset
        
        logger.info(f"Solving Joint State Model (L-BFGS-B)...")
        self.coef_ = fit_logistic_offset(
            X_sparse, y_float, base_margin, 
            l2_reg=self.l2_reg, verbose=False
        )
        
        logger.info(f"Joint State Model Fit Complete.")
        return self

    def predict_margin(self, df: pd.DataFrame, off_col: str, def_col: str) -> np.ndarray:
        """
        Predict the joint margin adjustment (Offense + Defense).
        Returns: Adjustment vector (N_samples,)
        """
        if self.coef_ is None:
            return np.zeros(len(df))
            
        n_samples = len(df)
        
        # Map indices
        off_idx_raw = df[off_col].map(self.team_idx_)
        def_idx_raw = df[def_col].map(self.team_idx_)
        
        # If teams are unknown (e.g. out of sample), they get 0 adjustment
        valid_mask = (~off_idx_raw.isna()) & (~def_idx_raw.isna())
        valid_rows = np.where(valid_mask)[0]
        
        off_vals = off_idx_raw.values[valid_rows].astype(int)
        def_vals = def_idx_raw.values[valid_rows].astype(int)
        
        # Look up coefficients directly
        # coef_[0:N] are Offense, coef_[N:2N] are Defense
        off_adj = self.coef_[off_vals]
        def_adj = self.coef_[def_vals + self.n_teams_]
        
        result = np.zeros(n_samples)
        result[valid_rows] = off_adj + def_adj
        
        return result

    def get_coefficients(self) -> pd.DataFrame:
        """
        Extract the learned intercepts into a readable DataFrame.
        """
        if self.coef_ is None:
            return pd.DataFrame()
            
        records = []
        for t_i, team in enumerate(self.teams_):
            # Offense Profile
            records.append({
                'team': team,
                'role': 'Offense',
                'feature': 'intercept',
                'coef': self.coef_[t_i]
            })
            # Defense Profile
            records.append({
                'team': team,
                'role': 'Defense',
                'feature': 'intercept',
                'coef': self.coef_[t_i + self.n_teams_]
            })
            
        return pd.DataFrame(records)


class GameMixedEffectsXG(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 base_model_path: str = None, 
                 base_model = None,
                 feature_set: Union[List[str], str] = None,
                 use_tensor_splines: bool = False,
                 updater: str = 'shotgun',
                 component_model_type: str = 'intercept', # 'intercept' or 'slopes'
                 l2_reg: float = 1.0 # Added l2_reg just in case
                 ):
        
        self.base_model_path = base_model_path
        self.base_model_ = base_model
        # Resolve feature set string if provided
        if isinstance(feature_set, str):
            self.feature_set = feature_util.get_features(feature_set)
        else:
            self.feature_set = feature_set
            
        self.use_tensor_splines = use_tensor_splines
        self.updater = updater 
        self.component_model_type = component_model_type
        self.l2_reg = l2_reg
        
        self.tensor_transformer_ = None
        self.ohe_transformer_ = None
        self.final_feature_names_ = None
        
        # Sub-models per game state (Joint Off/Def)
        self.state_models_: Dict[str, StateMixedEffectsModel] = {}
        
    def _map_state(self, row):
        # We can just use the 'game_state' column directly if it's clean (5v5, 5v4, 4v5, 4v4, etc.)
        # daily.py ensures '5v5', '5v4', '4v5' are primary.
        return row['game_state']

    def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Transform raw dataframe into numeric feature matrix for mixed effects.
        
        If component_model_type is 'intercept', we only really need metadata,
        BUT we still might want features if we switch back.
        
        For simplicity, if intercept only, we can return dummy features or 
        keep consistent tensor transformation but ignore it in ComponentModel.
        
        Decision: Keep _prepare_features mechanics intact so we can switch types easily.
        """
        # 1. Select relevant columns (plus potential OHE targets)
        # We start with the configured feature set
        if self.feature_set is None:
            # Fallback to candidates if not set
            candidates = [
                'distance', 'angle_deg', 
                'time_since_last_event', 'speed', 
                'is_rebound', 'is_rush',
                'shot_type', 'shooter_role', 'shoots_catches'
            ]
            self.feature_set = [c for c in candidates if c in df.columns]
            
        # Ensure we have a working list
        features = list(self.feature_set)
        
        # 2. Tensor Splines
        X_parts = []
        
        # Handle Spatial (Tensor or Raw)
        spatial_cols = ['distance', 'angle_deg']
        if self.use_tensor_splines and all(c in features for c in spatial_cols):
            # Extract spatial cols for tensor
            if fit:
                if self.tensor_transformer_ is None:
                    self.tensor_transformer_ = TensorSpline(n_knots=7, degree=3, include_bias=False)
                self.tensor_transformer_.fit(df[spatial_cols])
                
            X_tensor = self.tensor_transformer_.transform(df[spatial_cols])
            tensor_names = self.tensor_transformer_.get_feature_names_out(spatial_cols)
            X_parts.append(pd.DataFrame(X_tensor, columns=tensor_names, index=df.index))
            
            # Remove raw spatial from list so we don't double count
            features = [f for f in features if f not in spatial_cols]
        
        # 3. Categoricals / OHE
        # Identify Categoricals
        # If we are fitting, we detect them. If transforming, we use stored OHE.
        if fit:
            cat_cols = [c for c in features if df[c].dtype == 'object' or isinstance(df[c].dtype, pd.CategoricalDtype)]
            if cat_cols:
                self.ohe_transformer_ = OneHotEncoder(sparse_output=False, handle_unknown='ignore', dtype=np.float32)
                self.ohe_transformer_.fit(df[cat_cols])
        
        # Apply OHE if it exists
        if self.ohe_transformer_ is not None:
            # We must have the cat columns available
            # Get feature names from OHE
            # Check if we have cat cols in current features list? 
            # We rely on self.feature_set being consistent.
            cat_cols = self.ohe_transformer_.feature_names_in_
            
            # Validate existence
            valid_cat = [c for c in cat_cols if c in df.columns]
            if len(valid_cat) == len(cat_cols):
                X_cat = self.ohe_transformer_.transform(df[cat_cols])
                cat_names = self.ohe_transformer_.get_feature_names_out(cat_cols)
                X_parts.append(pd.DataFrame(X_cat, columns=cat_names, index=df.index))
                
                # Remove raw cats from features
                features = [f for f in features if f not in cat_cols]
            else:
                logger.warning(f"Missing categorical columns for OHE: {set(cat_cols) - set(valid_cat)}")
        
        # 4. Remaining Numeric Features
        if features:
            # Fill NaNs with 0 for robustness
            X_num = df[features].fillna(0)
            
            # Apply Scaling
            if fit:
                 from sklearn.preprocessing import StandardScaler
                 self.scaler_ = StandardScaler()
                 X_num_scaled = self.scaler_.fit_transform(X_num)
                 # Keep as DataFrame
                 X_num = pd.DataFrame(X_num_scaled, columns=features, index=df.index)
            elif hasattr(self, 'scaler_') and self.scaler_ is not None:
                 X_num_scaled = self.scaler_.transform(X_num)
                 X_num = pd.DataFrame(X_num_scaled, columns=features, index=df.index)
            
            X_parts.append(X_num)
            
        # 5. Intercept
        if 'intercept' not in df.columns:
            # Create a series with index matching df
            X_parts.append(pd.Series(1.0, index=df.index, name='intercept'))
        else:
            X_parts.append(df['intercept'])
            
        # Concatenate
        X_final = pd.concat(X_parts, axis=1)
        
        if fit:
            self.final_feature_names_ = X_final.columns.tolist()
            logger.info(f"Feature Prep Complete. Input: {len(self.feature_set)} -> Output: {len(self.final_feature_names_)}")
            
        return X_final

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        
        # 1. Load Base Model
        if self.base_model_ is None:
            if self.base_model_path is None:
                # Default path
                p = Path("analysis/xgs/xg_model_nested_tensor.joblib")
                if p.exists():
                     self.base_model_path = str(p)
                else:
                    raise FileNotFoundError("Base model not found/specified.")
            
            logger.info(f"Loading Base Model: {self.base_model_path}")
            self.base_model_ = joblib.load(self.base_model_path)
            
        # 2. Predict Base Margins
        logger.info("Predicting Base Margins...")
        
        # Ensure required columns for base model exist
        required_cols = ['shoots_catches', 'shooter_role', 'is_rebound', 'is_rush']
        for c in required_cols:
            if c not in df.columns:
                logger.warning(f"Column '{c}' missing for base model. Filling with default.")
                if c == 'shoots_catches':
                    df[c] = 'L' 
                elif c == 'shooter_role':
                    df[c] = 'Center' 
                else:
                    df[c] = 0

        # 1.5 Ensure Off/Def Names
        if 'off_team_name' not in df.columns:
            if 'team_abbrev' in df.columns:
                 df['off_team_name'] = df['team_abbrev']
            elif 'team_id' in df.columns:
                 df['off_team_name'] = df['team_id'].astype(str)
                 
        if 'def_team_name' not in df.columns:
            if 'home_abb' in df.columns and 'away_abb' in df.columns:
                off_vec = df['off_team_name'].astype(str)
                home_vec = df['home_abb'].astype(str)
                away_vec = df['away_abb'].astype(str)
                is_home = (off_vec == home_vec)
                df['def_team_name'] = np.where(is_home, away_vec, home_vec)
            else:
                df['def_team_name'] = 'Unknown'

        base_probs = self.base_model_.predict_proba(df)[:, 1]
        
        # DEBUG LOG in fit
        logger.info(f"Base Model Stats (Fit): Mean={base_probs.mean():.4f}, Min={base_probs.min():.4f}, Max={base_probs.max():.4f}")
        
        eps = 1e-6
        base_probs = np.clip(base_probs, eps, 1-eps)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        # 3. Prepare Features (Tensor + OHE + Numeric)
        # This transforms the WHOLE dataframe.
        # Note: We do this ONCE for the whole dataset to learn OHE/Knots.
        # But we train per state.
        
        logger.info("Preparing Features (OHE + Splines)...")
        # Ensure 'intercept' is present before calling _prepare if needed, 
        # but _prepare handles it.
        
        X_transformed = self._prepare_features(df, fit=True)
        # Add metadata cols back for splitting
        X_transformed['game_state'] = df['game_state']
        X_transformed['off_team_name'] = df['off_team_name']
        X_transformed['def_team_name'] = df['def_team_name']
        
        # 4. Train per Game State
        states = df['game_state'].value_counts()
        # Limit to core game states as requested
        target_states = ['5v5', '5v4', '4v5']
        valid_states = [s for s in target_states if s in states.index and states[s] > 100]
        logger.info(f"Training models for states: {valid_states}")
        
        for state in valid_states:
            logger.info(f"--- Fitting State: {state} ---")
            mask = X_transformed['game_state'] == state
            X_sub = X_transformed[mask]
            # Must handle empty mask
            if len(X_sub) == 0:
                continue
            
            # Target
            y_sub = y[mask] if y is not None else (df.loc[mask, 'event'] == 'goal').astype(int)
            margin_sub = base_margins[mask]
            
            # Joint Model
            logger.info(f"Fitting Joint Offense/Defense ({state})...")
            state_model = StateMixedEffectsModel(
                l2_reg=self.l2_reg
            )
            state_model.fit(X_sub, y_sub, margin_sub, off_col='off_team_name', def_col='def_team_name')
            self.state_models_[state] = state_model
            
        return self

    def predict_proba(self, X: pd.DataFrame):
        df = X.copy()
        
        # 1. Base
        base_probs = self.base_model_.predict_proba(df)[:, 1]
        
        # DEBUG LOG
        logger.info(f"Base Model Stats: Mean={base_probs.mean():.4f}, Min={base_probs.min():.4f}, Max={base_probs.max():.4f}")
        
        eps = 1e-6
        base_probs = np.clip(base_probs, eps, 1-eps)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        final_margins = base_margins.copy()
        
        # 2. Add Adjustments per State
        if 'intercept' not in df.columns:
            df['intercept'] = 1.0
            
        # 1.5 Ensure Off/Def Names (Same as fit)
        if 'off_team_name' not in df.columns:
            if 'team_abbrev' in df.columns:
                 df['off_team_name'] = df['team_abbrev']
            elif 'team_id' in df.columns:
                 df['off_team_name'] = df['team_id'].astype(str)
                 
        if 'def_team_name' not in df.columns:
            if 'home_abb' in df.columns and 'away_abb' in df.columns:
                off_vec = df['off_team_name'].astype(str)
                home_vec = df['home_abb'].astype(str)
                away_vec = df['away_abb'].astype(str)
                is_home = (off_vec == home_vec)
                df['def_team_name'] = np.where(is_home, away_vec, home_vec)
            else:
                df['def_team_name'] = 'Unknown'

        # 3. Transform Features
        X_transformed = self._prepare_features(df, fit=False)
        X_transformed['game_state'] = df['game_state']
        X_transformed['off_team_name'] = df['off_team_name']
        X_transformed['def_team_name'] = df['def_team_name']

        # Determine states from available models
        # Support Both New (state_models_) and Legacy (models_)
        
        processed_states = set()
        
        state_models = getattr(self, 'state_models_', {}) or {}
        
        # New Style
        new_states = set(list(state_models.keys()))
        for state in new_states:
            mask = df['game_state'] == state
            if not mask.any(): continue
            processed_states.add(state)
            
            df_sub = df[mask]
            
            # Off/Def cols
            off_col = 'off_team_name'
            def_col = 'def_team_name'
            
            # Joint Prediction
            if state in state_models:
                adj = state_models[state].predict_margin(X_transformed[mask], off_col=off_col, def_col=def_col)
                final_margins[mask] += adj

        # Legacy Support (if hasattr models_)
        # Check if 'models_' exists in self (handle missing attribute)
        legacy_models = getattr(self, 'models_', {}) or {}
        
        if legacy_models:
            legacy_states = set(legacy_models.keys())
            # Only process states NOT already processed by new style? 
            # Or if new style is empty?
            # Safe bet: if specific state wasn't in new, try legacy
            
            for state in legacy_states:
                if state in processed_states: continue
                
                mask = df['game_state'] == state
                if not mask.any(): continue
                
                model = legacy_models[state]
                # Check if model has predict_margin (our alias class has it)
                if hasattr(model, 'predict_margin'):
                     X_sub_trans = X_transformed[mask]
                     adj = model.predict_margin(X_sub_trans, off_team_col='off_team_name', def_team_col='def_team_name')
                     final_margins[mask] += adj
            
        # 3. Sigmoid
        final_probs = 1.0 / (1.0 + np.exp(-final_margins))
        return np.column_stack((1 - final_probs, final_probs))
        
    def get_all_coefficients(self) -> pd.DataFrame:
        """
        Aggregate coefficients from all sub-models into a single DataFrame.
        """
        dfs = []
        # Joint Models
        for state, model in self.state_models_.items():
            df_curr = model.get_coefficients()
            if not df_curr.empty:
                df_curr['game_state'] = state
                dfs.append(df_curr)
             
        # Legacy
        if hasattr(self, 'models_') and self.models_:
             # If ParallelMixedEffectsModel has get_coefficients, verify it
             pass

        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)
    def save_summary(self, output_dir: str, teams_filter: List[str] = None):
        """
        Save model coefficients and generate summary plots.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Get Data
        df_coefs = self.get_all_coefficients()
        if df_coefs.empty:
            logger.warning("No coefficients to save.")
            return
            
        # 2. Save CSV
        csv_path = output_dir / "mixed_effects_coefficients.csv"
        df_coefs.to_csv(csv_path, index=False)
        logger.info(f"Saved coefficients to {csv_path}")
        
        # 3. Generate Plots
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Setup style
            sns.set_theme(style="whitegrid")
            
            # A. League Wide Scatter (Offense vs Defense Intercepts)
            # We want to compare Team Strength across states (e.g. 5v5)
            # Filter to intercept
            mask_intercept = df_coefs['feature'] == 'intercept'
            
            unique_states = df_coefs['game_state'].unique()
            
            for state in unique_states:
                df_plot = df_coefs[mask_intercept & (df_coefs['game_state'] == state)]
                if df_plot.empty:
                    continue
                    
                # Pivot to get Offense and Defense on same row per team
                # df columns: team, role, feature, coef, game_state
                df_pivot = df_plot.pivot(index='team', columns='role', values='coef')
                # Pivot columns will be 'Defense', 'Offense'
                
                if 'Offense' in df_pivot.columns and 'Defense' in df_pivot.columns:
                    plt.figure(figsize=(10, 8))
                    
                    # Scatter
                    sns.scatterplot(data=df_pivot, x='Offense', y='Defense')
                    
                    # Add team labels
                    for team, row in df_pivot.iterrows():
                        plt.text(row['Offense']+0.001, row['Defense']+0.001, team, fontsize=9)
                        
                    plt.title(f"Team Strength: {state} (Intercepts)\nPositive Offense = Good | Negative Defense = Good")
                    plt.axhline(0, color='gray', linestyle='--')
                    plt.axvline(0, color='gray', linestyle='--')
                    
                    # Invert Y axis? 
                    # Negative Defense coef means "Lowers xG against" -> Good Defense.
                    # So lower is better. Top left quadrant (High Off, Low Def) is best.
                    plt.gca().invert_yaxis()
                    plt.ylabel("Defensive Impact (Lower is Better)")
                    plt.xlabel("Offensive Impact (Higher is Better)")
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / f"scatter_intercepts_{state}.png")
                    plt.close()
                    
            # B. Top 10 Bars per State/Role
            for state in unique_states:
                for role in ['Offense', 'Defense']:
                    df_sub = df_coefs[(df_coefs['game_state'] == state) & 
                                      (df_coefs['role'] == role) & 
                                      (mask_intercept)]
                                      
                    if df_sub.empty:
                        continue
                        
                    # Sort
                    # For Offense: Descending (High is good)
                    # For Defense: Ascending (Low/Negative is good)
                    ascending = True if role == 'Defense' else False
                    df_sorted = df_sub.sort_values('coef', ascending=ascending).head(10)
                    
                    plt.figure(figsize=(10, 6))
                    sns.barplot(data=df_sorted, x='coef', y='team', palette='viridis')
                    plt.title(f"Top 10 {role} ({state})")
                    plt.xlabel("Coefficient Impact")
                    plt.tight_layout()
                    plt.savefig(output_dir / f"top10_{role}_{state}.png")
                    plt.close()

            # C. Team Specific Plots
            teams_dir = output_dir / "teams"
            teams_dir.mkdir(exist_ok=True)
            
            unique_teams = df_coefs['team'].unique()
            
            if teams_filter:
                # Normalize filter to match team names (usually abbrevs)
                unique_teams = [t for t in unique_teams if t in teams_filter]
                logger.info(f"Filtering to {len(unique_teams)} teams: {unique_teams}")
            
            # Plot only if we have reasonable number of teams or user request?
            # User said "plots for individual teams". We'll generate for all.
            
            logger.info(f"Generating individual plots for {len(unique_teams)} teams...")
            
            for team in unique_teams:
                # Filter to team
                df_team = df_coefs[df_coefs['team'] == team]
                
                # We want to show coefficients for this team across states/features
                # Maybe a FacetGrid or just a simple bar chart of intercepts?
                # Let's show Intercepts across states first
                
                # 1. Intercepts (Overall Strength)
                df_int = df_team[df_team['feature'] == 'intercept']
                if not df_int.empty:
                    plt.figure(figsize=(8, 5))
                    sns.barplot(data=df_int, x='game_state', y='coef', hue='role')
                    plt.title(f"{team} - Overall Adjustments (Intercepts)")
                    plt.axhline(0, color='k', linewidth=0.5)
                    plt.ylabel("Impact on Log-Odds")
                    plt.tight_layout()
                    plt.savefig(teams_dir / f"{team}_intercepts.png")
                    plt.close()
                    
                # 2. Random Slopes (Continuous Features) - Optional/Advanced
                # If we have interesting features like 'distance', 'speed'
                # Let's verify if we have non-intercept features
                non_int = df_team[df_team['feature'] != 'intercept']
                if not non_int.empty:
                    # Plot heatmap of coefficients? Or bar chart?
                    # Facet by game_state
                    g = sns.catplot(
                        data=non_int, kind="bar",
                        x="coef", y="feature", hue="role", col="game_state",
                        col_wrap=2, height=4, aspect=1.5, sharex=False
                    )
                    g.fig.suptitle(f"{team} - Detailed Feature Adjustments", y=1.02)
                    g.set_axis_labels("Coefficient", "Feature")
                    plt.tight_layout()
                    plt.savefig(teams_dir / f"{team}_details.png")
                    plt.close()

        except ImportError:
            logger.warning("Could not import matplotlib/seaborn. Skipping plots.")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            import traceback
            traceback.print_exc()
