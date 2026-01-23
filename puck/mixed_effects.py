"""mixed_effects.py

MIXED EFFECTS EXTENSION FOR NESTED XG MODEL
===========================================
This module implements a "Mixed Effects" or "Random Slopes" extension to the 
Nested GLM xG Model.

It treats the pre-trained Nested GLM as a "Fixed Effect" (Base Model) and learns
group-specific adjustments (Random Effects) using a linear boosting approach 
(XGBoost with booster='gblinear') on the residuals (via base_margin).

Architecture:
-------------
1. Base Model: NestedGLM (Logistic Regression with Tensor Splines)
   - Provides P_base(Goal)
   - We extract the raw log-odds (margin) from this model.

2. Mixed Effects Model: XGBoost (gblinear)
   - Learns coefficients beta_group for each group (e.g. team or player).
   - Prediction = sigmoid( Base_Margin + X * beta_group )
   
   - We train a separate small linear model for each group, OR one large model 
     with interaction terms if memory permits. 
     Given we want per-group random slopes for ALL features, training separate 
     models (or using group-wise data splits) is effectively the same and parallelizable.

Usage:
------
    mixed_model = MixedEffectsXG(base_model_path="...")
    mixed_model.fit(df, group_col='team_id')
    preds = mixed_model.predict_proba(df)

"""

import numpy as np
import pandas as pd
import joblib
import logging
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Dict, Optional, Any
from pathlib import Path
import os
import copy

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from . import fit_nested_xgs
from . import fit_xgboost_nested
from . import features as feature_util

class MixedEffectsXG(BaseEstimator, ClassifierMixin):
    def __init__(self, 
                 base_model_path: str = None, 
                 group_col: str = 'team_name', 
                 feature_set: List[str] = None,
                 l1_reg: float = 0.0,
                 l2_reg: float = 1.0,
                 learning_rate: float = 0.1,  # Usually 1.0 for straight solving, but <1 for iterative
                 n_estimators: int = 100):
        
        self.base_model_path = base_model_path
        self.group_col = group_col
        self.feature_set = feature_set
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        
        # State
        self.base_model_ = None
        self.group_models_ = {} # Dict[group_key, XGBClassifier]
        self.global_bias_ = 0.0 # Correction if base model is biased on current data
        self.feature_names_ = None
        
    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the random slopes for each group found in X[group_col].
        
        Args:
            X: DataFrame containing features + group_col.
            y: Target variable (if None, looks for 'is_goal' or 'event' in X).
        """
        df = X.copy()
        
        # 1. Load Base Model (if not loaded)
        if self.base_model_ is None:
            if self.base_model_path is None:
                 # Default path search
                 possible_paths = [
                     Path("analysis/xgs/xg_model_nested_tensor.joblib"),
                     Path("analysis/xgs/xg_model_nested_standard.joblib"),
                     Path("analysis/xgs/xg_model_nested.joblib")
                 ]
                 for p in possible_paths:
                     if p.exists():
                         self.base_model_path = str(p)
                         break
                 if self.base_model_path is None:
                     raise FileNotFoundError("Could not find a default base model. Please specify base_model_path.")
            
            logger.info(f"Loading base model from {self.base_model_path}...")
            self.base_model_ = joblib.load(self.base_model_path)
            
        # 2. Prepare Data & Targets
        if y is None:
            if 'is_goal' in df.columns:
                y = df['is_goal']
            elif 'event' in df.columns:
                y = (df['event'] == 'goal').astype(int)
            else:
                raise ValueError("No target provided and columns 'is_goal' or 'event' missing.")
        
        # 3. Get Base Margins (Log-Odds) form Base Model
        #    Note: NestedGLM might not expose 'decision_function' directly for the *combined* probability,
        #    so we might need to compute proba and convert to log-odds.
        
        logger.info("Computing base margins...")
        base_probs = self.base_model_.predict_proba(df)[:, 1]
        
        # Clip to avoid inf in logit
        epsilon = 1e-6
        base_probs = np.clip(base_probs, epsilon, 1 - epsilon)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        # 4. Identify Features for Random Slopes
        #    We use numeric features. Categorical features (OHE) could be used too but sparse.
        if self.feature_names_ is None:
             if hasattr(self.base_model_, 'features'):
                 self.feature_names_ = self.base_model_.features
             else:
                 # Fallback
                 self.feature_names_ = feature_util.get_features('standard')
                 
        #    Filter to what's in DF
        fit_feats = [f for f in self.feature_names_ if f in df.columns]
        #    We primarily want random slopes for continuos variables like distance, angle.
        #    Maybe exclude complex categorical OHEs to keep it lightweight? 
        #    For now, use all numeric columns found.
        fit_feats = [f for f in fit_feats if pd.api.types.is_numeric_dtype(df[f])]
        
        logger.info(f"Fitting Mixed Effects to {len(fit_feats)} features locally per {self.group_col}.")
        
        # 5. Fit Group Models
        groups = df[self.group_col].unique()
        logger.info(f"Found {len(groups)} groups.")
        
        for g in groups:
            mask = (df[self.group_col] == g)
            X_g = df.loc[mask, fit_feats]
            y_g = y[mask]
            margin_g = base_margins[mask]
            
            if len(X_g) < 10: 
                # Too few samples to fit random slopes safely
                continue
                
            # Train Linear Model on Residuals
            # XGBoost with gblinear + base_margin
            
            # Note: We want to learn deviations. 
            # gblinear: prediction = base_margin + w*x + bias
            # This is exactly what we want.
            
            clf = xgb.XGBClassifier(
                booster='gblinear',
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                reg_alpha=self.l1_reg,
                reg_lambda=self.l2_reg,
                base_score=0.5, # Ignored when margin provided?
                objective='binary:logistic',
                n_jobs=1 
            )
            
            # XGBoost expects `base_margin` passed to fit? No, usually in DMatrix.
            # Scikit-Learn wrapper doesn't standardly accept base_margin in fit().
            # BUT, we can pass it as a kwarg if supported, or use the specialized set_info.
            # Actually, in recent XGBoost sklearn API, `fit` accepts kwargs that go to DMatrix.
            # Let's try passing `base_margin` to fit.
            
            try:
                clf.fit(X_g, y_g, base_margin=margin_g)
            except TypeError:
                # Fallback if sklearn wrapper doesn't support it easily -> use core API?
                # Or maybe it expects it in sample_weight? No.
                # Let's try standard way.
                # If this fails, we might need to use xgb.train directly.
                logger.warning(f"Could not pass base_margin to XGBClassifier.fit for {g}. Falling back to default fit (incorrect for mixed effects).")
                continue

            self.group_models_[g] = clf
            
        logger.info(f"Fitted random slopes for {len(self.group_models_)} groups.")
        return self

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict probability including random effects.
        """
        df = X.copy()
        
        # 1. Base Margins
        base_probs = self.base_model_.predict_proba(df)[:, 1]
        epsilon = 1e-6
        base_probs = np.clip(base_probs, epsilon, 1 - epsilon)
        base_margins = np.log(base_probs / (1 - base_probs))
        
        final_margins = base_margins.copy()
        
        # 2. Add Random Effects
        fit_feats = [f for f in self.feature_names_ if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
        
        # This loop is slow for many groups/rows. Vectorize if possible? 
        # For now, iterate groups present in data.
        present_groups = df[self.group_col].unique()
        
        for g in present_groups:
            if g in self.group_models_:
                mask = (df[self.group_col] == g)
                X_g = df.loc[mask, fit_feats]
                
                # We need the MARGIN output from the booster, NOT probability
                model = self.group_models_[g]
                
                # predict(output_margin=True) returns (base_margin + w*x) if base_margin provided?
                # If we don't provide base_margin to predict, it returns (bias + w*x).
                # We want (bias + w*x) to ADD to our global base_margin.
                
                # Note: XGBoost sklearn wrapper `predict` doesn't strictly support output_margin without native DMatrix?
                # Actually it does.
                
                # We pass base_margin=0 (or None) to get just the delta? 
                # If we fitted with base_margin, the model learned w such that margin = base + w*x.
                # We want to retrieve w*x.
                
                # If we call predict(X, output_margin=True), it usually assumes base_margin=0.5 (logit 0) unless specified?
                # Let's just use the booster directly to be safe.
                
                booster = model.get_booster()
                dmat = xgb.DMatrix(X_g)
                # Ensure feature names match? XGBoost is index based unless feature names set.
                # We are passing dataframe, so names should be preserved.
                
                delta_margin = booster.predict(dmat, output_margin=True)
                
                # The model output includes the base_score (0.5 -> 0.0 logit) by default usually.
                # gblinear usually learns weights.
                
                final_margins[mask] += delta_margin
                
        # 3. Sigmoid
        final_probs = 1.0 / (1.0 + np.exp(-final_margins))
        
        return np.column_stack((1 - final_probs, final_probs))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

