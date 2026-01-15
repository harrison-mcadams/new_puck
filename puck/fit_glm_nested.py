
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, SplineTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError
import logging

from . import features as feature_util
from . import config as puck_config

logger = logging.getLogger(__name__)

VOCAB_SHOT_TYPE = ['wrist', 'slap', 'snap', 'backhand', 'tip-in', 'wrap-around', 'deflected']

class NestedGLM(BaseEstimator, ClassifierMixin):
    """
    Nested Expected Goals Model using Polynomial Logistic Regression (GLM).
    
    Structure:
    1. Block Model: P(Unblocked | Shot)
    2. Accuracy Model: P(On Net | Unblocked)
    3. Finish Model: P(Goal | On Net)
    
    P(Goal) = P(Unblocked) * P(On Net) * P(Goal | On Net)
    
    Features:
    - Uses PolynomialFeatures(degree=3) for numeric columns to capture non-linear geometry (curves).
    - Uses OneHotEncoder for categorical columns.
    - Handles missing Shot Types via Marginalization (Weighted Average).
    """
    
    def __init__(self, features=None, poly_degree=3, use_splines=False, enable_marginalization=True):
        self.features = features or feature_util.get_features('all_inclusive')
        self.poly_degree = poly_degree
        self.use_splines = use_splines
        
        # Interaction Feature Name
        self.interact_col = 'dist_angle' if use_splines else None

        self.enable_marginalization = enable_marginalization
        
        # Sub-models
        self.model_block = None
        self.model_acc = None
        self.model_finish = None
        
        # Priors for marginalization
        self.shot_type_priors_ = {}
        
    def fit(self, X, y=None):
        logger.info(f"Fitting NestedGLM on {len(X)} rows. Poly Degree={self.poly_degree}, Splines={self.use_splines}")

        # --- Dynamic Interaction Term for Splines ---
        df = X.copy()
        if self.use_splines:
            df = self._enrich_interaction(df)
            # Add interaction column to features if not already present
            if self.interact_col and self.interact_col not in self.features:
                logger.info(f"Adding explicit interaction term '{self.interact_col}' to features.")
                self.features.append(self.interact_col)

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

        # 1. Block Model (Trained on ALL shots)
        # CRITICAL FIX (2025-01-13): Exclude 'shot_type' from Block Model.
        # Blocked shots always have shot_type='Unknown', causing massive leakage if included.
        block_features = [f for f in self.features if f != 'shot_type']
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

    def _build_pipeline(self, features=None):
        """Builds a standardized Sklearn pipeline for a single layer."""
        features = features or self.features
        
        # Split features
        cat_features = ['shot_type', 'shooter_role', 'shoots_catches', 'last_event_type', 'game_state']
        # Filter to only those present in features
        cat_features = [f for f in cat_features if f in features]
        num_features = [f for f in features if f not in cat_features]
        
        # Numeric Pipeline: Impute -> Poly/Spline -> Scale
        steps = [('imputer', SimpleImputer(strategy='median'))]
        
        if self.use_splines:
            # Splines (Flexible, Piecewise)
            # CRITICAL: SplineTransformer does NOT generate interactions between features.
            # We explicitly want Distance * Angle interaction.
            # We rely on the caller (fit/predict) to have added explicitly constructed interaction columns
            # to X before calling this pipeline if needed. 
            # If we are in Spline mode, we treat all numeric features (including custom interactions) with splines.
            steps.append(('spline', SplineTransformer(n_knots=7, degree=3, include_bias=False)))
        else:
            # Polynomials (Global curve) - Automatically generates interactions
            steps.append(('poly', PolynomialFeatures(degree=self.poly_degree, include_bias=False)))
            
        steps.append(('scaler', StandardScaler()))
        
        num_trans = Pipeline(steps)
        
        # Categorical Pipeline: Impute -> OHE
        cat_trans = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer([
            ('num', num_trans, num_features),
            ('cat', cat_trans, cat_features)
        ])
        
        # Classifier: Logistic Regression (Ridge by default, l2 penalty)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=5000)) 
        ])
        
        return pipeline

    def _enrich_interaction(self, df):
        """Adds Distance * Angle interaction term if using Splines."""
        if not self.use_splines:
            return df
            
        df = df.copy()
        
        # Ensure we have distance and angle
        if 'distance' in df.columns and 'angle_deg' in df.columns:
            # We use absolute angle because symmetry is usually assumed, 
            # but let's stick to raw product? 
            # Actually, angle is usually absolute in meaningfulness but signed for side.
            # PolynomialFeatures(degree=2) produces x*y.
            # If we want to capture "Sharp Angle at Long Distance" vs "Sharp Angle at Short Distance",
            # abs(angle) is probably what matters most for xG, unless we model strong/weak side issues.
            # But standard polynomial features would produce dist * angle (signed).
            # Let's standardize on ABSOLUTE angle for the interaction, 
            # as geometry for blockage/visible net is symmetric.
            
            # NOTE: We simply multiply them.
            # However, since 'angle_deg' can be negative, dist * angle would be negative.
            # Does left side vs right side matter for interaction? Probably not much if we assume symmetry.
            # Let's use ABS angle to force symmetry and keep the interaction monotonic with "difficulty".
            # Distance = Harder, Abs(Angle) = Harder.
            # Interaction = Distance * Abs(Angle) = Extremity.
            
            df[self.interact_col] = df['distance'] * df['angle_deg'].abs()
        else:
            # Should not happen in standard pipeline
            # If missing, fill 0
            df[self.interact_col] = 0.0
            
        return df

    def predict_proba_layer(self, X, layer):
        """Returns probability of success (1) for a specific layer.
           Applies marginalization for 'accuracy' and 'finish' layers if shot_type is unknown.
        """
        # Enrich interaction term
        X = self._enrich_interaction(X)
        
        if layer == 'block':
            # Block model predicts 'is_blocked'.
            # Must exclude shot_type. Marginalization not needed as shot_type is excluded.
            feats = [f for f in self.features if f != 'shot_type']
            return self.model_block.predict_proba(X[feats])[:, 1]
            
        # Accuracy & Finish need marginalization
        df = X[self.features].copy()
        
        # Identify rows needing marginalization
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
            
        # Select model
        if layer == 'accuracy':
            model = self.model_acc
        elif layer == 'finish':
            model = self.model_finish
        else:
            raise ValueError(f"Unknown layer: {layer}")
            
        # 1. Base Prediction (handles non-marginalized rows and baseline for others)
        p_final = model.predict_proba(df[self.features])[:, 1]
        
        # 2. Marginalization Loop
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
        """
        Predicts P(Goal) using the nested chain.
        Applies Marginalization for rows with missing shot_type.
        """
        # Enrich Interaction
        X = self._enrich_interaction(X)
        
        df = X[self.features].copy()
        
        # 1. Identify rows needing marginalization
        # Trigger on NaN OR 'Unknown' (case-insensitive)
        if 'shot_type' in df.columns:
            mask_nan = df['shot_type'].isna() | (df['shot_type'].astype(str).str.lower() == 'unknown')
        else:
            mask_nan = pd.Series([False]*len(df), index=df.index)
        
        # 2. Main Prediction Path (for everyone, initially)
        # Note: GLM handles NaNs via SimpleImputer(fill_value='Unknown') in the pipeline.
        # So we can run this once for everyone as a baseline (and it handles the non-marginalized rows).
        p_final = self._predict_single_pass(df)
        
        # 3. Marginalization Path
        if self.enable_marginalization and mask_nan.any() and self.shot_type_priors_:
            # logger.debug(f"Marginalizing {mask_nan.sum()} rows with missing shot_type...")
            
            df_nan = df[mask_nan].copy()
            accumulated_prob = np.zeros(len(df_nan))
            
            # Loop through known shot types
            for st, weight in self.shot_type_priors_.items():
                # Temporarily impute this shot type
                df_nan_imputed = df_nan.copy()
                df_nan_imputed['shot_type'] = st
                
                # Predict
                prob_st = self._predict_single_pass(df_nan_imputed)
                
                # Add weighted prob
                accumulated_prob += prob_st * weight
            
            # Update final results for these rows
            p_final[mask_nan] = accumulated_prob
            
        return np.column_stack((1 - p_final, p_final))

    def _predict_single_pass(self, df):
        """Helper to run the P(Unblocked)*P(OnNet)*P(Finish) chain without marginalization logic."""
        # 1. Block (Target=Blocked, so success=1-P)
        # EXCLUDE shot_type
        feats_block = [f for f in self.features if f != 'shot_type']
        p_blocked = self.model_block.predict_proba(df[feats_block])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        # 2. Accuracy
        p_on_net = self.model_acc.predict_proba(df[self.features])[:, 1]
        
        # 3. Finish
        p_finish = self.model_finish.predict_proba(df[self.features])[:, 1]
        
        # Combine
        return p_unblocked * p_on_net * p_finish
