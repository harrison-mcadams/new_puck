"""fit_nested_xgs.py

LAYERED (NESTED) EXPECTED GOALS MODEL
=====================================
This script implements a "Layered" or "Conditional" approach to xG modeling.
Instead of asking "Is this shot attempt a goal?" (Binary Classification), 
we break the event down into a chain of sequential hurdles:

    1. BLOCK LAYER:   Will the shot attempt get past the defense?
                      P(Unblocked) = 1.0 - P(Blocked)

    2. ACCURACY LAYER: Given it wasn't blocked, will it hit the net?
                      P(On Net | Unblocked)

    3. FINISH LAYER:  Given it's on net, will it beat the goalie?
                      P(Goal | On Net)

Total Expected Goals (xG) is the product of these probabilities:
    xG = P(Unblocked) * P(On Net) * P(Goal)

WHY DO THIS?
------------
1. **Data Integrity**: Blocked shots often lack detailed features (like shot type) because
   they never reach the net. Modeling them separately allows us to use "imputed" or
   simplified features for the Block Layer without polluting the Finish Layer (which
   detects goal probability based on clean shot-on-goal data).
2. **Granularity**: We can tell if a player has "poor xG" because they get blocked often
   (low P_unblocked) or because they miss the net often (low P_on_net).

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import json
import sys
import math
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any

# --- DEPENDENCIES ---
# We use scikit-learn for the Random Forest classifiers.
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import LabelEncoder


# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("NestedxG")

# Import Config for valid Data Directory
try:
    from . import config as puck_config
except ImportError:
    try:
        import config as puck_config
    except ImportError:
        # Provide a dummy config if strictly standalone and config missing (rare)
        class DummyConfig:
            DATA_DIR = 'data'
            ANALYSIS_DIR = 'analysis'
        puck_config = DummyConfig()


# --- CONFIGURATION ---
@dataclass
class LayerConfig:
    name: str
    target_col: str
    positive_label: int  # The value in target_col we are predicting probability OF
    feature_cols: List[str]
    n_estimators: int = 200
    max_depth: Optional[int] = 10

from sklearn.base import BaseEstimator, ClassifierMixin

class NestedXGClassifier(BaseEstimator, ClassifierMixin):
    """
    A scikit-learn compatible classifier that implements the Nested (Layered) xG Model.
    
    It trains three sequential Random Forest models:
      1. Block Model: P(Unblocked)
      2. Accuracy Model: P(On Net | Unblocked)
      3. Finish Model: P(Goal | On Net)
      
    Final Probability = P(Unblocked) * P(On Net) * P(Goal)
    
    UPDATED: Now uses dynamic features.
    """
    def __init__(self, random_state=42, n_estimators=200, unknown_shot_type_val='Unknown', 
                 max_depth=10, prevent_overfitting=True, features=None, feature_set_name=None):
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.unknown_shot_type_val = unknown_shot_type_val
        self.max_depth = max_depth if prevent_overfitting else None
        self.prevent_overfitting = prevent_overfitting
        
        # Features
        self.feature_set_name = feature_set_name
        if features is None and feature_set_name:
             try:
                 from . import features as puck_features
                 features = puck_features.get_features(feature_set_name)
             except (ImportError, ValueError):
                 import features as puck_features
                 features = puck_features.get_features(feature_set_name)
        
        if features is None:
             features = ['distance', 'angle_deg', 'game_state', 'shot_type']
             self.feature_set_name = self.feature_set_name or 'standard'

        self.features = features
        
        self.shot_type_priors = None 
        
        # Training Metadata
        self.final_features = [] # The actual columns after OHE
        self.categorical_cols = []
        self.numeric_cols = []
        
        self.model_block = None
        self.model_accuracy = None
        self.model_finish = None
        
        self.config_block = None
        self.config_accuracy = None
        self.config_finish = None
        
        self.shot_type_cols = []

    def fit(self, X, y=None):
        """
        Fit the nested models and learn shot type priors for imputation.
        Expects X to be a DataFrame. 
        Note: If OHE is already done externally, we identify cols. 
        BUT for consistency, we often expect Raw data and do OHE, 
        OR we expect clean_df_for_model to have done it?
        
        DECISION: To support analyze.py which uses clean_df_for_model, 
        we assume X ALREADY contains the OHE columns and we just need to identify them.
        However, fit_nested_xgs.py main() passes raw data.
        So we support both: if raw cols exist, we encode. If not, we assume we find them.
        """
        if not isinstance(X, pd.DataFrame):
             raise ValueError("NestedXGClassifier.fit expects a pandas DataFrame as X.")
             
        df = X.copy()
        
        # --- Preprocessing (Internal) ---
        if 'event' not in df.columns:
            raise ValueError("NestedXGClassifier requires 'event' column in X for training to determine layer targets.")

        # 1. Identify Numeric and Categorical Features from the requested list
        self.categorical_cols = [c for c in self.features if df[c].dtype == object]
        self.numeric_cols = [c for c in self.features if c not in self.categorical_cols]

        # 2. Perform OHE (if raw cols exist)
        if self.categorical_cols:
            df = pd.get_dummies(df, columns=self.categorical_cols, prefix_sep='_')
        
        # Identify columns after OHE
        self.final_features = []
        for c in self.features:
            if c in self.categorical_cols:
                # Add all resulting code columns
                self.final_features.extend([col for col in df.columns if col.startswith(f"{c}_")])
            else:
                self.final_features.append(c)

        # 3. Define Layer Features
        # Block Layer: everything EXCEPT shot_type related (since blocked shots don't have shot type)
        # We assume 'shot_type' is the conventional name.
        self.config_block = LayerConfig(
            name="Block Model", target_col='is_blocked', positive_label=1,
            feature_cols=[c for c in self.final_features if not c.startswith('shot_type_')],
            max_depth=self.max_depth, n_estimators=self.n_estimators
        )
        self.config_accuracy = LayerConfig(
            name="Accuracy Model", target_col='is_on_net', positive_label=1,
            feature_cols=self.final_features, max_depth=self.max_depth, n_estimators=self.n_estimators
        )
        self.config_finish = LayerConfig(
            name="Finish Model", target_col='is_goal_layer', positive_label=1,
            feature_cols=self.final_features, max_depth=self.max_depth, n_estimators=self.n_estimators
        )

        # Create Layer Targets
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
        df['is_goal_layer'] = (df['event'] == 'goal').astype(int)

        # 2. Train Layer 1: Block Model (All Data)
        self.model_block = RandomForestClassifier(
            n_estimators=self.config_block.n_estimators, 
            max_depth=self.config_block.max_depth, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.model_block.fit(df[self.config_block.feature_cols], df[self.config_block.target_col])
        
        # 3. Train Layer 2: Accuracy Model (Unblocked Only)
        df_unblocked = df[df['is_blocked'] == 0]
        self.model_accuracy = RandomForestClassifier(
            n_estimators=self.config_accuracy.n_estimators, 
            max_depth=self.config_accuracy.max_depth, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.model_accuracy.fit(df_unblocked[self.config_accuracy.feature_cols], df_unblocked[self.config_accuracy.target_col])

        # 4. Train Layer 3: Finish Model (On Net Only)
        df_on_net = df[df['is_on_net'] == 1]
        self.model_finish = RandomForestClassifier(
            n_estimators=self.config_finish.n_estimators, 
            max_depth=self.config_finish.max_depth, 
            random_state=self.random_state, 
            n_jobs=-1
        )
        self.model_finish.fit(df_on_net[self.config_finish.feature_cols], df_on_net[self.config_finish.target_col])
        
        # 5. Learn Shot Type Priors (Calculated from UNBLOCKED shots only)
        # We sum the OHE columns to get counts
        self.shot_type_cols = [c for c in self.final_features if c.startswith('shot_type_')]
        if self.shot_type_cols:
            stats = df_unblocked[self.shot_type_cols].sum()
            total = stats.sum()
            self.shot_type_priors = (stats / total).to_dict()
            
            # Remove 'Unknown' from priors if it exists
            unk_col = f'shot_type_{self.unknown_shot_type_val}'
            if unk_col in self.shot_type_priors:
                 del self.shot_type_priors[unk_col]
                 new_sum = sum(self.shot_type_priors.values())
                 if new_sum > 0:
                     self.shot_type_priors = {k: v/new_sum for k, v in self.shot_type_priors.items()}
        else:
            self.shot_type_priors = None
        
        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        Handles missing columns (aligns to training) and OHE-based Marginalization.
        """
        if not isinstance(X, pd.DataFrame):
             raise ValueError("NestedXGClassifier.predict_proba expects a pandas DataFrame.")
             
        df = X.copy()
        
        # 0. Feature Alignment
        # If input is raw, encode it. 
        cat_cols = getattr(self, 'categorical_cols', [])
        if cat_cols and not any(c in df.columns for c in self.final_features if c not in self.features):
             df = pd.get_dummies(df, columns=cat_cols, prefix_sep='_')
        
        # Add missing columns with 0
        for c in self.final_features:
            if c not in df.columns:
                df[c] = 0
                
        # 1. Block Prob (Independent of shot type)
        # Make sure we select columns in correct order? RF usually robust but better safe.
        # (sklearn doesn't require column order matching by name, it requires by POSITION. 
        # But we trained with pandas, so it shouldn't matter? Wait, sklearn converts pd to numpy array.
        # ORDER MATTERS.)
        
        # We MUST enforce column order to match training config
        p_blocked = self.model_block.predict_proba(df[self.config_block.feature_cols])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        # 2. Goal Prob (Conditional on Unblocked)
        # Determine rows to marginalize
        # Marginalize input where Shot Type is Unknown.
        # In OHE, "Unknown" means either 'shot_type_Unknown'==1 OR all 'shot_type_*'==0
        
        unk_col = f'shot_type_{self.unknown_shot_type_val}'
        if unk_col in df.columns:
            mask_impute = (df[unk_col] == 1)
        else:
            # If no unknown col, maybe check if sum of known cols is 0?
            known_cols = [c for c in self.shot_type_cols if c != unk_col and c in df.columns]
            if known_cols:
                mask_impute = (df[known_cols].sum(axis=1) == 0)
            else:
                mask_impute = np.zeros(len(df), dtype=bool)
                
        # Also force impute if any row has NaN in shot type cols? 
        # Assuming cleaned data.
        
        # Initialize results
        p_goal_given_unblocked = np.zeros(len(df))
        
        # A. Direct Calculation (Known Types)
        mask_direct = ~mask_impute
        if np.any(mask_direct):
            X_direct = df.loc[mask_direct]
            p_on_net_d = self.model_accuracy.predict_proba(X_direct[self.config_accuracy.feature_cols])[:, 1]
            p_finish_d = self.model_finish.predict_proba(X_direct[self.config_finish.feature_cols])[:, 1]
            p_goal_given_unblocked[mask_direct] = p_on_net_d * p_finish_d
            
        # B. Marginalized Calculation (Unknown Types)
        if np.any(mask_impute) and self.shot_type_priors:
            X_impute_base = df.loc[mask_impute].copy()
            weighted_prob_sum = np.zeros(len(X_impute_base))
            
            # Loop through each possible shot type (from priors)
            for st_col, st_prob in self.shot_type_priors.items():
                if st_col not in df.columns: 
                    continue # Should be rare
                    
                # Set this shot type to 1, all others to 0
                # We need to zero out ALL shot type cols first
                for c in self.shot_type_cols:
                    X_impute_base[c] = 0
                X_impute_base[st_col] = 1
                
                # Predict
                p_on_net_i = self.model_accuracy.predict_proba(X_impute_base[self.config_accuracy.feature_cols])[:, 1]
                p_finish_i = self.model_finish.predict_proba(X_impute_base[self.config_finish.feature_cols])[:, 1]
                
                # Accumulate
                weighted_prob_sum += (p_on_net_i * p_finish_i * st_prob)
                
            p_goal_given_unblocked[mask_impute] = weighted_prob_sum
            
        # 3. Final xG
        p_goal = p_unblocked * p_goal_given_unblocked
        
        return np.column_stack((1 - p_goal, p_goal))
    
    def predict_proba_layer(self, X, layer: str):
        """
        Return raw probability estimates for a specific layer.
        Useful for diagnostics/calibration plots.
        does NOT do marginalization - assumes X has valid features for that layer.
        """
        if not isinstance(X, pd.DataFrame):
             raise ValueError("NestedXGClassifier.predict_proba_layer expects a pandas DataFrame.")
             
        df = X.copy()
        
        # 0. Prep (OHE if needed)
        # Check if we need to do OHE ourselves (legacy support or raw input)
        if 'shot_type' in df.columns and not any(c in df.columns for c in self.shot_type_cols):
             df['shot_type'] = df['shot_type'].fillna('Unknown')
             df = pd.get_dummies(df, columns=['shot_type', 'game_state'], prefix_sep='_')
        
        # Add missing columns with 0
        all_feats = list(set(self.config_block.feature_cols + self.config_accuracy.feature_cols + self.config_finish.feature_cols))
        for c in all_feats:
            if c not in df.columns:
                df[c] = 0
                
        if layer == 'block':
            return self.model_block.predict_proba(df[self.config_block.feature_cols])[:, 1]
        elif layer == 'accuracy':
            return self.model_accuracy.predict_proba(df[self.config_accuracy.feature_cols])[:, 1]
        elif layer == 'finish':
            return self.model_finish.predict_proba(df[self.config_finish.feature_cols])[:, 1]
        else:
             raise ValueError(f"Unknown layer '{layer}'. Must be block, accuracy, or finish.")

    def predict(self, X):
        # Threshold at 0.5 (arbitrary for xG, but required for API)
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    
    
def load_data(path_pattern: str = None) -> pd.DataFrame:
    """Load and concatenate all available season data. Expects data/{year}/*.csv"""
    
    if path_pattern is None:
        path_pattern = str(Path(puck_config.DATA_DIR) / "**" / "*.csv")
        
    import glob
    files = glob.glob(path_pattern, recursive=True)
    
    # Deduplication logic: Map season -> file path
    # We prefer 'data/{year}/{year}_df.csv' over 'data/{year}/{year}.csv'
    # And we want to avoid loading the same season from multiple locations (e.g. data/processed vs data/)
    season_files = {}

    for f in files:
        p = Path(f)
        parent_name = p.parent.name
        
        # Check if parent is a year-like dir (e.g. "20252026")
        if parent_name.isdigit():
             # Check for valid filenames
             if p.name.endswith('_df.csv') or p.name == f"{parent_name}.csv":
                 
                 # Logic: If we haven't seen this season, take it.
                 # If we HAVE seen it, prefer the one that ends in '_df.csv' (new format)
                 # or prefer the one in 'data/' over 'data/processed/' if needed (implicit by shorter path? random?)
                 # Let's simple prefer '_df.csv' as the tiebreaker.
                 
                 if parent_name not in season_files:
                     season_files[parent_name] = f
                 else:
                     # If existing is NOT preferred format and NEW IS, swap.
                     current_path = season_files[parent_name]
                     if not current_path.endswith('_df.csv') and f.endswith('_df.csv'):
                         season_files[parent_name] = f
                     # If both are same format, duplicates in different folders?
                     # e.g. data/2025/file.csv vs data/processed/2025/file.csv
                     # Just keep the first one found or maybe the one with shorter path (closer to root)
                     pass
    
    data_files = list(season_files.values())
    
    if not data_files:
        logger.warning(f"No season files found matching pattern. Trying specific fallback.")
        fallback = str(Path(puck_config.DATA_DIR) / '20252026/20252026_df.csv')
        if Path(fallback).exists():
            data_files = [fallback]
        else:
            raise FileNotFoundError("Could not find any season CSV files.")
            
    logger.info(f"Loading data from {len(data_files)} files: {data_files}")
    dfs = []
    for f in data_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    full_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(full_df)} total rows.")
    return full_df


# --- PREPROCESSING ----
def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the dataframe for modeling.
    
    1. Filter to Shot Attempts only.
    2. Impute missing 'shot_type' for Blocked Shots as 'Unknown'.
    3. Encode categorical features.
    """
    # 1. Scope: Shot Attempts Only
    # We define the universe of 'Attempts' as: Goal, Shot (Saved), Miss, Block
    # Note: 'shot-on-goal' usually means saved shots in API data, but sometimes includes goals.
    # We'll normalize.
    
    # Standardize event names if needed (though our parser is usually consistent)
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df = df[df['event'].isin(valid_events)].copy()
    
    # 2. Handle Blocked Shots Missing Data
    # If the column is completely missing (e.g. older data), create it.
    if 'shot_type' not in df.columns:
        logger.warning("'shot_type' column missing from data. Filling with 'Unknown'.")
        df['shot_type'] = 'Unknown'

    # EXCLUDE EMPTY NET SHOTS from Training
    if 'is_net_empty' in df.columns:
        mask_empty = (df['is_net_empty'] == 1) | (df['is_net_empty'] == True)
        if mask_empty.any():
            logger.info(f"Filtering {mask_empty.sum()} empty net shots from training data.")
            df = df[~mask_empty].copy()

    # EXCLUDE 1v0 and 0v1 from Training
    if 'game_state' in df.columns:
        mask_extreme = df['game_state'].isin(['1v0', '0v1'])
        if mask_extreme.any():
            logger.info(f"Filtering {mask_extreme.sum()} rows with extreme game states (1v0/0v1).")
            df = df[~mask_extreme].copy()
    
    # Blocked shots usually have shot_type = NaN.
    # For the Block Model, this 'Missingness' is actually a signal (or at least, we need a value).
    # We will fill NaN with 'Unknown'.
    df['shot_type'] = df['shot_type'].fillna('Unknown')
    
    # EXCLUDE NON-REGULAR SEASON (02)
    if 'game_id' in df.columns:
        df['game_id_str'] = df['game_id'].astype(str)
        # Game IDs are YYYYTTNNNN, where TT=02 is regular season.
        mask_regular = (df['game_id_str'].str.len() >= 6) & (df['game_id_str'].str[4:6] == '02')
        if not mask_regular.all():
            logger.info(f"Filtering {len(df) - mask_regular.sum()} events from non-regular season games.")
            df = df[mask_regular].copy()
        df.drop(columns=['game_id_str'], inplace=True)

    # 3. Create Target Columns for Each Layer
    
    # LAYER 1 TARGET: is_blocked?
    # 1 = Blocked, 0 = Not Blocked (Missed, Saved, Goal)
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    # LAYER 2 TARGET: is_on_net? (Condition: Only Unblocked)
    # 1 = On Net (Goal, Saved), 0 = Missed
    # Note: 'shot-on-goal' in our data usually implies a save. 'goal' is a goal. 
    # Both are "On Net".
    df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
    
    # LAYER 3 TARGET: is_goal? (Condition: Only On Net)
    # 1 = Goal, 0 = Saved
    df['is_goal_layer'] = (df['event'] == 'goal').astype(int)

    return df


# --- TRAINING LOGIC ---

def train_layer(name: str, df_layer: pd.DataFrame, config: LayerConfig):
    """Train a single model layer."""
    
    X = df_layer[config.feature_cols]
    y = df_layer[config.target_col]
        
    logger.info(f"Training {name}... (N={len(df_layer)})")
    logger.info(f"  Target: {config.target_col} (Positive Rate: {y.mean():.1%})")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_prob = clf.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_prob)
    ll = log_loss(y_test, y_prob)
    
    logger.info(f"  Result -> AUC: {auc:.4f} | LogLoss: {ll:.4f}")
    
    return clf, X_test, y_test, y_prob


# --- PLOTTING LOGIC ---

def plot_calibration_curve_layer(y_true, y_prob, name: str, ax=None):
    """Plot calibration for a specific layer."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(prob_pred, prob_true, marker='o', label=name)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title(f'Calibration: {name}')
    ax.legend()
    ax.grid(True, alpha=0.2)
    return ax

def plot_feature_importance(clf, feature_names, name: str, ax=None):
    """Bar chart of feature importances."""
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    
    if ax is None:
        fig, ax = plt.subplots()
        
    ax.barh(range(len(indices)), importances[indices], align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(f'{name} Features')
    return ax


# --- MAIN PIPELINE ---

def main():
    logger.info("Starting Nested xG Training...")
    
    # 1. Load & Preprocess
    df = load_data()
    
    # 2. Basic Cleaning (Events, Empty Net, Game State)
    df = preprocess_features(df)
    
    try:
        from . import impute
    except ImportError:
        import impute
        
    logger.info("Applying 'mean_6' imputation for blocked shots...")
    df_imputed = impute.impute_blocked_shot_origins(df, method='mean_6')
    
    # 2. Evaluation Split
    logger.info("Splitting data for evaluation (80/20)...")
    df_train, df_test = train_test_split(df_imputed, test_size=0.2, random_state=42)
    
    # 3. Train Eval Model
    logger.info("Training evaluation model on 80% split...")
    clf_eval = NestedXGClassifier(n_estimators=100) # Faster for eval
    clf_eval.fit(df_train)
    
    # 4. Generate Diagnostics
    visuals_dir = Path(puck_config.ANALYSIS_DIR) / 'nested_xgs'
    visuals_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating diagnostic plots in {visuals_dir}...")
    
    # Re-create targets for test set (mimicking fit logic)
    df_test = df_test.copy()
    df_test['is_blocked'] = (df_test['event'] == 'blocked-shot').astype(int)
    # Note: is_on_net for validation of Accuracy Layer should be restricted to Unblocked shots
    df_test['is_on_net'] = df_test['event'].isin(['shot-on-goal', 'goal']).astype(int)
    # Note: is_goal for validation of Finish Layer should be restricted to On Net shots
    df_test['is_goal_layer'] = (df_test['event'] == 'goal').astype(int)
    
    # --- Plot A: Block Model ---
    p_block = clf_eval.predict_proba_layer(df_test, 'block')
    plot_calibration_curve_layer(df_test['is_blocked'], p_block, "Block Model")
    plt.savefig(visuals_dir / 'calibration_block.png')
    plt.close()
    
    auc_block = roc_auc_score(df_test['is_blocked'], p_block)
    ll_block = log_loss(df_test['is_blocked'], p_block)
    logger.info(f"Block Model Eval: AUC={auc_block:.4f}, LogLoss={ll_block:.4f}")
    
    plot_feature_importance(clf_eval.model_block, clf_eval.config_block.feature_cols, "Block Model")
    plt.savefig(visuals_dir / 'importance_block.png')
    plt.close()
    
    # --- Plot B: Accuracy Model (Unblocked Only) ---
    mask_unblocked = (df_test['is_blocked'] == 0)
    if mask_unblocked.sum() > 0:
        df_test_acc = df_test[mask_unblocked]
        p_acc = clf_eval.predict_proba_layer(df_test_acc, 'accuracy')
        plot_calibration_curve_layer(df_test_acc['is_on_net'], p_acc, "Accuracy Model")
        plt.savefig(visuals_dir / 'calibration_accuracy.png')
        plt.close()
        
        auc_acc = roc_auc_score(df_test_acc['is_on_net'], p_acc)
        ll_acc = log_loss(df_test_acc['is_on_net'], p_acc)
        logger.info(f"Accuracy Model Eval (Unblocked): AUC={auc_acc:.4f}, LogLoss={ll_acc:.4f}")
        
        plot_feature_importance(clf_eval.model_accuracy, clf_eval.config_accuracy.feature_cols, "Accuracy Model")
        plt.savefig(visuals_dir / 'importance_accuracy.png')
        plt.close()
    
    # --- Plot C: Finish Model (On Net Only) ---
    mask_on_net = (df_test['is_blocked'] == 0) & (df_test['is_on_net'] == 1)
    if mask_on_net.sum() > 0:
        df_test_finish = df_test[mask_on_net]
        p_finish = clf_eval.predict_proba_layer(df_test_finish, 'finish')
        plot_calibration_curve_layer(df_test_finish['is_goal_layer'], p_finish, "Finish Model")
        plt.savefig(visuals_dir / 'calibration_finish.png')
        plt.close()
        
        auc_fin = roc_auc_score(df_test_finish['is_goal_layer'], p_finish)
        ll_fin = log_loss(df_test_finish['is_goal_layer'], p_finish)
        logger.info(f"Finish Model Eval (On Net): AUC={auc_fin:.4f}, LogLoss={ll_fin:.4f}")
        
        plot_feature_importance(clf_eval.model_finish, clf_eval.config_finish.feature_cols, "Finish Model")
        plt.savefig(visuals_dir / 'importance_finish.png')
        plt.close()
        
    # --- Plot D: Overall xG Calibration ---
    # Compare final xG pred vs Actual Goal
    p_xg = clf_eval.predict_proba(df_test)[:, 1]
    df_test['is_goal_final'] = (df_test['event'] == 'goal').astype(int)
    plot_calibration_curve_layer(df_test['is_goal_final'], p_xg, "Nested xG Model (Combined)")
    plt.savefig(visuals_dir / 'calibration_overall.png')
    plt.close()
    
    auc_xg = roc_auc_score(df_test['is_goal_final'], p_xg)
    ll_xg = log_loss(df_test['is_goal_final'], p_xg)
    sum_xg = p_xg.sum()
    sum_goals = df_test['is_goal_final'].sum()
    logger.info(f"Total xG Model Eval: AUC={auc_xg:.4f}, LogLoss={ll_xg:.4f}")
    logger.info(f"CALIBRATION CHECK: Sum(xG)={sum_xg:.2f} vs Sum(Goals)={sum_goals} -> Ratio: {sum_xg/sum_goals:.3f}")
    
    # 5. Train Final Model
    logger.info("Retraining final model on FULL data (all years)...")
    clf = NestedXGClassifier(
        n_estimators=200, 
        max_depth=10, 
        prevent_overfitting=True
    )
    clf.fit(df_imputed)
    
    # 6. Save
    out_dir = Path(puck_config.ANALYSIS_DIR) / 'xgs'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'xg_model_nested.joblib'
    
    logger.info(f"Saving model to {out_path}...")
    joblib.dump(clf, out_path)
    
    # Save Metadata
    meta = {
        'model_type': 'nested',
        'imputation': 'mean_6',
        'training_rows': len(df_imputed),
        'final_features': clf.final_features
    }
    with open(str(out_path) + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
        
    logger.info("Model saved successfully.")
    
    # 7. Example Inference
    logger.info("Running example inference...")
    sample = df_imputed.sample(10).copy()
    probs = clf.predict_proba(sample)[:, 1]
    sample['nested_xg'] = probs
    print(sample[['event', 'distance', 'angle_deg', 'nested_xg']].to_string())

if __name__ == "__main__":
    main()
