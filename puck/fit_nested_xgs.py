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
    """
    def __init__(self, random_state=42, n_estimators=200):
        self.random_state = random_state
        self.n_estimators = n_estimators
        
        self.model_block = None
        self.model_accuracy = None
        self.model_finish = None
        
        # Define configurations for internal layers
        # Note: We assume the input DataFrame has the necessary columns (or we create them)
        self.config_block = LayerConfig(
            name="Block Model", target_col='is_blocked', positive_label=1,
            feature_cols=['distance', 'angle_deg', 'is_net_empty', 'game_state_encoded']
        )
        self.config_accuracy = LayerConfig(
            name="Accuracy Model", target_col='is_on_net', positive_label=1,
            feature_cols=['distance', 'angle_deg', 'shot_type_encoded', 'is_net_empty', 'game_state_encoded']
        )
        self.config_finish = LayerConfig(
            name="Finish Model", target_col='is_goal_layer', positive_label=1,
            feature_cols=['distance', 'angle_deg', 'shot_type_encoded', 'is_net_empty', 'game_state_encoded']
        )

    def fit(self, X, y=None):
        """
        Fit the nested models.
        
        Args:
            X: pandas DataFrame containing feature columns AND 'event' column (for stratification).
               If 'event' is missing, valid training is impossible for layers.
            y: (Ignored), we derive targets from 'event' column in X.
        """
        if not isinstance(X, pd.DataFrame):
             raise ValueError("NestedXGClassifier.fit expects a pandas DataFrame as X to access column names.")
             
        df = X.copy()
        
        # --- Preprocessing (Internal) ---
        # 1. Ensure Targets Exist (derived from 'event' or passed in X)
        # We need 'event' to determine if blocked/missed/goal
        if 'event' not in df.columns:
            # Fallback? If y is provided and simple binary, we can't do nested training.
            # We strictly need event types.
            raise ValueError("NestedXGClassifier requires 'event' column in X for training to determine layer targets.")
            
        # Create Layer Targets
        df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
        df['is_on_net'] = df['event'].isin(['shot-on-goal', 'goal']).astype(int)
        df['is_goal_layer'] = (df['event'] == 'goal').astype(int)

        # 2. Train Layer 1: Block Model (All Data)
        self.model_block = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=self.random_state, n_jobs=-1)
        self.model_block.fit(df[self.config_block.feature_cols], df[self.config_block.target_col])
        
        # 3. Train Layer 2: Accuracy Model (Unblocked Only)
        df_unblocked = df[df['is_blocked'] == 0]
        self.model_accuracy = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=self.random_state, n_jobs=-1)
        self.model_accuracy.fit(df_unblocked[self.config_accuracy.feature_cols], df_unblocked[self.config_accuracy.target_col])

        # 4. Train Layer 3: Finish Model (On Net Only)
        df_on_net = df[df['is_on_net'] == 1]
        self.model_finish = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=10, random_state=self.random_state, n_jobs=-1)
        self.model_finish.fit(df_on_net[self.config_finish.feature_cols], df_on_net[self.config_finish.target_col])
        
        return self

    def predict_proba(self, X):
        """
        Return probability estimates for the test data X.
        Returns: array-like of shape (n_samples, 2) -> [P(NoGoal), P(Goal)]
        """
        if not isinstance(X, pd.DataFrame):
             # Try to convert if it's a known schema? 
             # For now, require DF.
             raise ValueError("NestedXGClassifier.predict_proba expects a pandas DataFrame.")
             
        p_blocked = self.model_block.predict_proba(X[self.config_block.feature_cols])[:, 1]
        p_unblocked = 1.0 - p_blocked
        
        p_on_net = self.model_accuracy.predict_proba(X[self.config_accuracy.feature_cols])[:, 1]
        
        p_finish = self.model_finish.predict_proba(X[self.config_finish.feature_cols])[:, 1]
        
        p_goal = p_unblocked * p_on_net * p_finish
        
        return np.column_stack((1 - p_goal, p_goal))
    
    def predict(self, X):
        # Threshold at 0.5 (arbitrary for xG, but required for API)
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)

    
    
def load_data(path_pattern: str = 'data/**/*.csv') -> pd.DataFrame:
    """Load and concatenate all available season data. Expects data/{year}/*.csv"""
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
        fallback = 'data/20252026/20252026_df.csv'
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
    
    # Blocked shots usually have shot_type = NaN.
    # For the Block Model, this 'Missingness' is actually a signal (or at least, we need a value).
    # We will fill NaN with 'Unknown'.
    df['shot_type'] = df['shot_type'].fillna('Unknown')
    
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

    # 4. Feature Engineering: Basic Encodings
    # Encode categorical features
    le_shot = LabelEncoder()
    df['shot_type_encoded'] = le_shot.fit_transform(df['shot_type'].astype(str))
    
    # Fill missing game_state with '5v5' (heuristic) before encoding
    df['game_state'] = df['game_state'].fillna('5v5')
    le_state = LabelEncoder()
    df['game_state_encoded'] = le_state.fit_transform(df['game_state'].astype(str))
    
    # Log mappings for user education
    logger.info(f"Shot Type Encoding: {dict(zip(le_shot.classes_, range(len(le_shot.classes_))))}")
    logger.info(f"Game State Encoding: {dict(zip(le_state.classes_, range(len(le_state.classes_))))}")
    
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
    df_clean = preprocess_features(df)
    
    # 2. Define Layers
    # Note on Features:
    # - 'shot_type_encoded' includes 'Unknown' mostly for blocked shots.
    # - 'is_net_empty' is crucial for Goal probability but less so for blocking.
    
    # LAYER 1: BLOCK MODEL
    # Target: 1 if Blocked, 0 if Not.
    # Note: We want P(Unblocked), so eventually we take (1 - P(Blocked)).
    config_block = LayerConfig(
        name="Layer 1 - Block Model",
        target_col='is_blocked',
        positive_label=1,
        feature_cols=['distance', 'angle_deg', 'is_net_empty', 'game_state_encoded']
    )
    
    # LAYER 2: ACCURACY MODEL
    # Filter: Only Unblocked Shots (df[is_blocked == 0])
    # Target: 1 if On Net (Goal/Save), 0 if Miss.
    config_accuracy = LayerConfig(
        name="Layer 2 - Accuracy Model",
        target_col='is_on_net',
        positive_label=1,
        feature_cols=['distance', 'angle_deg', 'shot_type_encoded', 'is_net_empty', 'game_state_encoded']
    )
    
    # LAYER 3: FINISH MODEL
    # Filter: Only Shots On Net (df[is_on_net == 1])
    # Target: 1 if Goal, 0 if Save.
    config_finish = LayerConfig(
        name="Layer 3 - Finish Model",
        target_col='is_goal_layer',
        positive_label=1,
        feature_cols=['distance', 'angle_deg', 'shot_type_encoded', 'is_net_empty', 'game_state_encoded']
    )

    # 3. Train Sequentially with Filtering
    
    # L1: Train on ALL attempts
    model_block, _, y_test_block, y_prob_block = train_layer(
        "Block Model", df_clean, config_block
    )
    
    # L2: Train on UNBLOCKED attempts only
    df_unblocked = df_clean[df_clean['is_blocked'] == 0].copy()
    model_accuracy, _, y_test_acc, y_prob_acc = train_layer(
        "Accuracy Model", df_unblocked, config_accuracy
    )
    
    # L3: Train on ON-NET attempts only
    df_on_net = df_clean[df_clean['is_on_net'] == 1].copy()
    model_finish, _, y_test_finish, y_prob_finish = train_layer(
        "Finish Model", df_on_net, config_finish
    )
    
    # 4. Generate Output Directory
    out_dir = Path('analysis/nested_xgs')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. Visualization Dashboard
    logger.info("Generating plots...")
    
    # Plot 1: Combined Calibration
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_calibration_curve_layer(y_test_block, y_prob_block, "P(Blocked)", ax=axes[0])
    plot_calibration_curve_layer(y_test_acc, y_prob_acc, "P(OnNet | Unblocked)", ax=axes[1])
    plot_calibration_curve_layer(y_test_finish, y_prob_finish, "P(Goal | OnNet)", ax=axes[2])
    plt.tight_layout()
    plt.savefig(out_dir / 'layer_calibrations.png')
    logger.info(f"Saved {out_dir / 'layer_calibrations.png'}")
    
    # Plot 2: Feature Importance
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_feature_importance(model_block, config_block.feature_cols, "Block Model", ax=axes[0])
    plot_feature_importance(model_accuracy, config_accuracy.feature_cols, "Accuracy Model", ax=axes[1])
    plot_feature_importance(model_finish, config_finish.feature_cols, "Finish Model", ax=axes[2])
    plt.tight_layout()
    plt.savefig(out_dir / 'layer_features.png')
    logger.info(f"Saved {out_dir / 'layer_features.png'}")
    
    # 6. Save Models (Moved before inference for safety)
    try:
        joblib.dump(model_block, out_dir / 'model_block.joblib')
        joblib.dump(model_accuracy, out_dir / 'model_accuracy.joblib')
        joblib.dump(model_finish, out_dir / 'model_finish.joblib')
        logger.info("Models saved.")
    except Exception as e:
        logger.error(f"Failed to save models: {e}")

    # 7. Example Inference (Sanity Check)
    logger.info("Running example inference on sample data...")
    # Take a sample from the original data
    sample = df_clean.sample(10, random_state=42).copy()
    
    # Get probabilities for each layer using SPECIFIC feature sets for each model
    p_blocked = model_block.predict_proba(sample[config_block.feature_cols])[:, 1]
    p_unblocked = 1.0 - p_blocked
    
    p_on_net = model_accuracy.predict_proba(sample[config_accuracy.feature_cols])[:, 1]
    
    p_finish = model_finish.predict_proba(sample[config_finish.feature_cols])[:, 1]
    
    # Calculate Final xG
    # xG = P(Unblocked) * P(OnNet) * P(Goal)
    sample['nested_xg'] = p_unblocked * p_on_net * p_finish
    
    # Simple console output
    print("\n--- SAMPLE PREDICTIONS ---")
    cols = ['event', 'shot_type', 'distance', 'nested_xg']
    print(sample[cols].to_string(index=False))

if __name__ == "__main__":
    main()
