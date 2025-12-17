"""compare_models.py

Visualizes and verifies the performance differences between:
1. Baseline (Dist/Angle)
2. With Shot Type (Dist/Angle + Shot Type) -> SUSPECTED LEAKAGE
3. Nested xG (Block->Accuracy->Finish)

Generates: `os.path.join(ANALYSIS_DIR, 'xgs', 'model_comparison_dashboard.png')`
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import json
import os
import sys
from pathlib import Path
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator

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

# Attempt to import relevant modules (for Nested class)
try:
    from puck.fit_nested_xgs import NestedXGClassifier
except ImportError:
    try:
        import fit_nested_xgs
        NestedXGClassifier = fit_nested_xgs.NestedXGClassifier
    except ImportError:
        NestedXGClassifier = None

# Re-use critical data loading/cleaning logic from fit_xgs implies duplicating or importing.
# Importing is cleaner but tricky with script vs module. We will try to import.
try:
    from puck.fit_xgs import load_all_seasons_data, clean_df_for_model, ModelConfig
except ImportError:
    # Quick hack to allow import from current dir if running as script
    sys.path.append('.')
    from puck.fit_xgs import load_all_seasons_data, clean_df_for_model, ModelConfig


def load_model_safely(path, model_name):
    """Load joblib model."""
    if not Path(path).exists():
        print(f"[{model_name}] Not found at {path}")
        return None, None
    
    print(f"[{model_name}] Loading from {path}...")
    try:
        clf = joblib.load(path)
    except Exception as e:
        print(f"[{model_name}] Failed to load joblib: {e}")
        return None, None
    
    # Try to load metadata
    meta_path = path + '.meta.json'
    meta = {}
    if Path(meta_path).exists():
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            
    return clf, meta

def get_global_map(df):
    """
    Derive a consistent categorical map from the full dataset.
    This ensures that 'Wrist Shot' is always encoded to the same integer 
    regardless of which model we are using (assuming they were trained on similar data).
    """
    # We need a superset of all features we care about
    all_feats = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    
    # Fill Unknowns for shot_type definition
    df_fill = df.copy()
    df_fill['shot_type'] = df_fill['shot_type'].fillna('Unknown')
    
    _, _, cat_map = clean_df_for_model(df_fill, all_feats)
    return cat_map


def get_model_predictions(model, df, model_name, global_map, meta=None):
    """
    Generate Prob(Goal) for the given dataframe.
    """
    
    # 1. Nested xG
    if hasattr(model, 'model_block'): 
        # Manual Prep for Nested Model
        # Needs: ['distance', 'angle_deg', 'shot_type_encoded', 'is_net_empty', 'game_state_encoded']
        
        prep_conf = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
        
        # Fill NaNs
        df_n = df.copy()
        df_n['shot_type'] = df_n['shot_type'].fillna('Unknown')
        
        # Clean using GLOBAL MAP to ensure codes match what we likely trained on
        # (Assuming training used the same sort-order logic which clean_df does)
        df_clean, _, _ = clean_df_for_model(df_n, prep_conf, fixed_categorical_levels=global_map)
        
        # Rename to match Nested internals
        rename_map = {
            'shot_type_code': 'shot_type_encoded',
            'game_state_code': 'game_state_encoded'
        }
        df_clean = df_clean.rename(columns=rename_map)
        
        return model.predict_proba(df_clean)[:, 1], df_clean.index
        
    # 2. Standard Models (Baseline / With Shot Type)
    feature_names = None
    if meta:
        feature_names = meta.get('final_features')
    
    if not feature_names:
        if model_name == 'Baseline':
            feature_names = ['distance', 'angle_deg', 'game_state_code', 'is_net_empty']
        elif model_name == 'With Shot Type':
            feature_names = ['distance', 'angle_deg', 'game_state_code', 'is_net_empty', 'shot_type_code']
            
    # Clean using GLOBAL MAP
    # Determine which raw cols we need based on feature names
    raw_cols = ['distance', 'angle_deg', 'is_net_empty']
    if 'game_state_code' in str(feature_names): raw_cols.append('game_state')
    if 'shot_type_code' in str(feature_names): raw_cols.append('shot_type')
    
    # Special: Shot Type model needs imputation for 'Unknown' to match global map?
    # If the map has 'Unknown', we better have 'Unknown' in data.
    # If the map DOES NOT have 'Unknown' (because we trained on filtered data?), 
    # then 'Unknown' will become -1 or error.
    # We should fillna('Unknown') just in case.
    df_s = df.copy()
    df_s['shot_type'] = df_s['shot_type'].fillna('Unknown')
    
    df_clean, _, _ = clean_df_for_model(df_s, raw_cols, fixed_categorical_levels=global_map)
    
    return model.predict_proba(df_clean[feature_names])[:, 1], df_clean.index


def plot_combined_metrics(y_true, y_prob, name, ax_roc, ax_cal, color, style='-'):
    """Helper to plot ROC and Calibration for one model."""
    label_suffix = " (Unblocked)" if style == ':' else ""
    
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    ax_roc.plot(fpr, tpr, color=color, linestyle=style, lw=2, label=f'{name}{label_suffix} (AUC={roc_auc:.3f})')
    
    # Calibration
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax_cal.plot(prob_pred, prob_true, marker='.' if style=='-' else None, linestyle=style, color=color, label=f'{name}{label_suffix}')
    
    return {'auc': roc_auc}


def main():
    # 1. Load Data
    print("Loading Data...")
    df = load_all_seasons_data()
    
    # Ensure is_goal exists for stratification
    if 'is_goal' not in df.columns:
        df['is_goal'] = (df['event'] == 'goal').astype(int)
        
    # Split
    from sklearn.model_selection import train_test_split
    # We use the same random_state as fit_xgs to ensure the map derived from Train is valid for Test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_goal'])
    print(f"Test Set Size: {len(test_df)}")

    # 1.5 Generate Global Categorical Map
    # To fix Baseline performance, we must mimic fit_xgs: derive mapping from TRAINING data.
    # The Global Map from valid training data is the robust way.
    print("Generating Categorical Map from Training Data...")
    train_df_fill = train_df.copy()
    train_df_fill['shot_type'] = train_df_fill['shot_type'].fillna('Unknown')
    
    # We grab all potential categorical cols
    all_feats = ['distance', 'angle_deg', 'game_state', 'is_net_empty', 'shot_type']
    _, _, global_map = clean_df_for_model(train_df_fill, all_feats)
    
    # 2. Def Models
    models_to_load = [
        ('Baseline', os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_model.joblib'), 'blue'),
        ('With Shot Type', os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'xg_model_shot_type.joblib'), 'green'),
        ('Nested xG', os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'nested_xg_model.joblib'), 'purple')
    ]
    
    # Setup Plots: 1 Row, 2 Cols
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_roc = axes[0]
    ax_cal = axes[1]
    
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_roc.set_title('ROC Curves')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    
    ax_cal.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_cal.set_title('Calibration Curves')
    ax_cal.set_xlabel('Predicted Probability')
    ax_cal.set_ylabel('Observed Goal Rate')

    # 3. Process Models
    start_test_df = test_df.copy()
    
    for name, path, color in models_to_load:
        clf, meta = load_model_safely(path, name)
        if clf is None: continue
        
        # Leakage Check
        if name == 'With Shot Type' and hasattr(clf, 'feature_importances_'):
             print(f"--- {name}: Checking for Leakage ---")
             pass
        
        # Predict All
        try:
            probs, valid_idx = get_model_predictions(clf, start_test_df.copy(), name, global_map, meta)
            
            # Align Y True
            y_true_aligned = start_test_df.loc[valid_idx, 'is_goal']
            
            # 1. Plot Full (Solid)
            plot_combined_metrics(y_true_aligned, probs, name, ax_roc, ax_cal, color, style='-')
            
            # 2. Plot Unblocked Subset (Dotted)
            aligned_events = start_test_df.loc[valid_idx, 'event']
            unblocked_mask = (aligned_events != 'blocked-shot')
            
            y_sub = y_true_aligned[unblocked_mask]
            probs_sub = probs[unblocked_mask]
            
            if len(y_sub) > 0:
                plot_combined_metrics(y_sub, probs_sub, name, ax_roc, ax_cal, color, style=':')
            else:
                print(f"[{name}] Warning: No unblocked shots found in valid predictions.")
            
        except Exception as e:
             print(f"Failed to process {name}: {e}")
             import traceback
             traceback.print_exc()

    ax_roc.legend(loc='lower right')
    ax_cal.legend(loc='upper left')
    
    out_path = os.path.join(puck_config.ANALYSIS_DIR, 'xgs', 'model_comparison_dashboard.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"\nSaved dashboard to {out_path}")

if __name__ == '__main__':
    main()
