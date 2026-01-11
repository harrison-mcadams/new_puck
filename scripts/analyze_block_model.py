
"""
Script to analyze the 'Blocked Shot Submodel' within the nested XGBoost framework.
1. Loads the trained pipeline (`puck/data/xg_model.joblib`).
2. Extracts the `model_block` (the classifier for P(Blocked)).
3. Evaluates Feature Importance.
4. Calculates AUC on a validation set (or estimates it from training data if needed).
"""
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgs, features as feature_util, data_pipeline, fit_xgboost_nested

def analyze_block_model():
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # Extract the actual Nested Classifier
    # Usually pipeline is a Pipeline object or the model itself?
    # In train_xgboost_model.py, it dumps `model` which is `XGBNestedXGClassifier`.
    
    model = pipeline
    if hasattr(model, 'steps'): # If it's a generic sklearn pipeline
        model = model.steps[-1][1]
        
    if not hasattr(model, 'model_block'):
        print("Error: Loaded model does not appear to be an XGBNestedXGClassifier (no 'model_block' attribute).")
        return

    block_model = model.model_block
    print("\n--- Blocked Shot Submodel Analysis ---")
    print(f"Type: {type(block_model)}")
    
    # Feature Importance
    if hasattr(block_model, 'feature_importances_'):
        importances = block_model.feature_importances_
        # We need the feature names used for the block model.
        # model.config_block.feature_cols
        feature_names = model.config_block.feature_cols
        
        if len(importances) != len(feature_names):
            print(f"Warning: Feature count mismatch ({len(importances)} vs {len(feature_names)})")
            # Try getting feature names from the booster if possible
            try:
                feature_names = block_model.get_booster().feature_names
            except:
                pass
        
        if len(importances) == len(feature_names):
            df_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            print("\nTop 20 Features driving P(Blocked):")
            print(df_imp.head(20))
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(df_imp['Feature'].head(20)[::-1], df_imp['Importance'].head(20)[::-1])
            plt.title('Feature Importance: Blocked Shot Model (P(Blocked))')
            plt.xlabel('Importance (Gain/Gini)')
            plt.tight_layout()
            plt.savefig('analysis/nested_xgs/block_model_feature_importance.png')
            print("Saved importance plot to analysis/nested_xgs/block_model_feature_importance.png")
            
    # Evaluation (AUC)
    # We need data to test it. Loading a generic sample.
    print("\nLoading sample data for evaluation...")
    df = fit_xgs.load_data() # Load defaults (usually 2023-2024 or similar)
    if df.empty:
        print("No data loaded. Skipping evaluation.")
        return

    # Preprocess
    print("Preprocessing...")
    df_processed = data_pipeline.preprocess_features(
        df, 
        is_training=False, 
        apply_arena_adjustments=True,
        apply_imputation=True,
        apply_dithering=False
    )
    
    # Prepare Inputs for Block Model
    # We focus on shots that *could* be blocked (Goals, Saves, Misses, Blocks).
    # The 'is_blocked' target is 1 for blocks, 0 for unblocked attempts.
    
    valid_events = ['shot-on-goal', 'missed-shot', 'blocked-shot', 'goal']
    df_eval = df_processed[df_processed['event'].isin(valid_events)].copy()
    
    y_true = (df_eval['event'] == 'blocked-shot').astype(int)
    
    # X features
    # We need to transform categorical cols same as training
    # The XGBNested class has `_prepare_df`.
    
    # We can try to use the `block_model` directly if we match the columns.
    # Or use `model.predict_proba_block(X)` if that method exists?
    # No, usually not exposed.
    
    # Let's rely on the internal feature preparation if possible, or manually replicate.
    # The simplest way is to use `model._prepare_df` then predict with `block_model`.
    
    X_prepared = model._prepare_df(df_eval)
    
    # Filter columns to only those expected by block model
    feat_cols = model.config_block.feature_cols
    
    # Ensure cols exist
    missing = [c for c in feat_cols if c not in X_prepared.columns]
    if missing:
        print(f"Missing columns for analysis: {missing}")
        # Add NaNs
        for c in missing:
            X_prepared[c] = np.nan
            
    X_input = X_prepared[feat_cols]
    
    print("Predicting P(Blocked)...")
    y_pred_prob = block_model.predict_proba(X_input)[:, 1]
    
    auc = roc_auc_score(y_true, y_pred_prob)
    print(f"\nEvaluation on Default Dataset ({len(y_true)} shots):")
    print(f"Blocked Shot Model AUC: {auc:.4f}")
    
    # Confusion Matrix at 0.25 threshold (just for context)
    y_pred = (y_pred_prob > 0.25).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix (Threshold 0.25):")
    print(cm)
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    analyze_block_model()
