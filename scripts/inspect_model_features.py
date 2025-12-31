
import sys
import os
import joblib
import pandas as pd
from pathlib import Path

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from puck import config, fit_nested_xgs, fit_xgboost_nested

# --- HACK FOR JOBLIB LOADING ---
if not hasattr(sys.modules['__main__'], 'NestedXGClassifier'):
    setattr(sys.modules['__main__'], 'NestedXGClassifier', fit_nested_xgs.NestedXGClassifier)
if not hasattr(sys.modules['__main__'], 'LayerConfig'):
    setattr(sys.modules['__main__'], 'LayerConfig', fit_nested_xgs.LayerConfig)
if hasattr(fit_xgboost_nested, 'XGBNestedXGClassifier'):
    if not hasattr(sys.modules['__main__'], 'XGBNestedXGClassifier'):
        setattr(sys.modules['__main__'], 'XGBNestedXGClassifier', fit_xgboost_nested.XGBNestedXGClassifier)

def main():
    model_path = Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_nested.joblib'
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Model Type: {type(model).__name__}")
    
    # Check Finish Model Features
    if hasattr(model, 'config_finish'):
        print("\n--- Finish Model Features ---")
        feats = model.config_finish.feature_cols
        print(f"Total Features: {len(feats)}")
        print("Features:", feats)
        
        shot_type_feats = [f for f in feats if 'shot_type' in f]
        print(f"\nShot Type Features found ({len(shot_type_feats)}):")
        for f in shot_type_feats:
            print(f"  - {f}")
            
    # Check Shot Type Priors
    if hasattr(model, 'shot_type_priors'):
        print("\n--- Shot Type Priors ---")
        if model.shot_type_priors:
            print(model.shot_type_priors)
        else:
            print("None or Empty")

if __name__ == "__main__":
    main()
