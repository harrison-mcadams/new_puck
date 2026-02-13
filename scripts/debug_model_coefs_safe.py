import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

def main():
    with open("coef_debug_safe.txt", "w") as f:
        model_path = Path("analysis/xgs/mixed_effects_v2.joblib")
        if not model_path.exists():
            f.write(f"Model not found: {model_path}\n")
            return

        f.write(f"Loading model from {model_path}...\n")
        try:
            sys.path.append(os.getcwd())
            model = joblib.load(model_path)
            f.write("Model loaded successfully.\n")
        except Exception as e:
            f.write(f"Failed to load model: {e}\n")
            return

        f.write(f"Model Type: {type(model)}\n")
        f.write(f"Model Dir: {dir(model)}\n")
        
        # Check attributes
        off_models = getattr(model, 'off_models_', {})
        legacy_models = getattr(model, 'models_', {})
        
        f.write(f"off_models_ keys: {list(off_models.keys())}\n")
        f.write(f"models_ keys: {list(legacy_models.keys())}\n")
        
        target = None
        if '5v5' in off_models:
            target = off_models['5v5']
            f.write("Found 5v5 in off_models_\n")
        elif '5v5' in legacy_models:
            target = legacy_models['5v5']
            f.write("Found 5v5 in models_\n")
            
        if target:
            f.write(f"Target Model Type: {type(target)}\n")
            f.write(f"Target Model Dir: {dir(target)}\n")
            
            # Inspect coef_
            if hasattr(target, 'coef_'):
                coefs = target.coef_
                if isinstance(coefs, np.ndarray):
                    f.write(f"Coef Shape: {coefs.shape}\n")
                    f.write(f"Coef Stats:\n")
                    f.write(f"  Min: {coefs.min()}\n")
                    f.write(f"  Max: {coefs.max()}\n")
                    f.write(f"  Mean: {coefs.mean()}\n")
                    f.write(f"  Std: {coefs.std()}\n")
                    f.write(f"  1%: {np.percentile(coefs, 1)}\n")
                    f.write(f"  99%: {np.percentile(coefs, 99)}\n")
                    
                    # Check for large values
                    large_mask = np.abs(coefs) > 3.0
                    n_large = np.sum(large_mask)
                    f.write(f"  Count |coef| > 3.0: {n_large}\n")
                    
                    if n_large > 0:
                        # If meaningful vector, show indices
                        # We suspect this is a flattened vector of all params
                        # We can't easily map to features without knowing the layout
                        # But we can see IF they exist.
                        f.write(f"  Max Abs Indices: {np.argsort(np.abs(coefs))[-10:]}\n")
                        f.write(f"  Max Abs Values: {coefs[np.argsort(np.abs(coefs))[-10:]]}\n")

            # Check feature names for suspicious ones
            feats = getattr(model, 'final_feature_names_', [])
            f.write(f"\nFeature Analysis ({len(feats)}):\n")
            suspicious = ['time_since_last_event', 'is_rebound', 'is_rush', 'period_seconds']
            for s in suspicious:
                matches = [feat for feat in feats if s in feat]
                if matches:
                    f.write(f"Found suspicious features matching '{s}': {matches}\n")

if __name__ == "__main__":
    main()
