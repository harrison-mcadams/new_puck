import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

def main():
    model_path = Path("analysis/xgs/joint_mixed_effects.joblib")
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path}...")
    try:
        # Ensure current directory is in path for pickle imports
        sys.path.append(os.getcwd())
        model = joblib.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("\n--- Model Inspection ---")
    print(f"Feature Set: {model.feature_set}")
    
    # Check for legacy vs new
    off_models = getattr(model, 'off_models_', {})
    legacy_models = getattr(model, 'models_', {})
    
    with open("coef_results.txt", "w") as f:
        f.write(f"Final Feature Names ({len(getattr(model, 'final_feature_names_', []))}):\n")
        # f.write(str(getattr(model, 'final_feature_names_', 'Not found')) + "\n")
        
        target_model = None
        if '5v5' in off_models:
            f.write("\n--- Inspecting New Style 5v5 Offense Model ---\n")
            target_model = off_models['5v5']
        elif '5v5' in legacy_models:
            f.write("\n--- Inspecting Legacy Style 5v5 Model ---\n")
            target_model = legacy_models['5v5']
            
        if target_model:
            coefs = pd.DataFrame()
            if hasattr(target_model, 'get_coefficients'):
                 coefs = target_model.get_coefficients()
            elif hasattr(target_model, 'coef_'):
                 # Standard sklearn - ensuring feature names match length
                 feats = getattr(model, 'final_feature_names_', [])
                 if len(feats) == len(target_model.coef_[0]):
                     coefs = pd.DataFrame({'feature': feats, 'coef': target_model.coef_[0]})
                 else:
                     f.write(f"Mismatch: Features {len(feats)} vs Coefs {len(target_model.coef_[0])}\n")
            elif hasattr(target_model, 'model') and hasattr(target_model.model, 'coef_'):
                 feats = getattr(model, 'final_feature_names_', [])
                 vals = target_model.model.coef_[0]
                 if len(feats) == len(vals):
                     coefs = pd.DataFrame({'feature': feats, 'coef': vals})
                 else:
                     f.write(f"Mismatch in wrapped model: Features {len(feats)} vs Coefs {len(vals)}\n")

            if not coefs.empty and 'coef' in coefs.columns:
                coefs['abs_coef'] = coefs['coef'].abs()
                top_coefs = coefs.sort_values('abs_coef', ascending=False).head(50)
                
                f.write("\nTop 50 Coefficients:\n")
                f.write(top_coefs[['feature', 'coef']].to_string() + "\n")
            
                suspicious = ['time_since_last_event', 'is_rebound', 'is_rush']
                f.write("\nSuspicious Features:\n")
                for feat in suspicious:
                    matches = coefs[coefs['feature'].str.contains(feat, case=False)]
                    if not matches.empty:
                        f.write(matches[['feature', 'coef']].to_string() + "\n")
        else:
            f.write("No 5v5 model found to inspect.\n")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
