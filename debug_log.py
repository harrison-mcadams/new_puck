import sys
import os
import joblib
import traceback

# Add CWD
sys.path.insert(0, os.getcwd())

log_file = "debug_error.log"

with open(log_file, "w") as f:
    try:
        f.write(f"CWD: {os.getcwd()}\n")
        import puck.mixed_effects
        
        f.write("Attempting joblib load...\n")
        model = joblib.load("analysis/xgs/mixed_effects_v2.joblib")
        f.write("Load Success!\n")
        f.write(f"Type: {type(model)}\n")
        f.write(f"Attributes: {list(model.__dict__.keys())}\n")
        
        if hasattr(model, 'models_'):
             f.write(f"models_ FOUND. Type: {type(model.models_)}\n")
             f.write(f"Keys: {list(model.models_.keys())}\n")
             # Inspect one value
             if len(model.models_) > 0:
                 k = list(model.models_.keys())[0]
                 val = model.models_[k]
                 f.write(f"Value for {k}: {type(val)}\n")
                 f.write(f"dim: {getattr(val, 'input_dim_', 'N/A')}\n")
                 teams = getattr(val, 'teams_', [])
                 feats = getattr(val, 'feature_names', [])
                 coef = getattr(val, 'coef_', [])
                 f.write(f"n_teams: {len(teams)}\n")
                 f.write(f"n_feats: {len(feats)}\n")
                 f.write(f"n_coef: {len(coef) if hasattr(coef, '__len__') else 'scalar'}\n")
                 f.write(f"Calc: n_teams * n_feats = {len(teams) * len(feats)}\n")
                 f.write(f"Calc: 2 * n_teams * n_feats = {2 * len(teams) * len(feats)}\n")
        else:
             f.write("models_ NOT FOUND.\n")
             
    except Exception as e:
        f.write("Error!\n")
        f.write(traceback.format_exc())
