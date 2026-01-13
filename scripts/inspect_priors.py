import joblib
import pandas as pd
import sys
import os

sys.path.append(os.getcwd())

model_path = "analysis/xgs/xg_model_nested.joblib"
if not os.path.exists(model_path):
    print("Model not found.")
    sys.exit(1)

model = joblib.load(model_path)

print("--- NestedGLM Priors ---")
if hasattr(model, 'shot_type_priors_'):
    print(model.shot_type_priors_)
else:
    print("No shot_type_priors_ found.")

print("\n--- Model Coefficients (Accuracy) ---")
# Check if we can inspect coefficients to see what 'Unknown' (zero vector) implies
try:
    # Access the logistic regression step
    clf = model.model_acc.named_steps['clf']
    print(f"Intercept: {clf.intercept_}")
    
    # This is rough, just checking if valid
except Exception as e:
    print(f"Could not inspect coeffs: {e}")
