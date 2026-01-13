import joblib
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

model_path = 'analysis/xgs/xg_model_nested.joblib'
print(f"Loading {model_path}...")
clf = joblib.load(model_path)

print(f"Model Class: {type(clf).__name__}")
print(f"use_splines: {getattr(clf, 'use_splines', 'MISSING')}")
print(f"poly_degree: {getattr(clf, 'poly_degree', 'MISSING')}")

try:
    # Check pipeline
    steps = clf.model_finish.named_steps['preprocessor'].transformers_
    for name, pipe, cols in steps:
        if name == 'num':
            print(f"Numeric Transformation steps: {[s[0] for s in pipe.steps]}")
except Exception as e:
    print(f"Pipeline check failed: {e}")
