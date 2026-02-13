import sys
import os
sys.path.append(os.getcwd())

try:
    import puck.mixed_effects
    print(f"puck.mixed_effects: {puck.mixed_effects}")
    print(f"Has GameMixedEffectsXG: {hasattr(puck.mixed_effects, 'GameMixedEffectsXG')}")
except Exception as e:
    print(f"Import failed: {e}")

try:
    import joblib
    print("Loading model...")
    model = joblib.load("analysis/xgs/mixed_effects_v2.joblib")
    print("Model loaded successfully.")
    print(f"Type: {type(model)}")
except Exception as e:
    print(f"Model load failed: {e}")
