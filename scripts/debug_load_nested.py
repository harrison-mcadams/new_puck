
import sys
import os
import joblib
import traceback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs

def test_load():
    model_path = 'analysis/xgs/xg_model_nested.joblib'
    print(f"Attempting to load {model_path}...")
    try:
        clf = joblib.load(model_path)
        print("Success!")
        print(f"Type: {type(clf)}")
        if isinstance(clf, fit_nested_xgs.NestedXGClassifier):
            print("Confirmed instance of NestedXGClassifier")
        else:
            print("Warning: Instance type mismatch")
            
    except Exception as e:
        print(f"Failed to load: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
