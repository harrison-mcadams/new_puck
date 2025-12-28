
import sys
import os
import pandas as pd
import joblib

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from puck import fit_nested_xgs, analyze

def verify_training():
    print("--- Verifying fit_nested_xgs.py ---")
    try:
        # Run main to train and save
        fit_nested_xgs.main()
        
        # Check if file exists
        model_path = 'analysis/xgs/xg_model_nested.joblib'
        if os.path.exists(model_path):
            print(f"SUCCESS: {model_path} created.")
        else:
            print(f"FAILURE: {model_path} not found.")
            return False
            
        # Try loading it back
        clf = joblib.load(model_path)
        print(f"SUCCESS: Loaded model: {type(clf)}")
        return True
    except Exception as e:
        print(f"FAILURE in training: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_analyze_loading():
    print("\n--- Verifying analyze._predict_xgs ---")
    try:
        # Create dummy dataframe
        df = pd.DataFrame({
            'event': ['shot-on-goal', 'blocked-shot', 'missed-shot'],
            'distance': [10.0, 20.0, 30.0],
            'angle_deg': [0.0, 10.0, 20.0],
            'game_state': ['5v5', '5v5', '5v5'],
            'is_net_empty': [0, 0, 0],
            'shot_type': ['Wrist Shot', 'Unknown', 'Slap Shot']
        })
        
        # Predict
        print("Calling _predict_xgs...")
        df_pred, clf, meta = analyze._predict_xgs(df)
        
        if 'xgs' in df_pred.columns:
            print("SUCCESS: 'xgs' column added.")
            print(df_pred[['event', 'xgs']])
        else:
            print("FAILURE: 'xgs' column missing.")
            return False
            
        # Check if it used the nested model
        # The class name should be NestedXGClassifier
        print(f"Loaded CLF: {type(clf)}")
        if 'NestedXGClassifier' in str(type(clf)):
             print("SUCCESS: Loaded NestedXGClassifier by default.")
        else:
             print(f"WARNING: Loaded {type(clf)}, expected NestedXGClassifier.")
             
        return True
    except Exception as e:
        print(f"FAILURE in analyze loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    v1 = verify_training()
    v2 = verify_analyze_loading()
    
    if v1 and v2:
        print("\nALL CHECKS PASSED")
        sys.exit(0)
    else:
        print("\nCHECKS FAILED")
        sys.exit(1)
