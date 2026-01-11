
import joblib
import sys
import os

sys.path.append(os.getcwd())

def main():
    try:
        clf = joblib.load('analysis/xgs/xg_model_nested_all.joblib')
        print(f"Loaded {type(clf).__name__}")
        
        if hasattr(clf, 'features'):
            print(f"Model Features List: {clf.features}")
        else:
            print("No .features attribute found on classifier.")

        # Access internal model
        if hasattr(clf, 'model_finish'):
            inner = clf.model_finish
            print(f"Finish model type: {type(inner).__name__}")
            
            if hasattr(inner, 'feature_names_in_'):
                print("Features:", inner.feature_names_in_)
            elif hasattr(inner, 'get_booster'):
                print("Booster Features:", inner.get_booster().feature_names)
            else:
                print("Could not find feature names on inner model.")
                
            # Check params for Depth
            if hasattr(inner, 'get_params'):
                params = inner.get_params()
                print("Max Depth:", params.get('max_depth', 'Unknown'))
                print("N Estimators:", params.get('n_estimators', 'Unknown'))
        else:
            print("No .model attribute found.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
