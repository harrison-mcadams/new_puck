
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score
from pathlib import Path
import joblib

# Add project root
sys.path.append(str(Path.cwd()))
from puck import fit_xgs, data_pipeline

def verify_block_auc():
    print("Loading data sample...")
    df = fit_xgs.load_data().sample(10000, random_state=42)
    
    print("Preprocessing...")
    df = data_pipeline.preprocess_features(df, is_training=False, verbose=False)
    
    print("Loading model...")
    model_path = Path('analysis/xgs/xg_model_nested.joblib')
    clf = joblib.load(model_path)
    
    print(f"Model type: {type(clf).__name__}")
    
    # Predict Block Prob
    if 'event' not in df.columns:
        print("Error: 'event' column missing")
        return
        
    y_true = (df['event'] == 'blocked-shot').astype(int)
    y_prob = clf.predict_proba_layer(df, 'block')
    
    auc = roc_auc_score(y_true, y_prob)
    print(f"\n--- VERIFICATION RESULT ---")
    print(f"Block Model AUC: {auc:.4f}")
    
    if auc > 0.95:
        print("FAILURE: Block AUC is suspiciously high (>0.95). Leakage likely persists.")
    elif auc < 0.5:
        print("WARNING: Block AUC is worse than random (<0.5). Model might be broken.")
    else:
        print("SUCCESS: Block AUC is in realistic range.")

if __name__ == "__main__":
    verify_block_auc()
