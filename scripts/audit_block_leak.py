
import sys
from pathlib import Path
# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Load Model
model_path = 'analysis/xgs/xg_model_nested.joblib'
print(f"Loading model from {model_path}...")
clf = joblib.load(model_path)

# Check Feature Importances for Block Model
print("\n--- Block Model Feature Importances ---")
if clf.model_block:
    feats = clf.config_block.feature_cols
    imps = clf.model_block.feature_importances_
    feat_imp = pd.DataFrame({'feature': feats, 'importance': imps}).sort_values('importance', ascending=False)
    print(feat_imp)
else:
    print("Block model not accessible.")

# Load Data Sample (Processed)
print("\n--- Loading Data Sample ---")
# We need processed data. We can reuse the debug output or load fresh.
debug_csv = 'analysis/debug_imputation_pipeline.csv'
if Path(debug_csv).exists():
    df = pd.read_csv(debug_csv)
    print(f"Loaded {len(df)} rows from debug csv.")
    
    # Check Distributions of Top Feature
    top_feat = feat_imp.iloc[0]['feature']
    print(f"\nAnalyzing Top Feature: {top_feat}")
    
    df['is_blocked'] = (df['event'] == 'blocked-shot').astype(int)
    
    print("\n[Blocked Shots] Stats:")
    print(df[df['is_blocked']==1][top_feat].describe())
    
    print("\n[Unblocked Shots] Stats:")
    print(df[df['is_blocked']==0][top_feat].describe())
    
    # Check for perfect separation
    # e.g. Count of Unblocked > 60ft vs Blocked > 60ft
    if top_feat == 'distance':
        thresh = 60
        n_block_far = len(df[(df['is_blocked']==1) & (df['distance'] > thresh)])
        n_unblock_far = len(df[(df['is_blocked']==0) & (df['distance'] > thresh)])
        print(f"\nEvents > {thresh}ft:")
        print(f"  Blocked: {n_block_far}")
        print(f"  Unblocked: {n_unblock_far}")
        print(f"  Ratio (Block/Total): {n_block_far / (n_block_far + n_unblock_far + 1e-9):.4f}")

else:
    print("Debug CSV not found. Cannot analyze distributions.")
