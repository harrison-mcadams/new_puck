"""train_xgboost_nested.py

Script to train the Nested XGBoost xG model on full historical data.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_nested, fit_xgs
from puck import config as puck_config

def main():
    print("--- Training Nested XGBoost Model (Full History) ---")
    
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    
    print("Loading all seasons data...")
    df_raw = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    
    print(f"Loaded {len(df_raw)} rows.")
    if len(df_raw) == 0:
        print("No rows found. Aborting.")
        return

    # 2. Define Save Paths
    save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_nested.joblib')
    out_dir = Path(puck_config.ANALYSIS_DIR) / 'xgboost_nested_xgs'
    
    # 3. Call Consolidated Routine
    print(f"Fitting model and saving to {save_path}...")
    fit_xgboost_nested.XGBNestedXGClassifier.train(df_raw, save_path=save_path, out_dir=str(out_dir), verbose=True)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model: {save_path}")
    print(f"Diagnostics: {out_dir}")

if __name__ == "__main__":
    main()
