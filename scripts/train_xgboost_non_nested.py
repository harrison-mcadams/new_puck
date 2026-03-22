"""train_xgboost_non_nested.py

Script to train the Non-Nested XGBoost xG model on full historical data.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_non_nested, fit_xgs
from puck import config as puck_config

def main():
    print("--- Training Non-Nested XGBoost Model (Full History) ---")
    
    # 1. Load Data
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    
    print("Loading all seasons data...")
    df_raw = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    
    print(f"Loaded {len(df_raw)} rows.")
    if len(df_raw) == 0:
        print("No rows found. Aborting.")
        return

    # 2. Define Save Path
    save_path = str(Path(puck_config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_non_nested.joblib')
    
    # 3. Call Consolidated Routine
    print(f"Fitting model and saving to {save_path}...")
    fit_xgboost_non_nested.XGBNonNestedXGClassifier.train(df_raw, save_path=save_path, verbose=True)

    print("\n=== TRAINING COMPLETE ===")
    print(f"Model: {save_path}")

if __name__ == "__main__":
    main()
