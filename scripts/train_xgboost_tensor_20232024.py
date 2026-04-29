"""train_xgboost_tensor_20232024.py

Training script for the Pure spatial XGBoost model on the 2023-2024 season.
Generates full dashboard and summary artifacts.
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_tensor, config

def main():
    print("Loading 2023-2024 data for training...")
    data_path = Path(config.DATA_DIR) / '20232024.csv'
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Save directory
    save_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_20232024.joblib')
    out_dir = str(Path(config.ANALYSIS_DIR) / 'xgboost_tensor_xgs')
    
    print("Starting training...")
    fit_xgboost_tensor.train_xgboost_tensor(
        df, 
        save_path=save_path, 
        out_dir=out_dir,
        verbose=True
    )
    
    print("\nTraining Complete.")
    print(f"Model saved to: {save_path}")
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
