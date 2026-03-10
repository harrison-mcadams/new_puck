"""train_glm.py

Script to train the Non-Nested Polynomial Logistic Regression (GLM) xG model.
Refactored to use consolidated routines in puck/fit_glm.py.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from puck import fit_glm, fit_xgs

def main():
    print("--- Training Non-Nested GLM (Poly/Tensor) Model ---")
    
    # 1. Load Data
    print("Loading data from ALL seasons...")
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'
    
    if data_dir.exists():
        df_raw = fit_xgs.load_all_seasons_data(base_dir=str(data_dir))
    else:
        df_raw = fit_xgs.load_data()
        
    print(f"Loaded {len(df_raw)} rows.")

    # 2. Call Consolidated Routine
    fit_glm.train_glm(df_raw, verbose=True)

    print("\n=== TRAINING COMPLETE ===")

if __name__ == "__main__":
    main()
