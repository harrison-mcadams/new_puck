
import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_tensor, config, analyze

def main():
    print("============================================================")
    print("FAST REBOUND FIX TRAINING: Pure Spatial XGBoost (Tensor)")
    print("============================================================")

    season = '20252026'
    try:
        csv_path = analyze.locate_season_csv(season)
        df = pd.read_csv(csv_path)
        print(f"Loaded {season}: {len(df)} events.")
    except Exception as e:
        print(f"Failed to load {season}: {e}")
        return

    # Sample to speed up
    if len(df) > 100000:
        df = df.sample(100000, random_state=42)
        print(f"Sampled to 100,000 events.")

    # 3. Train
    # We save to a TEMPORARY path first to verify
    save_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_PARITY.joblib')
    out_dir = str(Path(config.ANALYSIS_DIR) / 'xgboost_tensor_xgs_PARITY')
    
    print("\nStarting training session...")
    fit_xgboost_tensor.train_xgboost_tensor(
        df,
        save_path=save_path,
        out_dir=out_dir,
        verbose=True,
        use_balancing=False,
        apply_html_enrichment=False
    )
    
    print("\n============================================================")
    print("FAST TRAINING COMPLETE")
    print(f"Model: {save_path}")
    print("============================================================")

if __name__ == "__main__":
    main()
