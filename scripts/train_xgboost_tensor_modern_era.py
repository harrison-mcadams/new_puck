"""train_xgboost_tensor_modern_era.py

Training script for the Pure spatial XGBoost model on the full Modern Era (20202021+).
Aggregates all available modern seasons and generates full dashboard artifacts.
"""

import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_tensor, config
from scripts.evaluate_predictive_power import DataUtils

def main():
    print("============================================================")
    print("MODERN ERA TRAINING: Pure Spatial XGBoost (Tensor)")
    print("============================================================")

    # 1. Discover Modern Era Seasons
    all_seasons = DataUtils.get_available_seasons()
    modern_seasons = sorted([s for s in all_seasons if int(s) >= 20202021])
    
    if not modern_seasons:
        print("Error: No modern era seasons (20202021+) found in data/ directory.")
        return

    print(f"Loading and aggregating {len(modern_seasons)} seasons: {modern_seasons}")

    # 2. Load and Aggregate
    dfs = []
    for s in modern_seasons:
        try:
            from puck import analyze
            csv_path = analyze.locate_season_csv(s)
            df_s = pd.read_csv(csv_path)
            print(f"  [OK] {s}: {len(df_s)} events loaded.")
            dfs.append(df_s)
        except Exception as e:
            print(f"  [Error] Failed to load {s}: {e}")

    if not dfs:
        print("Error: No data loaded.")
        return

    df_modern = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Modern Era Dataset: {len(df_modern)} events.")

    # [VERIFY GOLD STREAM]
    print("\nVerifying Blocked Shot Enrichment...")
    blocks = df_modern[df_modern['event'] == 'blocked-shot']
    if len(blocks) > 0:
        counts = blocks['shot_type'].value_counts(dropna=False)
        print("Blocked Shot Type Distribution:")
        print(counts)
        unknown_count = counts.get('Unknown', 0) + counts.get('unknown', 0) + blocks['shot_type'].isna().sum()
        unknown_pct = 100.0 * unknown_count / len(blocks)
        print(f"Unknown Fraction: {unknown_pct:.1f}%")
        if unknown_pct > 90:
            print("WARNING: Data stream appears UNENRICHED. Shot type responsiveness will be minimal.")
        else:
            print("SUCCESS: Enriched data stream detected.")
    else:
        print("Warning: No blocked shots found in dataset.")

    # 3. Train
    # Use the train_xgboost_alternate helper which triggers model_summary generation
    save_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / 'xg_model_xgboost_tensor_modern_era.joblib')
    out_dir = str(Path(config.ANALYSIS_DIR) / 'xgboost_tensor_xgs_modern')
    
    print("\nStarting training session...")
    fit_xgboost_tensor.train_xgboost_tensor(
        df_modern,
        save_path=save_path,
        out_dir=out_dir,
        verbose=True,
        use_balancing=False,
        apply_html_enrichment=True
    )
    
    print("\n============================================================")
    print("MODERN ERA TRAINING COMPLETE")
    print(f"Model: {save_path}")
    print(f"Dashboard: {out_dir}/xg_model_xgboost_tensor_modern_era_dashboard.html")
    print("============================================================")

if __name__ == "__main__":
    main()
