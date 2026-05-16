"""train_xgboost_tensor_full_history.py

Training script for the Pure spatial XGBoost model on the FULL HISTORICAL dataset (2010-2025).
Aggregates all available seasons and generates full dashboard artifacts.
"""

import sys
import os
import pandas as pd
from pathlib import Path
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import fit_xgboost_tensor, config, analyze
from scripts.evaluate_predictive_power import DataUtils

def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Tensor model on full history.")
    parser.add_argument('--out-suffix', type=str, default='full_history', help="Suffix for model and dashboard names")
    parser.add_argument('--min-season', type=int, default=20102011, help="Minimum season to include")
    args = parser.parse_args()

    print("============================================================")
    print(f"FULL HISTORY TRAINING: Pure Spatial XGBoost (Tensor)")
    print("============================================================")

    # 1. Discover All Seasons
    all_seasons = DataUtils.get_available_seasons()
    # Filter by minimum season
    target_seasons = sorted([s for s in all_seasons if int(s) >= args.min_season])
    
    if not target_seasons:
        print(f"Error: No seasons found in data/ directory matching >= {args.min_season}.")
        return

    print(f"Loading and aggregating {len(target_seasons)} seasons: {target_seasons}")

    # 2. Load and Aggregate
    dfs = []
    for s in target_seasons:
        try:
            csv_path = analyze.locate_season_csv(s)
            if csv_path:
                df_s = pd.read_csv(csv_path, low_memory=False)
                # Ensure season column is set as a string for categorical treatment
                df_s['season'] = str(s)
                dfs.append(df_s)
                print(f"  [OK] {s}: {len(df_s)} events loaded.")
            else:
                print(f"  [MISSING] {s}: CSV not found.")
        except Exception as e:
            print(f"  [Error] Failed to load {s}: {e}")

    if not dfs:
        print("Error: No data loaded.")
        return

    df_full = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal Dataset: {len(df_full)} events.")

    # [VERIFY GOLD STREAM]
    print("\nVerifying Blocked Shot Enrichment...")
    blocks = df_full[df_full['event'] == 'blocked-shot']
    if len(blocks) > 0:
        counts = blocks['shot_type'].value_counts(dropna=False)
        print("Blocked Shot Type Distribution (Sample):")
        print(counts.head(10))
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
    model_name = f"xg_model_xgboost_tensor_{args.out_suffix}"
    save_path = str(Path(config.ANALYSIS_DIR) / 'xgs' / f"{model_name}.joblib")
    out_dir = str(Path(config.ANALYSIS_DIR) / f"xgboost_tensor_xgs_{args.out_suffix}")
    
    print(f"\nStarting training session for '{model_name}'...")
    fit_xgboost_tensor.train_xgboost_tensor(
        df_full,
        save_path=save_path,
        out_dir=out_dir,
        verbose=True,
        use_balancing=False,
        apply_html_enrichment=False # We assume data is already enriched
    )
    
    print("\n============================================================")
    print("FULL HISTORY TRAINING COMPLETE")
    print(f"Model: {save_path}")
    print(f"Dashboard: {out_dir}/{model_name}_dashboard.html")
    print("============================================================")

if __name__ == "__main__":
    main()
