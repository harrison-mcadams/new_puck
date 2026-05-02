import pandas as pd
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puck import analyze, data_pipeline, config

def permanent_sweep():
    seasons = ['20202021', '20212022', '20222023', '20232024', '20242025', '20252026']
    
    print("============================================================")
    print("PERMANENT ENRICHMENT SWEEP: HTML Blocked Shot Types")
    print("============================================================")
    
    for season in seasons:
        try:
            path = analyze.locate_season_csv(season)
            print(f"\nProcessing {season}...")
            print(f"  Source: {path}")
            
            df = pd.read_csv(path)
            initial_count = df[df['event'] == 'blocked-shot']['shot_type'].dropna().shape[0]
            total_blocks = df[df['event'] == 'blocked-shot'].shape[0]
            
            print(f"  Initial Enriched Blocks: {initial_count} / {total_blocks}")
            
            if initial_count == total_blocks and total_blocks > 0:
                print(f"  [SKIP] {season} is already 100% enriched.")
                continue
            
            # Enrich
            print(f"  Enriching {season} in parallel...")
            # We call preprocess_features but ONLY for enrichment. 
            # We don't want to flip coordinates in the saved CSV if they are already canonical.
            # Actually, preprocess_features handles canonicality check.
            # But to be safe and ONLY update shot_type, we'll call the enrichment logic directly 
            # or just use preprocess_features and trust it.
            # preprocess_features is safer because it handles the Parallel call correctly.
            df_enriched = data_pipeline.preprocess_features(df, apply_html_enrichment=True, verbose=True)
            
            final_count = df_enriched[df_enriched['event'] == 'blocked-shot']['shot_type'].dropna().shape[0]
            print(f"  Final Enriched Blocks: {final_count} / {total_blocks}")
            
            # Save back
            print(f"  Saving enriched data back to {path}...")
            # Create backup just in case
            backup_path = path + ".bak"
            if not os.path.exists(backup_path):
                import shutil
                shutil.copy2(path, backup_path)
                
            df_enriched.to_csv(path, index=False)
            print(f"  [OK] {season} Updated.")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {season}: {e}")

    print("\n============================================================")
    print("SWEEP COMPLETE")
    print("============================================================")

if __name__ == "__main__":
    permanent_sweep()
