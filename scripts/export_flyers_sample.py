
import sys
import pandas as pd
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgs, data_pipeline, enrich

def main():
    print("Loading 2025-2026 data...")
    # Load specific season file to be fast
    data_path = Path("data/20252026/20252026_df.csv")
    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} rows.")
        
        # Filter for Flyers (PHI)
        # Check team_abbrev, home_abb, away_abb
        mask_phi = (df['home_abb'] == 'PHI') | (df['away_abb'] == 'PHI')
        df_phi = df[mask_phi].copy()
        print(f"Found {len(df_phi)} Flyers events.")
        
        if df_phi.empty:
            print("No Flyers games found.")
            return

        # Find most recent game_id
        last_game_id = df_phi['game_id'].max()
        print(f"Most recent Flyers Game ID: {last_game_id}")
        
        df_game = df_phi[df_phi['game_id'] == last_game_id].copy()
        
        print(f"Processing Game {last_game_id} ({len(df_game)} events)...")
        
        # Enrich Player Data (fetch shoots/catches if missing)
        print("Enriching player data (shoots_catches)...")
        enricher = enrich.PlayerEnricher()
        df_game = enricher.enrich_dataframe(df_game)
        
        # Run Pipeline
        # Enable filtering to show the "clean" output the model would see
        df_processed = data_pipeline.preprocess_features(
            df_game,
            is_training=False, # technically verifying inference flow features, but let's use filtering=True as requested? 
            # User said "load up a df ... pass it through this"
            # It's safer to use the inference settings (filtering=False) unless they want training data look.
            # actually, they asked to "examine" it. usually that implies seeing the model features.
            # I will apply filtering=False so they can see EVERYTHING, but with features calculated.
            # They verified filtering earlier. The crucial part now is the FEATURE ENGINEERING.
            apply_filtering=False,
            apply_arena_adjustments=True,
            apply_imputation=True,
            verbose=True
        )
        
        # Save output
        out_dir = Path("analysis")
        out_dir.mkdir(exist_ok=True)
        out_file = out_dir / "flyers_processed_sample.csv"
        
        # Select relevant columns to make manual inspection easier
        cols = ['game_id', 'event', 'period', 'period_time', 'home_abb', 'away_abb', 'team_id',
                'x', 'y', 'x_adj', 'y_adj', 'distance', 'angle_deg', 
                'is_rebound', 'is_rush', 'last_event_type', 'shot_type',
                'shoots_catches', 'shooter_role']
        # Add others if they exist
        for c in ['imputed_x', 'imputed_y', 'block_x', 'block_y']:
            if c in df_processed.columns: cols.append(c)
            
        # grab all columns just in case, but sort interesting ones first
        final_cols = [c for c in cols if c in df_processed.columns]
        remaining = [c for c in df_processed.columns if c not in final_cols]
        
        df_processed[final_cols + remaining].to_csv(out_file, index=False)
        print(f"Saved processed game data to: {out_file.resolve()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
