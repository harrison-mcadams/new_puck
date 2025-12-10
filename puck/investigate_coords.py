import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def load_sample_data():
    # Load just one file for speed
    import glob
    files = glob.glob('data/**/*.csv', recursive=True)
    df_files = [f for f in files if f.endswith('_df.csv')]
    if not df_files:
        return None
    
    logger.info(f"Loading {df_files[0]}...")
    return pd.read_csv(df_files[0])

def check_coords():
    df = load_sample_data()
    if df is None:
        logger.error("No data found.")
        return

    # Check X distribution
    logger.info("\n--- Coordinate Stats ---")
    logger.info(df[['x', 'y']].describe())
    
    # Check if 'distance' column exists
    if 'distance' in df.columns:
        logger.info("\n--- Existing Distance Stats ---")
        logger.info(df['distance'].describe())
        
        # Check calculation consistency
        # Assume net at 89, 0
        my_dist = np.sqrt((df['x'] - 89)**2 + (df['y'] - 0)**2)
        diff = np.abs(df['distance'] - my_dist)
        
        logger.info("\n--- Diff (My Calc vs Existing) ---")
        logger.info(diff.describe())
        
        # Check if maybe net is at -89?
        my_dist_neg = np.sqrt((df['x'] + 89)**2 + (df['y'] - 0)**2)
        diff_neg = np.abs(df['distance'] - my_dist_neg)
        
        logger.info(f"Mean Diff (Net +89): {diff.mean():.2f}")
        logger.info(f"Mean Diff (Net -89): {diff_neg.mean():.2f}")
        
    else:
        logger.warning("'distance' column not found in raw data.")

if __name__ == "__main__":
    check_coords()
