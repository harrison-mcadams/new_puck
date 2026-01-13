import pandas as pd
import sys
import os

# Adjust path to find puck package
sys.path.append(os.getcwd())
from puck import data_pipeline

# Load a sample or the full dataset
# Assuming 'data/all_pbp.parquet' or similar exists, based on previous context.
# If not, I'll use data_pipeline.load_data() if available.

try:
    print("Loading data...")
    df = data_pipeline.load_data(years=[20232024], processed=True) # Load recent data
    if 'shot_type' in df.columns:
        print("Unique shot_types:", df['shot_type'].unique())
        print("Value Counts:\n", df['shot_type'].value_counts(dropna=False))
    else:
        print("shot_type column missing.")
except Exception as e:
    print(f"Error loading data: {e}")
