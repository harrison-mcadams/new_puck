import pandas as pd
import sys
import os

sys.path.append(os.getcwd())
from puck import data_pipeline, fit_xgs

print("--- Investigating Shot Type Data Loss ---")

# Load raw-ish data
try:
    # Load directly to see state before preprocessing
    df_raw = fit_xgs.load_data(years=[20232024])
    print(f"Loaded {len(df_raw)} rows.")
except:
    print("Could not load via fit_xgs, trying direct CSV.")
    df_raw = pd.read_csv('data/20232024/20232024_df.csv')

if 'shot_type' in df_raw.columns:
    print("\nRaw 'shot_type' Value Counts:")
    print(df_raw['shot_type'].value_counts(dropna=False).head(10))
    print(f"Raw NaNs: {df_raw['shot_type'].isna().sum()}")
else:
    print("Raw data missing 'shot_type' column!")

# Run Preprocessing steps one by one to see where it drops
print("\n--- Running Preprocessing Pipeline ---")
df_proc = df_raw.copy()

# Step 1-4 (Standard stuff)
# ... skipping deep details, just running main wrapper
df_proc = data_pipeline.preprocess_features(df_proc, is_training=False, apply_imputation=False)

print("\nProcessed 'shot_type' Value Counts:")
if 'shot_type' in df_proc.columns:
    print(df_proc['shot_type'].value_counts(dropna=False).head(10))
else:
    print("Processed data missing 'shot_type' column!")

# Check 'Unknown' vs lowercase 'unknown'
unknowns = df_proc[df_proc['shot_type'].astype(str).str.lower() == 'unknown']
print(f"\nTotal Normalized 'unknown': {len(unknowns)}")
print(f"Percentage: {len(unknowns)/len(df_proc):.1%}")

# Sample some rows
print("\nSample 'Unknown' Rows:")
if len(unknowns) > 0:
    print(unknowns[['event', 'shot_type', 'x', 'y']].head())
