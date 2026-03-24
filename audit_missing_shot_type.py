import pandas as pd
import numpy as np
import os

# Updated to the correct path found via list_dir
data_path = 'data/20202021/20202021_df.csv'
if not os.path.exists(data_path):
    print(f"Data path {data_path} not found. Aborting.")
    exit()

df = pd.read_csv(data_path)

# In our raw data, 'event' is the type.
# shot_type is often 'secondary_type' in the raw feed, 
# but may have been renamed to 'shot_type' in df_df.csv.
print(f"Columns: {df.columns.tolist()}")

shot_col = 'shot_type' if 'shot_type' in df.columns else 'secondary_type'
if shot_col not in df.columns:
     # Try 'type'
     shot_col = 'secondary_type' # Default check

is_blocked = (df['event'] == 'blocked-shot')

blocked_shots = df[is_blocked]
unblocked_shots = df[~is_blocked & (df['event'].isin(['shot-on-goal', 'missed-shot', 'goal']))]

print(f"Total rows: {len(df)}")
print(f"Total blocked: {len(blocked_shots)}")
print(f"Total unblocked: {len(unblocked_shots)}")

def audit_missing(sub_df, label):
    if len(sub_df) == 0:
        print(f"{label}: No data")
        return
    
    # Check if column exists
    if shot_col not in sub_df.columns:
        print(f"{label}: Column '{shot_col}' missing entirely.")
        return

    missing_count = sub_df[shot_col].isna().sum()
    perc = (missing_count / len(sub_df)) * 100
    print(f"{label}: Missing '{shot_col}' = {missing_count} ({perc:.2f}%)")
    
    # Also check for 'nan' strings or unknowns
    str_ser = sub_df[shot_col].astype(str).str.upper()
    unknowns = sub_df[str_ser.isin(['', 'UNKNOWN', 'NONE', 'NAN'])].shape[0]
    print(f"{label}: 'Unknown' '{shot_col}' = {unknowns} ({(unknowns/len(sub_df))*100:.2f}%)")
    
    # Value counts
    print(f"{label} {shot_col} counts:\n{sub_df[shot_col].value_counts(dropna=False).head(10)}")

audit_missing(blocked_shots, "BLOCKED")
audit_missing(unblocked_shots, "UNBLOCKED")
