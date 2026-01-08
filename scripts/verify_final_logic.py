
import pandas as pd
import os

# Check Tampa 2018-2019 (Lightning had the large positive bias at 85)
csv_path = "data/20182019/20182019_df.csv"
df = pd.read_csv(csv_path)

# Filter for shots near X=85 (absolute)
# We need to find Tampa home games.
# In the CSV, we don't have home_team_name, but we can look for the specific adjustment value.
# Adjustment was 13.9xx (around 14).
# So x - x_adj should be ~14 (Bias).
df['diff_x'] = df['x'] - df['x_adj']

large_adjustments = df[df['diff_x'].abs() > 10]

if not large_adjustments.empty:
    sample = large_adjustments.iloc[0]
    print(f"Found large adjustment:")
    print(f"X: {sample['x']} -> X_adj: {sample['x_adj']} (Diff: {sample['diff_x']})")
    
    # Validation: If it was at 85 and diff is +14, then x_adj should be 71.
    # Current code: x_adj = abs_x - delta_x
    # If delta_x is 14, then 85 - 14 = 71. Correct.
    if sample['x_adj'] < sample['x']:
         print("SUCCESS: Adjustment moved shot AWAY from net (Subtraction applied).")
    else:
         print("FAILURE: Adjustment moved shot TOWARDS/BEHIND net (Addition still applied?)")
else:
    print("No large adjustments found in 20182019. Check if they were applied.")
