
import pandas as pd
import os

csv_path = "data/20182019/20182019_df.csv"
if not os.path.exists(csv_path):
    print("CSV not found")
    exit()

df = pd.read_csv(csv_path)
df['diff_x'] = (df['x'] - df['x_adj']).abs()

total = len(df)
adjusted = len(df[df['diff_x'] > 0.001])
over_5ft = len(df[df['diff_x'] >= 5])
over_10ft = len(df[df['diff_x'] >= 10])

print(f"2018-2019 Season Analysis:")
print(f"Total events: {total}")
print(f"Events with any adjustment: {adjusted} ({adjusted/total*100:.1f}%)")
print(f"Events shifted by >= 5ft: {over_5ft} ({over_5ft/total*100:.1f}%)")
print(f"Events shifted by >= 10ft: {over_10ft} ({over_10ft/total*100:.1f}%)")

# Breakdown by home team (assuming we can infer it or just looking at distributions)
# We don't have home_team in the CSV column, but we know the large ones are Tampa.
# Let's see the most common bias values
print("\nMost common X adjustment values (rounded):")
print(df[df['diff_x'] > 0.1]['diff_x'].round(0).value_counts().head(10))
