
import pandas as pd
import os

csv_path = "data/20182019/20182019_df.csv"
if not os.path.exists(csv_path):
    print("CSV not found")
    exit()

df = pd.read_csv(csv_path)
df['diff_x'] = (df['x'] - df['x_adj']).abs()

# Filter for significant adjustments
large = df[df['diff_x'] >= 5].copy()

print("Spatial Distribution of Large Adjustments (>= 5ft):")
print(f"Total significant adjustments: {len(large)}")

# Binning by X (Absolute)
# 0-30: Neutral Zone / Blue Line area
# 31-60: Mid Zone / Circles
# 60-89: Low Zone / Net Front
# 89+: Behind Net
large['x_bin'] = pd.cut(large['x'].abs(), bins=[0, 30, 60, 89, 100], labels=['Neutral/Point', 'Mid Slot', 'Net Front', 'Behind Net'])
print("\nBreakdown by Zone:")
print(large['x_bin'].value_counts())

print("\nSample of 'Net Front' large adjustments:")
print(large[large['x_bin'] == 'Net Front'][['x', 'x_adj', 'diff_x']].head(5))
