"""Analyze CDF mapping behavior to understand why imputation shift is too small."""
import joblib
import numpy as np

# Load mappings
m = joblib.load('puck/data/cdf_mappings.joblib')
print('CDF Mappings keys:', list(m.keys()))

g = m.get('global', m.get('F'))
if g is None:
    print("No mappings found!")
    exit()

cdf_block = g['cdf_block']
icdf_origin = g['icdf_origin']

print("\nSample CDF values (block distances -> percentiles):")
for d in [5, 10, 15, 20, 25, 30, 40, 50]:
    try:
        pct = float(cdf_block(d))
        print(f"  Block dist {d:3d}ft -> percentile {pct:.3f}")
    except:
        print(f"  Block dist {d:3d}ft -> ERROR")

print("\nSample iCDF values (percentiles -> origin distances):")
for p in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    try:
        origin_dist = float(icdf_origin(p))
        print(f"  Percentile {p:.2f} -> origin dist {origin_dist:.1f}ft")
    except:
        print(f"  Percentile {p:.2f} -> ERROR")

# Trace a specific example: block 10 ft from goal
print("\n--- Trace: Block at 10 ft from goal ---")
block_dist = 10.0
pct = float(cdf_block(block_dist))
origin_dist = float(icdf_origin(pct))
print(f"Block distance: {block_dist} ft")
print(f"CDF(block_dist) = percentile: {pct:.3f}")
print(f"iCDF(percentile) = origin distance: {origin_dist:.1f} ft")
print(f"Expected shift AWAY from goal: {origin_dist - block_dist:.1f} ft")
