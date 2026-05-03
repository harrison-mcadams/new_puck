"""
KEY QUESTION: Does the training pipeline apply correction TWICE?
If the _df.csv already has team_id swapped, and then preprocess_features()
swaps it again, the two swaps cancel out and the model trains on CORRECT data.
"""
import pandas as pd
import sys, os
sys.path.insert(0, os.getcwd())

from puck import data_pipeline

# Load the raw CSV (what the training script loads)
df = pd.read_csv("data/20242025/20242025_df.csv", nrows=5000, low_memory=False)
bs = df[df["event"] == "blocked-shot"]
sog = df[df["event"] == "shot-on-goal"]

print("=== STATE 1: Raw CSV (what training script loads) ===")
print(f"Blocked shots: {len(bs)}")
print(f"  BS avg distance: {bs['distance'].dropna().mean():.1f}")
print(f"  SOG avg distance: {sog['distance'].dropna().mean():.1f}")

if "team_id" in bs.columns and "home_id" in bs.columns:
    home_pct = (bs["team_id"].astype(str) == bs["home_id"].astype(str)).mean()*100
    print(f"  BS team_id == home_id: {home_pct:.1f}% (expect ~50%)")

# Now run preprocess_features (which runs correction.fix_blocked_shot_attribution)
print("\n=== STATE 2: After data_pipeline.preprocess_features() ===")
df2 = data_pipeline.preprocess_features(
    df.copy(), 
    is_training=True,
    verbose=True, 
    apply_html_enrichment=False,
    apply_filtering=True
)
bs2 = df2[df2["event"] == "blocked-shot"]
sog2 = df2[df2["event"] == "shot-on-goal"]

print(f"Blocked shots: {len(bs2)}")
print(f"  BS avg distance: {bs2['distance'].dropna().mean():.1f}")
print(f"  SOG avg distance: {sog2['distance'].dropna().mean():.1f}")

if "team_id" in bs2.columns and "home_id" in bs2.columns:
    home_pct2 = (bs2["team_id"].astype(str) == bs2["home_id"].astype(str)).mean()*100
    print(f"  BS team_id == home_id: {home_pct2:.1f}%")

# KEY QUESTION: What does the model ACTUALLY see?
print("\n=== CONCLUSION ===")
bs_dist_after = bs2["distance"].dropna().mean()
if bs_dist_after < 80:
    print("Model sees CORRECT distances (~35-45ft) for blocked shots.")
    print("The double-swap hypothesis is CONFIRMED.")
    print("The CSV has wrong data, but preprocess_features() swaps it BACK.")
    print("The model was trained correctly all along!")
else:
    print(f"Model sees WRONG distances (~{bs_dist_after:.0f}ft) for blocked shots.")
    print("The corruption persists through to training.")
