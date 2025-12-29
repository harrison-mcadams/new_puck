
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def analyze_rebound_factor():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_deep_dive_2025.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Filter for High MP xG (>0.7) where My Model is Low (<0.3) & Distances Agree (<5ft difference)
    # This isolates "Model Opinion" differences
    mask_disagreement = (df['xGoal'] > 0.7) & (df['xgs'] < 0.3) & (df['dist_diff'].abs() < 5.0)
    
    disagreements = df[mask_disagreement].copy()
    print(f"Found {len(disagreements)} cases where MP >> My Model (MP > 0.7, Mine < 0.3).")
    
    print("\n--- 'is_rebound' Status in these cases (My Model) ---")
    print(disagreements['is_rebound'].value_counts(normalize=True))
    print(disagreements['is_rebound'].value_counts())
    
    print("\n--- Event Type Distribution ---")
    print(disagreements['event'].value_counts())
    
    print("\n--- Shot Type Distribution ---")
    print(disagreements['shotType'].value_counts())

if __name__ == "__main__":
    analyze_rebound_factor()
