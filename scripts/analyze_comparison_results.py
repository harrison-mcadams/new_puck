
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from puck import config

def analyze_discrepancies():
    csv_path = os.path.join(config.ANALYSIS_DIR, 'shot_comparison_2025.csv')
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} shots.")
    
    # Sort by diff
    # diff = xgs - xGoal
    # Positive = My Model Higher
    # Negative = My Model Lower
    
    cols = ['game_id', 'period', 'sec', 'team', 'shooterName', 'event', 'shotType', 'xgs', 'xGoal', 'diff']
    
    print("\n=== My Model SIGNIFICANTLY HIGHER (Top 5) ===")
    top_high = df.sort_values('diff', ascending=False).head(5)
    print(top_high[cols].to_markdown(index=False))
    
    print("\n=== My Model SIGNIFICANTLY LOWER (Top 5) ===")
    top_low = df.sort_values('diff', ascending=True).head(5)
    print(top_low[cols].to_markdown(index=False))
    
    # Look for patterns
    # E.g. high discrepancy by shot type?
    print("\n--- Mean Diff by Event Type ---")
    print(df.groupby('event')['diff'].mean())
    
    print("\n--- Mean Diff by Shot Type (MP) ---")
    if 'shotType' in df.columns:
        print(df.groupby('shotType')['diff'].mean())

if __name__ == "__main__":
    analyze_discrepancies()
