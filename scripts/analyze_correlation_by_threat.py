import pandas as pd
import sys
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    print("--- Correlation by Threat Level (Unblocked) ---")
    
    # 1. Load Data
    pred_path = Path('analysis/nested_xgs/test_predictions.csv')
    if not pred_path.exists():
        print(f"Error: {pred_path} not found.")
        return
        
    df = pd.read_csv(pred_path)
    
    # 2. Filter Unblocked & Matched
    valid = df[
        (df['event'] != 'blocked-shot') & 
        (df['mp_xGoal'].notna())
    ].copy()
    
    print(f"Total Unblocked & Matched: {len(valid)}")
    print(f"Overall Correlation: {valid['xG'].corr(valid['mp_xGoal']):.4f}\n")
    
    # 3. Define Buckets (Based on OUR xG)
    # Why our xG? Because we want to know "When WE think it's high danger, does MoneyPuck agree?"
    bins = [0, 0.05, 0.15, 1.0]
    labels = ['Low (< 5%)', 'Medium (5-15%)', 'High (> 15%)']
    valid['threat_level'] = pd.cut(valid['xG'], bins=bins, labels=labels)
    
    # 4. Analyze by Bucket
    results = []
    for label in labels:
        subset = valid[valid['threat_level'] == label]
        n = len(subset)
        if n > 10:
            corr = subset['xG'].corr(subset['mp_xGoal'])
            mae = (subset['xG'] - subset['mp_xGoal']).abs().mean()
            mean_us = subset['xG'].mean()
            mean_mp = subset['mp_xGoal'].mean()
            
            results.append({
                'Threat Level': label,
                'Count': n,
                'Corr': f"{corr:.4f}",
                'MAE': f"{mae:.4f}",
                'Mean Us': f"{mean_us:.4f}",
                'Mean MP': f"{mean_mp:.4f}"
            })
        else:
            results.append({'Threat Level': label, 'Count': n, 'Corr': 'N/A'})
            
    # Print Table
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # 5. Scatter Plot by Threat
    # Optional: could generate a plot here if needed, but text is fine for now.

if __name__ == "__main__":
    main()
