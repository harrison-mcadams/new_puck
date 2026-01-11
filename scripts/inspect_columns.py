
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from puck import fit_xgs

print("Loading sample data...")
try:
    df = fit_xgs.load_data().sample(10)
    print("Columns found:")
    for col in sorted(df.columns):
        print(f"  {col}")
        
    print("\nSample Data (Goalie/Skaters):")
    cols_of_interest = [c for c in df.columns if 'goalie' in c or 'skater' in c or 'state' in c or 'net' in c]
    print(df[cols_of_interest].head())
    
except Exception as e:
    print(f"Error: {e}")
