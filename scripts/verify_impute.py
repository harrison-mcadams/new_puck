
import pandas as pd
import numpy as np
import sys
import os

# Add project root
sys.path.append(os.getcwd())
from puck import impute

def main():
    print("Testing Imputation...")
    # create dummy blocked shot
    df = pd.DataFrame({
        'x': [50.0], 'y': [0.0], 
        'event': ['blocked-shot'], 
        'shooter_role': ['F'],
        'distance': [10.0] # dummy
    })
    
    print(" calling impute_blocked_shot_origins(method='cdf_mapping')...")
    try:
        df_out = impute.impute_blocked_shot_origins(df, method='cdf_mapping')
        print("Imputation result:")
        print(df_out[['x', 'y', 'imputed_x', 'imputed_y']])
        
        if hasattr(impute, '_CDF_MAPPINGS'):
            mappings = impute._CDF_MAPPINGS
            if mappings and 'F' in mappings:
                print("Inspecting 'F' mapping...")
                m = mappings['F']
                cdf = m['cdf_block']
                icdf = m['icdf_origin']
                
                print(f"CDF Type: {type(cdf)}")
                print(f"ICDF Type: {type(icdf)}")
                
                # Test continuity
                dists = [10.0, 10.1, 10.5, 11.0, 15.0]
                print(f"Sampling CDF at {dists}:")
                for d in dists:
                    try:
                        val = float(cdf(d))
                        origin_dist = float(icdf(val))
                        print(f"  d={d:.1f} -> pct={val:.4f} -> origin_dist={origin_dist:.4f}")
                    except Exception as e:
                        print(f"  d={d:.1f} -> Error: {e}")

            else:
                print("CDF Mappings is EMPTY or missing 'F'.")
        else:
            print("CDF Mappings is None.")
            
    except Exception as e:
        print(f"Imputation Crashed: {e}")

if __name__ == "__main__":
    main()
