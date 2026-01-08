
import json
import os
import numpy as np

def inspect_outliers():
    path = os.path.join("data", "arena_adjustments.json")
    with open(path, 'r') as f:
        data = json.load(f)

    print("Checking for adjustments > 10 units...")
    
    found_large = False
    for season, s_data in data.items():
        if season == "20252026": continue
        
        for arena, a_data in s_data.items():
            # a_data has 'x': { '0.0': 1.2, ... } mapping coordinate -> offset
            # The structure is seemingly:
            # "x": { "coord_bucket": bias_value, ... }
            
            x_biases = a_data.get('x', {})
            y_biases = a_data.get('y', {})
            
            max_x = 0
            max_y = 0
            
            if x_biases:
                max_x = max([abs(float(v)) for v in x_biases.values()])
            if y_biases:
                max_y = max([abs(float(v)) for v in y_biases.values()])
                
            if max_x > 10 or max_y > 10:
                found_large = True
                print(f"\nSeason {season} - Arena '{arena}':")
                print(f"  Max X-Bias: {max_x}")
                print(f"  Max Y-Bias: {max_y}")
                
                # Print the specific entries causing this
                large_x = {k:v for k,v in x_biases.items() if abs(float(v)) > 10}
                if large_x:
                    print(f"  Large X-entries: {large_x}")
                
                large_y = {k:v for k,v in y_biases.items() if abs(float(v)) > 10}
                if large_y:
                    print(f"  Large Y-entries: {large_y}")

    if not found_large:
        print("No outlier adjustments > 10 found in JSON.")

if __name__ == "__main__":
    inspect_outliers()
