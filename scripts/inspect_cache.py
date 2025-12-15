
import numpy as np
import os
import json
import sys

# Path to a cache file
param_path = sys.argv[1]

print(f"Inspecting {param_path}")
try:
    with np.load(param_path, allow_pickle=True) as data:
        print("Keys:", list(data.keys()))
        if 'empty' in data:
            print("File is marked as empty.")
        else:
            for k in data.keys():
                if k.endswith('_stats'):
                    val = data[k]
                    print(f"\n--- {k} ---")
                    if val.shape == ():
                         s = str(val)
                         # Clean numpy wrapping
                         print(s)
                         try:
                             d = json.loads(s)
                             print("Parsed JSON:")
                             print(json.dumps(d, indent=2))
                         except:
                             pass
                    else:
                        print(val)
except Exception as e:
    print(f"Error: {e}")
