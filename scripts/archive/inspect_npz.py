
import numpy as np
import sys
import os
import glob
import json

if len(sys.argv) > 1:
    f = sys.argv[1]
else:
    f = 'data/cache/20252026/partials/2025010073_5v5.npz'
if not os.path.exists(f):
    print(f"File {f} not found.")
    sys.exit(0)

print(f"Inspecting {f}")
try:
    with np.load(f, allow_pickle=True) as data:
        print("Keys:", list(data.keys()))
        if 'empty' in data:
            print("File is marked EMPTY.")
        else:
            for k in data.keys():
                if 'team_' in k and '_grid_team' in k:
                    print(f"--- {k} ---")
                    val = data[k]
                    # Print raw value/type first
                    print(f"Type: {type(val)}")
                    print(f"Dtype: {val.dtype}")
                    print(f"Raw: {val}")
                    
                    if val.dtype.kind in {'U', 'S'}:
                        s_str = str(val.item())
                    else:
                        s_str = str(val)
                    
                    print(f"String to parse: {s_str!r}")
                    
                    try:
                        s = json.loads(s_str)
                        print(json.dumps(s, indent=2))
                    except Exception as e:
                        print(f"JSON Decode Error: {e}")

except Exception as e:
    print(f"Error: {e}")
