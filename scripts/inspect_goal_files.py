import os
import glob
import json
import pandas as pd

metrics_dir = r"c:\Users\harri\Desktop\new_puck\data\edge_goals\20252026"
files = glob.glob(os.path.join(metrics_dir, "game_202502*_edge.json"))

if not files:
    print("No regular season files found.")
else:
    # Find a pair that exists
    for f_json in files:
        f_csv = f_json.replace("_edge.json", "_positions.csv")
        if os.path.exists(f_csv):
            print(f"Inspecting Pair: {os.path.basename(f_json)}")
            
            # Read JSON
            try:
                with open(f_json, 'r') as f:
                    data = json.load(f)
                    print(f"JSON Type: {type(data)}")
                    if isinstance(data, list):
                        print(f"JSON List Len: {len(data)}")
                        if len(data) > 0:
                            print("First Item Keys:", data[0].keys())
                            print("First Item Sample:", str(data[0])[:200])
                    elif isinstance(data, dict):
                         print("JSON Keys:", data.keys())
                         if 'details' in data:
                             print("Details:", data['details'])
            except Exception as e:
                print(f"Error reading JSON: {e}")

            # Read CSV
            try:
                df = pd.read_csv(f_csv, nrows=5)
                print("CSV Columns:", list(df.columns))
                print("Sample Rows:\n", df.to_string())
            except Exception as e:
                print(f"Error reading CSV: {e}")
            
            break # Stop after first valid pair
    else:
        print("No valid JSON/positions CSV pair found.")
