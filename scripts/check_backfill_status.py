
import glob
import os
import pandas as pd

data_dir = "data"
seasons = sorted(glob.glob(os.path.join(data_dir, "20*")))

for s_dir in seasons:
    if not os.path.isdir(s_dir):
        continue
    
    season_name = os.path.basename(s_dir)
    csv_path = os.path.join(s_dir, f"{season_name}_df.csv")
    
    if os.path.exists(csv_path):
        try:
            # Check if x_adj exists and if it has any non-zero diffs from x
            df = pd.read_csv(csv_path, nrows=1000)
            if 'x_adj' in df.columns:
                diffs = (df['x'] != df['x_adj']).sum()
                print(f"Season {season_name}: OK (x_adj found, {diffs} diffs in first 1k rows)")
            else:
                print(f"Season {season_name}: MISSING x_adj column")
        except Exception as e:
            print(f"Season {season_name}: ERROR {e}")
    else:
        print(f"Season {season_name}: NO CSV FOUND")
