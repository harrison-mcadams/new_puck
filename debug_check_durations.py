import pandas as pd
import glob
import os

def check_durations():
    files = glob.glob('data/edge_goals/20242025/*_positions.csv')
    print(f"Scanning {len(files)} files...")
    
    long_count = 0
    total = 0
    
    maxlen = 0
    maxfile = ""
    
    for f in files[:1000]:
        try:
            df = pd.read_csv(f)
            if df.empty: continue
            
            t_min = df['timestamp'].min()
            t_max = df['timestamp'].max()
            duration = (t_max - t_min) / 1000.0
            
            total += 1
            if duration > 1.0:
                print(f"File {os.path.basename(f)}: {duration:.2f}s")
                long_count += 1
                
            if duration > maxlen:
                maxlen = duration
                maxfile = f
                
        except: pass
        
    print(f"Scanned {total} files. Found {long_count} > 1.0s.")
    print(f"Max Duration: {maxlen:.2f}s in {maxfile}")

if __name__ == "__main__":
    check_durations()
