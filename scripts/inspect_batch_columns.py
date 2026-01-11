
import pandas as pd
import os

def inspect():
    path = os.path.join('analysis', 'blocked_shots', 'blocked_shots_summary_batch.csv')
    df = pd.read_csv(path, nrows=1)
    
    with open('analysis/columns.txt', 'w') as f:
        f.write(str(list(df.columns)))
        
if __name__ == "__main__":
    inspect()
