
import pandas as pd
import sys

try:
    df = pd.read_csv('audit_shift_stats.csv', names=['game_id', 'home', 'away', 'api_attempts', 'shift_attempts', 'diff', 'abs_diff', 'pct_err', 'status'])
    # Skip header row if it exists (it seems to have no header based on cat output, but names arg handles it. 
    # Wait, if file has header, names overrides it and makes header a row. 
    # Let's assume standard logic: verify if first row is non-numeric.
    
    if not isinstance(df.iloc[0]['game_id'], (int, float)) and not str(df.iloc[0]['game_id']).isdigit():
        df = df.iloc[1:]
        
    df['diff'] = pd.to_numeric(df['diff'])
    df['abs_diff'] = df['diff'].abs()
    
    print("--- Top 20 Worst Discrepancies ---")
    print(df.sort_values('abs_diff', ascending=False).head(20).to_string(index=False))
    
    print("\n--- Zero Shift Games? ---")
    zeros = df[pd.to_numeric(df['shift_attempts']) == 0]
    if len(zeros) > 0:
        print(zeros.to_string(index=False))
    else:
        print("No games with 0 shift attempts found!")

except Exception as e:
    print(e)
