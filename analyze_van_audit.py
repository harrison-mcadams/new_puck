
import pandas as pd
import sys

try:
    df = pd.read_csv('audit_shift_stats.csv', names=['game_id', 'home', 'away', 'api_attempts', 'shift_attempts', 'diff', 'abs_diff', 'pct_err', 'status'])
    if not isinstance(df.iloc[0]['game_id'], (int, float)) and not str(df.iloc[0]['game_id']).isdigit():
        df = df.iloc[1:]
        
    df['diff'] = pd.to_numeric(df['diff'])
    df['abs_diff'] = df['diff'].abs()
    
    # Filter for VAN
    van_df = df[(df['home'] == 'VAN') | (df['away'] == 'VAN')]
    
    print("--- VAN Worst Discrepancies ---")
    print(van_df.sort_values('abs_diff', ascending=False).head(20).to_string(index=False))

except Exception as e:
    print(e)
