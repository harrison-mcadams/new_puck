
import pandas as pd
import sys

try:
    # Load audit results
    # game_id,home,away,api_attempts,shift_attempts,diff,abs_diff,pct_err,status
    df = pd.read_csv('audit_shift_stats.csv', names=['game_id', 'home', 'away', 'api_attempts', 'shift_attempts', 'diff', 'abs_diff', 'pct_err', 'status'])
    
    # Check for header and skip if present
    if not isinstance(df.iloc[0]['game_id'], (int, float)) and not str(df.iloc[0]['game_id']).isdigit():
        df = df.iloc[1:]
        
    cols = ['api_attempts', 'shift_attempts', 'diff', 'abs_diff']
    for c in cols:
        df[c] = pd.to_numeric(df[c])

    # Transform to team-level
    # We want a row for Home team and a row for Away team for each game
    home_df = df[['home', 'api_attempts', 'shift_attempts']].rename(columns={'home': 'Team'})
    away_df = df[['away', 'api_attempts', 'shift_attempts']].rename(columns={'away': 'Team'})
    
    team_df = pd.concat([home_df, away_df])
    
    # Group by Team
    stats = team_df.groupby('Team').sum().reset_index()
    
    stats['diff'] = stats['shift_attempts'] - stats['api_attempts']
    stats['pct_diff'] = (stats['diff'] / stats['api_attempts']) * 100
    stats['abs_pct_diff'] = stats['pct_diff'].abs()
    
    # Sort by absolute percent difference
    stats = stats.sort_values('abs_pct_diff', ascending=False)
    
    print(f"--- Team Totals (Season) ---")
    print(f"{'Team':<5} {'API':<8} {'Shift':<8} {'Diff':<6} {'% Diff':<8}")
    print("-" * 40)
    
    for _, row in stats.iterrows():
        print(f"{row['Team']:<5} {row['api_attempts']:<8} {row['shift_attempts']:<8} {row['diff']:<6} {row['pct_diff']:.2f}%")

    print("\n--- Summary ---")
    print(f"Worst Discrepancy: {stats.iloc[0]['Team']} ({stats.iloc[0]['pct_diff']:.2f}%)")
    print(f"Average Discrepancy: {stats['abs_pct_diff'].mean():.2f}%")

except Exception as e:
    print(f"Error: {e}")
