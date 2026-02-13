
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def main():
    df = pd.read_csv("data/20252026.csv")
    gid = 2025020006
    
    # Target Interval: 531.0 - 593.0 (Period 1 typically)
    # Check Period 1 events around this time
    
    g_df = df[df['game_id'] == gid]
    
    # Compute period_seconds
    def time_to_sec(x):
        try:
            m, s = x.split(':')
            return int(m)*60 + int(s)
        except: return 0
    g_df['period_seconds'] = g_df['period_time'].apply(time_to_sec)
    
    # Filter for Period 1, Time 500-600
    mask = (g_df['period'] == 1) & (g_df['period_seconds'] >= 500) & (g_df['period_seconds'] <= 650)
    
    print("Columns:", g_df.columns.tolist())
    
    subset = g_df[mask].copy()
    cols = ['period_time', 'period_seconds', 'event', 'game_state']
    for c in ['team_name', 'team_id_for', 'description', 'secondary_type', 'player_1_name']:
         if c in subset.columns: cols.append(c)
         
    print(f"--- Game {gid} Events around 531s ---")
    print(subset[cols])
    
if __name__ == "__main__":
    main()
