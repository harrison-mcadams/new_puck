import pandas as pd
df = pd.read_csv('analysis/gravity/player_gravity_season.csv')
targets = ['Matvei Michkov', 'Trevor Zegras', 'Travis Konecny']
res = df[df['player_name'].isin(targets)]
print(f"{'Player':<20} {'Season':<10} {'Rel On-Puck':<15} {'Rel Off-Puck':<15}")
for _, row in res.iterrows():
    print(f"{row['player_name']:<20} {row['season']:<10} {row['rel_on_puck_mean_dist_ft']:<15.2f} {row['rel_off_puck_mean_dist_ft']:<15.2f}")
