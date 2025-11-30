
import analyze

season = '20252026'
teams = ['ANA', 'BOS', 'BUF', 'CAR', 'CBJ'] # Just a few teams for testing
out_dir = f'static/league/{season}'

print(f"Generating Special Teams plots for {season}...")
analyze.generate_special_teams_plot(season, teams, out_dir)
print("Done.")
