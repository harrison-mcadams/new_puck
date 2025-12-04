
import analyze

season = '20252026'
teams = None # Process all teams
out_dir = f'static/league/{season}'

print(f"Generating Special Teams plots for {season}...")
analyze.generate_special_teams_plot(season, teams, out_dir)
print("Done.")
