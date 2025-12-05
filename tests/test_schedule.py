import nhl_api
import pandas as pd

season = '20252026'
print(f"Fetching schedule for {season}...")
schedule = nhl_api.get_season(season=season)
print(f"Schedule length: {len(schedule)}")
if schedule:
    print("Sample game:", schedule[0])
    print("Sample ID:", schedule[0].get('id'))
    print("Sample Date:", schedule[0].get('gameDate') or schedule[0].get('startTimeUTC'))
else:
    print("Schedule is empty.")
