import cloudscraper
import pandas as pd
import io

scraper = cloudscraper.create_scraper()
print("Fetching Moneypuck data...")
r = scraper.get('https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv')
if r.status_code == 200:
    df = pd.read_csv(io.StringIO(r.text), nrows=10)
    print("Columns:", df.columns.tolist())
else:
    print("Failed to fetch:", r.status_code)
