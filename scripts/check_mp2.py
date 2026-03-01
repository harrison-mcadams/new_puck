import cloudscraper
import pandas as pd
import io

scraper = cloudscraper.create_scraper()
r = scraper.get('https://moneypuck.com/moneypuck/playerData/careers/gameByGame/all_teams.csv')
if r.status_code == 200:
    df = pd.read_csv(io.StringIO(r.text), nrows=10)
    print("Odds Columns:", [c for c in df.columns if 'odd' in c.lower() or 'line' in c.lower() or 'money' in c.lower() or 'prob' in c.lower()])
