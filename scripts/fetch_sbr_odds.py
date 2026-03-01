import cloudscraper
import pandas as pd
import os

scraper = cloudscraper.create_scraper()
url_sbr = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%202023-24.xlsx"
print("Downloading Sportsbook Review data...")
response_sbr = scraper.get(url_sbr)

if response_sbr.status_code == 200:
    with open('tmp_sbr.xlsx', 'wb') as f:
        f.write(response_sbr.content)
    try:
        df_sbr = pd.read_excel('tmp_sbr.xlsx', engine='openpyxl')
        os.makedirs('analysis/market', exist_ok=True)
        df_sbr.to_csv('analysis/market/nhl_odds_2023_24.csv', index=False)
        print("Success! Saved SBR odds data to analysis/market/nhl_odds_2023_24.csv")
    except Exception as e:
        print("Failed to parse Excel:", e)
else:
    print("SBR download failed with status code:", response_sbr.status_code)
        
