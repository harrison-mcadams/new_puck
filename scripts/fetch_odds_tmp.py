import pandas as pd
import requests
import os

url = "https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhl%20odds%202023-24.xlsx"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
response = requests.get(url, headers=headers)
if response.status_code == 200:
    with open('tmp_odds.xlsx', 'wb') as f:
        f.write(response.content)
    df = pd.read_excel('tmp_odds.xlsx')
    os.makedirs('analysis/market', exist_ok=True)
    df.to_csv('analysis/market/nhl_odds_2023_24.csv', index=False)
    print("Successfully downloaded and saved to analysis/market/nhl_odds_2023_24.csv")
else:
    print(f"Failed to download. Status code: {response.status_code}")
