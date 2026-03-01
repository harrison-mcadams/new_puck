import cloudscraper
import pandas as pd
from bs4 import BeautifulSoup
import os

url = "https://checkbestodds.com/hockey-odds/nhl/archive-nhl/2023-2024"
scraper = cloudscraper.create_scraper()
response = scraper.get(url)

print(response.status_code)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table') # look for a table
    if table:
        df = pd.read_html(str(table))[0]
        print(len(df))
        print(df.head())
    else:
        print("No table found. Might be loaded via JS.")
        print(response.text[:500])
