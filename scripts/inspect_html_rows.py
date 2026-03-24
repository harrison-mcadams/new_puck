from bs4 import BeautifulSoup
import sys
import os
import requests

sys.path.append(os.getcwd())
from puck import nhl_api

def inspect_html(game_id):
    html_text = nhl_api.get_pbp_from_nhl_html(game_id)
    if not html_text:
        print("Empty HTML")
        return
        
    soup = BeautifulSoup(html_text, 'html.parser')
    rows = soup.find_all('tr')
    print(f"Total rows found: {len(rows)}")
    
    count = 0
    event_codes = {}
    for i, row in enumerate(rows):
        tds = row.find_all('td')
        if len(tds) < 5:
            continue
            
        txt = [td.get_text(strip=True) for td in tds]
        if len(txt) > 4 and txt[0].isdigit():
            code = txt[4].upper()
            event_codes[code] = event_codes.get(code, 0) + 1
            
            if "BLOCK" in txt[4] or "BLOCK" in txt[5].upper():
                print(f"BLOCK-like Row {i} (P{txt[1]} {txt[3]}): {txt[4]} | {txt[5][:50]}")

    print("\nEvent Code Counts:")
    for code, count in sorted(event_codes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {code}: {count}")

inspect_html("2024020151")
