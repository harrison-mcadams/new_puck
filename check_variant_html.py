
import requests

GAME_ID = 2025020642
url = f"http://www.nhl.com/scores/htmlreports/20252026/TV020642.HTM"

try:
    resp = requests.get(url)
    html = resp.text
    
    print(f"HTML Length: {len(html)}")

    with open('variant_dump.html', 'w', encoding='utf-8') as f:
        f.write(html[:20000])
    print("Dumped 20k chars to variant_dump.html")
        
except Exception as e:
    print(e)
