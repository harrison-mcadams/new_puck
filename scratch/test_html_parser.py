import sys
from puck import html_enrichment, nhl_api

game_id = '2025020001' # First game of 25/26 season
print(f"Fetching HTML for {game_id}...")
html_text = nhl_api.get_pbp_from_nhl_html(game_id)
if not html_text:
    print("Failed to fetch HTML")
    sys.exit(1)

print(f"Fetched {len(html_text)} bytes")
events = html_enrichment.parse_html_pbp(html_text)

blocks = [e for e in events if e['event_code'] == 'BLOCK']
print(f"Found {len(blocks)} blocks")

for b in blocks[:5]:
    print(f"Desc: {b['description']}")
    print(f"Type: {b['shot_type']}")
    print("---")
