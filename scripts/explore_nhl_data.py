import requests
import json
import time

GAME_ID = "2025020583"
# Using the season from the user's URL context: 2025/12/23 -> 2025-2026 season
# NHL API usually uses 20252026 for the season string in these URLs
SEASON = "20252026" 

def check_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nhl.com/",
        "Origin": "https://www.nhl.com"
    }
    try:
        r = requests.head(url, headers=headers, timeout=2)
        return r.status_code
    except:
        return 999

def main():
    # 1. Get PBP
    pbp_url = f"https://api-web.nhle.com/v1/gamecenter/{GAME_ID}/play-by-play"
    print(f"Fetching PBP from {pbp_url}...")
    try:
        pbp = requests.get(pbp_url).json()
    except Exception as e:
        print(f"Failed to fetch PBP: {e}")
        return

    print("Successfully fetched PBP.")
    
    events_with_ppt = []
    event_types_probed = {}
    
    plays = pbp.get('plays', [])
    print(f"Found {len(plays)} plays.")

    for play in plays:
        event_id = play.get('eventId')
        type_desc = play.get('typeDescKey', 'UNKNOWN')
        
        # Check if it advertises a replay URL explicitly
        if 'pptReplayUrl' in play:
            events_with_ppt.append((event_id, type_desc, play['pptReplayUrl']))
        
        # We want to probe a few of each type that DOESN'T have it explicitly
        if type_desc not in event_types_probed:
            event_types_probed[type_desc] = []
        
        if len(event_types_probed[type_desc]) < 3: # Probe up to 3 of each type
            event_types_probed[type_desc].append(event_id)

    print("\n--- Events with explicit pptReplayUrl ---")
    if events_with_ppt:
        for eid, desc, url in events_with_ppt:
            print(f"Event {eid} ({desc}): {url}")
    else:
        print("None found.")

    print("\n--- Probing other event types ---")
    # Base URL format from user: https://wsr.nhle.com/sprites/20252026/2025020583/ev187.json
    base_url = f"https://wsr.nhle.com/sprites/{SEASON}/{GAME_ID}"
    
    # We'll probe a sample of events
    for type_desc, ids in event_types_probed.items():
        print(f"Checking {type_desc}...")
        for eid in ids:
            url = f"{base_url}/ev{eid}.json"
            status = check_url(url)
            print(f"  Event {eid}: {status}")
            if status == 200:
                print(f"    FOUND! {url}")
            time.sleep(0.1) # Be nice

if __name__ == "__main__":
    main()
