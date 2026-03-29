#!/usr/bin/env python3
import sys
import os
import argparse
import requests
import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import quote, urlparse

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))
from puck import config

def get_live_games(base_domain):
    """Scrape the schedule for live games."""
    url = f"https://{base_domain}/schedule"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://{base_domain}/"
    }
    
    print(f"[*] Scraping schedule at {url}...")
    try:
        # Note: We use a simple requests call. If Cloudflare blocks this on the Pi, 
        # we may need to switch to cloudscraper or a headless browser.
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        # On this site, games are usually in 'a' tags with specific structure
        links = soup.find_all('a', href=re.compile(r'/watch/'))
        for link in links:
            title = link.text.strip()
            if title:
                url_path = link['href']
                games.append((title, f"https://{base_domain}{url_path}"))
        return games
    except Exception as e:
        print(f"(!) Error fetching schedule: {e}")
        return []

def find_game_page(team_name, base_domain="strmd.link"):
    """Find the specific game page for a team."""
    games = get_live_games(base_domain)
    for title, url in games:
        if team_name.lower() in title.lower():
            print(f"[+] Found Game Matching '{team_name}': {title}")
            return url
    print(f"[-] No live game found for '{team_name}'.")
    return None

def extract_m3u8(game_url):
    """Extract a playable m3u8 from the game page using SvelteKit data logic."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": game_url
    }
    
    print(f"[*] Analyzing stream sources...")
    try:
        response = requests.get(game_url, headers=headers, timeout=10)
        
        # SvelteKit Data Extraction (from Pulsar logic)
        # The site stores source information in a JSON-like string within a script tag
        sources_match = re.search(r'sources:\s*\[(.*?)\]', response.text)
        if sources_match:
            sources_json_str = sources_match.group(1)
            # This is a bit brittle but avoids a full JS parser
            items = sources_json_str.split('},{')
            
            # Form stream candidates
            streams = []
            for item in items:
                item = item.strip('{}')
                name_match = re.search(r'source:"?(\w+)"?', item)
                viewers_match = re.search(r'viewers:"?(\d+)"?', item)
                
                if name_match:
                    name = name_match.group(1)
                    viewers = int(viewers_match.group(1)) if viewers_match else 0
                    streams.append({
                        'name': name,
                        'viewers': viewers,
                        'url': f"{game_url}/{name}/1"
                    })
            
            if streams:
                # Pick the most popular stream (usually the most stable)
                best = max(streams, key=lambda x: x['viewers'])
                print(f"[+] Selected '{best['name']}' source with {best['viewers']} viewers.")
                
                # Visit the final page to get the iframe/m3u8
                final_response = requests.get(best['url'], headers=headers, timeout=10)
                
                # Look for direct m3u8 in scripts
                m3u8_match = re.search(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)', final_response.text)
                if m3u8_match:
                    return m3u8_match.group(1)
                    
                # Look for embed iframe (embedsports.top)
                embed_match = re.search(r'https://embedsport[sy]\.top/embed/[^"\'\s]+', final_response.text)
                if embed_match:
                    return embed_match.group(0)
                    
        return None
    except Exception as e:
        print(f"(!) Extraction error: {e}")
        return None

def play_url(url, user_agent=None, referer=None, origin=None):
    """
    Sends a play request to Kodi via JSON-RPC.
    Kodi supports headers via a 'pipe' suffix: URL|Header1=Val&Header2=Val
    
    This technique is critical for bypassing 403 Forbidden errors.
    """
    # Use verified Pulsar headers by default
    if not user_agent:
        user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
    if not referer:
        referer = "https://embedsports.top/"
    if not origin:
        origin = "https://embedsports.top"

    from urllib.parse import quote
    
    # Construct the Kodi-style URL with headers
    # Values MUST be URL-encoded so Kodi's internal parser doesn't break on spaces or ampersands
    headers_str = f"User-Agent={quote(user_agent)}&Referer={quote(referer)}&Origin={quote(origin)}"
    kodi_url = f"{url}|{headers_str}"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "Player.Open",
        "params": {
            "item": {
                "file": kodi_url
            }
        },
        "id": 1
    }
    
    auth = (config.KODI_USER, config.KODI_PASS)
    rpc_url = f"http://{config.KODI_HOST}:{config.KODI_PORT}/jsonrpc"
    
    print(f"[*] Dispatching stream to Kodi...")
    print(f"[*] Target: {config.KODI_HOST}:{config.KODI_PORT}")

    try:
        response = requests.post(
            rpc_url, 
            json=payload, 
            auth=auth,
            timeout=10
        )
        response.raise_for_status()
        result = response.json()
        
        if result.get('result') == "OK":
            print("[+] Success: Playback command accepted by Kodi.")
        else:
            print(f"[!] Kodi returned an unexpected response: {result}")
            
    except Exception as e:
        print(f"(!) RPC Call Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kodi Remote Discovery & Launcher")
    parser.add_argument("--url", help="Direct HLS/m3u8 URL to play")
    parser.add_argument("--team", help="Team name to discover (e.g. Flyers)")
    parser.add_argument("--test", action="store_true", help="Launch a public test stream")
    parser.add_argument("--ua", help="Custom User-Agent")
    parser.add_argument("--ref", help="Custom Referer")

    args = parser.parse_args()

    if args.test:
        test_url = "https://gg.poocloud.in/cdr_guadalajara/index.m3u8"
        play_url(test_url)
    elif args.url:
        play_url(args.url, user_agent=args.ua, referer=args.ref)
    elif args.team:
        # Step 1: Find the game page on the working mirror
        game_url = find_game_page(args.team, base_domain="strmd.link")
        if game_url:
            # Step 2: Extract the playable link (m3u8 or embed)
            stream_url = extract_m3u8(game_url)
            if stream_url:
                # Step 3: Play on Kodi
                play_url(stream_url, user_agent=args.ua, referer=args.ref)
            else:
                print(f"[-] Could not extract a playable link for '{args.team}'.")
    else:
        parser.print_help()
