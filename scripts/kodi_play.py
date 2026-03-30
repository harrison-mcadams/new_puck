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
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        games = []
        links = soup.find_all('a', href=re.compile(r'/watch/'))
        for link in links:
            title = link.text.strip()
            if title:
                url_path = link['href']
                games.append((title, f"https://{base_domain}{url_path}"))
        return games
    except Exception as e:
        print(f"(!) Error fetching schedule from {base_domain}: {e}")
        return []

def find_game_page(team_name, preferred_domain="streamed.pk"):
    """Find the specific game page for a team across multiple mirrors."""
    domains = [preferred_domain, "strmd.link", "streamed.su"]
    for d in domains:
        games = get_live_games(d)
        if games:
            for title, url in games:
                if team_name.lower() in title.lower():
                    print(f"[+] Found Game Matching '{team_name}' on {d}: {title}")
                    return url
    
    print(f"[-] No live game found for '{team_name}' on any monitored domain.")
    return None

def extract_m3u8(game_url):
    """Extract a playable m3u8 from the game page using SvelteKit data logic."""
    parsed = urlparse(game_url)
    referer_host = f"{parsed.scheme}://{parsed.netloc}/"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": referer_host
    }
    
    print(f"[*] Analyzing stream sources at {game_url}...")
    try:
        response = requests.get(game_url, headers=headers, timeout=10)
        
        # SvelteKit Data Extraction
        sources_match = re.search(r'sources:\s*\[(.*?)\]', response.text)
        if sources_match:
            sources_json_str = sources_match.group(1)
            items = sources_json_str.split('},{')
            
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
                        'url': f"{game_url}/{name}/1" if not game_url.endswith('/') else f"{game_url}{name}/1"
                    })
            
            if streams:
                best = max(streams, key=lambda x: x['viewers'])
                print(f"[+] Selected '{best['name']}' source with {best['viewers']} viewers.")
                
                final_response = requests.get(best['url'], headers=headers, timeout=10)
                
                # Try finding direct m3u8
                m3u8_match = re.search(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)', final_response.text)
                if m3u8_match:
                    return m3u8_match.group(1)
                    
                # Try finding embed
                embed_match = re.search(r'https://embedsport[sy]\.top/embed/[^"\'\s]+', final_response.text)
                if embed_match:
                    return embed_match.group(0)
        else:
            print("[*] No SvelteKit source block found. Searching for direct links...")
            m3u8_match = re.search(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)', response.text)
            if m3u8_match:
                return m3u8_match.group(1)
                    
        return None
    except Exception as e:
        print(f"(!) Extraction error: {e}")
        return None

def play_url(url, user_agent=None, referer=None, origin=None):
    """Sends play request to Kodi via JSON-RPC."""
    if not user_agent:
        user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
    if not referer:
        referer = "https://embedsports.top/"
    if not origin:
        origin = "https://embedsports.top"

    from urllib.parse import quote
    
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

    try:
        response = requests.post(rpc_url, json=payload, auth=auth, timeout=10)
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
    parser.add_argument("--game-url", help="Direct game page URL (e.g. from streamed.pk)")
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
    elif args.game_url:
        stream_url = extract_m3u8(args.game_url)
        if stream_url:
            play_url(stream_url, user_agent=args.ua, referer=args.ref)
    elif args.team:
        game_url = find_game_page(args.team, preferred_domain="streamed.pk")
        if game_url:
            stream_url = extract_m3u8(game_url)
            if stream_url:
                play_url(stream_url, user_agent=args.ua, referer=args.ref)
            else:
                print(f"[-] Could not extract a playable link for '{args.team}'.")
    else:
        parser.print_help()
