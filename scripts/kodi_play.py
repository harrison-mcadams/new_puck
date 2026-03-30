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
    """Scrape the schedule for live games with updated selectors."""
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
        # Current site structure: a tags containing h1/h2 or just text with team names
        links = soup.find_all('a', href=re.compile(r'/watch/'))
        for link in links:
            # Check h1 inside link or the link text itself
            title_tag = link.find(['h1', 'h2', 'h3', 'p', 'span'])
            title = title_tag.text.strip() if title_tag else link.text.strip()
            
            if title:
                url_path = link['href']
                full_url = f"https://{base_domain}{url_path}" if url_path.startswith('/') else url_path
                games.append((title, full_url))
        
        print(f"[*] Found {len(games)} potential game links.")
        return games
    except Exception as e:
        print(f"(!) Error fetching schedule from {base_domain}: {e}")
        return []

def find_game_page(team_name, preferred_domain="streamed.pk"):
    """Find the specific game page for a team across multiple mirrors."""
    domains = [preferred_domain, "strmd.link"]
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
    """Extract a playable m3u8 or embed URL with refined source discovery."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": game_url
    }
    
    parsed_game = urlparse(game_url)
    base_url = f"{parsed_game.scheme}://{parsed_game.netloc}"
    
    print(f"[*] Analyzing stream sources at {game_url}...")
    try:
        response = requests.get(game_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Try to find provider links directly (Admin, Delta, etc.)
        # These usually look like /watch/[id]/admin/1
        game_id = game_url.strip('/').split('/')[-1]
        provider_links = soup.find_all('a', href=re.compile(rf'/watch/{game_id}/\w+/\d+'))
        
        if not provider_links:
            # Fallback to any /watch/ link on the page that isn't the game URL itself
            provider_links = [a for a in soup.find_all('a', href=re.compile(r'/watch/')) if len(a['href'].split('/')) > 3]

        if provider_links:
            # Sort by priority: Admin > Delta > Echo > others
            priority = ["admin", "delta", "echo", "golf", "bravo"]
            selected_link = provider_links[0] # Default
            for p in priority:
                for link in provider_links:
                    if p in link['href'].lower():
                        selected_link = link
                        break
                else: continue
                break
            
            href = selected_link['href']
            stream_page = f"{base_url}{href}" if href.startswith('/') else href
            print(f"[+] Following stream source: {stream_page}")
            
            # 2. Visit the stream page to find the iframe
            stream_response = requests.get(stream_page, headers=headers, timeout=10)
            
            # Search for direct m3u8 first (some sources have it in script tags)
            m3u8_match = re.search(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)', stream_response.text)
            if m3u8_match:
                url = m3u8_match.group(1).replace('\\', '')
                print(f"[+] Found direct m3u8: {url}")
                return url
                
            # Search for embed iframe (embedsports.top)
            embed_match = re.search(r'https://embedsport[sy]\.top/embed/[^"\'\s]+', stream_response.text)
            if embed_match:
                embed_url = embed_match.group(0)
                print(f"[*] Found Embed URL: {embed_url}")
                return embed_url
        
        # 3. Last resort: check if SvelteKit data exists
        sources_match = re.search(r'sources:\s*\[(.*?)\]', response.text)
        if sources_match:
            # ... Pulsar logic for sources (already in previous version) ...
            pass
            
        print("[-] Could not find any stream sources on the game page.")
        return None
    except Exception as e:
        print(f"(!) Extraction error: {e}")
        return None

def play_url(url, user_agent=None, referer=None, origin=None):
    """Sends play request to Kodi via JSON-RPC."""
    if not user_agent:
        user_agent = "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
    
    # If it's an embedsports URL, we MUST use embedsports headers
    if "embedsport" in url:
        referer = "https://embedsports.top/"
        origin = "https://embedsports.top"
    elif not referer:
        referer = "https://streamed.su/" # Fallback
        
    if not origin:
        origin = "https://streamed.su"

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
    print(f"[*] Payload: {kodi_url[:150]}...")

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
