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

# Optional cloudscraper support for Cloudflare bypass
try:
    import cloudscraper
    SCRAPER = cloudscraper.create_scraper()
except ImportError:
    SCRAPER = requests.Session()

def get_live_games(base_domain):
    """Scrape the schedule for live games with Cloudflare bypass."""
    url = f"https://{base_domain}/schedule"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": f"https://{base_domain}/"
    }
    
    print(f"[*] Scraping schedule at {url}...")
    try:
        response = SCRAPER.get(url, headers=headers, timeout=10)
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
    """Extract a playable m3u8 or embed URL with 'Brute-Force' fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": game_url
    }
    
    parsed_game = urlparse(game_url)
    base_url = f"{parsed_game.scheme}://{parsed_game.netloc}"
    
    print(f"[*] Analyzing stream sources at {game_url}...")
    
    # Brute-force guessing priority list (common paths for Admin, Delta, etc.)
    priority_paths = ["/admin/1", "/delta/1", "/echo/1", "/golf/1", "/bravo/1"]
    
    # Try actual scraping first
    try:
        response = SCRAPER.get(game_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Look for direct links in the HTML
        game_id = game_url.strip('/').split('/')[-1]
        provider_links = soup.find_all('a', href=re.compile(rf'/watch/{game_id}/\w+/\d+'))
        
        if provider_links:
            # Prioritize found links
            found_paths = [link['href'] for link in provider_links]
            # Add to brute-force list at the front
            priority_paths = found_paths + priority_paths
            
        # 2. Visit prioritized paths and look for iframe/m3u8
        for path in priority_paths:
            stream_page = f"{base_url}{path}" if path.startswith('/') else path
            # Normalizing path in case of double slashes
            if f"{base_url}//" in stream_page: stream_page = stream_page.replace(f"{base_url}//", f"{base_url}/")
            
            print(f"[*] Testing stream source: {stream_page}...")
            try:
                stream_response = SCRAPER.get(stream_page, headers=headers, timeout=10)
                
                # Check for direct m3u8 in scripts
                m3u8_match = re.search(r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)', stream_response.text)
                if m3u8_match:
                    url = m3u8_match.group(1).replace('\\', '')
                    print(f"[+] Found direct m3u8: {url}")
                    return url
                    
                # Check for embed iframe (embedsports.top)
                embed_match = re.search(r'https://embedsport[sy]\.top/embed/[^"\'\s]+', stream_response.text)
                if embed_match:
                    embed_url = embed_match.group(0)
                    print(f"[*] Found Embed URL: {embed_url}")
                    return embed_url
            except:
                continue
                
        # 3. Last resort: check if SvelteKit data exists in the base page
        sources_match = re.search(r'sources:\s*\[(.*?)\]', response.text)
        if sources_match:
            # ... Pulsar-style regex parsing if needed ...
            pass
            
    except Exception as e:
        print(f"(!) Scraping error: {e}")
        
    print("[-] Extraction failed. Falling back to direct embed guessing...")
    # If all above fails, return a guess based on the game-url ID
    game_id = game_url.strip('/').split('/')[-1]
    return f"https://embedsports.top/embed/admin/{game_id}/1"

def play_url(url, user_agent=None, referer=None, origin=None):
    """Sends play request to Kodi via JSON-RPC."""
    # Standard desktop UA often works best for these CDNs
    if not user_agent:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # If it's a specific provider domain, set recommended headers
    if "embedsport" in url:
        referer = "https://embedsports.top/"
        origin = "https://embedsports.top"
    elif "modifiles.fans" in url:
        # Modifiles often requires high-priority headers
        referer = "https://streamed.su/" 
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
    print(f"[*] Payload Final: {kodi_url[:150]}...")

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
