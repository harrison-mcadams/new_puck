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
        links = soup.find_all('a', href=re.compile(r'/watch/'))
        for link in links:
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

def deep_extract_m3u8(page_url):
    """Deep-dive into a specific page (embed or stream) to find a direct .m3u8 link."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": page_url
    }
    
    print(f"[*] Deep-diving into: {page_url}...")
    try:
        response = SCRAPER.get(page_url, headers=headers, timeout=10)
        
        # 1. Broad Look for m3u8 patterns (handles many poocloud/modifiles patterns)
        # Often looks like: file: "https://lb1.modifiles.fans/secure/TOKEN/index.m3u8"
        # Or: source: "https://..."
        m3u8_patterns = [
            r'file:\s*["\'](https://[^"\'\s]+\.m3u8[^"\'\s]*)["\']',
            r'source:\s*["\'](https://[^"\'\s]+\.m3u8[^"\'\s]*)["\']',
            r'(https://[^"\'\s]+\.m3u8[^"\'\s]*)'
        ]
        
        for pattern in m3u8_patterns:
            matches = re.findall(pattern, response.text)
            for m in matches:
                url = m.replace('\\', '')
                if "placeholder" not in url.lower():
                    print(f"[+] Successfully extracted direct stream: {url}")
                    return url
                    
        # 2. Look for nested iframes that might contain the stream
        if "embedsport" in page_url:
            # If we are already on an embed page and found no m3u8, check if there's another iframe
            iframe_match = re.search(r'<iframe[^>]+src=["\'](https://[^"\'\s]+)["\']', response.text)
            if iframe_match:
                nested_url = iframe_match.group(1)
                # Avoid recursion loop
                if nested_url != page_url:
                    return deep_extract_m3u8(nested_url)

        return None
    except Exception as e:
        print(f"(!) Deep-extraction failure on {page_url}: {e}")
        return None

def extract_m3u8(game_url):
    """Extract a playable m3u8 with exhaustive search and brute-force fallback."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Referer": game_url
    }
    
    parsed_game = urlparse(game_url)
    base_url = f"{parsed_game.scheme}://{parsed_game.netloc}"
    
    print(f"[*] Analyzing stream sources at {game_url}...")
    
    # Brute-force guessing priority list
    priority_suffixes = ["/admin/1", "/delta/1", "/echo/1", "/golf/1", "/bravo/1"]
    
    try:
        response = SCRAPER.get(game_url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. Identify direct links on the game page
        game_id = game_url.strip('/').split('/')[-1]
        found_links = soup.find_all('a', href=re.compile(r'/watch/'))
        
        candidate_paths = []
        for a in found_links:
            href = a['href']
            # We want paths that look like provider stream links
            if any(s in href for s in priority_suffixes) or len(href.split('/')) > 3:
                candidate_paths.append(href)
        
        # Combine found links with priority guesses (uniques only)
        final_search_paths = candidate_paths + priority_suffixes
        seen = set()
        final_search_paths = [x for x in final_search_paths if not (x in seen or seen.add(x))]
        
        # 2. Iterate and deep-dive
        for path in final_search_paths:
            stream_page = f"{base_url}{path}" if path.startswith('/') else path
            # Cleanup double slashes
            if "://" in stream_page:
                parts = stream_page.split("://")
                stream_page = parts[0] + "://" + parts[1].replace("//", "/")
            
            # First, check if there's an embed iframe on the stream page
            print(f"[*] Checking path: {stream_page}...")
            try:
                stream_response = SCRAPER.get(stream_page, headers=headers, timeout=10)
                
                # Check for embed iframe (embedsports.top)
                embed_match = re.search(r'https://embedsport[sy]\.top/embed/[^"\'\s]+', stream_response.text)
                if embed_match:
                    embed_url = embed_match.group(0)
                    # Deep-dive into the embed
                    m3u8 = deep_extract_m3u8(embed_url)
                    if m3u8: return m3u8
                
                # Also try deep-diving into the stream page itself (it might have the m3u8)
                m3u8 = deep_extract_m3u8(stream_page)
                if m3u8: return m3u8
                
            except:
                continue
                
    except Exception as e:
        print(f"(!) Primary extraction error: {e}")
        
    print("[-] Extraction failed to find a direct link. Falling back to manual embed dispatch...")
    # Last resort fallback: dispatch the guessed embed URL to Kodi (unlikely to work without plugin)
    game_id = game_url.strip('/').split('/')[-1]
    return f"https://embedsports.top/embed/admin/{game_id}/1"

def play_url(url, user_agent=None, referer=None, origin=None):
    """Sends play request to Kodi via JSON-RPC with smart header piping."""
    if not user_agent:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    
    # Auto-detect necessary headers based on domain
    parsed = urlparse(url)
    if "embedsport" in parsed.netloc:
        referer = "https://embedsports.top/"
        origin = "https://embedsports.top"
    elif "modifiles.fans" in parsed.netloc or "poocloud" in parsed.netloc:
        # These are the modern Pulsar-verified headers for Modifiles
        referer = "https://streamed.su/" 
        origin = "https://streamed.su"
    elif not referer:
        # Generic fallback
        referer = f"{parsed.scheme}://{parsed.netloc}/"
        
    if not origin and referer:
        origin = referer.rstrip('/')

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
    print(f"[*] Using CDN: {parsed.netloc}")

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
