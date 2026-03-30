#!/usr/bin/env python3
import sys
import os
import argparse
import requests
import re
import subprocess
from pathlib import Path
from urllib.parse import quote, urlparse

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))
from puck import config

def extract_via_playwright(page_url):
    """Call the Turbo Playwright extractor with a long 120s timeout."""
    print(f"[*] Turbo Headless Discovery in progress (120s limit)...")
    script_path = Path(__file__).parent / "playwright_extract.py"
    
    try:
        # Long timeout for the Pi browser to resolve the manifest
        result = subprocess.run(
            [sys.executable, str(script_path), "--url", page_url],
            capture_output=True, text=True, timeout=120
        )
        
        if result.returncode == 0:
            m3u8 = result.stdout.strip()
            if m3u8:
                print(f"[+] Turbo Success: {m3u8}")
                return m3u8
        else:
            print(f"(!) Turbo Extraction Failed: {result.stderr}")
            
    except Exception as e:
        print(f"(!) Failed to invoke Playwright script: {e}")
        
    return None

def play_url(url, user_agent=None, referer=None, origin=None):
    """Sends play request to Kodi via JSON-RPC with smart header piping."""
    if not user_agent:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    
    parsed = urlparse(url)
    if "embedsport" in parsed.netloc:
        referer = "https://embedsports.top/"
        origin = "https://embedsports.top"
    elif "modifiles.fans" in parsed.netloc or "poocloud" in parsed.netloc:
        referer = "https://streamed.su/" 
        origin = "https://streamed.su"
        
    from urllib.parse import quote
    headers_str = f"User-Agent={quote(user_agent)}&Referer={quote(referer)}&Origin={quote(origin)}"
    kodi_url = f"{url}|{headers_str}"
    
    payload = {
        "jsonrpc": "2.0",
        "method": "Player.Open",
        "params": {"item": {"file": kodi_url}},
        "id": 1
    }
    
    auth = (config.KODI_USER, config.KODI_PASS)
    rpc_url = f"http://{config.KODI_HOST}:{config.KODI_PORT}/jsonrpc"
    
    print(f"[*] Dispatching stream to Kodi...")
    print(f"[*] Target: {parsed.netloc}")
    try:
        response = requests.post(rpc_url, json=payload, auth=auth, timeout=10)
        response.raise_for_status()
        if response.json().get('result') == "OK":
            print("[+] Success: Playback command accepted by Kodi.")
    except Exception as e:
        print(f"(!) RPC Call Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kodi Turbo Launcher")
    parser.add_argument("--url", help="Direct HLS/m3u8 URL to play")
    parser.add_argument("--game-url", help="Direct game page URL")
    parser.add_argument("--team", help="Team name to discover (e.g. Flyers)")
    parser.add_argument("--ua", help="Custom User-Agent")
    parser.add_argument("--ref", help="Custom Referer")

    args = parser.parse_args()

    if args.url:
        play_url(args.url, user_agent=args.ua, referer=args.ref)
    elif args.game_url:
        stream_url = extract_via_playwright(args.game_url)
        if stream_url:
            play_url(stream_url, user_agent=args.ua, referer=args.ref)
    elif args.team:
        print(f"[*] Discovery blocked by Cloudflare. Running brute-force guess for '{args.team}'...")
        # Friendly URL guess
        game_url_guess = f"https://streamed.pk/watch/ppv-dallas-stars-vs-philadelphia-flyers"
        stream_url = extract_via_playwright(game_url_guess)
        if stream_url:
            play_url(stream_url, user_agent=args.ua, referer=args.ref)
    else:
        parser.print_help()
