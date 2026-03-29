#!/usr/bin/env python3
import sys
import os
import argparse
import requests
import json
from pathlib import Path

# Add project root to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))
from puck import config

def play_url(url, user_agent=None, referer=None, origin=None):
    """
    Sends a play request to Kodi via JSON-RPC.
    Kodi supports headers via a 'pipe' suffix: URL|Header1=Val&Header2=Val
    
    This technique is critical for bypassing 403 Forbidden errors on matches.
    """
    # Use verified Pulsar headers by default
    if not user_agent:
        user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    if not referer:
        referer = "https://streamed.su/"
    if not origin:
        origin = "https://streamed.su"

    # Construct the Kodi-style URL with headers
    # The | character initiates the header block in Kodi's stream URL parser
    kodi_url = f"{url}|User-Agent={user_agent}&Referer={referer}&Origin={origin}"
    
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
    print(f"[*] Payload URL (with pipe): {kodi_url[:100]}...")

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
            
    except requests.exceptions.ConnectionError:
        print(f"(!) Error: Could not connect to Kodi at {config.KODI_HOST}.")
        print("    Ensure the Raspberry Pi is reachable via Tailscale and Kodi is running.")
    except Exception as e:
        print(f"(!) RPC Call Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kodi Remote Stream Launcher")
    parser.add_argument("--url", help="Direct HLS/m3u8 URL to play")
    parser.add_argument("--test", action="store_true", help="Launch a public test stream")
    parser.add_argument("--ua", help="Custom User-Agent")
    parser.add_argument("--ref", help="Custom Referer")

    args = parser.parse_args()

    if args.test:
        # Public test stream (no headers usually required)
        test_url = "https://gg.poocloud.in/cdr_guadalajara/index.m3u8"
        play_url(test_url)
    elif args.url:
        play_url(args.url, user_agent=args.ua, referer=args.ref)
    else:
        parser.print_help()
