#!/usr/bin/env python3
import sys
import argparse
import time
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=15):
    """
    Uses Playwright to navigate to a stream page and intercept the .m3u8 request.
    This bypasses Cloudflare and ensures the token is for the Pi's IP.
    """
    target_m3u8 = None
    print(f"[*] Launching headless browser for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        # Using a realistic User-Agent to match what we will send to Kodi
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Listener to intercept the .m3u8 URL
        def handle_request(request):
            nonlocal target_m3u8
            # Look for the master playlist or index from known CDNs
            if ".m3u8" in request.url and ("modifiles.fans" in request.url or "poocloud" in request.url):
                if not target_m3u8:
                    target_m3u8 = request.url
                    print(f"[+] Intercepted Stream: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        try:
            # Navigate and wait for some time for the player to load
            page.goto(url, wait_until="load", timeout=20000)
            
            # Wait a few seconds for network activity to settle/player to start
            start_time = time.time()
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(0.5)

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Stream Extractor")
    parser.add_argument("--url", required=True, help="Embed/Stream URL to analyze")
    parser.add_argument("--timeout", type=int, default=15, help="Seconds to wait for interception")
    
    args = parser.parse_args()
    
    result = extract_stream(args.url, timeout_secs=args.timeout)
    if result:
        # Print ONLY the URL to stdout for easy parsing by kodi_play.py
        print(result)
        sys.exit(0)
    else:
        print("[-] No stream link found within timeout.", file=sys.stderr)
        sys.exit(1)
