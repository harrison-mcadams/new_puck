#!/usr/bin/env python3
import sys
import argparse
import time
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=30):
    """
    Nuclear extractor with diagnostic support.
    Saves a debug_pi_view.png if it fails.
    """
    target_m3u8 = None
    data_dir = Path("/home/spoon/new_puck/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = data_dir / "debug_pi_view.png"
    
    print(f"[*] Launching Nuclear Headless Extractor for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a high-end desktop UA to avoid mobile/bot redirects
        context = browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Listener to intercept the .m3u8 URL
        def handle_request(request):
            nonlocal target_m3u8
            # Look for ANY .m3u8 link that isn't a placeholder
            if ".m3u8" in request.url:
                if not target_m3u8 and "placeholder" not in request.url.lower():
                    target_m3u8 = request.url
                    print(f"[+] Intercepted Stream Link: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        try:
            # 1. Navigate to the game page
            print(f"[*] Navigating to initial page...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=45000)
            time.sleep(3) # Wait for initial rendering
            
            # 2. Check for the "Friendly" vs "ID" game page links
            # If we don't have a stream URL yet, look for provider links
            if not target_m3u8:
                print(f"[*] Analyzing page structure for provider links...", file=sys.stderr)
                # Broader link search: any link with /watch/ that has more than 3 segments
                links = page.query_selector_all('a[href*="/watch/"]')
                provider_urls = []
                for link in links:
                    href = link.get_attribute("href")
                    if href and len(href.rstrip('/').split('/')) > 3:
                        provider_urls.append(href)
                
                if provider_urls:
                    # Pick Admin or Delta selectively
                    selected = provider_urls[0]
                    for p_url in provider_urls:
                        p_low = p_url.lower()
                        if "admin" in p_low or "delta" in p_low or "echo" in p_low:
                            selected = p_url
                            break
                    
                    target_url = selected if "://" in selected else f"https://{url.split('/')[2]}{selected}"
                    print(f"[*] Navigating to selected provider: {target_url}", file=sys.stderr)
                    page.goto(target_url, wait_until="load", timeout=45000)
                else:
                    # If no provider links found, maybe we're already on a stream page or it's hidden
                    # Try clicking any button that looks like a play button or provider name
                    print(f"[*] No direct links found. Trying to find button elements...", file=sys.stderr)
                    buttons = page.query_selector_all('button, div[role="button"], a.btn')
                    for btn in buttons:
                        text = btn.inner_text().lower()
                        if any(p in text for p in ["admin", "delta", "echo", "golf", "stream"]):
                            print(f"[*] Found likely button '{text}'. Clicking...", file=sys.stderr)
                            btn.click()
                            time.sleep(2)
                            break

            # 3. Final Wait for interception
            start_time = time.time()
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(1)

            # 4. Diagnostic Screenshot on Failure
            if not target_m3u8:
                print(f"[-] Failed to intercept stream. Saving diagnostic screenshot to {screenshot_path}", file=sys.stderr)
                page.screenshot(path=str(screenshot_path))
                # Dump HTML log for deep inspection
                with open(data_dir / "debug_page_source.html", "w", encoding="utf-8") as f:
                    f.write(page.content())

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
            # Take a screenshot even on crash
            try: page.screenshot(path=str(screenshot_path))
            except: pass
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulsar-X Diagnostic Extractor")
    parser.add_argument("--url", required=True, help="Game or Embed URL")
    parser.add_argument("--timeout", type=int, default=30, help="Wait timeout")
    
    args = parser.parse_args()
    
    result = extract_stream(args.url, timeout_secs=args.timeout)
    if result:
        print(result) # Print ONLY the URL
        sys.exit(0)
    else:
        sys.exit(1)
