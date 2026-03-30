#!/usr/bin/env python3
import sys
import argparse
import time
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=80):
    """
    Turbo-optimised Console Miner for Raspberry Pi.
    Blocks images/ads to save CPU and uses long timeouts.
    """
    target_m3u8 = None
    data_dir = Path("/home/spoon/new_puck/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = data_dir / "debug_pi_view.png"
    
    print(f"[*] Turbo Launch for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        # Launch Chromium (optimized) - HEADLESS=FALSE is required
        # Player blocks headless=True, causing the 120s timeouts!
        browser = p.chromium.launch(headless=False, args=['--no-sandbox'])
        context = browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Mask Playwright automation (Stealth Mode) so the player's obfuscated JS doesn't silently block us
        page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page.add_init_script("window.navigator.chrome = { runtime: {} };")
        page.add_init_script("Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3]});")
        # and likely aborted the .m3u8 fetch itself. The Pi will just have to load the full page.
        
        # 2. Network Listener (Total Intercept)
        def handle_request(request):
            nonlocal target_m3u8
            u = request.url.lower()
            if ".m3u8" in u or "manifest" in u or "master.json" in u:
                if not target_m3u8 and "placeholder" not in u:
                    target_m3u8 = request.url
                    print(f"[!] Traffic Found: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        # 3. Console Listener (Mine for URLs)
        def handle_console(msg):
            nonlocal target_m3u8
            text = msg.text
            if "http" in text and (".m3u8" in text or "manifest" in text):
                urls = re.findall(r'(https?://[^\s\'\"]+\.m3u8[^\s\'\"]*)', text)
                if urls and not target_m3u8:
                    target_m3u8 = urls[0]
                    print(f"[!] Console Found: {target_m3u8}", file=sys.stderr)

        page.on("console", handle_console)

        try:
            # Step A: Load page (Longer timeout for Pi)
            print(f"[*] Navigating (90s limit)...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=90000)
            time.sleep(10) 
            
            # Step B: Faster Discovery (Skip one navigation if possible)
            if "/watch/" in url and not any(p in url for p in ["/admin/", "/delta/", "/echo/"]):
                print(f"[*] Game page detected. Jumping to provider...", file=sys.stderr)
                links = page.query_selector_all('a[href*="/watch/"]')
                for link in links:
                    href = link.get_attribute("href")
                    if href and ("admin/1" in href or "delta/1" in href):
                        target_url = href if "://" in href else f"https://{url.split('/')[2]}{href}"
                        page.goto(target_url, wait_until="domcontentloaded", timeout=60000)
                        break
                time.sleep(10)

            # Step C: Heavy Interaction (Frame Penetration)
            if not target_m3u8:
                print(f"[*] Still searching. Triggering player inside nested iframes...", file=sys.stderr)
                time.sleep(5)
                # Scroll to wake up lazy-loaded iframes
                page.evaluate("window.scrollTo(0, 500)")
                time.sleep(5)
                
                # Iterate through all frames (flattened hierarchy) and explicitly click inside the player frames
                for frame in page.frames:
                    url_low = frame.url.lower()
                    if "pooembed" in url_low or "embed" in url_low or "modifiles" in url_low:
                        print(f"[*] Deep-clicking inside frame: {frame.url[:50]}...", file=sys.stderr)
                        try:
                            # Force click the specific container we saw in its HTML! 
                            # (Removed 'body' from fallback because .first was catching the background!)
                            frame.locator('#player, video, .jw-video, button_parent, .fp-ui').first.click(force=True, timeout=5000)
                        except Exception as e:
                            print(f"[!] Force-click ignored: {e}", file=sys.stderr)
                            try:
                                frame.evaluate('document.elementFromPoint(window.innerWidth/2, window.innerHeight/2)?.click()')
                            except: pass
                
                time.sleep(10)

            # Step D: Final Monitoring
            start_time = time.time()
            print(f"[*] Final {timeout_secs}s monitoring...", file=sys.stderr)
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(1)

            if not target_m3u8:
                page.screenshot(path=str(screenshot_path))
                print(f"[-] Silent failure. Check {screenshot_path}", file=sys.stderr)

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
            try: page.screenshot(path=str(screenshot_path))
            except: pass
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turbo extraction")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    
    result = extract_stream(args.url)
    if result:
        print(result) # Master HLS link
        sys.exit(0)
    else:
        sys.exit(1)
