#!/usr/bin/env python3
import sys
import argparse
import time
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=40):
    """
    Nuclear extractor with interactivity.
    Blindly clicks the center of the player to trigger loading.
    """
    target_m3u8 = None
    data_dir = Path("/home/spoon/new_puck/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = data_dir / "debug_pi_view.png"
    
    print(f"[*] Launching Nuclear Headless Extractor for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        # Launch Chromium (usually better for these sites than Firefox)
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1280, 'height': 720},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Listener to intercept the .m3u8 URL
        def handle_request(request):
            nonlocal target_m3u8
            # Look for the master playlist or index from known CDNs
            # Filter out ads/placeholders
            if ".m3u8" in request.url and "placeholder" not in request.url.lower():
                if not target_m3u8:
                    target_m3u8 = request.url
                    print(f"[+] Intercepted Stream Link: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        try:
            # 1. First Pass: Load initial page
            print(f"[*] Navigating to initial page...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=45000)
            time.sleep(4) 
            
            # 2. Base Game Page Detection (Finding provider buttons)
            if "/watch/" in url and not any(p in url for p in ["/admin/", "/delta/", "/echo/", "/golf/"]):
                print(f"[*] Finding provider links...", file=sys.stderr)
                # Broader search for provider links
                # Usually look like /watch/[id]/admin/1
                links = page.query_selector_all('a[href*="/watch/"]')
                provider_urls = []
                for link in links:
                    href = link.get_attribute("href")
                    if href and len(href.rstrip('/').split('/')) > 3:
                        provider_urls.append(href)
                
                if provider_urls:
                    selected = provider_urls[0]
                    for p_url in provider_urls:
                        p_low = p_url.lower()
                        if "admin" in p_low or "delta" in p_low:
                            selected = p_url
                            break
                    target_url = selected if "://" in selected else f"https://{url.split('/')[2]}{selected}"
                    print(f"[*] Navigating to provider: {target_url}", file=sys.stderr)
                    page.goto(target_url, wait_until="load", timeout=45000)
                    time.sleep(3)

            # 3. INTERACTION PASS (Trigger the player)
            if not target_m3u8:
                print(f"[*] No stream yet. Triggering blind clicks on player area...", file=sys.stderr)
                # Click center of screen (usually where the 'Play' button is located)
                page.mouse.click(640, 360) 
                time.sleep(1)
                page.mouse.click(640, 360) # Double click for good measure
                
                # Also try to find any iframe and click inside it
                iframes = page.frames
                for frame in iframes:
                    try:
                        if "embedsport" in frame.url or "modifiles" in frame.url:
                            print(f"[*] Found player frame. Attempting click inside...", file=sys.stderr)
                            # Clicking center of the first found frame
                            frame.wait_for_load_state("load")
                            time.sleep(2)
                            page.mouse.click(640, 360) # Still click at absolute center of viewport
                    except:
                        pass

            # 4. Final Verification
            start_time = time.time()
            print(f"[*] Waiting for HLS interception...", file=sys.stderr)
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(1)

            if not target_m3u8:
                print(f"[-] Interception failed. Screenshotting debug...", file=sys.stderr)
                page.screenshot(path=str(screenshot_path))

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
            try: page.screenshot(path=str(screenshot_path))
            except: pass
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulsar-X Interact")
    parser.add_argument("--url", required=True, help="URL to analyze")
    parser.add_argument("--timeout", type=int, default=25, help="Seconds to wait")
    
    args = parser.parse_args()
    
    result = extract_stream(args.url, timeout_secs=args.timeout)
    if result:
        print(result) # Master URL
        sys.exit(0)
    else:
        sys.exit(1)
