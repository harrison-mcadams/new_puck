#!/usr/bin/env python3
import sys
import argparse
import time
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=45):
    """
    Wide-Net extractor with total interception.
    Attempts to catch ANY m3u8 request by interacting aggressively.
    """
    target_m3u8 = None
    data_dir = Path("/home/spoon/new_puck/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = data_dir / "debug_pi_view.png"
    
    print(f"[*] Launching Wide-Net Extractor for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Larger viewport to catch full player layouts
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # UNFILTERED Listener: Log EVERY m3u8
        def handle_request(request):
            nonlocal target_m3u8
            if ".m3u8" in request.url:
                print(f"[*] Saw HLS Link: {request.url}", file=sys.stderr)
                if not target_m3u8 and "placeholder" not in request.url.lower():
                    target_m3u8 = request.url
                    print(f"[!] Target Locked: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        try:
            # 1. Load initial page
            print(f"[*] Loading page...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=60000)
            time.sleep(5) 
            
            # 2. Base Game Page Detection (Finding provider buttons)
            if "/watch/" in url and not any(p in url for p in ["/admin/", "/delta/", "/echo/", "/golf/"]):
                print(f"[*] Discovery phase: Navigating to stream provider...", file=sys.stderr)
                links = page.query_selector_all('a[href*="/watch/"]')
                provider_urls = []
                for link in links:
                    href = link.get_attribute("href")
                    if href and len(href.rstrip('/').split('/')) > 3:
                        provider_urls.append(href)
                
                if provider_urls:
                    # Prefer Admin 1
                    selected = provider_urls[0]
                    for p_url in provider_urls:
                        if "admin/1" in p_url.lower():
                            selected = p_url
                            break
                    target_url = selected if "://" in selected else f"https://{url.split('/')[2]}{selected}"
                    print(f"[*] Navigating to provider: {target_url}", file=sys.stderr)
                    page.goto(target_url, wait_until="load", timeout=60000)
                    time.sleep(5)

            # 3. INTERACTION PASS: Click and Scroll
            if not target_m3u8:
                print(f"[*] No stream yet. Triggering aggressive interaction...", file=sys.stderr)
                # Scroll down to ensure player is in view
                page.evaluate("window.scrollTo(0, 300)")
                time.sleep(1)
                
                # Click center of screen (usually where the 'Play' button / iframe is)
                coords = [(640, 360), (960, 540), (800, 450)]
                for x, y in coords:
                    if target_m3u8: break
                    print(f"[*] Clicking at {x}, {y}...", file=sys.stderr)
                    page.mouse.click(x, y) 
                    time.sleep(2)
                
                # Broadest search for any iframe and clicking inside
                iframes = page.frames
                print(f"[*] Scanning {len(iframes)} frames for player triggers...", file=sys.stderr)
                for frame in iframes:
                    if target_m3u8: break
                    try:
                        if any(cdn in frame.url for cdn in ["embed", "poo", "stream"]):
                            print(f"[*] Found likely player iframe: {frame.url[:50]}", file=sys.stderr)
                            # Clicking center of the first found frame
                            page.mouse.click(960, 500) # Re-click center
                            time.sleep(3)
                    except: pass

            # 4. Final Verification
            start_time = time.time()
            print(f"[*] Waiting for HLS interception...", file=sys.stderr)
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(1)

            # Diagnostic dump on failure
            if not target_m3u8:
                print(f"[-] Interception failed. Screenshotting debug...", file=sys.stderr)
                page.screenshot(path=str(screenshot_path))
                with open(data_dir / "debug_page_source.html", "w", encoding="utf-8") as f:
                    f.write(page.content())

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
            try: page.screenshot(path=str(screenshot_path))
            except: pass
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulsar-X Wide-Net")
    parser.add_argument("--url", required=True, help="URL to analyze")
    parser.add_argument("--timeout", type=int, default=15, help="Seconds to wait")
    
    args = parser.parse_args()
    
    result = extract_stream(args.url, timeout_secs=args.timeout)
    if result:
        print(result) # Master URL
        sys.exit(0)
    else:
        sys.exit(1)
