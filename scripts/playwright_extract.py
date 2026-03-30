#!/usr/bin/env python3
import sys
import argparse
import time
import re
import os
from pathlib import Path
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=50):
    """
    Console-Mining extractor.
    Captures browser logs and broad network traffic to find the stream.
    """
    target_m3u8 = None
    data_dir = Path("/home/spoon/new_puck/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = data_dir / "debug_pi_view.png"
    
    print(f"[*] Launching Console-Mining Extractor for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # 1. Total Network Listener (Broad Patterns)
        def handle_request(request):
            nonlocal target_m3u8
            u = request.url.lower()
            if any(term in u for term in [".m3u8", "manifest", "master.json", "/chunklist", "/playlist.m3u8"]):
                print(f"[*] Traffic: {request.url[:80]}...", file=sys.stderr)
                if not target_m3u8 and "placeholder" not in u:
                    target_m3u8 = request.url
                    print(f"[!] Target Found via Network: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        # 2. Console Listener (Mine for "http" + "m3u8")
        def handle_console(msg):
            nonlocal target_m3u8
            text = msg.text
            if "http" in text and (".m3u8" in text or "manifest" in text):
                urls = re.findall(r'(https?://[^\s\'\"]+\.m3u8[^\s\'\"]*)', text)
                if urls and not target_m3u8:
                    target_m3u8 = urls[0]
                    print(f"[!] Target Found via Console: {target_m3u8}", file=sys.stderr)

        page.on("console", handle_console)

        try:
            # Step A: Load page
            print(f"[*] Loading page...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=60000)
            time.sleep(5) 
            
            # Step B: Discovery (Provider Links)
            if "/watch/" in url and not any(p in url for p in ["/admin/", "/delta/", "/echo/"]):
                links = page.query_selector_all('a[href*="/watch/"]')
                provider_urls = []
                for link in links:
                    href = link.get_attribute("href")
                    if href and len(href.rstrip('/').split('/')) > 3:
                        provider_urls.append(href)
                
                if provider_urls:
                    selected = provider_urls[0]
                    for p_url in provider_urls:
                        if "admin/1" in p_url.lower():
                            selected = p_url
                            break
                    target_url = selected if "://" in selected else f"https://{url.split('/')[2]}{selected}"
                    print(f"[*] Switching to provider: {target_url}", file=sys.stderr)
                    page.goto(target_url, wait_until="load", timeout=60000)
                    time.sleep(5)

            # Step C: Interaction (Aggressive Clicks)
            if not target_m3u8:
                print(f"[*] Triggering Aggressive Clicks...", file=sys.stderr)
                page.mouse.click(960, 540) # Absolute center
                time.sleep(2)
                
                iframes = page.frames
                for frame in iframes:
                    if target_m3u8: break
                    try:
                        f_url = frame.url.lower()
                        if "poo" in f_url or "embed" in f_url:
                            print(f"[*] Found Frame: {frame.url[:50]}", file=sys.stderr)
                            # Scroll to it and click center
                            page.mouse.click(960, 500)
                            time.sleep(5) # Long wait after click for HLS init
                    except: pass

            # Step D: Final Monitoring
            start_time = time.time()
            print(f"[*] Final monitoring for {timeout_secs}s...", file=sys.stderr)
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(1)

            if not target_m3u8:
                page.screenshot(path=str(screenshot_path))
                print(f"[-] No link found. Inspect screenshot at {screenshot_path}", file=sys.stderr)

        except Exception as e:
            print(f"(!) Error: {e}", file=sys.stderr)
            try: page.screenshot(path=str(screenshot_path))
            except: pass
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pulsar-X Console Miner")
    parser.add_argument("--url", required=True, help="URL")
    args = parser.parse_args()
    
    result = extract_stream(args.url)
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)
