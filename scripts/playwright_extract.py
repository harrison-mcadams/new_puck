#!/usr/bin/env python3
import sys
import argparse
import time
import re
from playwright.sync_api import sync_playwright

def extract_stream(url, timeout_secs=25):
    """
    Uses Playwright for the ENTIRE extraction process.
    Bypasses Cloudflare on game pages AND embed pages.
    """
    target_m3u8 = None
    print(f"[*] Launching Nuclear Headless Extractor for: {url}", file=sys.stderr)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        # Listener to intercept the .m3u8 URL
        def handle_request(request):
            nonlocal target_m3u8
            if ".m3u8" in request.url and any(cdn in request.url for cdn in ["modifiles.fans", "poocloud", "streamed"]):
                if not target_m3u8:
                    target_m3u8 = request.url
                    print(f"[+] Intercepted Stream: {target_m3u8}", file=sys.stderr)

        page.on("request", handle_request)

        try:
            # 1. Navigate to the game page (or embed page)
            print(f"[*] Navigating to initial page...", file=sys.stderr)
            page.goto(url, wait_until="load", timeout=30000)
            
            # 2. Scrape the page for provider links if it's a base game page
            if "/watch/" in url and (not any(p in url for p in ["/admin/", "/delta/", "/echo/", "/golf/"])):
                print(f"[*] Base game page detected. Finding provider links...", file=sys.stderr)
                time.sleep(2) # Wait for JS to render links
                
                # Look for Admin 1 or Delta 1 links
                # Usually hrefs contain /watch/[id]/admin/1
                links = page.query_selector_all('a[href*="/watch/"]')
                provider_urls = []
                for link in links:
                    href = link.get_attribute("href")
                    if href and len(href.split('/')) > 3:
                        provider_urls.append(href)
                
                if provider_urls:
                    # Prefer Admin or Delta
                    selected = provider_urls[0]
                    for p_url in provider_urls:
                        if "admin" in p_url.lower():
                            selected = p_url
                            break
                    
                    target_url = selected if "://" in selected else f"https://{url.split('/')[2]}{selected}"
                    print(f"[*] Navigating to provider: {target_url}", file=sys.stderr)
                    page.goto(target_url, wait_until="load", timeout=30000)

            # 3. Wait for the stream to appear (intercepted by the listener)
            start_time = time.time()
            while not target_m3u8 and (time.time() - start_time) < timeout_secs:
                time.sleep(0.5)

        except Exception as e:
            print(f"(!) Browser error: {e}", file=sys.stderr)
        finally:
            browser.close()

    return target_m3u8

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nuclear Stream Extractor")
    parser.add_argument("--url", required=True, help="Game or Embed URL to analyze")
    parser.add_argument("--timeout", type=int, default=25, help="Seconds to wait")
    
    args = parser.parse_args()
    
    result = extract_stream(args.url, timeout_secs=args.timeout)
    if result:
        print(result)
        sys.exit(0)
    else:
        sys.exit(1)
