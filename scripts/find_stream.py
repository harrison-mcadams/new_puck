import asyncio
import argparse
import sys
import datetime
from playwright.async_api import async_playwright

def log(msg):
    with open("find_stream_log.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")
    print(msg) # Still print for local dev if needed

async def find_stream(team_name):
    log(f"🔍 Searching for '{team_name}' stream on streamed.pk...")
    
    async with async_playwright() as p:
        # Launch browser (headless=True for background, False for debugging)
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Variable to store the found stream
        found_stream = None

        # Define the network sniffer
        async def handle_request(request):
            nonlocal found_stream
            if ".m3u8" in request.url and "master" not in request.url and found_stream is None:
                # We want the index/playlist, not the master if possible, or whatever plays
                # Filtering out 'master' is sometimes risky, let's just grab the first valid m3u8
                # that isn't a segment
                if "segment" not in request.url and ".ts" not in request.url:
                    log(f"🎯 FOUND SIGNATURE: {request.url[:50]}...")
                    found_stream = {
                        "url": request.url,
                        "headers": request.headers
                    }

        # Attach sniffer
        page.on("request", handle_request)

        try:
            # 1. Go to Schedule
            log("Navigating to schedule...")
            await page.goto("https://streamed.pk/schedule", timeout=20000, wait_until="domcontentloaded")
            
            # 2. Click "Hockey" (It's usually a button or a tab)
            # Based on the HTML we saw, it might be in a carousel or list.
            # Let's search for the text "Hockey" and click it just to be safe filters are valid
            # await page.get_by_text("Hockey", exact=True).click() # Might not be necessary if "All" shows everything
            # Actually, standard view shows everything. Let's just search for the team.

            # 3. Search for the Team
            # use a broad selector
            log(f"👀 Looking for team '{team_name}' in matches...")
            # Wait a bit for svelte to hydrate
            await page.wait_for_timeout(3000)
            
            match_locator = None
            if team_name == "FIRST_AVAILABLE":
                # Create a locator for ANY match card/link. 
                # Inspecting similar sites, they usually have "match-preview" or similar classes.
                # Or we can just look for the first link inside the main specific container if we knew it.
                # Let's try to match a generic "vs" text which is usually in match titles
                log("🤷‍♂️ No specific team invited. Picking the first match with 'vs'...")
                match_matches = page.get_by_text("vs") # This might be risky if 'vs' is elsewhere
                if await match_matches.count() > 0:
                    match_locator = match_matches
            else:
                # Try to click the match card containing the team name
                match_locator = page.get_by_text(team_name)
            
            if match_locator and await match_locator.count() > 0:
                log("✅ Match found! Clicking...")
                # Note: Might be multiple (e.g. "Flyers vs ..."), click the first
                await match_locator.first.click()
            else:
                log("❌ Match not found in schedule")
                # Debug screenshot to see what IS there
                await page.screenshot(path="debug_schedule.png")
                await browser.close()
                return None

            # 4. Select Stream Source
            log("⏳ Waiting for stream selection page...")
            await page.wait_for_timeout(3000)
            
            # Debug: See what this page looks like
            await page.screenshot(path="debug_stream_page.png")
            log("📸 Captured debug_stream_page.png")

            # Try strict "Stream 1" then loose "Stream"
            log("👀 Looking for 'Stream 1'...")
            
            # Check for iframes or new windows? (Usually not, but checking frames)
            # The click logic:
            clicked = False
            
            # Try specific "Stream 1" button/link
            stream_1 = page.get_by_text("Stream 1", exact=True)
            if await stream_1.count() > 0:
                 log("✅ Found exact 'Stream 1'. Clicking.")
                 await stream_1.first.click()
                 clicked = True
            else:
                 log("⚠️ Exact 'Stream 1' not found. Checking partial match...")
                 stream_partial = page.get_by_text("Stream 1") # case sensitive? default is insensitive mostly
                 if await stream_partial.count() > 0:
                      log("✅ Found partial 'Stream 1'. Clicking.")
                      await stream_partial.first.click()
                      clicked = True
                 else:
                      log("⚠️ 'Stream 1' failed. Trying generic 'Stream' bit....")
                      # Maybe it's "HD Stream" or "SD Stream"
                      stream_generic = page.locator("text=Stream").first
                      if await stream_generic.count() > 0:
                           log("✅ Found generic 'Stream' element. Clicking.")
                           await stream_generic.click()
                           clicked = True
            
            if not clicked:
                 log("❌ No known stream links found. Aborting.")
                 await browser.close()
                 return None
            
            # 5. Wait for Player to Load
            log("⏳ Waiting for player playback (m3u8 request)...")
            
            # Anti-Sandbox Strategy B: Extract Iframe SRC and Navigate
            log("🕵️ Checking for blocked iframes...")
            # We look for iframes with sandbox attribute or just the main video iframe
            # Usually it's the only iframe in the video container
            
            iframe_src = await page.evaluate("""() => {
                const iframes = document.querySelectorAll('iframe');
                for (const f of iframes) {
                    // Check if it looks like a player (often has embed or allowed fullscreen)
                    if (f.src && (f.src.includes('embed') || f.src.includes('xyz') || f.hasAttribute('allowfullscreen'))) {
                        return f.src;
                    }
                }
                return null;
            }""")

            if iframe_src:
                log(f"🚀 Found player iframe: {iframe_src}")
                log("🚀 Navigating DIRECTLY to iframe to bypass sandbox...")
                try:
                    # 'commit' means wait until the server responds and we start receiving data.
                    await page.set_extra_http_headers({"Referer": "https://streamed.pk/"})
                    
                    # Anti-Sandbox Strategy C: Pre-emptive MutationObserver
                    # Injected BEFORE the page loads so it catches iframes on creation
                    await page.add_init_script("""
                        const observer = new MutationObserver((mutations) => {
                            for (const mutation of mutations) {
                                for (const node of mutation.addedNodes) {
                                    if (node.tagName === 'IFRAME') {
                                        if (node.hasAttribute('sandbox')) {
                                            console.log('🚫 Stripping sandbox from new iframe:', node);
                                            node.removeAttribute('sandbox');
                                        }
                                    }
                                }
                                if (mutation.type === 'attributes' && mutation.target.tagName === 'IFRAME') {
                                     if (mutation.attributeName === 'sandbox' && mutation.target.hasAttribute('sandbox')) {
                                          console.log('🚫 Stripping executed sandbox attribute change:', mutation.target);
                                          mutation.target.removeAttribute('sandbox');
                                     }
                                }
                            }
                        });
                        observer.observe(document.documentElement, {
                            childList: true,
                            subtree: true,
                            attributes: true,
                            attributeFilter: ['sandbox']
                        });
                        console.log('🛡️ Anti-Sandbox Observer Active');
                    """)

                    await page.goto(iframe_src, timeout=15000, wait_until="commit")
                    log("✅ Navigation committed. Waiting for m3u8...")

                    # Re-inject anti-sandbox in this new context/page
                    log("💉 Re-injecting anti-sandbox script for nested iframes...")
                    await page.evaluate("""
                        setInterval(() => {
                            document.querySelectorAll('iframe').forEach(f => {
                                if (f.hasAttribute('sandbox')) {
                                    console.log('Removing sandbox from', f);
                                    f.removeAttribute('sandbox');
                                }
                            });
                        }, 500);
                    """)
                    
                    # Screenshot to see the player
                    await page.wait_for_timeout(2000)
                    await page.screenshot(path="debug_iframe.png")
                    log("📸 Captured debug_iframe.png")
                    
                    # Try to click center of screen to trigger
                    log("👆 Clicking center of screen...")
                    try:
                        await page.mouse.click(500, 300)
                    except:
                        pass
                    
                except Exception as e:
                    log(f"⚠️ Navigation warning (might be blocked): {e}")
            else:
                 log("⚠️ No obvious player iframe found. Continuing...")

            # The player usually takes a few seconds to initialize and request the m3u8
            # We wait up to 25 seconds for the m3u8 request to fire
            for i in range(25):
                if found_stream:
                    break
                
                # Try clicking play overlay again
                try:
                    play_overlay = page.locator(".vjs-big-play-button") 
                    if await play_overlay.count() > 0 and await play_overlay.is_visible():
                         log("▶️ Clicking Play Overlay...")
                         await play_overlay.click()
                except:
                    pass
                await asyncio.sleep(1)
            
            if found_stream:
                log("\n🎉 STREAM SECURED!")
                log(f"URL: {found_stream['url']}")
                return found_stream['url']
            else:
                log("⚠️ Timeout: Video player loaded but no .m3u8 request detected.")
                log("📸 Taking debug screenshot: debug_fail.png")
                await page.screenshot(path="debug_fail.png")
                
        except Exception as e:
            log(f"❌ Error: {e}")
            await page.screenshot(path="debug_error.png")
        finally:
            await browser.close()

if __name__ == "__main__":
    # Clear log
    with open("find_stream_log.txt", "w") as f:
        f.write("--- Start ---\n")

    parser = argparse.ArgumentParser()
    parser.add_argument("team", nargs="?", help="Team name to find (e.g. 'Flyers'). If omitted, picks first match.")
    args = parser.parse_args()
    
    # If no team provided, pass None to trigger "pick first" logic (need to update find_stream to handle this)
    target = args.team if args.team else "FIRST_AVAILABLE"
    
    url = asyncio.run(find_stream(target))
    if url:
        print(f"\nCOMMAND_TO_RUN:\npython3 scripts/stream.py \"{url}\"")
        log(f"Outputting command for: {url}")
