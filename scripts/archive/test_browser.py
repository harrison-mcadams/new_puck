import asyncio
from playwright.async_api import async_playwright
import datetime

def log(msg):
    with open("browser_log.txt", "a") as f:
        f.write(f"{datetime.datetime.now()} - {msg}\n")

async def test():
    log("Starting test")
    async with async_playwright() as p:
        log("Launching browser")
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        log("Navigating to example.com")
        await page.goto("https://example.com")
        title = await page.title()
        log(f"Title: {title}")
        await browser.close()
    log("Done")

if __name__ == "__main__":
    asyncio.run(test())
