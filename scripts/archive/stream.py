#!/usr/bin/env python3
import subprocess
import sys
import os
import argparse
from urllib.parse import urlparse

"""
stream.py - Raspberry Pi 4 Optimized Sports Streamer

Usage:
    python3 scripts/stream.py [URL] [--team TEAM_NAME]

Examples:
    python3 scripts/stream.py "https://gg.poocloud.in/cdr_guadalajara/index.m3u8"
    python3 scripts/stream.py "hls://..."
"""

# ----------------- Configuration ----------------- #

# Known headers for common streaming backends (EmbedSports, PooCloud, etc.)
# These mimic a mobile Android device to bypass some PC-targeted ads/blocks.
DEFAULT_HEADERS = [
    "Referer=https://embedsports.top/",
    "Origin=https://embedsports.top",
    "User-Agent=Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
]

# Alternate headers for "Streamed.pk" if the direct link is used (experimental)
STREAMED_HEADERS = [
    "Referer=https://streamed.pk/",
    "Origin=https://streamed.pk",
    "User-Agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

# Raspberry Pi 4 Performance Tunings (mpv)
# --profile=fast: Disables high-quality scalers (spline36) which are too heavy for Pi
# --vo=gpu: Uses the GPU for video output
# --hwdec=v4l2m2m_copy: The most stable hardware decoding path for Pi 4 (V4L2 Memory-to-Memory)
# --framedrop=vo: Drops video frames instead of freezing if synchronization is lost
# --ao=alsa: REMOVED (Let mpv auto-detect)
# --osd-level=1: REMOVED (Clean output)


# ----------------- functions ----------------- #

def get_headers_for_url(url):
    """Selects the best headers based on the URL domain."""
    if any(domain in url for domain in ["embedsports", "poocloud", "xyz"]):
        return DEFAULT_HEADERS
    elif any(domain in url for domain in ["streamed.pk", "strmd.top", "modifiles.fans", "fans"]):
        return STREAMED_HEADERS
    else:
        # Default to the mobile headers as they are generally more permissive
        return DEFAULT_HEADERS

def play_stream(stream_url, quality="720p,best", referer=None, user_agent=None, origin=None):
    """
    Constructs and runs the optimized streamlink command.
    """
    # Ensure streamlink parses it as HLS if not specified
    if not stream_url.startswith("hls://") and not stream_url.startswith("http"):
         print(f"❌ Invalid URL format: {stream_url}")
         return
    
    # 2. Build Command
    PLAYER_ARGS_CLEAN = r"--fs --profile=fast --vo=gpu --hwdec=v4l2m2m_copy --framedrop=vo --ao=alsa --audio-device=alsa/hdmi:CARD=vc4hdmi0,DEV=0 --x11-bypass-compositor=yes"
    
    cmd = [
        "streamlink",
        f"hls://{stream_url}" if "hls://" not in stream_url else stream_url,
        quality,
        "--hls-live-edge", "5",
        "--ringbuffer-size", "32M",
        "--player", "mpv",
        "--player-args", PLAYER_ARGS_CLEAN
    ]

    # 3. Add Headers
    headers_list = get_headers_for_url(stream_url)
    
    # Convert list to dict for easier override
    headers_dict = {}
    for h in headers_list:
        if "=" in h:
            k, v = h.split("=", 1)
            headers_dict[k] = v

    # Manual overrides
    if referer:
        headers_dict["Referer"] = referer
        if not origin:
            # Auto-set origin from referer if not specified
            headers_dict["Origin"] = referer.rstrip("/")
    if user_agent:
        headers_dict["User-Agent"] = user_agent
    if origin:
        headers_dict["Origin"] = origin

    for k, v in headers_dict.items():
        cmd.extend(["--http-header", f"{k}={v}"])

    # 4. Environment Variables
    env = os.environ.copy()
    env["DISPLAY"] = ":0"

    print(f"\n📺 Starting Stream on Display :0")
    print(f"🔗 URL: {stream_url}")
    print(f"📡 Headers summary:")
    for k, v in headers_dict.items():
        print(f"   🔹 {k}: {v[:60]}{'...' if len(v) > 60 else ''}")
    print(f"🚀 Optimization: Pi 4 Mode (v4l2m2m_copy, {quality} pref)")
    print("-" * 60)
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError:
        print("\n❌ Streamlink exited with error (or stream ended).")
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped by user.")
    except FileNotFoundError:
        print("\n❌ Error: 'streamlink' is not installed. Run: sudo apt install streamlink")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play sports streams on Raspberry Pi 4")
    parser.add_argument("url", nargs="?", help="The .m3u8 stream URL")
    parser.add_argument("--team", help="Team name to search for (Not yet implemented)")
    parser.add_argument("--quality", default="720p,best", help="Stream quality (default: 720p,best). Use 'best' for 1080p.")
    parser.add_argument("--referer", help="Custom Referer header")
    parser.add_argument("--user-agent", help="Custom User-Agent header")
    parser.add_argument("--origin", help="Custom Origin header")
    
    args = parser.parse_args()

    if args.url:
        play_stream(args.url, quality=args.quality, referer=args.referer, user_agent=args.user_agent, origin=args.origin)
    elif args.team:
        print(f"🔍 Auto-discovery for '{args.team}' is not yet implemented (Anti-bot protection).")
        print("   Please extract the .m3u8 link manually from the browser DevTools (Network tab).")
    else:
        # Fallback to the known good test stream if nothing provided
        print("⚠️ No URL provided. Playing default test stream (Guadalajara).")
        play_stream("https://gg.poocloud.in/cdr_guadalajara/index.m3u8", quality=args.quality, referer=args.referer, user_agent=args.user_agent, origin=args.origin)
