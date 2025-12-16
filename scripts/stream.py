#!/usr/bin/env python3
import subprocess
import sys
import os

"""
stream.py - Raspberry Pi 4 Optimized Sports Streamer

This script encapsulates the configuration needed to play high-framerate sports streams
smoothly on a Raspberry Pi 4. It handles header spoofing to bypass website protections
and sets specific mpv/decoder flags to ensure hardware acceleration works correctly.

Usage:
    python3 scripts/stream.py [optional_m3u8_url]

Prerequisites on Pi:
    sudo apt install streamlink mpv
    (Optional) Force 1080p60 in /boot/cmdline.txt if connected to a 4K TV.
"""

# Default URL for the specific stream source we found (EmbedSports / Poocloud)
# This will likely change for different games, so override via command line args.
DEFAULT_STREAM_URL = "https://gg.poocloud.in/cdr_guadalajara/index.m3u8"

# Headers required to bypass the "Google HTML" block / 403 Forbidden
# These mimic the browser request from the embedding site.
HEADERS = [
    "Referer=https://embedsports.top/",
    "Origin=https://embedsports.top",
    "User-Agent=Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"
]

# Raspberry Pi 4 Performance Tunings for mpv
# --profile=fast: Disables high-quality scalers (spline36) which are too heavy for Pi
# --vo=gpu: Uses the GPU for video output
# --hwdec=v4l2m2m_copy: The most stable hardware decoding path for Pi 4 (V4L2 Memory-to-Memory)
# --framedrop=vo: Drops video frames instead of freezing if synchronization is lost
# --ao=alsa: Uses ALSA directly for audio to reduce latency/lag vs PulseAudio
PLAYER_ARGS = "--fs --profile=fast --vo=gpu --hwdec=v4l2m2m_copy --framedrop=vo --ao=alsa"

# with frame drop tracking
PLAYER_ARGS = r"--fs --profile=fast --vo=gpu --hwdec=v4l2m2m_copy --framedrop=vo --ao=alsa --osd-level=1 --osd-msg1='FPS: ${estimated-vf-fps} / Dropped: ${vo-drop-frame-count}'"

def play_stream(stream_url):
    """
    Constructs and runs the optimized streamlink command.
    """
    # Ensure streamlink parses it as HLS if not specified
    if not stream_url.startswith("hls://") and not stream_url.startswith("http"):
         # Assume it's a direct url if neither, but streamlink handles http, we just prepend hls:// to force HLS mode if ambiguous
         pass
    
    # Streamlink command construction
    cmd = [
        "streamlink",
        f"hls://{stream_url}" if "hls://" not in stream_url else stream_url,
        "720p,best",                    # Prioritize 720p (smoother on Pi) before falling back to source/best
        "--hls-live-edge", "5",         # Buffer stability: stay 5 segments behind live
        "--ringbuffer-size", "32M",     # Network buffer: 32MB to handle Wi-Fi dips
        "--player", "mpv",
        "--player-args", PLAYER_ARGS
    ]

    # Add all headers
    for header in HEADERS:
        cmd.extend(["--http-header", header])

    # Environment variables to force display to the HDMI port (Display :0)
    # This ensures the video window opens on the TV even if run via SSH.
    env = os.environ.copy()
    env["DISPLAY"] = ":0"

    print(f"\n📺 Starting Stream on Display :0")
    print(f"🔗 URL: {stream_url}")
    print(f"🚀 Optimization: Pi 4 Mode (v4l2m2m_copy, 720p pref)")
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
    # Use command line argument if provided, otherwise default
    target_url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STREAM_URL
    play_stream(target_url)
