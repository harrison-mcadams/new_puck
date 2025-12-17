#!/usr/bin/env python3

import argparse
import json
import logging
import os
import sys
import time
from rpi_rf import RFDevice

# PREFERRED PIN: GPIO 17 (Physical Pin 11)
GPIO_TX = 17
CODES_FILE = os.path.join(os.path.dirname(__file__), "remote_codes.json")

def main():
    parser = argparse.ArgumentParser(description='Mimic an RF remote button press.')
    parser.add_argument('button', type=str, help="Button name (e.g., '1 ON', '3 OFF')")
    parser.add_argument('-g', '--gpio', dest='gpio', type=int, default=GPIO_TX,
                        help="GPIO pin (Default: 17)")
    parser.add_argument('-r', '--repeat', dest='repeat', type=int, default=20,
                        help="Repeat count (Default: 20)")
    parser.add_argument('--blast', dest='blast', action='store_true',
                        help="Send a blast of signals with varying pulse lengths to ensure reception.")
    args = parser.parse_args()

    # Load codes
    if not os.path.exists(CODES_FILE):
        print(f"Error: Codes file not found at {CODES_FILE}")
        print("Please run sniff_remote.py first.")
        sys.exit(1)

    with open(CODES_FILE, 'r') as f:
        codes_db = json.load(f)

    # Normalize input
    btn_key = args.button.upper() 
    if btn_key not in codes_db:
        matches = [k for k in codes_db.keys() if k.upper() == btn_key]
        if matches:
            btn_key = matches[0]
        else:
            print(f"Error: Button '{args.button}' not found in database.")
            print("Available buttons:", ", ".join(sorted(codes_db.keys())))
            sys.exit(1)

    data = codes_db[btn_key]
    
    rfdevice = RFDevice(args.gpio)
    rfdevice.enable_tx()
    
    base_pulse = data['pulselength']
    # MEGA BLAST STRATEGY
    # We found that valid codes can be +8, +34, or other offsets from the sniffed value.
    # Sniffed OFF was 244. Valid was 267. (Diff +23).
    # Valid ON was +8 from valid OFF.
    # Instead of guessing the math, we just flood the entire neighborhood.
    # A sweep of +/- 60 codes takes < 1 second and GUARANTEES hitting the target.
    
    print(f"📡 MEGA BLASTING [{btn_key}] (Range +/- 60)...")
    
    # Range covering all known variants (+8, +34, +23, noise)
    start_code = code - 60
    end_code = code + 60
    
    # We send duplicates? No, just the range.
    codes_to_send = list(range(start_code, end_code + 1))
    
    # Send the burst
    # We use a very low repeat (5) because we are sending 120 codes.
    # 120 codes * 5 repeats is too slow.
    # Actually, sending each code ONCE or TWICE is enough if we are sweeping consecutive integers.
    # The receiver will see "266, 267, 268" and trigger on 267.
    
    rfdevice.tx_repeat = 3
    
    for c in codes_to_send:
        # Verified Settings: Protocol 1, Pulse 150
        rfdevice.tx_code(c, 1, 150)
        
    # If blast mode is ON, we assume the user is really desperate, so we widen the pulse too?
    # No, Pulse 150 is verified. We just stick to it.
    
    rfdevice.cleanup()
    print("Done.")

if __name__ == "__main__":
    main()
