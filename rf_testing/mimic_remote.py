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
    protocol = data['protocol']
    code = data['code']

    # Smart Code Blasting
    # The sniffer might capture a "noisy" variant (e.g. 225 vs 259).
    # Instead of guessing, we send the "Neighborhood" of probable valid codes.
    # Pattern seen: +/- 32, +/- 34, +/- 1
    
    codes_to_try = [code]
    offsets = [32, -32, 34, -34, 1, -1]
    
    for off in offsets:
        codes_to_try.append(code + off)
        
    # Remove duplicates
    codes_to_try = sorted(list(set(codes_to_try)))
    
    print(f"📡 Sending neighborhood blast ({len(codes_to_try)} codes) for [{btn_key}]...")
    
    # Send the burst
    for c in codes_to_try:
        # Verified Settings: Protocol 1, Pulse 150
        rfdevice.tx_repeat = 10 # Shorter repeat per code to keep total time low
        rfdevice.tx_code(c, 1, 150)
        
    # If blast mode is ON, we do it even harder (pulse variations + code variations)
    if args.blast:
        print(f"💥 SUPER BLASTING (Pulse Variations)...")
        pulse_offsets = [-4, 4, -8, 8]
        for c in codes_to_try:
            for p_off in pulse_offsets:
                rfdevice.tx_code(c, 1, 150 + p_off)
            
    rfdevice.cleanup()
    print("Done.")

if __name__ == "__main__":
    main()
