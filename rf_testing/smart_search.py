#!/usr/bin/env python3

import argparse
import time
import json
import os
import sys
from rpi_rf import RFDevice

# PREFERRED PIN: GPIO 17 (Physical Pin 11)
GPIO_TX = 17
FILES_DIR = os.path.dirname(__file__)
CODES_FILE = os.path.join(FILES_DIR, "remote_codes.json")

# Verified settings
PROTO = 1
PULSE = 150

    key = args.button.upper()
    if key not in data:
        # Fallback if key not found but user wants to run it: allow creating new entry
        print(f"Key '{key}' not in file. Using default or manual entry logic?")
        # Actually better to just ask for the seed code if not in file?
        # For now, let's assume if it's not in file, we can't seed it.
        # OR better: allow command line arg for seed?
        # Let's keep it simple: We use the value from the file if exists.
        print(f"Error: Button '{key}' not found in remote_codes.json. Please run sniff or add manually.")
        sys.exit(1)

    seed_code = data[key]['code']
    # Calculate the "Hex Page"
    # e.g. 4477185 = 0x445101
    # Page = 0x445100
    
    # Mask out the last byte (0xFF)
    base_prefix = seed_code & 0xFFFFFF00
    
    print(f"🧠 DYNAMIC SMART SEARCH for [{key}]")
    print(f"Seed Code: {seed_code} ({seed_code:#x})")
    print(f"Sweeping Hex Page: {base_prefix:#x} to {base_prefix + 0xFF:#x}")
    print(f"Range: {base_prefix} to {base_prefix + 255}")
    print("Press Ctrl+C IMMEDIATELY when the outlet reacts!")
    print("------------------------------------------------")
    
    start_code = base_prefix
    end_code = base_prefix + 255
    
    last_sent = 0
    
    try:
        rfdevice.tx_repeat = 6 # Fast but reliable
        
        for code in range(start_code, end_code + 1):
            last_sent = code
            hex_str = f"{code:#0{8}x}" # Format as 0x......
            print(f"👉 Testing: {code} ({hex_str})", end='\r')
            
            rfdevice.tx_code(code, PROTO, PULSE)
            time.sleep(0.12) # ~30 seconds for full byte sweep
            
        print("\n❌ Reached end of range without user interrupt.")
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 STOPPED at ~{last_sent} ({last_sent:#x})!")
        print("Entering FINE TUNE mode to lock it in.")
        
        current_code = last_sent - 3 # Backtrack slightly
        
        print(" Controls:")
        print("  [a] -1  (Previous)")
        print("  [d] +1  (Next)")
        print("  [s] FIRE Current")
        print("  [y] SAVE this code")
        
        while True:
            cmd = input(f"\rCurrent: {current_code} ({current_code:#x}) > ").strip().lower()
            
            if cmd == 'a': current_code -= 1
            elif cmd == 'd': current_code += 1
            elif cmd == 's': 
                rfdevice.tx_repeat = 15
                rfdevice.tx_code(current_code, PROTO, PULSE)
                print(" Fired.")
            elif cmd == 'y' or cmd == 'save':
                if not os.path.exists(CODES_FILE):
                     data = {}
                else:
                    with open(CODES_FILE, 'r') as f:
                        data = json.load(f)
                        
                print(f"💾 Saving {current_code} for [{key}]...")
                data[key] = {
                    "code": current_code,
                    "protocol": PROTO,
                    "pulselength": PULSE
                }
                with open(CODES_FILE, 'w') as f:
                    json.dump(data, f, indent=2)
                return
                
    finally:
        rfdevice.cleanup()

if __name__ == "__main__":
    main()
