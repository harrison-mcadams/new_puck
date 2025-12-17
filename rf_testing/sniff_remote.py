#!/usr/bin/env python3

import argparse
import time
import json
import os
import sys
from rpi_rf import RFDevice

# Configuration
GPIO_RX = 27
BUTTONS = [
    "1 ON", "1 OFF", 
    "2 ON", "2 OFF", 
    "3 ON", "3 OFF", 
    "4 ON", "4 OFF", 
    "5 ON", "5 OFF"
]
SAMPLES_NEEDED = 5
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "remote_codes.json")

def capture_button(rfdevice, button_name):
    timestamp = None
    samples = []
    
    print(f"\n--- RECORDING: [{button_name}] ---")
    print(f"Please press and HOLD the '{button_name}' button on your remote...")
    
    while len(samples) < SAMPLES_NEEDED:
        if rfdevice.rx_code_timestamp != timestamp:
            timestamp = rfdevice.rx_code_timestamp
            code = rfdevice.rx_code
            pulse = rfdevice.rx_pulselength
            proto = rfdevice.rx_proto
            
            # Simple noise filter: ignore extremely short/long pulses if needed, 
            # but usually rpi-rf handles this. 
            # We will just collect raw valid data for now.
            
            samples.append({
                "code": code,
                "pulselength": pulse,
                "protocol": proto
            })
            print(f"  Captured signal {len(samples)}/{SAMPLES_NEEDED}: Code={code} Pulse={pulse}")
        
        time.sleep(0.01)
    
    # Process samples to find the most common code (mode)
    codes = [s['code'] for s in samples]
    most_common_code = max(set(codes), key=codes.count)
    
    # Average the pulse length for the correct code
    valid_pulses = [s['pulselength'] for s in samples if s['code'] == most_common_code]
    avg_pulse = int(sum(valid_pulses) / len(valid_pulses))
    
    # Use the protocol from the first match
    protocol = next(s['protocol'] for s in samples if s['code'] == most_common_code)
    
    result = {
        "code": most_common_code,
        "pulselength": avg_pulse,
        "protocol": protocol
    }
    print(f"✅ Success! Saved '{button_name}': {result}")
    time.sleep(1) # Short pause before next button
    return result

def main():
    rfdevice = RFDevice(GPIO_RX)
    rfdevice.enable_rx()
    
    codes_db = {}
    
    print("Welcome to the RF Remote Sniffer!")
    print(f"We will cycle through {len(BUTTONS)} buttons.")
    print("For each button, press and hold it until we confirm capture.")
    print("---------------------------------------------------------")
    
    try:
        for btn in BUTTONS:
            input(f"\nPress ENTER when ready to record [{btn}] (or Ctrl+C to stop)...")
            codes_db[btn] = capture_button(rfdevice, btn)
            
    except KeyboardInterrupt:
        print("\n\nStopping capture...")
    finally:
        rfdevice.cleanup()
    
    if codes_db:
        print(f"\nSaving {len(codes_db)} codes to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(codes_db, f, indent=2)
        print("Done!")
    else:
        print("No codes saved.")

if __name__ == "__main__":
    main()
