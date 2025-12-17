#!/usr/bin/env python3

import argparse
import time
from rpi_rf import RFDevice

# PREFERRED PIN: GPIO 17 (Physical Pin 11)
GPIO_TX = 17

# Candidates
CODE_A_ON = 4478225
CODE_B_ON = 4478259

# Using verified OFF code from file
CODE_OFF = 4478212 

VERIFIED_PROTO = 2
VERIFIED_PULSE = 150

def main():
    parser = argparse.ArgumentParser(description='Validate exactly which code works.')
    parser.add_argument('-g', '--gpio', dest='gpio', type=int, default=GPIO_TX, help="GPIO pin (Default: 17)")
    args = parser.parse_args()

    rfdevice = RFDevice(args.gpio)
    rfdevice.enable_tx()
    rfdevice.tx_repeat = 20

    print(f"🎯 Validating Codes at Pulse {VERIFIED_PULSE}, Proto {VERIFIED_PROTO}...")
    print("------------------------------------------------")

    try:
        while True:
            print(f"👉 Trying Code A: {CODE_A_ON} ...", end='\r')
            rfdevice.tx_code(CODE_A_ON, VERIFIED_PROTO, VERIFIED_PULSE)
            time.sleep(1)
            
            print(f"   Sending OFF ({CODE_OFF}) ...       ", end='\r')
            rfdevice.tx_code(CODE_OFF, VERIFIED_PROTO, VERIFIED_PULSE)
            time.sleep(1)

            print(f"👉 Trying Code B: {CODE_B_ON} ...", end='\r')
            rfdevice.tx_code(CODE_B_ON, VERIFIED_PROTO, VERIFIED_PULSE)
            time.sleep(1)
            
            print(f"   Sending OFF ({CODE_OFF}) ...       ", end='\r')
            rfdevice.tx_code(CODE_OFF, VERIFIED_PROTO, VERIFIED_PULSE)
            time.sleep(1)
            
            print("   (Looping...)                         ", end='\r')
            
    except KeyboardInterrupt:
        print("\n\n🛑 STOPPED!")
        
    finally:
        rfdevice.cleanup()
        print("\nDone.")

if __name__ == "__main__":
    main()
