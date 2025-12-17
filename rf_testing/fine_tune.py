#!/usr/bin/env python3

import argparse
import time
from rpi_rf import RFDevice

# PREFERRED PIN: GPIO 17 (Physical Pin 11)
GPIO_TX = 17

# ON Codes (We know 259 works)
CODE_ON = 4478259

# OFF Codes (In file: 4478212. Jitter variant likely +34 = 4478246)
# We will use the 'variant' 4478246 to match the logic of the ON code
CODE_OFF = 4478246 

TARGET_PROTO = 2

def main():
    parser = argparse.ArgumentParser(description='Fine tune RF pulse length via Blinking.')
    parser.add_argument('-g', '--gpio', dest='gpio', type=int, default=GPIO_TX, help="GPIO pin (Default: 17)")
    args = parser.parse_args()

    rfdevice = RFDevice(args.gpio)
    rfdevice.enable_tx()
    
    # Needs a few repeats to reliably safeguard against noise
    rfdevice.tx_repeat = 15

    print(f"🎯 Fine Tuning Blink Test (Proto {TARGET_PROTO})...")
    print("Sweeping Pulse Length from 120 to 180.")
    print("If it BLINKS (On then Off), that pulse is GOOD.")
    print("------------------------------------------------")

    try:
        # Sweep around the 150 area
        for pulse in range(120, 182, 2):
            msg = f"👉 Testing Pulse: {pulse} | "
            print(msg + "Sending ON ...", end='\r')
            
            # Send ON
            rfdevice.tx_code(CODE_ON, TARGET_PROTO, pulse)
            time.sleep(1.0) # Time to see it turn ON
            
            print(msg + "Sending OFF...", end='\r')
            
            # Send OFF
            rfdevice.tx_code(CODE_OFF, TARGET_PROTO, pulse)
            time.sleep(1.0) # Time to see it turn OFF
            
            # Clear line
            print(f"   Done Pulse:    {pulse}              ")
            
    except KeyboardInterrupt:
        print("\n\n🛑 STOPPED!")
        
    finally:
        rfdevice.cleanup()
        print("\nDone.")

if __name__ == "__main__":
    main()
