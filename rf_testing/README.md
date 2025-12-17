# RF Testing

This folder contains scripts to test your SMAKN 433MHz RF Transmitter and Receiver kit.

## Wiring (Fan-Safe Version)

Since your fan occupies Pin 4 and Pin 6, we will use alternative power and ground pins.

### Transmitter (Square Board - 3 Pins)
- **VCC** -> **Pin 2** (5V) - *The other 5V pin, top right corner.*
- **GND** -> **Pin 14** (GND) - *Or Pin 20, 25, 30, 34, 39.*
- **DATA** -> **Pin 11** (GPIO 17)

### Receiver (Rectangular Board - 4 Pins)
- **VCC** (Left-most) -> **Pin 1** (3.3V)
- **DATA** (Middle) -> **Pin 13** (GPIO 27)
- **GND** (Right-most) -> **Pin 9** (GND)

## Usage

1. **Install Dependencies** (on the Pi):
   ```bash
   sudo pip3 install rpi-rf
   ```

2. **Run Receiver**:
   Open a terminal and run:
   ```bash
   python3 receive.py
   ```

3. **Run Sender**:
   Open a second terminal window and run:
   ```bash
   python3 send.py
   ```

You should see numbers generated in the sender window appear in the receiver window!
