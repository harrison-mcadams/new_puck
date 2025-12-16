#!/bin/bash

# setup_drive.sh
# Helper script to mount an external hard drive on Raspberry Pi
# 
# Usage: sudo ./scripts/setup_drive.sh

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (sudo ./scripts/setup_drive.sh)"
  exit 1
fi

echo "=========================================="
echo "      Raspberry Pi External Drive Setup   "
echo "=========================================="
echo ""
echo "scanning for block devices..."
echo ""

# List devices excluding loop, ram, and the SD card (mmcblk0) usually
lsblk -o NAME,FSTYPE,SIZE,MOUNTPOINT,LABEL,MODEL | grep -v "loop" | grep -v "ram"

echo ""
echo "------------------------------------------"
echo "Identify your external drive from the list above."
echo "It is usually named 'sda1' or 'sdb1'."
echo "------------------------------------------"
read -p "Enter the device name (e.g. sda1): " DEVICE_NAME

DEVICE="/dev/$DEVICE_NAME"

if [ ! -b "$DEVICE" ]; then
    echo "Error: Device $DEVICE not found."
    exit 1
fi

echo "Detailed info for $DEVICE:"
blkid "$DEVICE"

# Extract UUID and TYPE
UUID=$(blkid -s UUID -o value "$DEVICE")
FSTYPE=$(blkid -s TYPE -o value "$DEVICE")

if [ -z "$UUID" ]; then
    echo "Error: Could not determine UUID. Is the drive formatted?"
    exit 1
fi

echo ""
echo "Detected Filesystem: $FSTYPE"
echo "Detected UUID:       $UUID"
echo ""

# Suggest mount point
DEFAULT_MOUNT="/mnt/puck_data"
read -p "Enter mount point (default: $DEFAULT_MOUNT): " MOUNT_POINT
MOUNT_POINT=${MOUNT_POINT:-$DEFAULT_MOUNT}

# Create mount point
if [ ! -d "$MOUNT_POINT" ]; then
    echo "Creating directory $MOUNT_POINT..."
    mkdir -p "$MOUNT_POINT"
    # Set permissions so the pi user can read/write if needed (commonly uid 1000)
    # We'll adjust after mounting or via mount options
else
    echo "Directory $MOUNT_POINT already exists."
fi

# Determine mount options based on FSTYPE
OPTIONS="defaults,noatime"
if [ "$FSTYPE" == "ntfs" ]; then
    OPTIONS="defaults,noatime,uid=1000,gid=1000"
    if ! command -v ntfs-3g &> /dev/null; then
        echo "Warning: ntfs-3g not found. Installing..."
        apt-get update && apt-get install -y ntfs-3g
    fi
elif [ "$FSTYPE" == "vfat" ] || [ "$FSTYPE" == "exfat" ]; then
    OPTIONS="defaults,noatime,uid=1000,gid=1000,umask=000"
fi

# Attempt temporary mount
echo "Attempting to mount $DEVICE to $MOUNT_POINT..."
mount -t "$FSTYPE" -o "$OPTIONS" "$DEVICE" "$MOUNT_POINT"

if [ $? -eq 0 ]; then
    echo "Success! Drive mounted."
    echo "Contents:"
    ls -F "$MOUNT_POINT" | head -n 5
    
    echo ""
    read -p "Do you want to enable auto-mount on boot (add to /etc/fstab)? (y/n) " CONFIRM
    if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
        # Check if already in fstab
        if grep -q "$UUID" /etc/fstab; then
            echo "Entry already exists in /etc/fstab.bak."
        else
            echo "Backing up /etc/fstab to /etc/fstab.bak..."
            cp /etc/fstab /etc/fstab.bak
            
            FSTAB_ENTRY="UUID=$UUID  $MOUNT_POINT  $FSTYPE  $OPTIONS  0  2"
            echo "Adding entry:"
            echo "$FSTAB_ENTRY"
            echo "$FSTAB_ENTRY" >> /etc/fstab
            echo "fstab updated."
        fi
    fi
    
    echo ""
    echo "Setup Complete!"
    echo "Your drive is available at: $MOUNT_POINT"
else
    echo "Mount failed. logical setup aborted."
fi
