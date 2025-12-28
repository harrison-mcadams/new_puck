#!/bin/bash

# --------------------------------------------------------------------------
# Setup Remote Access for Raspberry Pi (Git Bash Version)
# --------------------------------------------------------------------------

# 1. Setup Variables
KEY_SOURCE="./scripts/id_rsa_puck"
SSH_DIR="$HOME/.ssh"
KEY_DEST="$SSH_DIR/id_rsa_puck"
CONFIG_FILE="$SSH_DIR/config"

echo "--------------------------------------------------------"
echo "Configuring SSH Access for 'puck-server.local'..."
echo "--------------------------------------------------------"

# 2. Key Installation
if [ ! -f "$KEY_SOURCE" ]; then
    if [ -f "$KEY_DEST" ]; then
        echo "Key not found in scripts/, but found in ~/.ssh. Creating config only..."
    else
        echo "ERROR: Key file '$KEY_SOURCE' not found! Please place it there first."
        exit 1
    fi
else
    # Create .ssh dir
    if [ ! -d "$SSH_DIR" ]; then
        mkdir -p "$SSH_DIR"
        chmod 700 "$SSH_DIR"
        echo "Created $SSH_DIR"
    fi

    # Move Key
    mv "$KEY_SOURCE" "$KEY_DEST"
    chmod 600 "$KEY_DEST"
    echo "Files: Moved private key to $KEY_DEST"
fi

# 3. SSH Config Update
# Check if "Host puck" already exists
if grep -q "Host puck" "$CONFIG_FILE" 2>/dev/null; then
    echo "Config: 'Host puck' entry already exists in $CONFIG_FILE"
else
    echo "" >> "$CONFIG_FILE"
    echo "Host puck" >> "$CONFIG_FILE"
    echo "    HostName puck-server.local" >> "$CONFIG_FILE"
    echo "    User spoon" >> "$CONFIG_FILE"
    echo "    IdentityFile $KEY_DEST" >> "$CONFIG_FILE"
    echo "    StrictHostKeyChecking accept-new" >> "$CONFIG_FILE"
    echo "Config: Added 'puck' alias to $CONFIG_FILE"
fi

# 4. Connection Test
echo "--------------------------------------------------------"
echo "Testing SSH Connection..."
echo "--------------------------------------------------------"
ssh -o BatchMode=yes -o ConnectTimeout=5 puck "echo 'SUCCESS: Connected to Raspberry Pi via SSH!'"
if [ $? -ne 0 ]; then
    echo "WARNING: Connection check failed. Ensure the Pi is on and 'puck-server.local' is reachable."
    echo "You can try manually: ssh puck"
else
    echo "Verified."
fi

# 5. SMB / File Explorer Instructions
echo "--------------------------------------------------------"
echo "To access files in File Explorer (SMB):"
echo "--------------------------------------------------------"
echo "Run this command in a standard Windows Command Prompt (or PowerShell) to mount drive P:"
echo "    net use P: \\\\puck-server.local\\spoon /PERSISTENT:YES"
echo ""
echo "Note: If asked for credentials, use User: 'spoon' and your Pi password."
