---
description: Set up Samba (SMB) on Raspberry Pi for Finder access
---

This workflow guides you through setting up a Samba server on your Raspberry Pi so you can mount its filesystem on your Mac and browse it with Finder.

### 1. SSH into your Raspberry Pi
Run this from your Mac terminal:
```bash
ssh pi@<your-pi-ip-address>
```

### 2. Install Samba
Update packages and install Samba:
```bash
sudo apt update
sudo apt install samba samba-common-bin -y
```

### 3. Create a Shared Directory (Optional)
If you want to share a specific folder (e.g., `puck_share`), create it. If you want to share your entire home directory (`/home/pi`), skip this step.
```bash
mkdir -p /home/pi/puck_share
```

### 4. Configure Samba
Edit the configuration file:
```bash
sudo nano /etc/samba/smb.conf
```

Scroll to the bottom and add the following block. 
*Option A: Share Home Directory (Easiest)*
```ini
[pi_home]
   path = /home/pi
   writeable = yes
   browseable = yes
   public = no
   create mask = 0644
   directory mask = 0755
   force user = pi
```

*Option B: Share Specific Folder*
```ini
[puck]
   path = /home/pi/puck_share
   writeable = yes
   browseable = yes
   public = no
   force user = pi
```

Press `Ctrl+X`, then `Y`, then `Enter` to save and exit.

### 5. Set a Samba Password
You need to set a separate password for Samba access (can be the same as your SSH password):
```bash
sudo smbpasswd -a pi
```

### 6. Restart Samba
Apply the changes:
```bash
sudo systemctl restart smbd
```

### 7. Connect from Mac
1. Open **Finder**.
2. Press `Cmd + K` (or go to **Go > Connect to Server**).
3. Enter `smb://<your-pi-ip-address>` (e.g., `smb://192.168.1.50`).
4. Click **Connect**.
5. Keep "Registered User" selected.
6. Enter name `pi` and the password you set in step 5.
7. Select the volume to mount (e.g., `pi_home` or `puck`).

Now the Pi will appear in Finder under "Locations" or on your Desktop!
