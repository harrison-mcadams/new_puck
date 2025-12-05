# Deploying New Puck to Raspberry Pi 4

This guide assumes you are starting with a fresh installation of Raspberry Pi OS (64-bit recommended) on your Raspberry Pi 4.

## 1. System Preparation

First, update your system packages to ensure everything is current.

```bash
sudo apt update && sudo apt upgrade -y
```

Install necessary system dependencies. While `pip` handles Python packages, some might need system-level libraries (like `numpy` or `pandas` dependencies).

```bash
sudo apt install -y git python3-pip python3-venv libopenblas-dev gfortran
```

## 2. Clone the Repository

Navigate to your home directory and clone the project.

```bash
cd ~
git clone https://github.com/harrison-mcadams/new_puck new_puck
cd new_puck
git checkout public-release-clean
```
*(Replace `<YOUR_REPO_URL>` with your actual Git repository URL. If you haven't pushed it yet, you'll need to do that first!)*

## 3. Set Up Virtual Environment

Create a virtual environment to isolate dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

Install the required Python packages using the `requirements.txt` file we created.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*Note: Installing pandas/numpy on a Pi can sometimes take a while if it needs to build from source. Using the 64-bit OS usually allows fetching pre-built wheels, which is much faster.*

## 5. Test the Application

Before creating a service, run the app manually to make sure it works.

```bash
# Temporarily set the bind address to 0.0.0.0 to access from your network
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```
Open a browser on your computer and go to `http://192.168.1.125:5000`. If it loads, you're good to go! Press `Ctrl+C` to stop the server.

## 6. Set Up Systemd Service

We want the app to start automatically when the Pi boots. We'll use the service file created in `deploy/new_puck.service`.

1. **Copy the service file**:
   ```bash
   sudo cp deploy/new_puck.service /etc/systemd/system/new_puck.service
   ```

2. **Reload systemd**:
   ```bash
   sudo systemctl daemon-reload
   ```

3. **Enable and Start the service**:
   ```bash
   sudo systemctl enable new_puck
   sudo systemctl start new_puck
   ```

4. **Check status**:
   ```bash
   sudo systemctl status new_puck
   ```

## 7. Accessing the App

Your app is now running with Gunicorn on port **8000**.

Access it at: `http://192.168.1.125:8000`

## Troubleshooting

- **Browser Won't Load (Chrome vs Safari)**:
  - If Chrome fails but Safari works, Chrome is likely forcing **HTTPS**.
  - **Fix**: Ensure you type `http://` explicitly (e.g., `http://puck-server.local:5000`).
  - **Deep Fix**: If it keeps redirecting to HTTPS, go to `chrome://net-internals/#hsts` in Chrome, type `puck-server.local` in "Delete domain security policies", and click Delete.

- **Logs**: Check service logs with `journalctl -u new_puck -f`
- **Permissions**: Ensure the `User` in `new_puck.service` matches your Pi user (default is `pi`).
