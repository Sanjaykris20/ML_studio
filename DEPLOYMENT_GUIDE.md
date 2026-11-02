# ML Studio – Comprehensive Cloud Deployment & Hosting Guide

This guide details all methods to host and deploy the **ML Studio** interactive machine learning platform online.

---

## 📦 Project Archive & Backup

A clean release package has already been generated:
* **Path**: `C:\Users\GURU PRASATH JV\Downloads\NVDIA\ML_Studio_FullProject.zip`
* **Size**: ~3.7 MB (includes full backend, React frontend build, datasets, and configuration; excludes bulky `node_modules`).
* **Regenerate Anytime**: Double-click `export_project.bat` or run `python export_zip.py`.

---

## 🚀 Option 1: Free Cloud Hosting on Render.com (Recommended)

Render offers free hosting with automated SSL/HTTPS and Git push integration.

### Steps:
1. **Push or Upload to GitHub**:
   * Create a new GitHub repository (e.g. `ml-studio`).
   * Unzip or push this project to your repository.
2. **Deploy on Render**:
   * Go to [Render.com](https://render.com) and log in.
   * Click **New +** $\rightarrow$ **Blueprint**.
   * Connect your GitHub repository.
   * Render will automatically detect `render.yaml` and configure:
     * Python 3.11 environment
     * Frontend build (`npm run build`)
     * Backend server (`python run_server.py`)
     * Persistent disk for datasets and model artifacts
   * Click **Apply**.
3. **Done!**
   * Render will provide a free public HTTPS URL: `https://ml-studio-xxxx.onrender.com`.

---

## 🚂 Option 2: Deploy to Railway.app

Railway provides fast containerized deployments with automatic Nixpacks detection.

### Steps:
1. Go to [Railway.app](https://railway.app).
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Select your repository.
4. Railway will automatically detect `railway.json` and `Dockerfile`, build the React frontend, install dependencies, and launch on a dynamic `$PORT`.
5. Under service settings, click **Generate Domain** to get your public HTTPS URL (e.g., `https://ml-studio-production.up.railway.app`).

---

## 🐳 Option 3: VPS / Cloud Server with Docker (AWS, DigitalOcean, GCP, Linode)

For full server control with custom domains on Ubuntu/Debian/CentOS.

### Using Single-Container (Simplest):
```bash
# 1. Clone or copy files to your VPS
git clone <your-repo-url> ml-studio
cd ml-studio

# 2. Build and run unified container
docker build -t ml-studio .
docker run -d -p 80:8000 -v ml_studio_data:/app/backend/artifacts --name ml_studio ml-studio
```
The application will be live at `http://your-server-ip`.

### Using Multi-Container (Docker Compose + Nginx):
```bash
docker compose up -d --build
```
* **Frontend**: `http://your-server-ip` (Port 80 via Nginx)
* **Backend**: `http://your-server-ip:8000` (Port 8000)

---

## 🌐 Option 4: Instant Public Access from Your Local Machine (Zero Cloud Setup)

If you want to share your running application with anyone on the internet right now without deploying to a cloud server:

### Method A: Cloudflare Tunnel (100% Free, No Account Required)
1. Download [cloudflared for Windows](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/).
2. Run in PowerShell:
   ```powershell
   cloudflared tunnel --url http://localhost:5173
   ```
3. Cloudflare will give you an instant public HTTPS link (e.g., `https://random-words.trycloudflare.com`).

### Method B: ngrok
1. Run:
   ```bash
   ngrok http 5173
   ```
2. Share the generated public forwarding URL.

---

## ⚙️ Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8000` | Application port (assigned dynamically by cloud platforms) |
| `ENVIRONMENT` | `development` | Set to `production` in live deployments |
| `DATABASE_URL` | `sqlite:///ml_studio.db` | Database connection string |
| `DATA_DIR` | `backend/artifacts/datasets` | Storage directory for uploaded CSVs |
| `ARTIFACTS_DIR` | `backend/artifacts/models` | Storage directory for trained model bundles |
| `MAX_UPLOAD_SIZE_MB` | `25` | Maximum CSV upload size limit |

---

## 💻 Local Windows Quick Start (After Downloading Zip)
1. Extract `ML_Studio_FullProject.zip`.
2. Double-click `start_project.bat`.
3. Opens:
   * **Web Dashboard**: `http://localhost:5173`
   * **API Docs**: `http://127.0.0.1:8000/docs`
