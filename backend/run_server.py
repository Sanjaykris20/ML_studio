import os
import subprocess
import sys
import time

def run_supervisor():
    port = os.getenv("PORT", "8000")
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("PORT") else "127.0.0.1")
    print(f"🚀 Starting ML Studio High-Availability Backend Supervisor on http://{host}:{port}...", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        port,
        "--log-level",
        "info",
    ]
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONFAULTHANDLER"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"


    while True:
        try:
            proc = subprocess.Popen(cmd, env=env)
            ret = proc.wait()
            print(f"⚠️ Backend process exited with code {ret}. Auto-recovering in 1s...", flush=True)
            time.sleep(1)
        except KeyboardInterrupt:
            print("🛑 Server stopped by operator.")
            break
        except Exception as e:
            print(f"[SUPERVISOR] Exception: {e}. Restarting in 1s...", file=sys.stderr, flush=True)
            time.sleep(1)

if __name__ == "__main__":
    run_supervisor()
