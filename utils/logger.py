import os
from datetime import datetime, timedelta

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

transcript_log = []

def log_event(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    transcript_log.append(line)
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def cleanup_logs(days=7):
    cutoff = datetime.now() - timedelta(days=days)
    for f in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, f)
        if os.path.isfile(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)
