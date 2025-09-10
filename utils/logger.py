import os
from datetime import datetime, timedelta
from collections import defaultdict

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

transcript_log = defaultdict(list)

def log_event(radio, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} - {radio}] {msg}"
    transcript_log[radio].append(line)
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

def get_radio_log(radio, num_lines=100):
    if radio in transcript_log:
        return transcript_log[radio][-num_lines:]
    return []