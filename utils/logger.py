import os
from datetime import datetime, timedelta
from collections import defaultdict

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

MAX_TRANSCRIPT_LINES = 1000

transcript_log = defaultdict(list)

def log_event(radio, msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} - {radio}] {msg}"
    transcript_log[radio.upper()].append(line)
    if len(transcript_log[radio.upper()]) > MAX_TRANSCRIPT_LINES:
        transcript_log[radio.upper()] = transcript_log[radio.upper()][-MAX_TRANSCRIPT_LINES:]
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def log_ai_event(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}_ai.log")
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

def get_radio_log(radio="", num_lines=100):
    if not radio:
        # return combined log
        combined = []
        for lines in transcript_log.values():
            combined.extend(lines)
        combined = sorted(combined, reverse=True)[:num_lines]
        return combined
    if radio.upper() in transcript_log:
        return transcript_log[radio.upper()][-num_lines:]
    # see if it's the beginning of a radio name
    for name, log in transcript_log.items():
        if name.startswith(radio.upper()):
            return log[-num_lines:]
    return []