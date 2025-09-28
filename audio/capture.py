import subprocess
from utils import logger
import time


def capture_stream(q, stream_url, controller):
    cmd = [
        "ffmpeg",
        "-i",
        stream_url,
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last_alert_time = 0
    while controller.running:
        data = proc.stdout.read(16384)
        if not data:
            continue
        now = time.time()
        if len(q) == q.maxlen and now - last_alert_time > 300:
            logger.log_event(
                controller.RADIO_CONF["NAME"], "Audio queue full, dropping audio data"
            )
            last_alert_time = now
        q.append(data)
    proc.terminate()
    proc.wait()
