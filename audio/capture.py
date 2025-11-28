import subprocess
from utils import logger
import time


NO_AUDIO_TIMEOUT_SECONDS = 180  # Time to wait for audio before treating stream as stalled


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
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    last_alert_time = 0
    last_audio_time = time.time()
    try:
        while controller.running:
            data = proc.stdout.read(16384)
            now = time.time()
            if not data:
                if now - last_audio_time > NO_AUDIO_TIMEOUT_SECONDS:
                    logger.log_event(
                        controller.RADIO_CONF.get("NAME", "UNKNOWN"),
                        "No audio data received; restarting capture",
                    )
                    raise RuntimeError(
                        f"No audio data for {NO_AUDIO_TIMEOUT_SECONDS} seconds"
                    )
                continue
            last_audio_time = now
            if len(q) == q.maxlen and now - last_alert_time > 300:
                logger.log_event(
                    controller.RADIO_CONF["NAME"],
                    "Audio queue full, dropping audio data",
                )
                last_alert_time = now
            q.append(data)
    finally:
        proc.terminate()
        proc.wait()
