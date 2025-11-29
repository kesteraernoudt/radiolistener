import subprocess
from utils import logger
import time

NO_AUDIO_TIMEOUT_SECONDS = 180  # Time to wait for audio before treating stream as stalled

def capture_stream(q, stream_url, controller):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-reconnect",
        "1",
        "-reconnect_streamed",
        "1",
        "-reconnect_delay_max",
        "30",
        "-rw_timeout",
        "5000000",  # 5s read timeout to force reconnects on stalled sockets
        "-i",
        stream_url,
        "-f",
        "s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-af",
        "acompressor=threshold=-20dB:ratio=2:attack=5:release=50,volume=+3dB",
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
