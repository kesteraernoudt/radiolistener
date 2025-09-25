import subprocess
from utils import logger

def capture_stream(q, stream_url, controller):
    cmd = [
        "ffmpeg", "-i", stream_url,
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while controller.running:
        data = proc.stdout.read(4096)
        if not data:
            continue
        if len(q) == q.maxlen:
            logger.log_event(controller.RADIO_CONF['NAME'], "Audio queue full, dropping audio data")
        q.append(data)
    proc.terminate()
    proc.wait()
    
