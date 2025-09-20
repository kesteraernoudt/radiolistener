import subprocess

def capture_stream(q, stream_url, controller):
    cmd = [
        "ffmpeg", "-i", stream_url,
        "-f", "s16le", "-ar", "16000", "-ac", "1", "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    while controller.running:
        data = proc.stdout.read(4096)
        if not data:
            break
        q.put(data)
    proc.terminate()
    proc.wait()
    
