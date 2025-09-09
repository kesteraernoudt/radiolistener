import os, wave
from datetime import datetime

CLIP_DIR = "clips"
os.makedirs(CLIP_DIR, exist_ok=True)

def save_clip(audio_bytes):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CLIP_DIR, f"clip_{ts}.wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    return filename
