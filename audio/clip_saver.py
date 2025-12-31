import os, wave
from datetime import datetime
from typing import Optional

CLIP_DIR = "clips"
os.makedirs(CLIP_DIR, exist_ok=True)

def save_clip(audio_bytes, capture_ts: Optional[float] = None):
    """
    Persist the provided audio bytes to a WAV file.

    If a capture timestamp is provided it is used to tag the filename so we can
    correlate the file with the time the audio was first seen in the pipeline.
    """
    ts = datetime.fromtimestamp(capture_ts) if capture_ts else datetime.now()
    ts_label = ts.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CLIP_DIR, f"clip_{ts_label}.wav")
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(audio_bytes)
    return filename
