"""
Offline eval runner: feed a short clip through StreamProcessor and compare the
recombined transcript against a golden reference. Useful for testing smaller
buffer sizes/overlaps without touching live streams.
"""

import argparse
import difflib
import logging
import os
import json
import subprocess
import sys
import time
from collections import deque
import threading
from pathlib import Path

# Allow running as a script from the repository root without installing as a package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio import process
from utils import genai as genai_module


logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("offline-eval")


class OfflineController:
    """Minimal controller stub so StreamProcessor can run offline."""

    def __init__(self, name: str):
        self.running = True
        self.RADIO_CONF = {"NAME": name, "PHRASES": []}

    def send_message(self, text: str):
        log.info(f"[telegram stub] {text}")

    def send_audio(self, file_path: str, caption: str = ""):
        log.info(f"[audio stub] saved clip at {file_path} {caption}")


def decode_audio_to_pcm(path: str, sample_rate: int) -> bytes:
    """Use ffmpeg to decode any input into 16-bit PCM at the given sample rate."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    try:
        return subprocess.check_output(cmd)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed to decode audio: {exc}") from exc
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found on PATH")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def load_base_config(config_path: str | None) -> dict:
    if not config_path:
        return {}
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        return {}
    try:
        with open(cfg_file) as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"Failed to load config {config_path}: {exc}")
        return {}


def resolve_pre_prompt(base_config: dict, config_path: str | None) -> str:
    pre_prompt = base_config.get("AI_PRE_PROMPT_FILE", "")
    if not pre_prompt:
        return ""
    p = Path(pre_prompt)
    if p.exists():
        return str(p)
    if config_path:
        cfg_dir = Path(config_path).resolve().parent
        candidate = cfg_dir / pre_prompt
        if candidate.exists():
            return str(candidate)
    return ""


def disable_genai():
    """Monkey-patch GenAIHandler to a no-op for offline testing."""
    class DummyGenAIHandler:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, *args, **kwargs):
            return None

    genai_module.GenAIHandler = DummyGenAIHandler


def try_enqueue_tail(proc) -> bool:
    """
    If there is remaining audio smaller than a full frame, pad with silence and enqueue
    one last frame so short clips still get transcribed.
    """
    lock = getattr(proc, "lock", None)
    frame_size = getattr(proc, "frame_size", 0)
    rolling = getattr(proc, "rolling_buffer", b"")
    window_start = getattr(proc, "window_start", 0)
    if frame_size <= 0 or len(rolling) <= window_start:
        return False
    pending = len(rolling) - window_start
    if pending >= frame_size:
        return False
    if lock:
        lock.acquire()
    try:
        rolling = bytes(proc.rolling_buffer)
        window_start = proc.window_start
        pending = len(rolling) - window_start
        if pending <= 0 or pending >= frame_size:
            return False
        start = max(0, len(rolling) - frame_size)
        tail = rolling[start:]
        if len(tail) < frame_size:
            tail += b"\x00" * (frame_size - len(tail))
        # Advance window so we don't reprocess tail
        proc.window_start = len(rolling)
    finally:
        if lock:
            lock.release()
    try:
        proc.transcribe_in.put_nowait(tail)
        log.info("Enqueued padded tail frame to cover short clip.")
        return True
    except queue.Full:
        log.warning("Tail frame dropped: transcribe queue full.")
        return False


def diff_summary(pred: str, golden: str) -> str:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(golden)
    ratio = difflib.SequenceMatcher(None, gold_norm.split(), pred_norm.split()).ratio()
    diff_lines = list(
        difflib.unified_diff(
            gold_norm.split(),
            pred_norm.split(),
            lineterm="",
            n=5,
        )
    )
    head = "\n".join(diff_lines[:60])  # cap noise
    return f"match ratio: {ratio:.3f}\nDiff (golden -> predicted):\n{head}"


def run_offline_eval(
    audio_path: str,
    golden_path: str | None,
    model: str,
    buffer_seconds: float,
    buffer_overlap: float,
    sample_rate: int,
    clip_duration: float,
    chunk_bytes: int,
    config_path: str | None = None,
    use_genai: bool = False,
    timeout: float = 60.0,
    realtime_feed: bool = True,
    verbose: bool = False,
    idle_grace: float = 5.0,
    allow_tail: bool = True,
):
    ctrl = OfflineController(name=os.path.basename(audio_path))
    base_config = load_base_config(config_path)
    genai_api_key = os.getenv("GEMINI_API_KEY") or base_config.get("GEMINI_API_KEY", "")

    if use_genai and genai_api_key:
        pre_prompt_file = resolve_pre_prompt(base_config, config_path)
    else:
        disable_genai()
        genai_api_key = ""
        pre_prompt_file = ""

    audio_bytes = decode_audio_to_pcm(audio_path, sample_rate)

    audio_queue = deque(maxlen=200)
    chunk_count = 0
    total_samples = len(audio_bytes) // 2  # int16 mono
    total_seconds = total_samples / sample_rate
    proc = process.StreamProcessor(
        ctrl.RADIO_CONF,
        asr_whisper_model=model,
        buffer_seconds=buffer_seconds,
        buffer_overlap=buffer_overlap,
        sample_rate=sample_rate,
        CLIP_DURATION=clip_duration,
        GENAI_API_KEY=genai_api_key,
        pre_prompt_file=pre_prompt_file,
        controller=ctrl,
        log_enabled=False,
    )

    proc_thread = threading.Thread(target=proc.process_audio, args=(audio_queue,), daemon=True)
    proc_thread.start()

    feed_start = time.time()
    for idx, offset in enumerate(range(0, len(audio_bytes), chunk_bytes)):
        audio_queue.append(audio_bytes[offset : offset + chunk_bytes])
        chunk_count += 1
        if realtime_feed:
            chunk_duration = chunk_bytes / (sample_rate * 2)
            target_time = feed_start + idx * chunk_duration
            sleep_for = target_time - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

    # If we have leftover audio smaller than a frame, optionally enqueue a tail padded with silence
    if allow_tail:
        try_enqueue_tail(proc)

    log.info(
        f"Queued {chunk_count} chunks ({total_seconds:.2f}s audio) "
        f"frame={buffer_seconds}s overlap={buffer_overlap}s model={model}"
    )

    # Wait until everything is drained before stopping threads
    idle_ticks = 0
    last_log = 0.0
    last_progress = time.time()
    last_events = 0
    idle_grace = args.idle_grace  # seconds without progress before stopping once queues are empty
    while True:
        active = getattr(proc, "active_transcribes", 0)
        queues_empty = (
            len(audio_queue) == 0
            and proc.transcribe_in.empty()
            and proc.transcribe_out.empty()
            and active == 0
        )
        now = time.time()
        if now - last_log > 1.0:
            last_log = now
            log.info(
                "Draining..."
                f" audio_q={len(audio_queue)}"
                f" in_q={proc.transcribe_in.qsize()}"
                f" out_q={proc.transcribe_out.qsize()}"
                f" active={active}"
            )

        # Track progress via processed events
        n_events = proc.stats.get("n_events", 0)
        if n_events != last_events:
            last_progress = time.time()
            if verbose and proc.previous_texts:
                new_texts = proc.previous_texts[last_events:n_events] if n_events > last_events else []
                if not new_texts and proc.previous_texts:
                    new_texts = [proc.previous_texts[-1]]
                for idx, txt in enumerate(new_texts, start=last_events + 1):
                    log.info(f"[event {idx}] {txt}")
            last_events = n_events

        if time.time() - last_progress > timeout:
            log.warning(f"No progress for {timeout}s; stopping.")
            break

        if queues_empty and (now - last_progress) >= idle_grace:
            break
        time.sleep(0.1)

    ctrl.running = False
    proc_thread.join(timeout=5)
    # Ensure transcriber thread exits cleanly
    transcriber = getattr(proc, "transcriber_thread", None)
    if transcriber:
        transcriber.join(timeout=2)

    combined = " ".join(proc.previous_texts)
    segments = [{"seq": idx, "start_sec": None, "text": text} for idx, text in enumerate(proc.previous_texts)]
    log.info(f"\nCombined transcript ({len(segments)} segments):\n{combined}\n")

    if golden_path:
        with open(golden_path) as f:
            golden = f.read()
        log.info(diff_summary(combined, golden))

    # Print a small timeline to inspect segmentation
    log.info("\nSegments (seq @ start_sec):")
    for seg in segments:
        start_val = seg.get("start_sec")
        start_desc = f"{start_val:.1f}s" if isinstance(start_val, (int, float)) else "n/a"
        log.info(f"{seg.get('seq')} @ {start_desc}: {seg.get('text')}")

    log.info("\nStats:")
    log.info(proc.get_stats())

    return {
        "transcript": combined,
        "segments": segments,
        "stats": proc.get_stats(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Offline Whisper eval harness")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--golden", help="Path to golden transcript text file")
    parser.add_argument("--model", default="small.en", help="Whisper model name")
    parser.add_argument("--buffer-seconds", type=float, default=4.0)
    parser.add_argument("--buffer-overlap", type=float, default=0.6)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--clip-duration", type=float, default=20.0)
    parser.add_argument(
        "--chunk-bytes",
        type=int,
        default=16384,
        help="Chunk size pushed into the queue (matches capture read size)",
    )
    parser.add_argument(
        "--config",
        default="config/config.json",
        help="Path to config JSON for defaults/pre-prompt (optional)",
    )
    parser.add_argument(
        "--use-genai",
        action="store_true",
        help="Use GenAIHandler with GEMINI_API_KEY (env) instead of stubbing it out",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Seconds without progress before stopping (offline safety)",
    )
    parser.add_argument(
        "--realtime-feed",
        action="store_true",
        default=True,
        help="Pace input chunks to real-time (default on; disable with --no-realtime-feed)",
    )
    parser.add_argument(
        "--no-realtime-feed",
        dest="realtime_feed",
        action="store_false",
        help="Disable real-time pacing (will likely overflow queues)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each processed segment as it arrives",
    )
    parser.add_argument(
        "--idle-grace",
        type=float,
        default=5.0,
        help="Seconds to wait with empty queues and no progress before stopping",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_offline_eval(
            audio_path=args.audio,
            golden_path=args.golden,
            model=args.model,
            buffer_seconds=args.buffer_seconds,
            buffer_overlap=args.buffer_overlap,
            sample_rate=args.sample_rate,
            clip_duration=args.clip_duration,
            chunk_bytes=args.chunk_bytes,
            config_path=args.config,
            use_genai=args.use_genai,
            timeout=args.timeout,
            realtime_feed=args.realtime_feed,
            verbose=args.verbose,
            idle_grace=args.idle_grace,
        )
    except RuntimeError as exc:
        log.error(exc)
        sys.exit(1)
