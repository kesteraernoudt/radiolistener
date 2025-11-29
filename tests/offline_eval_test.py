import os
import difflib
import pytest

from tools.offline_eval import run_offline_eval, normalize_text


def _env_path(var_name):
    path = os.getenv(var_name, "").strip()
    return path if path else None


@pytest.mark.skipif(
    not _env_path("OFFLINE_EVAL_AUDIO") or not _env_path("OFFLINE_EVAL_GOLDEN"),
    reason="Set OFFLINE_EVAL_AUDIO and OFFLINE_EVAL_GOLDEN to run offline eval",
)
def test_offline_eval_matches_golden():
    audio_path = _env_path("OFFLINE_EVAL_AUDIO")
    golden_path = _env_path("OFFLINE_EVAL_GOLDEN")
    model = os.getenv("OFFLINE_EVAL_MODEL", "small.en")
    buffer_seconds = float(os.getenv("OFFLINE_EVAL_BUFFER_SECONDS", "4.0"))
    buffer_overlap = float(os.getenv("OFFLINE_EVAL_BUFFER_OVERLAP", "0.6"))
    min_ratio = float(os.getenv("OFFLINE_EVAL_MIN_RATIO", "0.90"))
    realtime_feed = os.getenv("OFFLINE_EVAL_REALTIME_FEED", "false").lower() in ("1", "true", "yes")
    allow_tail = os.getenv("OFFLINE_EVAL_ALLOW_TAIL", "true").lower() in ("1", "true", "yes")
    idle_grace = float(os.getenv("OFFLINE_EVAL_IDLE_GRACE", "5.0"))

    result = run_offline_eval(
        audio_path=audio_path,
        golden_path=golden_path,
        model=model,
        buffer_seconds=buffer_seconds,
        buffer_overlap=buffer_overlap,
        sample_rate=16000,
        clip_duration=20.0,
        chunk_bytes=16384,
        realtime_feed=realtime_feed,
        allow_tail=allow_tail,
        idle_grace=idle_grace,
    )

    with open(golden_path) as f:
        golden_text = f.read()

    predicted = normalize_text(result["transcript"])
    expected = normalize_text(golden_text)
    ratio = difflib.SequenceMatcher(None, expected.split(), predicted.split()).ratio()

    assert ratio >= min_ratio, f"Similarity {ratio:.3f} below threshold {min_ratio}"
    assert result["segments"], "No segments were produced by StreamProcessor"
