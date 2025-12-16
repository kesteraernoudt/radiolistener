import os
import threading
from datetime import datetime, timedelta
from collections import defaultdict

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

MAX_TRANSCRIPT_LINES = 1000
MAX_AI_LINES = 1000

transcript_log = defaultdict(list)
ai_log = list()

_retention_days = 7
_clip_dir = "clips"
_last_cleanup_date = None
_cleanup_lock = threading.Lock()


def _extract_timestamp(line):
    try:
        if "[" in line and "]" in line:
            header = line.split("[", 1)[1].split("]", 1)[0]
            ts_part = header.split(" - ")[0]
            return datetime.strptime(ts_part, "%Y-%m-%d %H:%M:%S")
    except (ValueError, IndexError):
        pass
    return datetime.min


def _extract_radio_tag(line):
    try:
        if "[" in line and "]" in line:
            header = line.split("[", 1)[1].split("]", 1)[0]
            parts = header.split(" - ")
            if len(parts) == 2:
                return parts[1].strip().upper()
    except (ValueError, IndexError):
        pass
    return ""


def _should_include(line, start_datetime):
    if start_datetime is None:
        return True
    return _extract_timestamp(line) >= start_datetime


def _dedupe_preserve_order(lines):
    seen = set()
    unique = []
    for line in lines:
        if line in seen:
            continue
        seen.add(line)
        unique.append(line)
    return unique


def _finalize_lines(lines, num_lines, start_datetime=None, reverse=False, reverse_after_slice=False):
    # When start_datetime is provided, lines are already filtered; keep chronological order unless caller wants reverse
    if start_datetime is not None:
        lines.sort(key=_extract_timestamp, reverse=False)
        result = lines[:num_lines]
        if reverse_after_slice:
            result.reverse()
        return result

    # Default: newest first
    lines.sort(key=_extract_timestamp, reverse=True)
    result = lines[:num_lines]
    if reverse:
        result.reverse()
    return result


def configure_cleanup(keep_days=None, clip_dir=None):
    """
    Configure log/clip cleanup behavior.
    """
    global _retention_days, _clip_dir
    if keep_days is not None:
        try:
            _retention_days = max(int(keep_days), 0)
        except (TypeError, ValueError):
            pass
    if clip_dir:
        _clip_dir = clip_dir


def _cleanup_dir(path, keep_days, allowed_exts=None):
    """
    Remove files in a directory older than keep_days. Returns list of removed paths.
    """
    removed = []
    if keep_days is None:
        return removed

    try:
        entries = os.listdir(path)
    except FileNotFoundError:
        return removed

    cutoff = datetime.now() - timedelta(days=max(keep_days, 0))
    for name in entries:
        full_path = os.path.join(path, name)
        if not os.path.isfile(full_path):
            continue
        if allowed_exts:
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_exts:
                continue
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(full_path))
        except OSError:
            continue
        if mtime < cutoff:
            try:
                os.remove(full_path)
                removed.append(full_path)
            except OSError:
                continue
    return removed


def cleanup_logs(keep_days=None):
    """
    Delete log files older than keep_days days.
    """
    days = _retention_days if keep_days is None else keep_days
    return _cleanup_dir(LOG_DIR, days, allowed_exts={".log"})


def cleanup_clips(keep_days=None, clip_dir=None):
    """
    Delete clip files older than keep_days days.
    """
    days = _retention_days if keep_days is None else keep_days
    directory = clip_dir or _clip_dir
    return _cleanup_dir(directory, days, allowed_exts={".wav", ".mp3", ".flac", ".aac"})


def ensure_daily_cleanup(force=False):
    """
    Run cleanup at most once per calendar day (or immediately when forced).
    """
    global _last_cleanup_date
    today = datetime.now().date()
    if not force and _last_cleanup_date == today:
        return
    with _cleanup_lock:
        if not force and _last_cleanup_date == today:
            return
        cleanup_logs()
        cleanup_clips()
        _last_cleanup_date = today


def run_startup_cleanup(keep_days=None, clip_dir=None):
    """
    Configure retention and perform an immediate cleanup (once per process start).
    """
    configure_cleanup(keep_days=keep_days, clip_dir=clip_dir)
    ensure_daily_cleanup(force=True)


def log_event(radio, msg):
    global transcript_log
    ensure_daily_cleanup()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} - {radio}] {msg}"
    transcript_log[radio.upper()].append(line)
    if len(transcript_log[radio.upper()]) > MAX_TRANSCRIPT_LINES:
        transcript_log[radio.upper()] = transcript_log[radio.upper()][-MAX_TRANSCRIPT_LINES:]
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}-{radio}.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def log_ai_event(msg, radio=""):
    global ai_log
    ensure_daily_cleanup()
    msg = msg.rstrip()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    radio_tag = (radio or "GLOBAL").upper()
    line = f"[{ts} - {radio_tag}] {msg}".rstrip()
    ai_log.append(line)
    if len(ai_log) > MAX_AI_LINES:
        ai_log = ai_log[-MAX_AI_LINES:]
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}_ai.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def get_radio_log(radio="", num_lines=100, start_datetime=None, reverse=False):
    """
    Get radio log entries from both in-memory logs and log files.
    Deduplicates entries to avoid showing the same line twice.
    
    Args:
        radio: Radio station name (empty string for all radios)
        num_lines: Maximum number of lines to return
        start_datetime: Optional datetime object - only return entries >= this datetime
        reverse: Optional boolean - reverse the order of the logs
    """
    global transcript_log
    all_lines = []
    seen_lines = set()  # Track seen lines to avoid duplicates
    radio_upper = radio.upper()

    # Read from log files first (oldest to newest)
    if os.path.exists(LOG_DIR):
        for filename in sorted(os.listdir(LOG_DIR)):
            if not filename.endswith(".log") or filename.endswith("_ai.log"):
                continue

            # Extract radio name from filename (format: YYYY-MM-DD-{radio}.log)
            if "-" in filename:
                parts = filename.rsplit("-", 1)
                if len(parts) == 2:
                    file_radio = parts[1].replace(".log", "").upper()
                    # Skip if radio specified and doesn't match
                    if radio and not file_radio.startswith(radio_upper):
                        continue
                elif radio:
                    continue
            elif radio:
                continue

            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and line not in seen_lines and _should_include(line, start_datetime):
                            seen_lines.add(line)
                            all_lines.append(line)
            except (IOError, OSError):
                continue

    # Add in-memory logs (these are the most recent, may overlap with latest log file)
    if not radio:
        for lines in transcript_log.values():
            for line in lines:
                if line not in seen_lines and _should_include(line, start_datetime):
                    seen_lines.add(line)
                    all_lines.append(line)
    else:
        matching_radios = []
        if radio_upper in transcript_log:
            matching_radios.append(radio_upper)
        else:
            for name in transcript_log.keys():
                if name.startswith(radio_upper):
                    matching_radios.append(name)

        for radio_name in matching_radios:
            for line in transcript_log[radio_name]:
                if line not in seen_lines and _should_include(line, start_datetime):
                    seen_lines.add(line)
                    all_lines.append(line)

    all_lines = _dedupe_preserve_order(all_lines)

    # With start_datetime: return oldest first but reversed for display (keep behavior)
    return _finalize_lines(
        all_lines,
        num_lines,
        start_datetime=start_datetime,
        reverse=reverse,
        reverse_after_slice=start_datetime is not None,
    )

def get_radio_ai_log(radio="", num_lines=100, start_datetime=None):
    """
    Get AI log entries from both in-memory logs and log files.
    Deduplicates entries to avoid showing the same line twice.
    
    Args:
        radio: Radio station name (empty string for all radios)
        num_lines: Maximum number of lines to return
        start_datetime: Optional datetime object - only return entries >= this datetime
    """
    global ai_log
    all_lines = []
    seen_lines = set()  # Track seen lines to avoid duplicates

    def matches_radio(line):
        if not radio:
            return True
        radio_tag = _extract_radio_tag(line)
        if radio_tag:
            return radio_tag.startswith(radio.upper())
        # Fallback for legacy lines without radio info: include only when radio is empty
        return False

    # Read from AI log files first (oldest to newest)
    if os.path.exists(LOG_DIR):
        for filename in sorted(os.listdir(LOG_DIR)):
            if not filename.endswith("_ai.log"):
                continue

            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if line and matches_radio(line):
                            if line not in seen_lines and _should_include(line, start_datetime):
                                seen_lines.add(line)
                                all_lines.append(line)
            except (IOError, OSError):
                continue

    # Add in-memory AI logs (these are the most recent, may overlap with latest log file)
    for line in ai_log:
        norm = line.rstrip()
        if not matches_radio(norm):
            continue
        if norm not in seen_lines and _should_include(norm, start_datetime):
            seen_lines.add(norm)
            all_lines.append(norm)

    all_lines = _dedupe_preserve_order(all_lines)

    # For start_datetime we keep chronological order; otherwise newest first
    return _finalize_lines(
        all_lines,
        num_lines,
        start_datetime=start_datetime,
        reverse=False,
        reverse_after_slice=False,
    )

def search_radio_log(radio="", keyword="", max_results=50):
    """
    Search through log files for a specific keyword or phrase.
    
    Args:
        radio: Radio station name (empty string for all radios)
        keyword: Keyword or phrase to search for (case-insensitive)
        max_results: Maximum number of results to return
    
    Returns:
        List of matching log lines, sorted by date (most recent first)
    """
    if not keyword:
        return []
    
    keyword_lower = keyword.lower()
    results = []
    
    # Search in-memory logs first
    global transcript_log
    if radio:
        radios_to_search = [r for r in transcript_log.keys() if r.startswith(radio.upper())]
    else:
        radios_to_search = list(transcript_log.keys())
    
    for radio_name in radios_to_search:
        for line in transcript_log[radio_name]:
            if keyword_lower in line.lower():
                results.append(line)
    
    # Search log files on disk
    if os.path.exists(LOG_DIR):
        for filename in sorted(os.listdir(LOG_DIR), reverse=True):
            if not filename.endswith('.log') or filename.endswith('_ai.log'):
                continue
            
            # Extract radio name from filename (format: YYYY-MM-DD-{radio}.log)
            if '-' in filename:
                parts = filename.rsplit('-', 1)
                if len(parts) == 2:
                    file_radio = parts[1].replace('.log', '').upper()
                    # Skip if radio specified and doesn't match
                    if radio and not file_radio.startswith(radio.upper()):
                        continue
                elif radio:
                    # If filename doesn't match expected format and radio is specified, skip
                    continue
            elif radio:
                continue
            
            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and keyword_lower in line.lower():
                            results.append(line)
            except (IOError, OSError):
                continue
            
            # Stop if we have enough results
            if len(results) >= max_results:
                break
    
    # Helper function to extract timestamp for sorting
    def extract_timestamp(line):
        try:
            # Extract date/time from [YYYY-MM-DD HH:MM:SS format
            if '[' in line and ']' in line:
                timestamp_str = line.split('[')[1].split(']')[0].split(' - ')[0]
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass
        return datetime.min
    
    # Deduplicate results - in-memory logs overlap with latest log file entries
    unique_results = []
    seen_lines = set()
    for line in results:
        if line not in seen_lines:
            seen_lines.add(line)
            unique_results.append(line)
    
    # Sort by timestamp (most recent first) and limit results
    unique_results.sort(key=extract_timestamp, reverse=True)
    return unique_results[:max_results]
