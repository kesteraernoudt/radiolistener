import os
from datetime import datetime, timedelta
from collections import defaultdict

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

MAX_TRANSCRIPT_LINES = 1000
MAX_AI_LINES = 1000

transcript_log = defaultdict(list)
ai_log = list()

def log_event(radio, msg):
    global transcript_log
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts} - {radio}] {msg}"
    transcript_log[radio.upper()].append(line)
    if len(transcript_log[radio.upper()]) > MAX_TRANSCRIPT_LINES:
        transcript_log[radio.upper()] = transcript_log[radio.upper()][-MAX_TRANSCRIPT_LINES:]
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}-{radio}.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def log_ai_event(msg):
    global ai_log
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    ai_log.append(line)
    if len(ai_log) > MAX_AI_LINES:
        ai_log = ai_log[-MAX_AI_LINES:]
    logfile = os.path.join(LOG_DIR, f"{datetime.now().date()}_ai.log")
    with open(logfile, "a") as f:
        f.write(line + "\n")
    return line

def cleanup_logs(days=7):
    cutoff = datetime.now() - timedelta(days=days)
    for f in os.listdir(LOG_DIR):
        path = os.path.join(LOG_DIR, f)
        if os.path.isfile(path):
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            if mtime < cutoff:
                os.remove(path)

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
    
    # Helper function to extract timestamp for sorting and filtering
    def extract_timestamp(line):
        try:
            if '[' in line and ']' in line:
                timestamp_str = line.split('[')[1].split(']')[0].split(' - ')[0]
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass
        return datetime.min
    
    def should_include(line):
        """Check if line should be included based on start_datetime"""
        if start_datetime is None:
            return True
        line_timestamp = extract_timestamp(line)
        return line_timestamp >= start_datetime
    
    # Read from log files first (oldest to newest)
    if os.path.exists(LOG_DIR):
        file_lines = []
        for filename in sorted(os.listdir(LOG_DIR)):
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
                    continue
            elif radio:
                continue
            
            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and line not in seen_lines and should_include(line):
                            seen_lines.add(line)
                            file_lines.append(line)
            except (IOError, OSError):
                continue
        
        all_lines.extend(file_lines)
    
    # Add in-memory logs (these are the most recent, may overlap with latest log file)
    if not radio:
        # return combined log from all radios
        for lines in transcript_log.values():
            for line in lines:
                if line not in seen_lines and should_include(line):
                    seen_lines.add(line)
                    all_lines.append(line)
    else:
        # Find matching radio(s)
        matching_radios = []
        if radio.upper() in transcript_log:
            matching_radios.append(radio.upper())
        else:
            # see if it's the beginning of a radio name
            for name in transcript_log.keys():
                if name.startswith(radio.upper()):
                    matching_radios.append(name)
        
        for radio_name in matching_radios:
            for line in transcript_log[radio_name]:
                if line not in seen_lines and should_include(line):
                    seen_lines.add(line)
                    all_lines.append(line)
    
    # Sort by timestamp
    # If start_datetime is provided, we want entries right after the timestamp (chronological order)
    # Otherwise, we want the most recent entries (reverse chronological order)
    if start_datetime is not None:
        # Sort chronologically (oldest first) to get entries right after the timestamp
        all_lines.sort(key=extract_timestamp, reverse=False)
        # Take the first num_lines entries (right after the timestamp)
        result = all_lines[:num_lines]
        # Reverse for display (most recent last)
        result.reverse()
        return result
    else:
        # Sort reverse (most recent first) and limit
        all_lines.sort(key=extract_timestamp, reverse=True)
        result = all_lines[:num_lines]
        if reverse:
            result.reverse()
        return result

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
    
    # Helper function to extract timestamp for sorting and filtering
    def extract_timestamp(line):
        try:
            if '[' in line and ']' in line:
                timestamp_str = line.split('[')[1].split(']')[0]
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
        except (ValueError, IndexError):
            pass
        return datetime.min
    
    def should_include(line):
        """Check if line should be included based on start_datetime"""
        if start_datetime is None:
            return True
        line_timestamp = extract_timestamp(line)
        return line_timestamp >= start_datetime
    
    # Read from AI log files first (oldest to newest)
    if os.path.exists(LOG_DIR):
        file_lines = []
        for filename in sorted(os.listdir(LOG_DIR)):
            if not filename.endswith('_ai.log'):
                continue
            
            filepath = os.path.join(LOG_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            # Filter by radio if specified
                            if radio and radio.upper() not in line.upper():
                                continue
                            if line not in seen_lines and should_include(line):
                                seen_lines.add(line)
                                file_lines.append(line)
            except (IOError, OSError):
                continue
        
        all_lines.extend(file_lines)
    
    # Add in-memory AI logs (these are the most recent, may overlap with latest log file)
    for line in ai_log:
        if radio and radio.upper() not in line.upper():
            continue
        if line not in seen_lines and should_include(line):
            seen_lines.add(line)
            all_lines.append(line)
    
    # Sort by timestamp
    # If start_datetime is provided, we want entries right after the timestamp (chronological order)
    # Otherwise, we want the most recent entries (reverse chronological order)
    if start_datetime is not None:
        # Sort chronologically (oldest first) to get entries right after the timestamp
        all_lines.sort(key=extract_timestamp, reverse=False)
        # Take the first num_lines entries (right after the timestamp)
        # Return in chronological order (oldest first) - bot will handle display order
        return all_lines[:num_lines]
    else:
        # Sort reverse (most recent first) and limit
        all_lines.sort(key=extract_timestamp, reverse=True)
        return all_lines[:num_lines]

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
