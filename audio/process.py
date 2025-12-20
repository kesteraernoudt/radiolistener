import time, json, re
from audio import clip_saver
from datetime import datetime
import numpy as np
from utils import logger, telegram_notifier, genai
import threading
import logging
import queue
import whisper
from collections import deque


class StreamProcessor:
    CONTEXT_LEN = 3
    MAX_MSG_STORAGE = 100
    MAX_CODEWORD_STORAGE = 10

    def __init__(
        self,
        radio_conf,
        asr_whisper_model="base",
        buffer_seconds: float = 5.0,
        buffer_overlap: float = 1.0,
        sample_rate: int = 16000,
        CLIP_DURATION: float = 10.0,
        GENAI_API_KEY: str = "",
        GROQ_API_KEY: str = "",
        MISTRAL_API_KEY: str = "",
        AI_PROVIDER: str = "auto",
        pre_prompt_file: str = "",
        controller=None,
        log_enabled: bool = True,
        asr_backend: str = "whisper",
        asr_device: str = "auto",
        asr_compute_type: str = "auto",
        asr_no_speech_threshold: float = 0.6,
        asr_vad_filter: bool = False,
        asr_vad_min_silence_ms: int = 500,
        asr_min_rms: float = 0.0,
        asr_language: str | None = "en",
    ):
        self.radio_conf = radio_conf
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        self.asr_backend = (asr_backend or "whisper").lower()
        self.log_enabled = log_enabled
        self.asr_device = (asr_device or "auto").lower()
        if self.asr_device == "gpu":
            self.asr_device = "cuda"
        self.asr_compute_type = asr_compute_type
        self.asr_no_speech_threshold = asr_no_speech_threshold
        self.asr_vad_filter = asr_vad_filter
        self.asr_vad_min_silence_ms = asr_vad_min_silence_ms
        self.asr_min_rms = asr_min_rms
        self.asr_language = asr_language
        self.whisper_model = None
        self.faster_whisper_model = None
        if self.asr_backend == "faster-whisper":
            self._init_faster_whisper(asr_whisper_model)
        if self.asr_backend == "whisper":
            self.whisper_model = whisper.load_model(
                asr_whisper_model,
                download_root="config",
                device=self.asr_device,
            )
        self.buffer_seconds = buffer_seconds
        self.buffer_overlap = buffer_overlap
        self.sample_rate = sample_rate
        self.CLIP_DURATION = CLIP_DURATION
        self.genAIHandler = genai.GenAIHandler(
            gemini_api_key=GENAI_API_KEY,
            pre_prompt_file=pre_prompt_file,
            groq_api_key=GROQ_API_KEY,
            mistral_api_key=MISTRAL_API_KEY or (getattr(self.controller, "CONFIG", {}) or {}).get("MISTRAL_API_KEY", ""),
            provider=AI_PROVIDER,
        )
        self.controller = controller

        self.rolling_buffer = b""
        # Track when audio enters the pipeline so we can map bytes back to capture time.
        # Each entry is (start_offset, end_offset, captured_at_epoch_seconds)
        self.segment_times = deque()
        # Pending clip requests (each may carry its own timeline/context)
        self.pending_clip_requests = deque()
        self.clip_window_before = CLIP_DURATION / 2.0
        self.clip_window_after = CLIP_DURATION / 2.0
        self.max_buffer_size = int(sample_rate * CLIP_DURATION * 2)
        self.frame_size = int(sample_rate * buffer_seconds * 2)
        self.step_size = int(sample_rate * (buffer_seconds - buffer_overlap) * 2)
        self.window_start = 0
        self.previous_texts = []
        self.previous_codewords = []
        self.last_alert_time = 0
        self.last_match_time = 0.0
        self.last_event_time = 0.0
        self.do_save_full_clip = 0
        self.lock = threading.Lock()
        # Single background transcription thread with queues
        self.transcribe_in = queue.Queue(maxsize=4)
        self.transcribe_out = queue.Queue()
        self.transcriber_thread = threading.Thread(target=self._transcriber_loop, daemon=True)
        self.transcriber_thread.start()
        self.active_transcribes = 0
        self.stats = {
            "t_transcribe": 0.0,
            "t_ai": 0.0,
            "t_match": 0.0,
            "t_process": 0.0,
            "t_max_process": 0.0,
            "t_max_ai": 0.0,
            "avg_transcribe": 0.0,
            "avg_ai": 0.0,
            "avg_match": 0.0,
            "avg_process": 0.0,
            "n_events": 0,
            "n_ai_events": 0,
            "t_transcribe_in_len": 0,
            "t_transcribe_out_len": 0,
            "transcribe_queue_max": self.transcribe_in.maxsize,
            "frames_dropped_backlog": 0,
            "frames_silence_skipped": 0,
            "rolling_buffer_max": self.max_buffer_size,
            "last_event_ts": 0.0,
            "last_match_ts": 0.0,
            "last_alert_ts": 0.0,
            "events_per_min": 0.0,
            "events_per_5min": 0.0,
            "p50_process": 0.0,
            "p90_process": 0.0,
            "p99_process": 0.0,
            "p50_transcribe": 0.0,
            "p90_transcribe": 0.0,
            "p99_transcribe": 0.0,
            "p50_ai": 0.0,
            "p90_ai": 0.0,
            "p99_ai": 0.0,
            "last_t_process": 0.0,
            "last_t_transcribe": 0.0,
            "last_t_ai": 0.0,
        }

        # Internal accumulators for averages
        self._sum_transcribe = 0.0
        self._sum_ai = 0.0
        self._sum_match = 0.0
        self._sum_process = 0.0
        self._count_events = 0
        self._count_ai_events = 0
        # Rolling windows for percentiles and throughput
        self._proc_samples = deque(maxlen=200)
        self._asr_samples = deque(maxlen=200)
        self._ai_samples = deque(maxlen=200)
        self._event_times = deque(maxlen=600)  # store timestamps (epoch seconds)
        self.frames_dropped_backlog = 0
        self.frames_silence_skipped = 0

    def get_stats(self):  # todo, update with processing times
        stats = {
            # "buffer_seconds": self.buffer_seconds,
            # "buffer_overlap": self.buffer_overlap,
            # "sample_rate": self.sample_rate,
            # "clip_duration": self.CLIP_DURATION,
            "rolling_buffer_size": len(self.rolling_buffer),
            # "max_buffer_size": self.max_buffer_size,
            # "frame_size": self.frame_size,
            # "step_size": self.step_size,
            # "window_start": self.window_start,
            "previous_texts_stored": len(self.previous_texts),
            "asr_backend": self.asr_backend,
            "previous_codewords": self._codeword_texts(),
            "recent_codewords": [
                {"word": cw, "ts": ts.isoformat()} for cw, ts in self.previous_codewords[-5:]
            ],
        }
        # Merge stat dictionaries without mutating types that confuse the linter
        merged = {**stats, **self.stats}
        return merged

    def phrase_matches(self, text, phrases):
        for phrase in phrases:
            t = phrase["text"].lower()
            mode = phrase.get("mode", "contains")
            if mode == "contains" and t in text.lower():
                return t
            elif mode == "exact" and text.lower().strip() == t:
                return t
            elif mode == "regex" and re.search(t, text, re.IGNORECASE):
                return t
        return str()

    def send_alert(self, matches, code_word, context="", timeline=None):
        now = time.time()
        alert_msg = f"🚨 '{matches}' heard - codeword is {code_word}"
        if "SMS_NUMBER" in self.radio_conf:
            alert_msg += f": text to {self.radio_conf['SMS_NUMBER']}"
        elif "CALL_NUMBER" in self.radio_conf:
            alert_msg += f": call {self.radio_conf['CALL_NUMBER']}"
        elif "URL" in self.radio_conf:
            alert_msg += f": submit at {self.radio_conf['URL']}"
        timeline_summary = self._format_timeline_summary(timeline or {})
        if timeline_summary:
            alert_msg += f"\n⏱️ {timeline_summary}"
        if self.log_enabled:
            logger.log_event(self.radio_conf["NAME"], alert_msg)
        print(self.radio_conf["NAME"] + ": " + alert_msg)
        if now - self.last_alert_time > 300:  # MIN_ALERT_INTERVAL
            # telegram_notifier.send_telegram(alert_msg)
            ctrl = getattr(self, 'controller', None)
            if ctrl is not None and hasattr(ctrl, 'send_message'):
                if timeline is not None:
                    timeline["telegram_sent_at"] = time.time()
                ctrl.send_message(
                    self.radio_conf["NAME"]
                    + ": "
                    + alert_msg
                    + (f", context: {context}" if context else "")
                )
            # self.controller.send_sms_message(code_word)
            self.last_alert_time = now
            # Request a clip tied to this transcript timeline
            self.request_clip(timeline or {"captured_at": now})

    def _codeword_texts(self):
        return [code_word for code_word, _ in self.previous_codewords]

    def get_codewords(self, limit=None):
        codewords = self._codeword_texts()
        if limit is not None:
            return codewords[-limit:]
        return codewords

    def _clear_codewords_if_stale(self, now: datetime | None = None):
        if not self.previous_codewords:
            return
        now = now or datetime.now()
        last_ts = self.previous_codewords[-1][1]
        if last_ts.date() != now.date():
            self.previous_codewords.clear()
            if self.log_enabled:
                logger.log_event(self.radio_conf.get("NAME", "UNKNOWN"), "Cleared previous codewords after day change")

    def _has_codeword(self, code_word: str) -> bool:
        return any(stored_word == code_word for stored_word, _ in self.previous_codewords)

    def _add_codeword(self, code_word: str, now: datetime | None = None):
        now = now or datetime.now()
        self.previous_codewords.append((code_word, now))
        if len(self.previous_codewords) > self.MAX_CODEWORD_STORAGE:
            self.previous_codewords = self.previous_codewords[-self.MAX_CODEWORD_STORAGE :]

    def _timestamp_for_offset(self, offset: int) -> float | None:
        bytes_per_second = self.sample_rate * 2
        for start, end, ts in self.segment_times:
            if offset < end:
                position = max(0, offset - start)
                return ts + position / bytes_per_second
        return None

    def _add_buffer_segment(self, data: bytes, captured_at: float):
        """
        Track captured timestamps alongside the rolling buffer so we can report
        latency for each clip.
        """
        start_offset = len(self.rolling_buffer)
        self.rolling_buffer += data
        end_offset = start_offset + len(data)
        self.segment_times.append((start_offset, end_offset, captured_at))

        if len(self.rolling_buffer) > self.max_buffer_size:
            trim_amount = len(self.rolling_buffer) - self.max_buffer_size
            self.rolling_buffer = self.rolling_buffer[-self.max_buffer_size :]
            self.window_start = max(0, self.window_start - trim_amount)
            self._trim_segment_times(trim_amount)

    def _trim_segment_times(self, trim_amount: int):
        updated = deque()
        for start, end, ts in self.segment_times:
            start -= trim_amount
            end -= trim_amount
            if end <= 0:
                continue
            updated.append((max(start, 0), end, ts))
        self.segment_times = updated

    def _frame_capture_timestamp(self, frame_start: int, frame_end: int) -> float:
        """
        Approximate the capture time of a frame by taking the first timestamped
        segment that overlaps the requested window.
        """
        for start, end, ts in self.segment_times:
            if end > frame_start:
                return ts
        return time.time()

    def _timestamp_to_offset(self, target_ts: float) -> int | None:
        """
        Map a capture timestamp back to a buffer offset, if the timestamp falls
        within the retained rolling buffer.
        """
        bytes_per_second = self.sample_rate * 2
        for start, end, ts in self.segment_times:
            segment_duration = (end - start) / bytes_per_second
            end_ts = ts + segment_duration
            if ts <= target_ts <= end_ts:
                return start + int((target_ts - ts) * bytes_per_second)
        return None

    def _extract_clip_bytes(
        self,
        timeline: dict | None,
        seconds_before: float | None = None,
        seconds_after: float | None = None,
    ) -> tuple[bytes | None, float | None]:
        """
        Pull a clip from the rolling buffer centered on the provided captured timestamp.
        Returns (audio_bytes, capture_ts_for_filename).
        """
        if timeline is None:
            return None, None
        captured_at = timeline.get("captured_at")
        if not captured_at:
            return None, None

        pre = seconds_before if seconds_before is not None else self.clip_window_before
        post = seconds_after if seconds_after is not None else self.clip_window_after
        bytes_per_second = self.sample_rate * 2

        with self.lock:
            offset = self._timestamp_to_offset(captured_at)
            if offset is None:
                return None, None

            start_offset = max(0, offset - int(pre * bytes_per_second))
            end_offset = min(len(self.rolling_buffer), offset + int(post * bytes_per_second))
            audio_bytes = self.rolling_buffer[start_offset:end_offset]
            capture_ts_for_file = self._timestamp_for_offset(start_offset) or captured_at

        if not audio_bytes:
            return None, None
        return bytes(audio_bytes), capture_ts_for_file

    def request_clip(self, timeline=None):
        """
        Enqueue a clip request that will be fulfilled on the next processing loop.
        """
        self.pending_clip_requests.append(timeline or {})

    def _format_timeline_summary(self, timeline: dict) -> str:
        """
        Turn raw stage timestamps into a human readable latency summary.
        """
        if not timeline:
            return ""
        fmt = lambda ts: datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        captured_at = timeline.get("captured_at")
        pieces = []
        if captured_at:
            pieces.append(f"captured {fmt(captured_at)}")

        def _stage(label, key):
            if not captured_at:
                return
            ts_val = timeline.get(key)
            if ts_val:
                delta_ms = (ts_val - captured_at) * 1000
                pieces.append(f"{label} +{delta_ms:.0f}ms")

        _stage("extracted", "frame_extracted_at")
        _stage("queued", "queued_for_transcription_at")
        _stage("asr_start", "transcription_start_at")
        _stage("asr_end", "transcription_end_at")
        _stage("processed", "processing_completed_at")
        _stage("clip", "clip_saved_at")
        _stage("telegram", "telegram_sent_at")

        if captured_at and timeline.get("processing_completed_at"):
            total_ms = (timeline["processing_completed_at"] - captured_at) * 1000
            pieces.append(f"total {total_ms:.0f}ms")
        return "; ".join(pieces)

    def handle_match(self, match, text, timeline=None):
        prev_context = (
            " ".join(self.previous_texts[-self.CONTEXT_LEN:])
            if self.previous_texts and len(self.previous_texts) > self.CONTEXT_LEN
            else ""
        )
        context = f"{prev_context} {text}".strip()
        self.last_match_time = time.time()
        code_word = self.genAIHandler.generate(context, radio=self.radio_conf.get("NAME", ""))
        if code_word:
            # Skip duplicate code words to avoid repeat alerts/logs
            now = datetime.now()
            self._clear_codewords_if_stale(now)
            if self._has_codeword(code_word):
                return None
            self._add_codeword(code_word, now)
            self.send_alert(match, code_word, context, timeline=timeline)
        else:
            if self.log_enabled:
                logger.log_event(
                    self.radio_conf["NAME"],
                    f"No code word found for match: {match}, text was: {context}",
                )
        return code_word

    def process_audio(self, audio_queue):
        last_match = ""
        while getattr(self, 'controller', None) and getattr(self.controller, 'running', False):
            try:
                packet = audio_queue.popleft()
                got_audio = True
            except IndexError:
                got_audio = False

            if got_audio:
                captured_at = time.time()
                if isinstance(packet, dict):
                    data = packet.get("data", b"")
                    captured_at = packet.get("captured_at", captured_at)
                else:
                    data = packet
                # add incoming data under lock
                with self.lock:
                    self._add_buffer_segment(data, captured_at)

            # Process as many windows as possible. We grab each frame under lock
            # but do transcription in a single background thread and drain results FIFO.
            while True:
                with self.lock:
                    can_take_frame = self.window_start + self.frame_size <= len(self.rolling_buffer)
                    if can_take_frame:
                        frame_start = self.window_start
                        frame_end = self.window_start + self.frame_size
                        frame_ts = self._frame_capture_timestamp(frame_start, frame_end)
                        # Advance immediately; if the queue is full we drop this frame to stay near real-time
                        self.window_start += self.step_size
                        buffer = self.rolling_buffer[frame_start:frame_end]
                    else:
                        buffer = None
                        frame_ts = None

                if buffer is None:
                    break
                
                # Try to enqueue for transcription without blocking infinitely
                try:
                    frame_timeline = {
                        "captured_at": frame_ts or time.time(),
                        "frame_extracted_at": time.time(),
                    }
                    # Optional RMS gate to skip near-silence frames
                    if self._rms_passes(buffer):
                        self.transcribe_in.put_nowait({"buffer": buffer, "timeline": frame_timeline})
                    else:
                        self.frames_silence_skipped += 1
                except queue.Full:
                    # Stop taking more frames this cycle if transcriber is backlogged; window has advanced
                    self.frames_dropped_backlog += 1
                    break

            if not got_audio:
                # Avoid busy-spin when queue is empty; give capture thread time to fill
                time.sleep(0.01)

            # Drain all available results FIFO (single worker preserves order)
            while True:
                try:
                    result = self.transcribe_out.get_nowait()
                except queue.Empty:
                    break

                text = result.get("text", "")
                t_transcribe = result.get("t_transcribe", 0)
                timeline = dict(result.get("timeline", {}) or {})

                t0_post = time.time()
                t_ai = 0
                t_match = 0
                timeline.setdefault("processing_started_at", t0_post)
                timeline.setdefault("captured_at", timeline.get("captured_at", t0_post))
                # Consume any pending generic clip request before we overwrite the flag below
                if self.do_save_full_clip:
                    self.request_clip(timeline)
                    self.do_save_full_clip = 0

                if text:
                    if last_match:
                        t_ai_start = time.time()
                        self.handle_match(last_match, text, timeline=timeline)
                        t_ai_end = time.time()
                        t_ai += t_ai_end - t_ai_start
                        last_match = ""
                    if self.do_save_full_clip:
                        with self.lock:
                            self.do_save_full_clip = 0
                            # legacy flag converted to a clip request using current timeline
                            self.request_clip(timeline)

                    if self.log_enabled:
                        logger.log_event(self.radio_conf.get("NAME", "UNKNOWN"), text)

                    t_match_start = time.time()
                    matches = self.phrase_matches(text, self.radio_conf["PHRASES"])
                    t_match_end = time.time()
                    t_match = t_match_end - t_match_start
                    if matches:
                        t_ai_start = time.time()
                        code_word = self.handle_match(matches, text, timeline=timeline)
                        t_ai_end = time.time()
                        t_ai += t_ai_end - t_ai_start
                        if not code_word:
                            last_match = matches

                    with self.lock:
                        self.previous_texts.append(text)
                        if len(self.previous_texts) > self.MAX_MSG_STORAGE:
                            self.previous_texts = self.previous_texts[-self.MAX_MSG_STORAGE :]

                timeline["processing_completed_at"] = time.time()

                t_process = time.time() - t0_post + t_transcribe
                now = time.time()
                self.stats["t_process"] = t_process
                if t_process > self.stats["t_max_process"]:
                    self.stats["t_max_process"] = t_process
                self.stats["t_transcribe"] = t_transcribe
                self.stats["t_ai"] = t_ai
                if t_ai > self.stats["t_max_ai"]:
                    self.stats["t_max_ai"] = t_ai
                self.stats["t_match"] = t_match
                self.stats["t_transcribe_in_len"] = self.transcribe_in.qsize()
                self.stats["t_transcribe_out_len"] = self.transcribe_out.qsize()
                self.stats["frames_dropped_backlog"] = self.frames_dropped_backlog
                self.stats["frames_silence_skipped"] = self.frames_silence_skipped

                # Update rolling sums and averages
                self._count_events += 1
                self._sum_transcribe += t_transcribe
                self._sum_match += t_match
                self._sum_process += t_process
                self.stats["n_events"] = self._count_events
                self.stats["avg_transcribe"] = (
                    self._sum_transcribe / self._count_events if self._count_events else 0.0
                )
                self.stats["avg_match"] = (
                    self._sum_match / self._count_events if self._count_events else 0.0
                )
                self.stats["avg_process"] = (
                    self._sum_process / self._count_events if self._count_events else 0.0
                )

                # Update AI-only averages and counters only when AI work occurred
                if t_ai > 0:
                    self._sum_ai += t_ai
                    self._count_ai_events += 1
                    self.stats["n_ai_events"] = self._count_ai_events
                    self.stats["avg_ai"] = (
                        self._sum_ai / self._count_ai_events if self._count_ai_events else 0.0
                    )
                # Rolling percentiles and throughput
                self._proc_samples.append(t_process)
                self._asr_samples.append(t_transcribe)
                if t_ai > 0:
                    self._ai_samples.append(t_ai)
                self._event_times.append(now)
                while self._event_times and now - self._event_times[0] > 300:
                    self._event_times.popleft()
                per_min = len([t for t in self._event_times if now - t <= 60])
                per_5min = len(self._event_times) / 5.0 if self._event_times else 0.0
                self.stats["events_per_min"] = per_min
                self.stats["events_per_5min"] = per_5min
                self.stats["p50_process"] = float(np.percentile(self._proc_samples, 50)) if self._proc_samples else 0.0
                self.stats["p90_process"] = float(np.percentile(self._proc_samples, 90)) if self._proc_samples else 0.0
                self.stats["p99_process"] = float(np.percentile(self._proc_samples, 99)) if self._proc_samples else 0.0
                self.stats["p50_transcribe"] = float(np.percentile(self._asr_samples, 50)) if self._asr_samples else 0.0
                self.stats["p90_transcribe"] = float(np.percentile(self._asr_samples, 90)) if self._asr_samples else 0.0
                self.stats["p99_transcribe"] = float(np.percentile(self._asr_samples, 99)) if self._asr_samples else 0.0
                self.stats["p50_ai"] = float(np.percentile(self._ai_samples, 50)) if self._ai_samples else 0.0
                self.stats["p90_ai"] = float(np.percentile(self._ai_samples, 90)) if self._ai_samples else 0.0
                self.stats["p99_ai"] = float(np.percentile(self._ai_samples, 99)) if self._ai_samples else 0.0
                self.stats["last_t_process"] = t_process
                self.stats["last_t_transcribe"] = t_transcribe
                self.stats["last_t_ai"] = t_ai
                self.last_event_time = now
                self.stats["last_event_ts"] = now
                self.stats["last_match_ts"] = self.last_match_time
                self.stats["last_alert_ts"] = self.last_alert_time
                self.stats["last_timeline"] = {
                    "summary": self._format_timeline_summary(timeline),
                    "raw": timeline,
                }
                logging.debug(
                    f"{self.radio_conf.get('NAME','UNKNOWN')}: process_time={t_process:.2f}, transcribe_time={t_transcribe:.2f}, ai_time={t_ai:.2f}, match_time={t_match:.2f}, buffer_seconds = {self.buffer_seconds}"
                )

                # Fulfill any queued clip requests using the latest timeline
                self._fulfill_clip_requests(timeline)

    def _transcriber_loop(self):
        while getattr(self, 'controller', None) and getattr(self.controller, 'running', False):
            try:
                frame_info = self.transcribe_in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if isinstance(frame_info, dict):
                    buffer_bytes = frame_info.get("buffer", b"")
                    timeline = dict(frame_info.get("timeline", {}) or {})
                else:
                    buffer_bytes = frame_info
                    timeline = {}
                timeline.setdefault("queued_for_transcription_at", time.time())
                with self.lock:
                    self.active_transcribes += 1
                t0 = time.time()
                timeline["transcription_start_at"] = t0
                audio_np = np.frombuffer(buffer_bytes, np.int16).astype("float32") / 32768.0
                text = self._run_asr(audio_np)
                end_ts = time.time()
                t_transcribe = end_ts - t0
                timeline["transcription_end_at"] = end_ts
                self.transcribe_out.put({"text": text, "t_transcribe": t_transcribe, "timeline": timeline})
            except Exception as e:
                logging.exception(f"Transcriber loop error: {e}")
                self.transcribe_out.put({"text": "", "t_transcribe": 0, "timeline": timeline if 'timeline' in locals() else {}})
            finally:
                with self.lock:
                    self.active_transcribes = max(0, self.active_transcribes - 1)

    def _fulfill_clip_requests(self, timeline):
        """
        Drain pending clip requests and send any clips built around the provided timeline.
        """
        if not self.pending_clip_requests:
            return

        ctrl = getattr(self, "controller", None)
        while self.pending_clip_requests:
            req_timeline = self.pending_clip_requests.popleft() or {}
            merged_timeline = dict(timeline or {})
            # give request-specific fields priority when present
            merged_timeline.update({k: v for k, v in req_timeline.items() if v is not None})
            audio_bytes, capture_ts = self._extract_clip_bytes(merged_timeline)
            if not audio_bytes:
                continue
            clip_path = clip_saver.save_clip(audio_bytes, capture_ts=capture_ts)
            merged_timeline["clip_saved_at"] = time.time()
            summary = self._format_timeline_summary(merged_timeline)

            if ctrl is not None and hasattr(ctrl, "send_audio"):
                caption = (
                    f"{self.radio_conf.get('NAME','UNKNOWN')} clip"
                    + (f"\n⏱️ {summary}" if summary else "")
                )
                ctrl.send_audio(clip_path, caption=caption)

    def _run_asr(self, audio_np: np.ndarray) -> str:
        if self.asr_backend == "faster-whisper" and self.faster_whisper_model:
            try:
                segments, _ = self.faster_whisper_model.transcribe(
                    audio_np,
                    language=self.asr_language,
                    no_speech_threshold=self.asr_no_speech_threshold,
                    vad_filter=self.asr_vad_filter,
                    vad_parameters={"min_silence_duration_ms": self.asr_vad_min_silence_ms},
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                return text
            except Exception as e:
                logging.exception(f"faster-whisper failed, falling back to whisper: {e}")
                self.asr_backend = "whisper"
        if self.whisper_model:
            whisper_result = self.whisper_model.transcribe(
                audio_np,
                fp16=self.whisper_model.device.type == "cuda",
                language=self.asr_language,
                no_speech_threshold=self.asr_no_speech_threshold,
                condition_on_previous_text=False,
            )
            text_val = whisper_result.get("text", "")
            text = str(text_val).strip() if not isinstance(text_val, str) else text_val.strip()
            return text
        return ""

    def _init_faster_whisper(self, model_name: str):
        """
        Initialize faster-whisper with graceful fallback on unsupported device/compute combinations.
        """
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            logging.exception(f"faster-whisper not installed: {e}")
            self.asr_backend = "whisper"
            return

        device = self.asr_device or "auto"
        compute_type = self.asr_compute_type or "auto"

        def try_load(dev, ctype):
            return WhisperModel(
                model_name,
                device=dev,
                compute_type=ctype,
                download_root="config",
            )

        attempts = []
        # user-requested first
        attempts.append((device, compute_type))
        # fallbacks on same device
        if compute_type != "auto":
            attempts.append((device, "auto"))
        if device == "cuda":
            attempts.append((device, "float16"))

        # Always try CPU-friendly fallbacks as a last resort to avoid cuDNN issues
        cpu_dev = "cpu"
        attempts.extend(
            [
                (cpu_dev, "int8_float32"),
                (cpu_dev, "int8"),
                (cpu_dev, "float32"),
                (cpu_dev, "auto"),
            ]
        )

        for dev, ctype in attempts:
            try:
                self.faster_whisper_model = try_load(dev, ctype)
                self.asr_backend = "faster-whisper"
                if dev != device or ctype != compute_type:
                    logging.info(
                        f"faster-whisper loaded with fallback device/compute: device={dev}, compute={ctype}"
                    )
                return
            except Exception as e:
                logging.warning(f"faster-whisper init failed for device={dev}, compute={ctype}: {e}")
                continue

        logging.error("All faster-whisper init attempts failed; falling back to whisper.")
        self.asr_backend = "whisper"

    def _rms_passes(self, buffer: bytes) -> bool:
        if self.asr_min_rms <= 0:
            return True
        try:
            arr = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
            if arr.size == 0:
                return False
            rms = np.sqrt(np.mean(arr * arr))
            return rms >= self.asr_min_rms
        except Exception:
            return True
