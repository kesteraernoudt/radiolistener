import time, json, re
from audio import clip_saver
from datetime import datetime
import numpy as np
from utils import logger, telegram_notifier, genai
import whisper
import threading
import logging
import queue


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
        pre_prompt_file: str = "",
        controller=None,
    ):
        self.radio_conf = radio_conf
        self.whisper_model = whisper.load_model(
            asr_whisper_model, download_root="config"
        )
        self.buffer_seconds = buffer_seconds
        self.buffer_overlap = buffer_overlap
        self.sample_rate = sample_rate
        self.CLIP_DURATION = CLIP_DURATION
        self.genAIHandler = genai.GenAIHandler(GENAI_API_KEY, pre_prompt_file)
        self.controller = controller

        self.rolling_buffer = b""
        self.max_buffer_size = int(sample_rate * CLIP_DURATION * 2)
        self.frame_size = int(sample_rate * buffer_seconds * 2)
        self.step_size = int(sample_rate * (buffer_seconds - buffer_overlap) * 2)
        self.window_start = 0
        self.previous_texts = []
        self.previous_codewords = []
        self.last_alert_time = 0
        self.do_save_full_clip = 0
        self.lock = threading.Lock()
        # Single background transcription thread with queues
        self.transcribe_in = queue.Queue(maxsize=4)
        self.transcribe_out = queue.Queue()
        self.transcriber_thread = threading.Thread(target=self._transcriber_loop, daemon=True)
        self.transcriber_thread.start()
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
        }

        # Internal accumulators for averages
        self._sum_transcribe = 0.0
        self._sum_ai = 0.0
        self._sum_match = 0.0
        self._sum_process = 0.0
        self._count_events = 0
        self._count_ai_events = 0

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

    def send_alert(self, matches, code_word, context=""):
        now = time.time()
        alert_msg = f"🚨 '{matches}' heard - codeword is {code_word}"
        if "SMS_NUMBER" in self.radio_conf:
            alert_msg += f": text to {self.radio_conf['SMS_NUMBER']}"
        elif "CALL_NUMBER" in self.radio_conf:
            alert_msg += f": call {self.radio_conf['CALL_NUMBER']}"
        elif "URL" in self.radio_conf:
            alert_msg += f": submit at {self.radio_conf['URL']}"
        logger.log_event(self.radio_conf["NAME"], alert_msg)
        print(self.radio_conf["NAME"] + ": " + alert_msg)
        if now - self.last_alert_time > 300:  # MIN_ALERT_INTERVAL
            # telegram_notifier.send_telegram(alert_msg)
            ctrl = getattr(self, 'controller', None)
            if ctrl is not None and hasattr(ctrl, 'send_message'):
                ctrl.send_message(
                    self.radio_conf["NAME"]
                    + ": "
                    + alert_msg
                    + (f", context: {context}" if context else "")
                )
            # self.controller.send_sms_message(code_word)
            self.last_alert_time = now
            self.do_save_full_clip = 1

    def handle_match(self, match, text):
        context = f"{" ".join(self.previous_texts[-self.CONTEXT_LEN:]) if self.previous_texts and len(self.previous_texts) > self.CONTEXT_LEN else ''} {text}"
        code_word = self.genAIHandler.generate(context)
        if code_word:
            self.previous_codewords.append(code_word)
            if len(self.previous_codewords) > self.MAX_CODEWORD_STORAGE:
                self.previous_codewords = self.previous_codewords[-self.MAX_CODEWORD_STORAGE :]
            self.send_alert(match, code_word, context)
        else:
            logger.log_event(
                self.radio_conf["NAME"],
                f"No code word found for match: {match}, text was: {context}",
            )
        return code_word

    def process_audio(self, audio_queue):
        last_match = ""
        while getattr(self, 'controller', None) and getattr(self.controller, 'running', False):
            try:
                data = audio_queue.popleft()
            except:
                # Avoid busy-spin when queue is empty; give capture thread time to fill
                time.sleep(0.01)
                continue

            # add incoming data under lock
            with self.lock:
                self.rolling_buffer += data
                # Trim buffer to max size
                if len(self.rolling_buffer) > self.max_buffer_size:
                    trim_amount = len(self.rolling_buffer) - self.max_buffer_size
                    self.rolling_buffer = self.rolling_buffer[-self.max_buffer_size :]
                    self.window_start = max(0, self.window_start - trim_amount)

            # Process as many windows as possible. We grab each frame under lock
            # but do transcription in a single background thread and drain results FIFO.
            while True:
                with self.lock:
                    can_take_frame = self.window_start + self.frame_size <= len(self.rolling_buffer)
                    if can_take_frame:
                        frame_start = self.window_start
                        frame_end = self.window_start + self.frame_size
                        self.window_start += self.step_size
                        buffer = self.rolling_buffer[frame_start:frame_end]
                    else:
                        buffer = None

                if buffer is None:
                    break
                
                # Try to enqueue for transcription without blocking infinitely
                try:
                    self.transcribe_in.put_nowait(buffer)
                except queue.Full:
                    # Stop taking more frames this cycle if transcriber is backlogged
                    break

            # Drain all available results FIFO (single worker preserves order)
            while True:
                try:
                    result = self.transcribe_out.get_nowait()
                except queue.Empty:
                    break

                text = result.get("text", "")
                t_transcribe = result.get("t_transcribe", 0)

                t0_post = time.time()
                t_ai = 0
                t_match = 0

                if text:
                    if last_match:
                        t_ai_start = time.time()
                        self.handle_match(last_match, text)
                        t_ai_end = time.time()
                        t_ai += t_ai_end - t_ai_start
                        last_match = ""
                    if self.do_save_full_clip:
                        with self.lock:
                            snapshot = bytes(self.rolling_buffer)
                            self.do_save_full_clip = 0
                        last_clip_path = clip_saver.save_clip(snapshot)
                        ctrl = getattr(self, 'controller', None)
                        if ctrl is not None and hasattr(ctrl, 'send_audio'):
                            ctrl.send_audio(last_clip_path, caption="")

                    logger.log_event(self.radio_conf.get("NAME", "UNKNOWN"), text)

                    t_match_start = time.time()
                    matches = self.phrase_matches(text, self.radio_conf["PHRASES"])
                    t_match_end = time.time()
                    t_match = t_match_end - t_match_start
                    if matches:
                        t_ai_start = time.time()
                        code_word = self.handle_match(matches, text)
                        t_ai_end = time.time()
                        t_ai += t_ai_end - t_ai_start
                        if not code_word:
                            last_match = matches

                    with self.lock:
                        self.previous_texts.append(text)
                        if len(self.previous_texts) > self.MAX_MSG_STORAGE:
                            self.previous_texts = self.previous_texts[-self.MAX_MSG_STORAGE :]

                t_process = time.time() - t0_post + t_transcribe
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
                logging.debug(
                    f"{self.radio_conf.get('NAME','UNKNOWN')}: process_time={t_process:.2f}, transcribe_time={t_transcribe:.2f}, ai_time={t_ai:.2f}, match_time={t_match:.2f}, buffer_seconds = {self.buffer_seconds}"
                )

    def _transcriber_loop(self):
        while getattr(self, 'controller', None) and getattr(self.controller, 'running', False):
            try:
                buffer_bytes = self.transcribe_in.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                t0 = time.time()
                audio_np = np.frombuffer(buffer_bytes, np.int16).astype("float32") / 32768.0
                whisper_result = self.whisper_model.transcribe(
                    audio_np,
                    fp16=self.whisper_model.device.type == "cuda",
                    no_speech_threshold=0.6,
                    condition_on_previous_text=False,
                )
                # Ensure we treat text as a string
                text_val = whisper_result.get("text", "")
                text = str(text_val).strip() if not isinstance(text_val, str) else text_val.strip()
                t_transcribe = time.time() - t0
                self.transcribe_out.put({"text": text, "t_transcribe": t_transcribe})
            except Exception as e:
                logging.exception(f"Transcriber loop error: {e}")
                self.transcribe_out.put({"text": "", "t_transcribe": 0})
