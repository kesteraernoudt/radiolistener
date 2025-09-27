import time, json, re
from audio import clip_saver
from datetime import datetime
import numpy as np
from utils import logger, telegram_notifier, genai
import whisper
import threading
import logging


class StreamProcessor:
    CONTEXT_LEN = 3
    MAX_MSG_STORAGE = 100

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
        self.last_alert_time = 0
        self.do_save_full_clip = 0
        self.lock = threading.Lock()

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
        return stats

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
        if "CALL_NUMBER" in self.radio_conf:
            alert_msg += f": call {self.radio_conf['CALL_NUMBER']}"
        logger.log_event(self.radio_conf["NAME"], alert_msg)
        print(self.radio_conf["NAME"] + ": " + alert_msg)
        if now - self.last_alert_time > 300:  # MIN_ALERT_INTERVAL
            # telegram_notifier.send_telegram(alert_msg)
            self.controller.send_message(
                self.radio_conf["NAME"]
                + ": "
                + alert_msg
                + (f", context: {context}" if context else "")
            )
            # self.controller.send_sms_message(code_word)
            self.last_alert_time = now
            self.do_save_full_clip = 1

    def handle_match(self, match, text):
        context = f"{self.previous_texts[-1] if self.previous_texts else ''} {text}"
        code_word = self.genAIHandler.generate(context)
        if code_word:
            self.send_alert(match, code_word, context)
        else:
            logger.log_event(
                self.radio_conf["NAME"],
                f"No code word found for match: {match}, text was: {self.previous_texts[-1] if self.previous_texts else ''} {text}",
            )
        return code_word

    def process_audio(self, queue):
        last_match = ""
        while self.controller.running:
            try:
                data = queue.popleft()
            except:
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
            # but do transcription and heavy work outside the lock.
            while True:
                with self.lock:
                    if self.window_start + self.frame_size <= len(self.rolling_buffer):
                        frame_start = self.window_start
                        frame_end = self.window_start + self.frame_size
                        # move window_start now to avoid reprocessing while we transcribe
                        self.window_start += self.step_size
                        buffer = self.rolling_buffer[frame_start:frame_end]
                    else:
                        buffer = None

                if buffer is None:
                    break

                t0 = time.time()

                # Heavy work outside the lock
                audio_np = np.frombuffer(buffer, np.int16).astype("float32") / 32768.0
                whisper_result = self.whisper_model.transcribe(audio_np, fp16=False)
                text = whisper_result.get("text", "").strip()

                t_transcribe = time.time() - t0

                t_ai = 0
                t_match = 0

                if text:
                    if last_match:  # there was a previous match but no codeword found
                        t_ai_start = time.time()
                        self.handle_match(last_match, text)
                        t_ai_end = time.time()
                        t_ai = t_ai + t_ai_end - t_ai_start
                        last_match = ""  # reset last match after using it
                    if self.do_save_full_clip:
                        with self.lock:
                            snapshot = bytes(self.rolling_buffer)
                            self.do_save_full_clip = 0
                        last_clip_path = clip_saver.save_clip(snapshot)
                        self.controller.send_audio(last_clip_path, caption="")

                    logger.log_event(self.radio_conf["NAME"], text)

                    # determine matches and update previous_texts under lock
                    t_match_start = time.time()
                    matches = self.phrase_matches(text, self.radio_conf["PHRASES"])
                    t_match_end = time.time()
                    t_match = t_match_end - t_match_start
                    if matches:
                        t_ai_start = time.time()
                        code_word = self.handle_match(matches, text)
                        t_ai_end = time.time()
                        t_ai = t_ai + t_ai_end - t_ai_start
                        if not code_word:  # no codeword found, save for next round
                            last_match = matches

                    with self.lock:
                        self.previous_texts.append(text)
                        if len(self.previous_texts) > self.MAX_MSG_STORAGE:
                            self.previous_texts = self.previous_texts[
                                -self.MAX_MSG_STORAGE :
                            ]

                t_process = time.time() - t0
                logging.debug(
                    f"{self.radio_conf.get('NAME','UNKNOWN')}: process_time={t_process:.2f}, transcribe_time={t_transcribe:.2f}, ai_time={t_ai:.2f}, match_time={t_match:.2f}, buffer_seconds = {self.buffer_seconds}"
                )
