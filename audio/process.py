import time, json, re
from audio import clip_saver
from datetime import datetime
import numpy as np
from utils import logger, telegram_notifier, genai
import whisper
import threading

class StreamProcessor:
    CONTEXT_LEN = 3
    MAX_MSG_STORAGE = 100

    def __init__(self, radio_conf, asr_whisper_model="base",
                 buffer_seconds: float = 5.0, buffer_overlap: float = 1.0,
                 sample_rate: int = 16000, 
                 CLIP_DURATION: float = 10.0,
                 GENAI_API_KEY: str = "",
                 pre_prompt_file: str = "",
                 controller=None):
        self.radio_conf = radio_conf
        self.whisper_model = whisper.load_model(asr_whisper_model, download_root='config')
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
        logger.log_event(self.radio_conf['NAME'], alert_msg)
        print(self.radio_conf['NAME'] + ": " + alert_msg)
        if now - self.last_alert_time > 300:  # MIN_ALERT_INTERVAL
            #telegram_notifier.send_telegram(alert_msg)
            self.controller.send_message(self.radio_conf['NAME'] + ": " + alert_msg + (f", context: {context}" if context else ""))
            #self.controller.send_sms_message(code_word)
            self.last_alert_time = now
            self.do_save_full_clip = 1

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
                    self.rolling_buffer = self.rolling_buffer[-self.max_buffer_size:]
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

                # Heavy work outside the lock
                audio_np = np.frombuffer(buffer, np.int16).astype("float32") / 32768.0
                whisper_result = self.whisper_model.transcribe(audio_np, fp16=False)
                text = whisper_result.get("text", "").strip()

                if text:
                    # handle last_match branch (use last_match variable)
                    if last_match:
                        context = f"{self.previous_texts[-1] if self.previous_texts else ''} {text}"
                        code_word = self.genAIHandler.generate(context)
                        if code_word:
                            self.send_alert(last_match, code_word, context)
                        else:
                            logger.log_event(self.radio_conf['NAME'], f"No code word found for match: {last_match}, text was: {self.previous_texts[-1] if self.previous_texts else ''} {text}")
                        last_match = ""  # reset last match after using it
                    if self.do_save_full_clip:
                        with self.lock:
                            snapshot = bytes(self.rolling_buffer)
                            self.do_save_full_clip = 0
                        last_clip_path = clip_saver.save_clip(snapshot)
                        self.controller.send_audio(last_clip_path, caption="")

                    logger.log_event(self.radio_conf['NAME'], text)

                    # determine matches and update previous_texts under lock
                    matches = self.phrase_matches(text, self.radio_conf["PHRASES"])
                    if matches:
                        context = f"{self.previous_texts[-1] if self.previous_texts else ''} {text}"
                        code_word = self.genAIHandler.generate(context)
                        if code_word:
                            self.send_alert(matches, code_word, context)
                        else:
                            logger.log_event(self.radio_conf['NAME'], f"No code word found for match: {matches}, text was: {self.previous_texts[-1] if self.previous_texts else ''} {text}")
                        last_match = matches

                    with self.lock:
                        self.previous_texts.append(text)
                        if len(self.previous_texts) > self.MAX_MSG_STORAGE:
                            self.previous_texts = self.previous_texts[-self.MAX_MSG_STORAGE:]

