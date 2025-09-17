import threading
import queue
import json
from audio import capture, process
from utils import logger, telegram_notifier, telegram_bot


class ExceptionThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exception = None

    def run(self):
        try:
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            self.exception = e

class RadioController:
    def __init__(self, config, radio_conf_path, listener):
        self.radio_conf_path = radio_conf_path
        self.audio_queue = queue.Queue()
        self.capture_thread = None
        self.process_thread = None
        self.processor = None
        self.running = False
        self.CONFIG = config
        self.listener = listener
        self.RADIO_CONF = {}
        self.monitor_thread = None
        self.load_configs()

    def load_configs(self):
        with open(self.radio_conf_path) as f:
            self.RADIO_CONF = json.load(f)

    def start(self):
        if self.running:
            return
        self.load_configs()
        self.running = True
        self.capture_thread = ExceptionThread(target=self._start_capture, daemon=True)
        self.process_thread = ExceptionThread(target=self._start_processing, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()
        self.monitor_thread = threading.Thread(target=self._monitor_threads, daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        if self.process_thread:
            self.process_thread.join(timeout=5)
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    def _monitor_threads(self):
        while self.running:
            # Monitor capture thread
            if self.capture_thread and (self.capture_thread.exception or not self.capture_thread.is_alive()):
                if self.capture_thread.exception:
                    print(f"Capture thread crashed: {self.capture_thread.exception}. Restarting...")
                    logger.log_event(f"{self.RADIO_CONF.get('NAME','UNKNOWN')}","Capture thread crashed: {self.capture_thread.exception}. Restarting...")
                else:
                    print("Capture thread stopped unexpectedly. Restarting...")
                    logger.log_event(f"{self.RADIO_CONF.get('NAME','UNKNOWN')}","Capture thread stopped unexpectedly. Restarting...")
                self.capture_thread = ExceptionThread(target=self._start_capture, daemon=True)
                self.capture_thread.start()
            # Monitor process thread
            if self.process_thread and (self.process_thread.exception or not self.process_thread.is_alive()):
                if self.process_thread.exception:
                    print(f"Process thread crashed: {self.process_thread.exception}. Restarting...")
                    logger.log_event(f"{self.RADIO_CONF.get('NAME','UNKNOWN')}","Process thread crashed: {self.process_thread.exception}. Restarting...")
                else:
                    print("Process thread stopped unexpectedly. Restarting...")
                    logger.log_event(f"{self.RADIO_CONF.get('NAME','UNKNOWN')}","Process thread stopped unexpectedly. Restarting...")
                self.process_thread = ExceptionThread(target=self._start_processing, daemon=True)
                self.process_thread.start()
            threading.Event().wait(5)

    def restart(self):
        self.stop()
        self.start()

    def send_message(self, text):
        self.listener.telegramBot.send_message(text)

    def send_audio(self, file_path, caption=""):
        self.listener.telegramBot.send_audio(file_path, caption)

    def send_sms_message(self, text):
        self.listener.telegramBot.send_sms_message(self.RADIO_CONF["SMS_NUMBER"], text)

    def _start_capture(self):
        capture.capture_stream(self.audio_queue, self.RADIO_CONF["STREAM_URL"], self)

    def _start_processing(self):
        self.processor = process.StreamProcessor(
            self.RADIO_CONF,
            self.CONFIG["WHISPER_MODEL"],
            self.CONFIG["BUFFER_SECONDS"],
            self.CONFIG["BUFFER_OVERLAP"],
            16000,
            self.CONFIG["CLIP_DURATION"],
            self.CONFIG["GEMINI_API_KEY"],
            self.CONFIG["AI_PRE_PROMPT_FILE"],
            self
        )
        self.processor.process_audio(self.audio_queue)
