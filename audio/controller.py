import threading
import queue
import json
from audio import capture, process
from utils import telegram_notifier, telegram_bot

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
        self.load_configs()

    def load_configs(self):
        with open(self.radio_conf_path) as f:
            self.RADIO_CONF = json.load(f)

    def start(self):
        if self.running:
            return
        self.load_configs()
        self.running = True
        #telegram_notifier.init_telegram(self.CONFIG["TELEGRAM_BOT_TOKEN"], self.CONFIG["TELEGRAM_CHAT_ID"])
        #telegram_notifier.send_telegram(f"✅ Radio Listener started for {self.RADIO_CONF['NAME']}.")
        #self.send_message(f"✅ Radio Listener started for {self.RADIO_CONF['NAME']}.")
        self.capture_thread = threading.Thread(target=self._start_capture, daemon=True)
        self.process_thread = threading.Thread(target=self._start_processing, daemon=True)
        self.capture_thread.start()
        self.process_thread.start()

    def stop(self):
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()
        if self.process_thread:
            self.process_thread.join()

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
            self
        )
        self.processor.process_audio(self.audio_queue)
