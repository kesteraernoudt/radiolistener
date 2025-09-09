from flask import Flask
import threading
import json
from web import routes
from audio.controller import RadioController
from audio import process
from utils.telegram_bot import TelegramBot
from dotenv import load_dotenv
import os

class RadioListener():
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.controller = None
        self.flask_thread = None
        self.CONFIG = None
        self.telegramBot = None

    def start(self):
        with open(self.config_path) as f:
            self.CONFIG = json.load(f)
        load_dotenv()  # Load environment variables from .env file
        self.CONFIG["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.CONFIG["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
        self.CONFIG["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
        self.telegramBot = TelegramBot(listener.CONFIG["TELEGRAM_BOT_TOKEN"], self)
        # for now only one controller
        self.controller = RadioController(self.CONFIG, "106.5.json", self)
        self.controller.start()
        self.flask_thread = threading.Thread(target=self._start_flask, daemon=True)
        self.flask_thread.start()
        self.telegramBot.bot_main()

    def get_recent_texts(self, num_lines=10, radio=""):
        # for now ignore radio and just return texts from current radio
        return "\n".join(self.controller.processor.previous_texts[-num_lines:])
    
    def _start_flask(self):
        routes.listener = self
        routes.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)

if __name__ == "__main__":
    listener = RadioListener("config.json")
    listener.start()
