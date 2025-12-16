#!/usr/bin/python3

from flask import Flask
import threading
import json
from web import routes
from audio.controller import RadioController
from audio import process, clip_saver
from utils.telegram_bot import TelegramBot
from utils import logger
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO)


class RadioListener:
    def __init__(self, config_path):
        super().__init__()
        self.config_path = config_path
        self.controllers = {}  # key: station name, value: RadioController instance
        self.flask_thread = None
        self.CONFIG = None
        self.telegramBot = None

    def controller(self, radio_name=""):
        if not radio_name:
            # return the first controller for now
            return next(iter(self.controllers.values()), None)
        if radio_name.upper() in self.controllers:
            return self.controllers[radio_name.upper()]
        # see if it's the beginning of a radio name
        for name, ctrl in self.controllers.items():
            if name.startswith(radio_name.upper()):
                return ctrl
        return None

    def start(self):
        with open(self.config_path) as f:
            self.CONFIG = json.load(f)
        load_dotenv()  # Load environment variables from .env file
        self.CONFIG["TELEGRAM_BOT_TOKEN"] = os.getenv("TELEGRAM_BOT_TOKEN")
        self.CONFIG["TELEGRAM_CHAT_ID"] = os.getenv("TELEGRAM_CHAT_ID")
        self.CONFIG["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
        self.CONFIG["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY") or ""
        self.CONFIG["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY") or ""
        self.CONFIG["AI_PROVIDER"] = os.getenv("AI_PROVIDER", self.CONFIG.get("AI_PROVIDER", "auto"))
        logger.run_startup_cleanup(self.CONFIG.get("KEEP_DAYS", 7), clip_saver.CLIP_DIR)
        self.telegramBot = TelegramBot(self.CONFIG["TELEGRAM_BOT_TOKEN"], self)
        for radio_conf_path in self.CONFIG["RADIO_CONFIGS"]:
            radio_conf_path = os.path.join("config", radio_conf_path)
            with open(radio_conf_path) as rf:
                radio_conf = json.load(rf)
                name = radio_conf["NAME"].upper()
                self.controllers[name] = RadioController(
                    self.CONFIG, radio_conf_path, self
                )
                self.controllers[name].start()
        # self.controller = RadioController(self.CONFIG, "106.5.json", self)
        # self.controller.start()
        self.flask_thread = threading.Thread(target=self._start_flask, daemon=True)
        self.flask_thread.start()
        self.telegramBot.bot_main()

    def get_recent_texts(self, num_lines=10, radio=""):
        controller = self.controller(radio)
        if controller is None or controller.processor is None:
            return ""
        return "\n".join(controller.processor.previous_texts[-num_lines:])

    def _start_flask(self):
        routes.listener = self
        routes.app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


if __name__ == "__main__":
    listener = RadioListener("config/config.json")
    listener.start()
