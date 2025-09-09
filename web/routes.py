from flask import Flask, render_template, request, jsonify, send_from_directory
import os, json
from utils import logger

app = Flask(__name__, template_folder="templates")

CLIPS_DIR = "clips"
LOGS_DIR = "logs"

listener = None

@app.route("/")
def index():
    return render_template("index.html", config=listener.CONFIG, radio_conf=listener.controller.RADIO_CONF, logs="\n".join(logger.transcript_log[-100:]))

@app.route("/update_config", methods=["POST"])
def update_config():
    CONFIG = request.json
    with open(listener.config_path, "w") as f: json.dump(CONFIG, f, indent=2)
    listener.controller.restart()
    return jsonify(success=True)

@app.route("/update_radio_conf", methods=["POST"])
def update_radio_conf():
    radio_conf = request.json
    with open(listener.controller.radio_conf_path, "w") as f: json.dump(radio_conf, f, indent=2)
    listener.controller.restart()
    return jsonify(success=True)

@app.route("/logs/live")
def live_log():
    return jsonify(logs=logger.transcript_log[-200:])

@app.route('/save_clip', methods=['POST'])
def save_clip():
    listener.controller.processor.do_save_full_clip = 1
    return jsonify({"message": f"Clip saved"})
