from flask import Flask, render_template, request, jsonify
import os, json
from utils import logger
import logging

app = Flask(__name__, template_folder="templates")

log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

listener = None  # Set this from app.py

# todo: logging per radio get_radio_log


@app.route("/")
def index():
    radios = []
    for name, ctrl in listener.controllers.items():
        radios.append(
            {
                "name": name,
                "conf": ctrl.RADIO_CONF,
                "stream_url": ctrl.RADIO_CONF.get("STREAM_URL", ""),
                "phrases": ctrl.RADIO_CONF.get("PHRASES", []),
                "logs": logger.get_radio_log(name, 100, reverse=True),
                "ai_logs": list(reversed(logger.get_radio_ai_log(name, 100))),
                "stats": ctrl.get_stats(),
            }
        )
    return render_template("index.html", config=listener.CONFIG, radios=radios)


@app.route("/update_config", methods=["POST"])
def update_config():
    CONFIG = request.json
    del CONFIG["TELEGRAM_BOT_TOKEN"]
    del CONFIG["TELEGRAM_CHAT_ID"]
    del CONFIG["GEMINI_API_KEY"]
    with open(listener.config_path, "w") as f:
        json.dump(CONFIG, f, indent=2)
    for ctrl in listener.controllers.values():
        ctrl.restart()
    return jsonify(success=True)


@app.route("/radio/<name>/update_conf", methods=["POST"])
def update_radio_conf(name):
    radio_conf = request.json
    ctrl = listener.controller(name)
    if not ctrl:
        return jsonify(success=False, error="Radio not found"), 404
    with open(ctrl.radio_conf_path, "w") as f:
        json.dump(radio_conf, f, indent=2)
    ctrl.restart()
    return jsonify(success=True)

@app.route("/radio/<name>/restart", methods=["POST"])
def restart_radio(name):
    ctrl = listener.controller(name)
    if not ctrl:
        return jsonify(success=False, error="Radio not found"), 404
    ctrl.restart()
    return jsonify(success=True)

@app.route("/radio/logs", defaults={"name": ""})
@app.route("/radio/<name>/logs")
def radio_logs(name):
    logs = logger.get_radio_log(name, 200, reverse=True)
    return jsonify(logs=logs)

@app.route("/radio/<name>/ai_logs")
def radio_ai_logs(name):
    logs = list(reversed(logger.get_radio_ai_log(name, 200)))
    return jsonify(logs=logs)

@app.route("/radio/<name>/codewords")
def radio_codewords(name):
    ctrl = listener.controller(name)
    if not ctrl or not ctrl.processor:
        return jsonify(success=False, error="Radio not found"), 404
    codewords = ctrl.processor.get_codewords()
    return jsonify(codewords=codewords)


@app.route("/radio/<name>/save_clip", methods=["POST"])
def save_clip(name):
    ctrl = listener.controller(name)
    if not ctrl:
        return jsonify(success=False, error="Radio not found"), 404
    ctrl.processor.do_save_full_clip = 1
    return jsonify({"message": f"Clip saved for {name}"})


@app.route("/radio/<name>/stats")
def get_radio_stats(name):
    ctrl = listener.controller(name)
    if not ctrl:
        return jsonify(success=False, error="Radio not found"), 404
    stats = ctrl.get_stats()
    return jsonify(stats=stats)

@app.route("/radio/<name>/stats/realtime")
def get_realtime_stats(name):
    """Get real-time stats for dashboard visualizations"""
    ctrl = listener.controller(name)
    if not ctrl:
        return jsonify(success=False, error="Radio not found"), 404
    
    stats = ctrl.get_stats()
    import time
    
    # Add timestamp for real-time updates
    stats_with_timestamp = {
        **stats,
        "timestamp": time.time(),
        "timestamp_readable": time.strftime("%H:%M:%S")
    }
    
    return jsonify(stats=stats_with_timestamp)
