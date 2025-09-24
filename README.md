## Radiolistener

A small service that captures radio audio clips, processes them with local AI models, and exposes a simple web UI and Telegram integration.

Quick highlights
- Captures audio and saves clips to `clips/`.
- Audio capture and processing code in `audio/`.
- Web routes and UI in `web/` (`web/templates/index.html`).
- Configs live in `config/` (station presets and models).
- Utilities and notifier integrations in `utils/`.

Requirements
- Python 3.9+ (tested on Linux)
- System audio tools depending on your capture backend
- See `requirements.txt` for Python dependencies

Installation (local)
1. Create and activate a virtualenv (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Configuration
- Copy or edit `config/config.json` and the files in `config/` for station-specific settings.
- Model files (`*.pt`) may already be present under `config/` (e.g. `base.pt`, `medium.pt`, `large-v3.pt`) — ensure they are readable by the service.
- Check `fb_config/` for the file browser DB used by the web UI.

`.env` (environment variables)
- The app loads environment variables from a `.env` file at startup. Do NOT commit real secrets to source control. Add a file named `.env` in the project root with entries similar to the example below (replace placeholders with your real values):

```ini
# Telegram bot token (string)
TELEGRAM_BOT_TOKEN="your-telegram-bot-token-here"

# Telegram chat id where alerts are posted (integer or string)
TELEGRAM_CHAT_ID=1234567890

# API key for the generative AI (the code uses this as GEMINI_API_KEY)
GEMINI_API_KEY="your-genai-or-gemini-api-key"

```

Descriptions
- `TELEGRAM_BOT_TOKEN`: Bot token obtained from BotFather (format: 123456:ABC-DEF...).
- `TELEGRAM_CHAT_ID`: The chat (group or private) id to send messages/audio to. Use `@username` for channels or the numeric id for groups.
- `GEMINI_API_KEY`: API key used by `utils/genai.py` to instantiate the GenAI client. Keep this secret.

Running

Run the app directly (development):

```bash
python app.py
```

Run with Docker (recommended for deployment):

```bash
docker-compose up --build
```

Where to look
- Saved audio clips: `clips/`
- Logs: `logs/` (daily logs and AI logs)
- Main entrypoint: `app.py`
- Core modules: `audio/`, `utils/`, `web/`

Development notes
- Add new station configs under `config/` as JSON files. See existing examples (`99.7now.json`, `106.5.json`, `alice.json`).
- Tests: none included — adding small unit tests for `audio/` and `utils/` is a good next step.

Troubleshooting
- Check `logs/` for runtime errors.
- Ensure model files in `config/` are present and have correct permissions.
- If audio capture fails, verify system audio devices and permissions.

Contributing
- Open an issue or submit a PR. Keep changes small and include a short test or manual verification steps.

License
- No license file included in the repository; treat this project as private by default or add a `LICENSE` file.

Contact
- For local developer questions, inspect `utils/telegram_notifier.py` and `utils/telegram_bot.py` for integration examples.

Enjoy!
