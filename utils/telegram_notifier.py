import requests

telegram_token = ""
telegram_chat_id = ""

def init_telegram(token, chat_id):
    global telegram_token, telegram_chat_id
    telegram_token = token
    telegram_chat_id = chat_id

def send_telegram(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        r = requests.post(url, json=payload)
        if r.status_code == 200:
            print("📲 Telegram message sent:", message)
        else:
            print("⚠️ Telegram error:", r.text)
    except Exception as e:
        print("⚠️ Could not send Telegram message:", e)

def send_telegram_audio(filepath, caption="Radio clip"):
    url = f"https://api.telegram.org/bot{telegram_token}/sendAudio"
    try:
        with open(filepath, "rb") as audio_file:
            r = requests.post(
                url,
                data={"chat_id": telegram_chat_id, "caption": caption},
                files={"audio": audio_file}
            )
        if r.status_code == 200:
            print(f"📲 Sent audio clip to Telegram: {filepath}")
        else:
            print("⚠️ Telegram audio error:", r.text)
    except Exception as e:
        print("⚠️ Could not send audio clip:", e)
