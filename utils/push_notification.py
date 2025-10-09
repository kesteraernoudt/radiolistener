import requests, urllib.parse

def send_radio_alert(number, codeword, station="Radio1"):
    # Build the Shortcut URL with parameters
    text_input = f"{number} {codeword} detected on {station}"
    shortcut_url = (
        "shortcuts://run-shortcut?name="
        + urllib.parse.quote("Send RadioListener SMS")
        + "&input="
        + urllib.parse.quote(text_input)
    )

    # Notification text
    message = f"Codeword '{codeword}' detected!\nTap below to send SMS:\n{shortcut_url}"

    # Push to ntfy.sh
    r = requests.post(
        "https://ntfy.sh/RadioListener",  # same channel you subscribed to in the app
        data=message.encode("utf-8"),
        headers={"Title": "Radio Alert"}
    )
    r.raise_for_status()


if __name__ == "__main__":
    send_radio_alert("6693245301", "codeword", "Radio1")
