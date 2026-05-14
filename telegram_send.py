import os
import cv2
import requests


def save_image(image):
    # Store captured alert images inside this project instead of a developer's
    # old absolute path. Flask/OpenCV can create this folder on first use.
    path_img = os.path.join(os.getcwd(), 'captured_images')
    os.makedirs(path_img, exist_ok=True)

    img_name = 'latest_patient_gesture.png'
    image_path = os.path.join(path_img, img_name)
    cv2.imwrite(image_path, image)

    # requests.post expects a file-like object for multipart uploads.
    files = {'photo': open(image_path, 'rb')}
    return files


def send_msg(caption, files):
    # Keep secrets outside source code. Set these in PowerShell before running:
    # $env:TELEGRAM_BOT_TOKEN="..."
    # $env:TELEGRAM_CHAT_ID="..."
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram alert skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing.")
        files['photo'].close()
        return None

    url = "https://api.telegram.org/bot"
    captions = {}
    file_path = 'caption.txt'
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            captions[i + 1] = line.strip()
    caption = captions.get(caption, "Patient gesture detected")

    try:
        return requests.post(
            url + token + "/sendPhoto",
            params={"chat_id": chat_id, "caption": caption},
            files=files,
            timeout=15,
        )
    finally:
        files['photo'].close()
