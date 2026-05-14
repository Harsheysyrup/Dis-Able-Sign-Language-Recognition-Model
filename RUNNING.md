# dis-able: local run notes

## What this project is

dis-able is a Flask web app that streams webcam frames through OpenCV,
MediaPipe, and TensorFlow/Keras models. It has three main demo pages:

- `/handGesture` uses `mp_hand_gesture/` and `gesture.names` to classify simple
  hand gestures such as thumbs up, stop, and peace.
- `/tmp` uses `ann_model.h5` plus `connections.csv` to classify sign-language
  characters from MediaPipe hand-landmark distances.
- `/tmp2` uses `skripsi.h5` to classify patient-care gestures such as pain,
  drink, medication, and help me. This page can optionally send a Telegram photo
  alert if Telegram environment variables are configured.

## Recommended setup

Use Python 3.9 or 3.10 on Windows. Python 3.13 is not a good target for this
project because TensorFlow 2.10.1 and this MediaPipe stack are older.

```powershell
cd "C:\Users\Harshvardhan\Downloads\Translate-Care-main\Translate-Care-main"
py -3.9 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& ".\.venv\Scripts\python.exe" app.py
```

Then open:

- `http://127.0.0.1:5000/`
- `http://127.0.0.1:5000/handGesture`
- `http://127.0.0.1:5000/tmp`
- `http://127.0.0.1:5000/tmp2`

## Optional Telegram alerts

The patient gesture flow no longer uses the committed bot token by default.
Set these only if you want `/tmp2` to send Telegram alerts:

```powershell
$env:TELEGRAM_BOT_TOKEN="your_bot_token"
$env:TELEGRAM_CHAT_ID="your_chat_id"
```

Without those variables, the app still runs and prints a warning instead of
trying to send a Telegram message.

## Optional Gemini AI enhancement

The landing page has an AI Writing Assistant. It works with a local fallback by
default, but real AI rewriting needs a Gemini API key in `.env`:

```text
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
```

Supabase is not required for the current local demo. Add Supabase only when you
want real login/authentication, saved users, stored gesture history, or a
persistent analytics dashboard.
