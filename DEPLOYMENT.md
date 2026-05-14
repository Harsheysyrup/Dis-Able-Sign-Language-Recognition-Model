# dis-able deployment guide

## Important deployment reality

The main landing page can be deployed like a normal Flask app. The old OpenCV
streaming endpoints (`/handGesture`, `/tmp`, `/tmp2`) use the server machine's
camera with `cv2.VideoCapture(0)`. On cloud hosting, there is usually no server
webcam, so those routes are mainly for local laptop demos.

For a public production version, keep webcam capture in the browser and send
frames to `/predict`. That is the correct architecture for deployed use.

## Environment variables

Set these on the hosting platform, not in a committed `.env` file:

```text
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-2.0-flash
```

Optional:

```text
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Render deployment

1. Push the project to GitHub.
2. Create a new Render Web Service.
3. Select Python environment.
4. Build command:

```bash
pip install -r requirements.txt
```

5. Start command:

```bash
gunicorn app:app --timeout 180
```

6. Add environment variables from above.

## Railway deployment

1. Push to GitHub.
2. Create a Railway project from the repo.
3. Add environment variables.
4. Railway should detect the `Procfile`. If not, set start command:

```bash
gunicorn app:app --timeout 180
```

## Local run

```powershell
cd "C:\Users\Harshvardhan\Downloads\Translate-Care-main\Translate-Care-main"
& ".\.venv\Scripts\python.exe" app.py
```

Open:

```text
http://127.0.0.1:5000/
```
