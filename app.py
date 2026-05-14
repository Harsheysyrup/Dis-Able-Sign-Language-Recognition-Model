import os
import time

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, Response, send_from_directory, jsonify, request
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

from tensorflow import keras
from generate_csv import get_connections_list, get_distance

from keypoint_detection import mp_hands, mediapipe_detection, draw_landmarks, extract_keypoints
from telegram_send import save_image, send_msg
from vis_prediction import visualize_prediction
from model import model as md

from tensorflow.keras.models import load_model


load_dotenv()

# This project keeps its HTML files in the repository root instead of the
# conventional Flask "templates/" folder, so point Flask at "." explicitly.
app = Flask(__name__, template_folder='.', static_folder='static')


def locally_enhance_text(raw_text, mode='improve'):
    """Small deterministic fallback when Gemini quota/network is unavailable."""
    cleaned = ' '.join(raw_text.strip().split())
    lowered = cleaned.lower()

    phrase_map = {
        'need help hospital': 'Hello, I need assistance at the hospital. Could someone please help me?',
        'need help': 'Hello, I need help. Could someone please assist me?',
        'need water': 'Hello, I need some water, please.',
        'need medicine': 'Hello, I need my medication, please.',
        'call nurse': 'Hello, could you please call the nurse?',
        'pain': 'Hello, I am in pain and need assistance.',
        'doctor': 'Hello, I would like to speak with a doctor.',
    }

    improved = phrase_map.get(lowered)
    if not improved:
        improved = cleaned[:1].upper() + cleaned[1:]
        if not improved.endswith(('.', '!', '?')):
            improved += '.'

    if mode == 'email':
        return (
            "Hello,\n\n"
            f"{improved}\n\n"
            "Please let me know how you can help.\n\n"
            "Thank you."
        )
    return improved

# handGesture

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

def handGesture_gen_frames():
    # Open the default webcam. OpenCV uses device index 0 for the built-in or
    # first connected camera on most laptops.
    cap = cv2.VideoCapture(0)
    while True:
        # Read, mirror, and convert each webcam frame before passing it to
        # MediaPipe. MediaPipe expects RGB images; OpenCV captures BGR images.
        success, frame = cap.read()
        if not success:
            break

        x, y, c = frame.shape

        # Mirror the frame so the preview behaves like a normal selfie camera.
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks, then feed the 21 landmark coordinates into the
        # saved TensorFlow gesture classifier.
        result = hands.process(framergb)

        # print(result)

        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames helps users place their hand in a
                # position the model can understand.
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # The model returns class probabilities; argmax selects the most
                # likely gesture label from gesture.names.
                prediction = model.predict([landmarks])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]

        # Encode the annotated frame as JPEG bytes for Flask's MJPEG stream.
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0,0,255), 2, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # release the webcam and destroy all active windows
    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Return a lightweight prediction response for the browser translator.

    The existing production-style camera demos stream predictions through
    /handGesture_video_feed and /video_feed22. This JSON endpoint gives the new
    landing-page UI something simple to call while keeping room to plug in a
    real frame-to-letter model later.
    """
    payload = request.get_json(silent=True) or {}
    image_data = payload.get('image', '')

    # Simulate a stable prediction from the frame string length. Replace this
    # block with real image decoding + model inference when you have a trained
    # letter model exposed for single-frame prediction.
    labels = ['A', 'E', 'H', 'L', 'O', 'R', 'S', 'T']
    seed = len(image_data) + int(time.time())
    letter = labels[seed % len(labels)]
    confidence = 88 + (seed % 10)

    return jsonify({
        'letter': letter,
        'confidence': confidence,
        'word': payload.get('word', ''),
        'sentence': payload.get('sentence', '')
    })

@app.route('/enhance-text', methods=['POST'])
def enhance_text():
    """Improve text with Gemini when GEMINI_API_KEY is configured.

    Without an API key, return a local fallback so the frontend remains usable
    during demos and development.
    """
    payload = request.get_json(silent=True) or {}
    raw_text = (payload.get('text') or '').strip()
    mode = payload.get('mode', 'improve')
    if not raw_text:
        return jsonify({'enhanced_text': '', 'source': 'empty'})

    api_key = (os.getenv('GEMINI_API_KEY') or '').strip()
    model_name = (os.getenv('GEMINI_MODEL') or 'gemini-2.0-flash').strip()
    if not api_key:
        return jsonify({
            'enhanced_text': locally_enhance_text(raw_text, mode),
            'source': 'local-fallback',
            'message': 'Add GEMINI_API_KEY to .env for Gemini responses.'
        })

    task = (
        "Generate a concise, professional email from this short sign-language message:"
        if mode == 'email'
        else "Correct grammar and expand this short sign-language message into one clear sentence:"
    )
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    try:
        response = requests.post(
            url,
            params={'key': api_key},
            json={'contents': [{'parts': [{'text': f"{task}\n\n{raw_text}"}]}]},
            timeout=20,
        )
        if not response.ok:
            print(f"Gemini HTTP status: {response.status_code}")
            fallback_source = 'gemini-quota-fallback' if response.status_code == 429 else 'gemini-fallback'
            return jsonify({
                'enhanced_text': locally_enhance_text(raw_text, mode),
                'source': fallback_source,
                'message': 'Gemini is currently unavailable, so dis-able used the local enhancer.'
            })
        response.raise_for_status()
        data = response.json()
        enhanced = data['candidates'][0]['content']['parts'][0]['text'].strip()
        return jsonify({'enhanced_text': enhanced, 'source': 'gemini'})
    except Exception as exc:
        print(f"Gemini enhancement failed: {exc.__class__.__name__}")
        return jsonify({
            'enhanced_text': locally_enhance_text(raw_text, mode),
            'source': 'gemini-fallback',
            'message': 'Gemini request failed, so dis-able used the local enhancer.'
        })

@app.route('/<path:filename>')
def serve_root_asset(filename):
    """Serve the legacy root-level HTML, CSS, JS, and image files.

    The original project links files like "homepage.css" and "about.html"
    directly from HTML. This route keeps those links working without moving the
    whole project into Flask's templates/static layout.
    """
    allowed_extensions = ('.html', '.css', '.js', '.jpg', '.jpeg', '.png')
    if filename.endswith(allowed_extensions):
        return send_from_directory('.', filename)
    return render_template('index.html'), 404

@app.route('/handGesture')
def handGesture():
    return render_template('index1.html')

@app.route('/handGesture_video_feed')
def handGesture_video_feed():
    return Response(handGesture_gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Character ASL 
def get_sign_list():
    # The CSV stores one row per training sample and the SIGN column is the
    # label. Keeping labels in CSV order keeps model outputs aligned to names.
    df = pd.read_csv('connections.csv', index_col=0)
    return df['SIGN'].unique()

sign_list = get_sign_list()
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
connections_dict = get_connections_list()
model1 = keras.models.load_model('ann_model.h5')

def gen_frames2():  
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
        while True:
            # Capture a frame, convert it for MediaPipe, and mirror it for a
            # user-friendly webcam preview.
            ret, frame = cap.read()
            if not ret:
                break
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)

            # Get result
            results = hands.process(image)
            if not results.multi_hand_landmarks:
                # If no hand detected, then just display the webcam frame
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            else:
                # Draw the detected hand skeleton so the user can see what the
                # classifier is reading.
                mp_drawing.draw_landmarks(
                    image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS
                )

                # Convert raw hand landmarks into distance features. Distances
                # make the classifier less sensitive to where the hand appears
                # in the frame.
                coordinates = results.multi_hand_landmarks[0].landmark
                data = []
                for _, values in connections_dict.items():
                    data.append(get_distance(coordinates[values[0]], coordinates[values[1]]))
                
                # Scale features into a stable range before prediction.
                data = np.array([data])
                max_value = data[0].max()
                if max_value:
                    data[0] /= max_value

                # Run the ANN model and map the winning output neuron to a sign.
                pred = np.array(model1(data))
                pred = sign_list[pred.argmax()]

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # Display text showing prediction
                image = cv2.putText(
                    image, pred, (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 
                    (255, 0, 0), 2
                )

                ret, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/tmp')
def tmp():
    """Video streaming home page."""
    return render_template('index2.html')

def process_image(image):
    # Write code here to process the image
    pass

def gen():
    while True:
        frame = yield
        process_image(frame)

@app.route('/video_feed22')
def video_feed22():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames2(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/tmp2')
def tmp2():
    return render_template('index3.html')


# patient Gesture
def gen3():
    # The patient gesture model uses short sequences of 10 frames. Once the same
    # high-confidence label is observed repeatedly, the app can send an alert.
    md.load_weights('skripsi.h5')
    sequence = []
    predictions = []
    threshold = 0.7
    output_label_counter = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, 800)
    cap.set(4, 600)

    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():

            start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, hands)
            if results.multi_hand_landmarks:
                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)
                # Keep only the newest 10 frames because the LSTM was trained on
                # 10-frame sequences.
                sequence.append(keypoints)
                sequence = sequence[-10:]

                if len(sequence) == 10:
                    # Predict a patient-care phrase from the recent motion
                    # sequence and draw the strongest predictions on the frame.
                    res = md.predict(np.expand_dims(sequence, axis=0))[0]
                    output_label = np.argmax(res)
                    predictions.append(output_label)
                    visualize_prediction(image, res)

                    if res[np.argmax(res)] > threshold:
                        for label in range(24):
                            if output_label == label:
                                output_label_counter += 1
                                if output_label_counter >= 50:
                                    cv2.putText(image, 'PESAN DIKIRIM', (250, 200),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
                                    files = save_image(image)
                                    send_msg(output_label + 1, files)
                                    output_label_counter = 0
                    else:
                        output_label_counter = 0

            end = time.time()
            totalTime = end - start

            fps = 1 / totalTime
            cv2.putText(image, f'FPS: {int(fps)}', (550, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2,
                        cv2.LINE_AA)
            frame = cv2.imencode('.jpg', image)[1].tobytes()
            yield b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            key = cv2.waitKey(20)
            if key == 27:
                break



@app.route('/templates/<filename>')
def serve_image(filename):
    # index3.html was written to request /templates/shc.jpg even though the
    # image is stored in the project root. Serve it from here for compatibility.
    return send_from_directory('.', filename, mimetype='image/jpeg')


@app.route('/video_feed33')
def video_feed33():
    return Response(gen3(), mimetype='multipart/x-mixed-replace; boundary=frame')



if __name__ == '__main__':
    # Keep the local server single-process. The debug reloader starts a second
    # Python process, which is awkward with webcam handles and background runs.
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=False, use_reloader=False)
